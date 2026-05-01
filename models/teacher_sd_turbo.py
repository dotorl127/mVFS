# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from .pose_guider import PoseGuider
from .id_adapter import IDAdapter


@dataclass
class TeacherOutput:
    model_pred: torch.Tensor
    target: torch.Tensor
    noisy_latents: torch.Tensor
    latents: torch.Tensor
    timesteps: torch.Tensor
    pred_x0: Optional[torch.Tensor] = None
    recon_image: Optional[torch.Tensor] = None


class MVFSTeacherSDTurbo(nn.Module):
    """
    Teacher 입력 구조 (수정 후)
      - main latent branch : noisy clean latent x_t (4ch)
      - pose branch        : landmark_map -> PoseGuider -> residual add (4ch)
      - blur branch        : blur RGB -> VAE latent z_blur (4ch)
      - final UNet input   : concat(x_t + pose_residual, z_blur) = 8ch
      - ID branch          : id_embed -> IDAdapter -> extra cross-attention tokens

    즉 landmark는 PoseGuider 전용, blur condition은 UNet 입력 concat 전용으로 분리된다.
    """
    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/sd-turbo",
        id_dim: int = 512,
        num_id_tokens: int = 4,
        condition_channels: int = 1,
        train_unet: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from diffusers import AutoPipelineForText2Image, DDPMScheduler

        load_dtype = torch.float32 if train_unet else dtype
        pipe = AutoPipelineForText2Image.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=load_dtype,
            safety_checker=None,
        )

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.device_name = device
        self.weight_dtype = dtype
        self.condition_channels = int(condition_channels)
        self.base_latent_channels = int(self.unet.config.in_channels)

        self._expand_unet_conv_in(extra_in_channels=self.base_latent_channels)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(train_unet)

        cross_dim = self.unet.config.cross_attention_dim
        self.pose_guider = PoseGuider(in_channels=self.condition_channels, out_channels=self.base_latent_channels)
        self.id_adapter = IDAdapter(id_dim=id_dim, cross_attention_dim=cross_dim, num_tokens=num_id_tokens)

        self.to(device)

        if dtype == torch.float16:
            self.vae.to(device=device, dtype=torch.float16)
            self.text_encoder.to(device=device, dtype=torch.float16)

        self.pose_guider.to(device=device, dtype=torch.float32)
        self.id_adapter.to(device=device, dtype=torch.float32)

        if train_unet:
            self.unet.to(device=device, dtype=torch.float32).train()
        else:
            self.unet.to(device=device, dtype=dtype).eval()

    def _expand_unet_conv_in(self, extra_in_channels: int):
        old_conv = self.unet.conv_in
        old_in = int(old_conv.in_channels)
        new_in = old_in + int(extra_in_channels)
        if old_in == new_in:
            return

        new_conv = nn.Conv2d(
            new_in,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :old_in].copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv
        self.unet.config.in_channels = new_in
        self.unet.register_to_config(in_channels=new_in)
        self.unet_input_channels = new_in

    @torch.no_grad()
    def encode_prompt(self, batch_size: int, prompt: str = "") -> torch.Tensor:
        toks = self.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        toks = {k: v.to(self.device_name) for k, v in toks.items()}
        return self.text_encoder(**toks).last_hidden_state

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        images = images.to(device=self.device_name, dtype=vae_dtype)
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = torch.nan_to_num(latents.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)
        vae_dtype = next(self.vae.parameters()).dtype
        z = (latents / self.vae.config.scaling_factor).to(device=self.device_name, dtype=vae_dtype)
        img = self.vae.decode(z).sample
        return torch.nan_to_num(img.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    def append_id_tokens(self, prompt_embeds: torch.Tensor, id_embed: Optional[torch.Tensor]) -> torch.Tensor:
        if id_embed is None:
            return prompt_embeds
        id_dtype = next(self.id_adapter.parameters()).dtype
        id_embed = id_embed.to(device=self.device_name, dtype=id_dtype)
        id_tokens = self.id_adapter(id_embed).to(device=self.device_name, dtype=prompt_embeds.dtype)
        return torch.cat([prompt_embeds, id_tokens], dim=1)

    def _prediction_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if pred_type == "epsilon":
            return noise
        if pred_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)
        if pred_type == "sample":
            return latents
        raise ValueError(f"Unsupported prediction_type: {pred_type}")

    def predict_x0(self, noisy_latents: torch.Tensor, model_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(noisy_latents.shape[0]):
            step_out = self.scheduler.step(model_pred[i:i + 1], timesteps[i], noisy_latents[i:i + 1], return_dict=True)
            outs.append(step_out.pred_original_sample)
        pred_x0 = torch.cat(outs, dim=0)
        return torch.nan_to_num(pred_x0.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30).to(dtype=noisy_latents.dtype)

    def forward(
        self,
        clean: torch.Tensor,
        blur_image: torch.Tensor,
        landmark_map: torch.Tensor,
        id_embed: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        prompt: str = "",
        decode_recon: bool = False,
        fixed_high_timestep: Optional[int] = None,
    ) -> TeacherOutput:
        b = clean.shape[0]
        clean = clean.to(self.device_name)
        blur_image = blur_image.to(self.device_name)
        landmark_map = landmark_map.to(self.device_name)

        if blur_image.ndim != 4 or blur_image.shape[1] != 3:
            raise ValueError(f"blur_image must be Bx3xHxW, got {tuple(blur_image.shape)}")
        if landmark_map.ndim != 4 or landmark_map.shape[1] != self.condition_channels:
            raise ValueError(
                f"landmark_map channel mismatch: got {landmark_map.shape[1]}, expected {self.condition_channels}"
            )

        latents = self.encode_images(clean)
        blur_latents = self.encode_images(blur_image).to(dtype=latents.dtype)

        noise = torch.randn_like(latents)
        if timesteps is None:
            if fixed_high_timestep is not None:
                timesteps = torch.full((b,), int(fixed_high_timestep), device=self.device_name, dtype=torch.long)
            else:
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=self.device_name, dtype=torch.long)
        else:
            timesteps = timesteps.to(self.device_name).long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        pose_dtype = next(self.pose_guider.parameters()).dtype
        pose_residual = self.pose_guider(
            landmark_map.to(device=self.device_name, dtype=pose_dtype),
            latent_hw=noisy_latents.shape[-2:],
        ).to(dtype=noisy_latents.dtype)

        latent_with_pose = noisy_latents + pose_residual
        unet_input = torch.cat([latent_with_pose, blur_latents], dim=1)

        with torch.no_grad():
            text_dtype = next(self.text_encoder.parameters()).dtype
            prompt_embeds = self.encode_prompt(b, prompt=prompt).to(dtype=text_dtype)
        prompt_embeds = self.append_id_tokens(prompt_embeds, id_embed)

        unet_dtype = next(self.unet.parameters()).dtype
        if unet_dtype == torch.float32:
            unet_input = unet_input.float()
            prompt_embeds = prompt_embeds.float()
        else:
            unet_input = unet_input.to(dtype=unet_dtype)
            prompt_embeds = prompt_embeds.to(dtype=unet_dtype)

        model_pred = self.unet(unet_input, timesteps, encoder_hidden_states=prompt_embeds).sample
        target = self._prediction_target(latents, noise, timesteps).to(dtype=model_pred.dtype)

        pred_x0 = self.predict_x0(noisy_latents.to(dtype=model_pred.dtype), model_pred, timesteps) if decode_recon else None
        recon_image = self.decode_latents(pred_x0) if pred_x0 is not None else None

        return TeacherOutput(model_pred, target, noisy_latents, latents, timesteps, pred_x0, recon_image)
