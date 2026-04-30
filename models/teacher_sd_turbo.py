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
    SD-Turbo based MVFS teacher.

    This version keeps the training loss/hyperparameters controlled by train_teacher.py.
    Fixes only implementation bugs:
      - fp16 condition vs fp32 PoseGuider dtype mismatch
      - fp16 id_embed vs fp32 IDAdapter dtype mismatch
      - unstable manual pred_x0 formula replaced by diffusers scheduler.step()
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/sd-turbo",
        id_dim: int = 512,
        num_id_tokens: int = 4,
        train_unet: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        try:
            from diffusers import AutoPipelineForText2Image, DDPMScheduler
        except Exception as e:
            raise ImportError("Install diffusers/transformers/accelerate to use MVFSTeacherSDTurbo") from e

        pipe = AutoPipelineForText2Image.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
            safety_checker=None,
        )

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.device_name = device
        self.weight_dtype = dtype

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(train_unet)

        cross_dim = self.unet.config.cross_attention_dim
        self.pose_guider = PoseGuider(in_channels=3, out_channels=self.unet.config.in_channels)
        self.id_adapter = IDAdapter(id_dim=id_dim, cross_attention_dim=cross_dim, num_tokens=num_id_tokens)

        self.to(device)

        # Keep new modules in fp32 unless the caller explicitly casts them.
        # This is safer and avoids Conv bias half/float mismatch.
        self.pose_guider.float()
        self.id_adapter.float()

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
        images = images.to(device=self.device_name, dtype=self.weight_dtype)
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Decode path is used for recon/debug. Guard only against invalid numeric values.
        latents = torch.nan_to_num(latents.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        latents = latents.clamp(-30.0, 30.0)
        z = (latents / self.vae.config.scaling_factor).to(device=self.device_name, dtype=self.weight_dtype)
        img = self.vae.decode(z).sample
        img = torch.nan_to_num(img.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        return img.clamp(-1, 1)

    def append_id_tokens(self, prompt_embeds: torch.Tensor, id_embed: Optional[torch.Tensor]) -> torch.Tensor:
        if id_embed is None:
            return prompt_embeds

        id_dtype = next(self.id_adapter.parameters()).dtype
        id_embed = id_embed.to(device=self.device_name, dtype=id_dtype)
        id_tokens = self.id_adapter(id_embed)
        id_tokens = id_tokens.to(device=self.device_name, dtype=prompt_embeds.dtype)
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

    def predict_x0(self, noisy_latents: torch.Tensor, model_pred: torch.Tensor, timesteps: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Use diffusers scheduler.step() instead of manual epsilon-only equation.
        This handles epsilon/v_prediction/sample according to scheduler config.
        """
        outs = []
        for i in range(noisy_latents.shape[0]):
            step_out = self.scheduler.step(
                model_pred[i:i+1],
                timesteps[i],
                noisy_latents[i:i+1],
                return_dict=True,
            )
            outs.append(step_out.pred_original_sample)
        pred_x0 = torch.cat(outs, dim=0)
        pred_x0 = torch.nan_to_num(pred_x0.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        return pred_x0.clamp(-30.0, 30.0).to(dtype=noisy_latents.dtype)

    def forward(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        id_embed: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        prompt: str = "",
        decode_recon: bool = False,
        fixed_high_timestep: Optional[int] = None,
    ) -> TeacherOutput:
        b = clean.shape[0]
        clean = clean.to(self.device_name)
        condition = condition.to(self.device_name)

        latents = self.encode_images(clean)
        noise = torch.randn_like(latents)

        if timesteps is None:
            if fixed_high_timestep is not None:
                timesteps = torch.full((b,), int(fixed_high_timestep), device=self.device_name, dtype=torch.long)
            else:
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (b,),
                    device=self.device_name,
                    dtype=torch.long,
                )
        else:
            timesteps = timesteps.to(self.device_name).long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        pose_dtype = next(self.pose_guider.parameters()).dtype
        pose_residual = self.pose_guider(
            condition.to(device=self.device_name, dtype=pose_dtype),
            latent_hw=noisy_latents.shape[-2:],
        ).to(dtype=noisy_latents.dtype)

        unet_input = noisy_latents + pose_residual

        with torch.no_grad():
            prompt_embeds = self.encode_prompt(b, prompt=prompt).to(dtype=self.weight_dtype)
        prompt_embeds = self.append_id_tokens(prompt_embeds, id_embed)

        model_pred = self.unet(unet_input, timesteps, encoder_hidden_states=prompt_embeds).sample
        target = self._prediction_target(latents, noise, timesteps)

        pred_x0 = self.predict_x0(noisy_latents, model_pred, timesteps) if decode_recon else None
        recon_image = self.decode_latents(pred_x0) if pred_x0 is not None else None

        return TeacherOutput(
            model_pred=model_pred,
            target=target,
            noisy_latents=noisy_latents,
            latents=latents,
            timesteps=timesteps,
            pred_x0=pred_x0,
            recon_image=recon_image,
        )
