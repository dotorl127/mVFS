# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """SD-Turbo based teacher for blur-condition reconstruction.

    The code is designed for light training on RTX 3060:
      - VAE frozen
      - text encoder frozen
      - UNet can be frozen or LoRA-trained externally
      - PoseGuider and IDAdapter are trainable

    Required external package:
      diffusers, transformers, accelerate
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

        pipe = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype)
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

    @torch.no_grad()
    def encode_prompt(self, batch_size: int, prompt: str = "") -> torch.Tensor:
        toks = self.tokenizer([prompt] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        toks = {k: v.to(self.device_name) for k, v in toks.items()}
        return self.text_encoder(**toks).last_hidden_state

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self.device_name, dtype=self.weight_dtype)
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents / self.vae.config.scaling_factor
        img = self.vae.decode(z).sample
        return img.clamp(-1, 1)

    def append_id_tokens(self, prompt_embeds: torch.Tensor, id_embed: Optional[torch.Tensor]) -> torch.Tensor:
        if id_embed is None:
            return prompt_embeds
        id_embed = id_embed.to(device=self.device_name, dtype=prompt_embeds.dtype)
        id_tokens = self.id_adapter(id_embed).to(dtype=prompt_embeds.dtype)
        return torch.cat([prompt_embeds, id_tokens], dim=1)

    def _prediction_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if pred_type == "epsilon":
            return noise
        if pred_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)
        raise ValueError(f"Unsupported prediction_type: {pred_type}")

    def predict_x0(self, noisy_latents: torch.Tensor, model_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # epsilon prediction path. For v-prediction, use scheduler.step in training script if needed.
        pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if pred_type != "epsilon":
            return None
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        a = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return (noisy_latents - (1.0 - a).sqrt() * model_pred) / a.sqrt()

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
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=self.device_name, dtype=torch.long)
        else:
            timesteps = timesteps.to(self.device_name).long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        pose_residual = self.pose_guider(condition.to(dtype=self.weight_dtype), latent_hw=noisy_latents.shape[-2:]).to(dtype=noisy_latents.dtype)
        unet_input = noisy_latents + pose_residual

        with torch.no_grad():
            prompt_embeds = self.encode_prompt(b, prompt=prompt).to(dtype=self.weight_dtype)
        prompt_embeds = self.append_id_tokens(prompt_embeds, id_embed)

        model_pred = self.unet(unet_input, timesteps, encoder_hidden_states=prompt_embeds).sample
        target = self._prediction_target(latents, noise, timesteps)

        pred_x0 = self.predict_x0(noisy_latents, model_pred, timesteps)
        recon_image = self.decode_latents(pred_x0) if (decode_recon and pred_x0 is not None) else None
        return TeacherOutput(model_pred=model_pred, target=target, noisy_latents=noisy_latents, latents=latents, timesteps=timesteps, pred_x0=pred_x0, recon_image=recon_image)
