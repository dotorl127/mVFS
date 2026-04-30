# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceNetIDLoss(nn.Module):
    """
    Lightweight differentiable ID loss for RTX 3060 12GB.

    Default backend:
        facenet-pytorch InceptionResnetV1(pretrained='vggface2')

    Why this backend?
        - PyTorch-native, so gradients flow from ID loss to recon image -> UNet.
        - Much lighter than ArcFace iresnet100 / Arc2Face style encoders.
        - Suitable as a first MVFS teacher ID regularizer.

    Input:
        pred / target: Bx3xHxW RGB in [-1, 1]
    """

    def __init__(
        self,
        pretrained: str = "vggface2",
        image_size: int = 160,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except Exception as e:
            raise ImportError(
                "FaceNetIDLoss requires facenet-pytorch. Install with:\n"
                "  uv pip install facenet-pytorch --no-deps"
            ) from e

        self.image_size = int(image_size)
        self.device_name = device
        self.model = InceptionResnetV1(pretrained=pretrained, classify=False).eval()
        self.model.to(device=device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        # facenet-pytorch InceptionResnetV1 expects roughly fixed-image-standardized input.
        # For aligned face crops, [-1,1] works well as a lightweight ID regularizer.
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.float().view(y.shape[0], -1)
        return F.normalize(y, dim=1, eps=1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_emb = self.embed(pred)
        with torch.no_grad():
            target_emb = self.embed(target)
        return (1.0 - (pred_emb * target_emb).sum(dim=1)).mean()


class TorchScriptIDLoss(nn.Module):
    """
    Optional custom TorchScript/PyTorch ID model loss.

    Model requirements:
        input : Bx3x112x112 RGB tensor
        range : selected by input_range
        output: BxD embedding
    """

    def __init__(
        self,
        model_path: str | Path,
        image_size: int = 112,
        input_range: Literal["minus1_1", "0_1", "imagenet"] = "minus1_1",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.image_size = int(image_size)
        self.input_range = input_range
        self.model = torch.jit.load(self.model_path, map_location=device).eval()
        self.model.to(device=device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.input_range == "minus1_1":
            return x
        if self.input_range == "0_1":
            return (x + 1.0) * 0.5
        if self.input_range == "imagenet":
            x01 = (x + 1.0) * 0.5
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            return (x01 - mean) / std
        raise ValueError(f"Unknown input_range={self.input_range}")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(self.preprocess(x))
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.float().view(y.shape[0], -1)
        return F.normalize(y, dim=1, eps=1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_emb = self.embed(pred)
        with torch.no_grad():
            target_emb = self.embed(target)
        return (1.0 - (pred_emb * target_emb).sum(dim=1)).mean()


def build_id_loss(
    backend: str,
    device: str = "cuda",
    model_path: str = "",
    facenet_pretrained: str = "vggface2",
    input_range: str = "minus1_1",
) -> nn.Module:
    backend = backend.lower()
    if backend == "facenet":
        return FaceNetIDLoss(pretrained=facenet_pretrained, image_size=160, device=device, dtype=torch.float32)
    if backend == "torchscript":
        if not model_path:
            raise ValueError("--id-loss-backend torchscript requires --id-loss-model")
        return TorchScriptIDLoss(model_path=model_path, image_size=112, input_range=input_range, device=device, dtype=torch.float32)
    raise ValueError(f"Unknown id loss backend: {backend}")
