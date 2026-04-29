# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseGuider(nn.Module):
    """Small CNN that maps condition image to latent-space residual.

    Input:  Bx3x512x512 condition image in [-1,1]
    Output: Bx4x64x64 residual to add to SD latent.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 4, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c, c, 3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c * 2, c * 2, 3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c * 4, c * 4, 3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(c * 4, out_channels, 3, stride=1, padding=1),
        )
        # zero-init final layer so it starts as no-op
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor, latent_hw: tuple[int, int] | None = None) -> torch.Tensor:
        y = self.net(condition)
        if latent_hw is not None and y.shape[-2:] != latent_hw:
            y = F.interpolate(y, size=latent_hw, mode="bilinear", align_corners=False)
        return y
