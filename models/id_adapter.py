# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class IDAdapter(nn.Module):
    """Map face-ID embedding to extra cross-attention tokens.

    This keeps implementation simple: append N identity tokens to the SD text tokens.
    Use a frozen ArcFace/AdaFace/InsightFace encoder outside this module to create id_embed.
    """

    def __init__(self, id_dim: int = 512, cross_attention_dim: int = 768, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(id_dim),
            nn.Linear(id_dim, cross_attention_dim * num_tokens),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, id_embed: torch.Tensor) -> torch.Tensor:
        if id_embed.ndim != 2:
            raise ValueError(f"id_embed must be BxD, got {tuple(id_embed.shape)}")
        b = id_embed.shape[0]
        return self.proj(id_embed).view(b, self.num_tokens, self.cross_attention_dim)
