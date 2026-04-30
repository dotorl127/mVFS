
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn

class LPIPSLoss(nn.Module):
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        super().__init__()
        try:
            import lpips
        except Exception as e:
            raise RuntimeError('Install lpips: pip install lpips') from e
        self.metric = lpips.LPIPS(net=net).to(device)
        self.metric.eval()
        for p in self.metric.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # expects range [-1, 1]
        out = self.metric(x, y)
        return out.mean()

def masked_blend_pair(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    # mask: [N,1,H,W] in [0,1]
    return x * mask + y * (1.0 - mask), y

def masked_l1(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = (x - y).abs() * mask
    denom = mask.sum() * x.shape[1] + eps
    return diff.sum() / denom
