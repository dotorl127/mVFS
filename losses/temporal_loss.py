# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_l1_loss(current: torch.Tensor, previous_warped: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Basic temporal consistency loss.

    current, previous_warped: BxCxHxW in same coordinate system.
    mask: optional Bx1xHxW visibility/confidence mask.
    """
    diff = (current.float() - previous_warped.float()).abs()
    if mask is not None:
        mask = mask.float()
        return (diff * mask).sum() / mask.sum().clamp_min(1.0)
    return diff.mean()


def alpha_temporal_loss(alpha_t: torch.Tensor, alpha_prev_warped: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(alpha_t.float(), alpha_prev_warped.float())


def feature_temporal_loss(feat_t: torch.Tensor, feat_prev_warped: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return temporal_l1_loss(feat_t, feat_prev_warped, mask)
