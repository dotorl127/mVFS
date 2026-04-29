# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn.functional as F


def noise_prediction_loss(model_pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(model_pred.float(), target.float())
    if loss_type == "l1":
        return F.l1_loss(model_pred.float(), target.float())
    raise ValueError(f"Unknown loss_type={loss_type}")


def reconstruction_l1_loss(recon: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    if recon is None:
        return clean.new_tensor(0.0)
    return F.l1_loss(recon.float(), clean.float())
