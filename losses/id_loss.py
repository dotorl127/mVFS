# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineIdentityLoss(nn.Module):
    """Generic identity loss.

    Pass a frozen face encoder that maps image tensors [-1,1] to BxD embeddings.
    This keeps the project independent from a specific ArcFace/AdaFace package.
    """

    def __init__(self, encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.encoder = encoder

    def forward(self, pred_image: torch.Tensor, id_image: torch.Tensor) -> torch.Tensor:
        if self.encoder is None or pred_image is None:
            return pred_image.new_tensor(0.0) if pred_image is not None else torch.tensor(0.0)
        pred_feat = F.normalize(self.encoder(pred_image), dim=-1)
        id_feat = F.normalize(self.encoder(id_image), dim=-1)
        return (1.0 - (pred_feat * id_feat).sum(dim=-1)).mean()


def cosine_id_embedding_loss(pred_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred_embed.float(), dim=-1)
    tgt = F.normalize(target_embed.float(), dim=-1)
    return (1.0 - (pred * tgt).sum(dim=-1)).mean()
