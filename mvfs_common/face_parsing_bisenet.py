# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List, Optional
import cv2
import numpy as np
import torch

# CelebAMask-HQ / facexlib parsing ids:
# 0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye,
# 6 eye_g, 7 l_ear, 8 r_ear, 9 ear_r, 10 nose,
# 11 mouth, 12 u_lip, 13 l_lip, 14 neck, 15 neck_l,
# 16 cloth, 17 hair, 18 hat
#
# For APPLE-style face-only blur, exclude hair, hat, neck, cloth, glasses/occlusion.
DEFAULT_INCLUDE_IDS = [1, 2, 3, 4, 5, 10, 11, 12, 13]


class FaceSegExtractor:
    def __init__(
        self,
        device: str = "cuda",
        include_ids: Optional[Iterable[int]] = None,
        mask_blur: int = 5,
    ):
        self.device = device
        self.include_ids: List[int] = list(include_ids) if include_ids is not None else list(DEFAULT_INCLUDE_IDS)
        self.mask_blur = int(mask_blur)

        try:
            from facexlib.parsing import init_parsing_model
        except Exception as e:
            raise RuntimeError(
                "facexlib is required for face segmentation. "
                "Install in MVFS env: uv pip install facexlib basicsr lpips"
            ) from e

        self.model = init_parsing_model(model_name="bisenet", device=device)
        self.model.eval()

    @torch.no_grad()
    def predict_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        logits = self.model(x)[0]
        parsing = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        mask = np.isin(parsing, self.include_ids).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.mask_blur > 0:
            k = self.mask_blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return mask
