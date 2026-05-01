# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageFilter


def build_face_blur_condition_rgb(
    img_bgr: np.ndarray,
    face_mask: np.ndarray,
    downsample_size: int = 8,
    gaussian_radius: float = 8.0,
    feather_sigma: float = 0.0,
) -> np.ndarray:
    """
    APPLE-style blur condition used by MVFS teacher.

    Steps:
      1. Resize whole image to downsample_size x downsample_size.
      2. Resize back to original size.
      3. Apply PIL GaussianBlur(radius=gaussian_radius).
      4. Composite the blurred image only inside face_mask.

    Returns RGB uint8 image.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)
    if downsample_size > 0:
        low = pil_img.resize((int(downsample_size), int(downsample_size)), Image.BILINEAR)
        up = low.resize((w, h), Image.BILINEAR)
    else:
        up = pil_img

    if gaussian_radius > 0:
        blur = up.filter(ImageFilter.GaussianBlur(radius=float(gaussian_radius)))
    else:
        blur = up

    blur_np = np.array(blur, dtype=np.uint8)

    mask = face_mask.astype(np.float32) / 255.0
    if feather_sigma and feather_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
        mask = np.clip(mask, 0.0, 1.0)

    cond = img_rgb.astype(np.float32) * (1.0 - mask[..., None]) + blur_np.astype(np.float32) * mask[..., None]
    return np.clip(cond, 0, 255).astype(np.uint8)
