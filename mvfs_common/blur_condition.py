
# -*- coding: utf-8 -*-
from __future__ import annotations
from io import BytesIO
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageFilter

def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    import PIL.Image
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    bio = BytesIO()
    PIL.Image.fromarray(mask, mode='L').save(bio, format='PNG')
    return bio.getvalue()

def mask_from_png_bytes(blob: bytes) -> np.ndarray:
    img = Image.open(BytesIO(blob)).convert('L')
    return np.array(img, dtype=np.uint8)

def build_face_blur_condition_rgb(
    img_bgr: np.ndarray,
    face_mask: np.ndarray,
    downsample_size: int = 8,
    gaussian_radius: float = 8.0,
    feather_sigma: float = 0.0,
) -> np.ndarray:
    """
    APPLE-style simple condition approximation requested by user:
    1) whole image downsample to 8x8
    2) resize back to original size
    3) PIL GaussianBlur(radius=8)
    4) composite only on face region using face mask
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    low = pil_img.resize((int(downsample_size), int(downsample_size)), Image.BILINEAR)
    up = low.resize((w, h), Image.BILINEAR)
    blur = up.filter(ImageFilter.GaussianBlur(radius=float(gaussian_radius)))
    blur_np = np.array(blur, dtype=np.uint8)

    mask = face_mask.astype(np.float32) / 255.0
    if feather_sigma and feather_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather_sigma)
        mask = np.clip(mask, 0.0, 1.0)
    mask = mask[..., None]

    cond = img_rgb.astype(np.float32) * (1.0 - mask) + blur_np.astype(np.float32) * mask
    cond = np.clip(cond, 0, 255).astype(np.uint8)
    return cond
