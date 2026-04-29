# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np

from .landmarks import draw_landmarks_68, ensure_lm68, landmark_bbox


def apple_style_blur_condition(
    image_bgr: np.ndarray,
    lm68: np.ndarray,
    downsample: int = 8,
    expand: float = 0.18,
    overlay_landmarks: bool = True,
    landmark_color=(0, 255, 0),
    landmark_thickness: int = 1,
) -> np.ndarray:
    """Build teacher condition image.

    Steps:
      1) Find an expanded landmark ROI.
      2) Downsample that ROI to downsample x downsample.
      3) Upsample it back to the ROI size.
      4) Paste it into original image context.
      5) Draw 68 landmarks over it.

    This keeps pose/illumination/background context while suppressing target ID details.
    """
    lm = ensure_lm68(lm68)
    cond = image_bgr.copy()
    x1, y1, x2, y2 = landmark_bbox(lm, cond.shape, expand=expand)
    roi = cond[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Empty ROI while building blur condition")
    tiny = cv2.resize(roi, (downsample, downsample), interpolation=cv2.INTER_AREA)
    # nearest keeps the APPLE-style blocky low-frequency cue. Use INTER_LINEAR if desired.
    up = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    cond[y1:y2, x1:x2] = up
    if overlay_landmarks:
        draw_landmarks_68(cond, lm, color=landmark_color, thickness=landmark_thickness, draw_points=True)
    return cond
