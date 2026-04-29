# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def landmark_bounds_score(lm68: np.ndarray, image_shape) -> float:
    """Return fraction of landmarks inside image bounds."""
    lm = np.asarray(lm68, dtype=np.float32)
    h, w = image_shape[:2]
    ok = (lm[:, 0] >= 0) & (lm[:, 0] < w) & (lm[:, 1] >= 0) & (lm[:, 1] < h)
    return float(ok.mean())


def bbox_area_from_landmarks(lm68: np.ndarray) -> float:
    lm = np.asarray(lm68, dtype=np.float32)
    xy_min = lm.min(axis=0)
    xy_max = lm.max(axis=0)
    wh = np.maximum(xy_max - xy_min, 0.0)
    return float(wh[0] * wh[1])


def sharpness_laplacian(gray_or_bgr: np.ndarray) -> float:
    import cv2
    img = gray_or_bgr
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())
