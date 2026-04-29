# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np

LANDMARK_PARTS_68 = {
    "jaw": (0, 17),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "nose": (27, 36),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "mouth": (48, 68),
}


def ensure_lm68(landmarks) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape != (68, 2):
        raise ValueError(f"Expected 68x2 landmarks, got {lm.shape}")
    return lm


def landmark_bbox(lm68: np.ndarray, image_shape, expand: float = 0.18) -> tuple[int, int, int, int]:
    lm = ensure_lm68(lm68)
    h, w = image_shape[:2]
    x1, y1 = lm.min(axis=0)
    x2, y2 = lm.max(axis=0)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= bw * expand
    x2 += bw * expand
    y1 -= bh * expand
    y2 += bh * expand
    return int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)), int(np.clip(x2, 1, w)), int(np.clip(y2, 1, h))


def face_hull_mask(image_shape, lm68: np.ndarray, expand_iter: int = 0) -> np.ndarray:
    lm = ensure_lm68(lm68).astype(np.int32)
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(lm)
    cv2.fillConvexPoly(mask, hull, 255)
    if expand_iter > 0:
        k = max(3, (min(h, w) // 64) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=expand_iter)
    return mask


def draw_landmarks_68(image: np.ndarray, lm68: np.ndarray, color=(0, 255, 0), thickness: int = 1, draw_points: bool = True) -> np.ndarray:
    lm = ensure_lm68(lm68).astype(np.int32)
    out = image
    parts = LANDMARK_PARTS_68
    for name in ["right_eyebrow", "jaw", "left_eyebrow"]:
        pts = lm[slice(*parts[name])]
        cv2.polylines(out, [pts], False, color, thickness, lineType=cv2.LINE_AA)
    nose = lm[slice(*parts["nose"])]
    cv2.polylines(out, [np.concatenate([nose, nose[-6:-5]], axis=0)], False, color, thickness, lineType=cv2.LINE_AA)
    for name in ["right_eye", "left_eye", "mouth"]:
        pts = lm[slice(*parts[name])]
        cv2.polylines(out, [pts], True, color, thickness, lineType=cv2.LINE_AA)
    if draw_points:
        for i, (x, y) in enumerate(lm):
            r = 2 if i < 17 else 1
            cv2.circle(out, (int(x), int(y)), r, color, -1, lineType=cv2.LINE_AA)
    return out


def landmark_map(size: int, lm68: np.ndarray, thickness: int = 2) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    return draw_landmarks_68(canvas, lm68, color=(255, 255, 255), thickness=thickness, draw_points=True)
