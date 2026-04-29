# -*- coding: utf-8 -*-
"""Small 3DMM coefficient utilities.

This file does not depend on a specific 3DMM implementation. It handles the
MVFS-side bookkeeping: quality-weighted canonical coeffs and view residuals.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    s = float(w.sum())
    if s <= 1e-12:
        return np.ones_like(w, dtype=np.float32) / max(len(w), 1)
    return w / s


def weighted_mean_coeff(coeffs: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    arr = np.stack([np.asarray(c, dtype=np.float32) for c in coeffs], axis=0)
    if weights is None:
        return arr.mean(axis=0).astype(np.float32)
    w = normalize_weights(np.asarray(weights, dtype=np.float32))
    if len(w) != len(arr):
        raise ValueError(f"weights length {len(w)} != coeff count {len(arr)}")
    return np.sum(arr * w[:, None], axis=0).astype(np.float32)


def make_view_residuals(canonical: np.ndarray, view_coeffs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    c = np.asarray(canonical, dtype=np.float32)
    return {name: np.asarray(v, dtype=np.float32) - c for name, v in view_coeffs.items()}


def blend_view_residuals(canonical: np.ndarray, residuals: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    out = np.asarray(canonical, dtype=np.float32).copy()
    for name, w in weights.items():
        if name in residuals:
            out += float(w) * np.asarray(residuals[name], dtype=np.float32)
    return out.astype(np.float32)


def save_identity_npz(path, canonical: np.ndarray, coeffs: Sequence[np.ndarray] | None = None, weights=None, **extra) -> None:
    data = {"id_coeff_canonical": np.asarray(canonical, dtype=np.float32)}
    if coeffs is not None:
        data["id_coeffs"] = np.stack([np.asarray(c, dtype=np.float32) for c in coeffs], axis=0)
    if weights is not None:
        data["quality_weights"] = np.asarray(weights, dtype=np.float32)
    for k, v in extra.items():
        data[k] = v
    np.savez(path, **data)
