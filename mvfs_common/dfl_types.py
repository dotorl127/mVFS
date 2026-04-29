# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from enum import IntEnum
from typing import List

import cv2
import numpy as np
import numpy.linalg as npla

# Minimal DFL-compatible helpers
# ----------------------------

class FaceType(IntEnum):
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20
    MARK_ONLY = 100

    @staticmethod
    def from_string(s: str) -> "FaceType":
        key = s.strip().lower()
        table = {
            "half": FaceType.HALF,
            "mid_full": FaceType.MID_FULL,
            "full": FaceType.FULL,
            "full_no_align": FaceType.FULL_NO_ALIGN,
            "whole_face": FaceType.WHOLE_FACE,
            "head": FaceType.HEAD,
            "head_no_align": FaceType.HEAD_NO_ALIGN,
            "mark_only": FaceType.MARK_ONLY,
        }
        if key not in table:
            raise ValueError(f"Unknown face type: {s}. valid={list(table.keys())}")
        return table[key]

    @staticmethod
    def to_string(ft: "FaceType") -> str:
        table = {
            FaceType.HALF: "half_face",
            FaceType.MID_FULL: "midfull_face",
            FaceType.FULL: "full_face",
            FaceType.FULL_NO_ALIGN: "full_face_no_align",
            FaceType.WHOLE_FACE: "whole_face",
            FaceType.HEAD: "head",
            FaceType.HEAD_NO_ALIGN: "head_no_align",
            FaceType.MARK_ONLY: "mark_only",
        }
        return table.get(ft, "full_face")


# DFL's normalized reference landmarks subset used for alignment.
landmarks_2D_new = np.array([
    [0.000213256, 0.106454],   # 17
    [0.0752622,   0.038915],   # 18
    [0.18113,     0.0187482],  # 19
    [0.29077,     0.0344891],  # 20
    [0.393397,    0.0773906],  # 21
    [0.586856,    0.0773906],  # 22
    [0.689483,    0.0344891],  # 23
    [0.799124,    0.0187482],  # 24
    [0.904991,    0.038915],   # 25
    [0.98004,     0.106454],   # 26
    [0.490127,    0.203352],   # 27
    [0.490127,    0.307009],   # 28
    [0.490127,    0.409805],   # 29
    [0.490127,    0.515625],   # 30
    [0.36688,     0.587326],   # 31
    [0.426036,    0.609345],   # 32
    [0.490127,    0.628106],   # 33
    [0.554217,    0.609345],   # 34
    [0.613373,    0.587326],   # 35
    [0.121737,    0.216423],   # 36
    [0.187122,    0.178758],   # 37
    [0.265825,    0.179852],   # 38
    [0.334606,    0.231733],   # 39
    [0.260918,    0.245099],   # 40
    [0.182743,    0.244077],   # 41
    [0.645647,    0.231733],   # 42
    [0.714428,    0.179852],   # 43
    [0.793132,    0.178758],   # 44
    [0.858516,    0.216423],   # 45
    [0.79751,     0.244077],   # 46
    [0.719335,    0.245099],   # 47
    [0.254149,    0.780233],   # 48
    [0.726104,    0.780233],   # 54
], dtype=np.float32)


FaceType_to_padding_remove_align = {
    FaceType.HALF: (0.0, False),
    FaceType.MID_FULL: (0.0675, False),
    FaceType.FULL: (0.2109375, False),
    FaceType.FULL_NO_ALIGN: (0.2109375, True),
    FaceType.WHOLE_FACE: (0.40, False),
    FaceType.HEAD: (0.70, False),
    FaceType.HEAD_NO_ALIGN: (0.70, True),
}


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool) -> np.ndarray:
    """Minimal umeyama similarity transform. Returns 3x3 matrix."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num

    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T


def transform_points(points: np.ndarray, mat: np.ndarray, invert: bool = False) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    mat = np.asarray(mat, dtype=np.float32)
    if invert:
        mat = cv2.invertAffineTransform(mat)
    pts = np.expand_dims(points, axis=1)
    pts = cv2.transform(pts, mat, pts.shape)
    return np.squeeze(pts)


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def estimate_averaged_yaw(lmrks: np.ndarray) -> float:
    # Simple DFL-like yaw proxy from mirrored landmark distances.
    lmrks = np.asarray(lmrks, dtype=np.float32)
    if len(lmrks) != 68:
        return 0.0
    left = np.mean(lmrks[[0, 1, 2, 3, 4], 0])
    right = np.mean(lmrks[[12, 13, 14, 15, 16], 0])
    nose = lmrks[30, 0]
    denom = max(right - left, 1e-6)
    return float(((nose - left) / denom - 0.5) * 2.0)


def get_transform_mat(image_landmarks: np.ndarray, output_size: int, face_type: FaceType, scale: float = 1.0) -> np.ndarray:
    image_landmarks = np.asarray(image_landmarks, dtype=np.float32)
    if image_landmarks.shape != (68, 2):
        raise ValueError(f"Expected landmarks shape (68,2), got {image_landmarks.shape}")

    padding, remove_align = FaceType_to_padding_remove_align.get(face_type, (0.2109375, False))

    # DFL uses landmarks[17:49] and landmark[54] for transform.
    src = np.concatenate([image_landmarks[17:49], image_landmarks[54:55]], axis=0)
    mat = umeyama(src, landmarks_2D_new, True)[0:2].astype(np.float32)

    g_p = transform_points(np.float32([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]), mat, True)
    g_c = g_p[4].astype(np.float32)

    tb_diag_vec = (g_p[2] - g_p[0]).astype(np.float32)
    tb_diag_vec /= max(npla.norm(tb_diag_vec), 1e-6)

    bt_diag_vec = (g_p[1] - g_p[3]).astype(np.float32)
    bt_diag_vec /= max(npla.norm(bt_diag_vec), 1e-6)

    mod = (1.0 / scale) * (npla.norm(g_p[0] - g_p[2]) * (padding * np.sqrt(2.0) + 0.5))

    if face_type == FaceType.WHOLE_FACE:
        vec = (g_p[0] - g_p[3]).astype(np.float32)
        vec_len = max(npla.norm(vec), 1e-6)
        vec /= vec_len
        g_c += vec * vec_len * 0.07

    elif face_type == FaceType.HEAD:
        yaw = estimate_averaged_yaw(transform_points(image_landmarks, mat, False))
        hvec = (g_p[0] - g_p[1]).astype(np.float32)
        hvec_len = max(npla.norm(hvec), 1e-6)
        hvec /= hvec_len
        yaw *= abs(math.tanh(yaw * 2.0))
        g_c -= hvec * (yaw * hvec_len / 2.0)

        vvec = (g_p[0] - g_p[3]).astype(np.float32)
        vvec_len = max(npla.norm(vvec), 1e-6)
        vvec /= vvec_len
        g_c += vvec * vvec_len * 0.50

    if not remove_align:
        l_t = np.array([
            g_c - tb_diag_vec * mod,
            g_c + bt_diag_vec * mod,
            g_c + tb_diag_vec * mod,
        ], dtype=np.float32)
    else:
        l_t4 = np.array([
            g_c - tb_diag_vec * mod,
            g_c + bt_diag_vec * mod,
            g_c + tb_diag_vec * mod,
            g_c - bt_diag_vec * mod,
        ], dtype=np.float32)
        area = polygon_area(l_t4[:, 0], l_t4[:, 1])
        side = np.float32(math.sqrt(max(area, 1e-6)) / 2)
        l_t = np.array([
            g_c + [-side, -side],
            g_c + [ side, -side],
            g_c + [ side,  side],
        ], dtype=np.float32)

    pts2 = np.float32(((0, 0), (output_size, 0), (output_size, output_size)))
    return cv2.getAffineTransform(l_t.astype(np.float32), pts2)


def rect_from_landmarks(lm68: np.ndarray, margin: float = 0.15) -> List[float]:
    lm68 = np.asarray(lm68, dtype=np.float32)
    x0, y0 = lm68.min(axis=0)
    x1, y1 = lm68.max(axis=0)
    w = x1 - x0
    h = y1 - y0
    return [
        float(x0 - w * margin),
        float(y0 - h * margin),
        float(x1 + w * margin),
        float(y1 + h * margin),
    ]


def draw_landmarks_68(img: np.ndarray, lm68: np.ndarray, color=(0, 255, 0), thickness: int = 1) -> np.ndarray:
    out = img.copy()
    lm = np.asarray(lm68, dtype=np.int32)

    parts_open = [
        lm[0:17],    # jaw
        lm[17:22],   # right brow
        lm[22:27],   # left brow
        lm[27:36],   # nose
    ]
    parts_closed = [
        lm[36:42],   # right eye
        lm[42:48],   # left eye
        lm[48:60],   # outer mouth
        lm[60:68],   # inner mouth
    ]

    for p in parts_open:
        cv2.polylines(out, [p], False, color, thickness, lineType=cv2.LINE_AA)
    for p in parts_closed:
        cv2.polylines(out, [p], True, color, thickness, lineType=cv2.LINE_AA)
    for x, y in lm:
        cv2.circle(out, (int(x), int(y)), 1, color, -1, lineType=cv2.LINE_AA)
    return out


