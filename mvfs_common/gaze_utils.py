# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# MediaPipe FaceMesh(refine_landmarks=True) 기준.
# 468~472: left iris 5 points
# 473~477: right iris 5 points
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]


def sidecar_gaze_path(image_path: str | Path) -> Path:
    p = Path(image_path)
    return p.with_suffix(p.suffix + ".mvfs_gaze.json")


def pad_image_bgr(img_bgr: np.ndarray, pad_ratio: float = 0.25):
    if pad_ratio <= 0:
        return img_bgr, 0, 0

    h, w = img_bgr.shape[:2]
    pad_h = int(round(h * pad_ratio))
    pad_w = int(round(w * pad_ratio))

    padded = cv2.copyMakeBorder(
        img_bgr,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded, pad_w, pad_h


class MediaPipeIrisExtractor:
    """
    MediaPipe FaceMesh(refine_landmarks=True)로 홍채 5점만 추출한다.
    분석할 때만 padding을 넣고, 저장 좌표는 원본 이미지 좌표계로 되돌린다.
    """
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "mediapipe import 실패. 현재 에러가 protobuf runtime_version 관련이면 "
                "`python -m pip install -U protobuf` 후 다시 시도해봐. "
                "그래도 충돌하면 mediapipe/tensorflow/protobuf 버전 충돌이라 별도 venv가 안전함."
            ) from e

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        try:
            self.mesh.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def extract_iris5(self, image_bgr: np.ndarray, pad_ratio: float = 0.25) -> Optional[dict]:
        if image_bgr is None or image_bgr.ndim != 3:
            return None

        h0, w0 = image_bgr.shape[:2]
        padded, pad_w, pad_h = pad_image_bgr(image_bgr, pad_ratio=pad_ratio)
        hp, wp = padded.shape[:2]

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if result.multi_face_landmarks is None or len(result.multi_face_landmarks) == 0:
            return None

        lms = result.multi_face_landmarks[0].landmark
        if len(lms) <= max(RIGHT_IRIS_IDX):
            return None

        pts_padded = np.array([[lm.x * wp, lm.y * hp] for lm in lms], dtype=np.float32)

        left = pts_padded[LEFT_IRIS_IDX].copy()
        right = pts_padded[RIGHT_IRIS_IDX].copy()

        # padded 좌표 -> 원본 이미지 좌표
        left[:, 0] -= pad_w
        left[:, 1] -= pad_h
        right[:, 0] -= pad_w
        right[:, 1] -= pad_h

        left[:, 0] = np.clip(left[:, 0], 0, w0 - 1)
        left[:, 1] = np.clip(left[:, 1], 0, h0 - 1)
        right[:, 0] = np.clip(right[:, 0], 0, w0 - 1)
        right[:, 1] = np.clip(right[:, 1], 0, h0 - 1)

        return {
            "success": True,
            "left_iris_5": left.astype(float).tolist(),
            "right_iris_5": right.astype(float).tolist(),
            "pad_ratio": float(pad_ratio),
            "image_size": [int(w0), int(h0)],
            "model": "mediapipe_facemesh_refine_landmarks",
        }


def save_gaze_sidecar(image_path: str | Path, gaze: dict) -> Path:
    out = sidecar_gaze_path(image_path)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(gaze, f, ensure_ascii=False, indent=2)
    return out


def load_gaze_sidecar(image_path: str | Path) -> Optional[dict]:
    p = sidecar_gaze_path(image_path)
    if not p.exists():
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict) or not data.get("success", False):
        return None

    return data


def gaze_iris_points_np(gaze: Optional[dict]) -> Optional[np.ndarray]:
    """
    sidecar dict에서 left/right iris 5점 총 10점을 (10,2) float32로 반환.
    없으면 None.
    """
    if not gaze:
        return None

    pts = []
    for key in ("left_iris_5", "right_iris_5"):
        arr = gaze.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            pts.append(arr[:, :2])

    if not pts:
        return None

    return np.concatenate(pts, axis=0).astype(np.float32)


def append_gaze_to_landmarks_2d(
    image_path: str | Path,
    landmarks_2d: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """
    기존 3DMM/3DDFA 2D landmarks 뒤에 MediaPipe iris 10점을 그대로 추가한다.
    즉 최종 landmark array:
        [3dmm landmarks..., left_iris_5, right_iris_5]

    sidecar gaze 좌표는 원본 이미지 기준이므로, dataset resize 비율을 같이 적용한다.
    """
    lm = np.asarray(landmarks_2d, dtype=np.float32)
    gaze = load_gaze_sidecar(image_path)
    gaze_pts = gaze_iris_points_np(gaze)

    if gaze_pts is None:
        return lm

    gaze_pts = gaze_pts.copy()
    gaze_pts[:, 0] *= float(scale_x)
    gaze_pts[:, 1] *= float(scale_y)

    return np.concatenate([lm, gaze_pts], axis=0).astype(np.float32)
