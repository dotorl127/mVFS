# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mvfs_common.dfljpg_utils import read_dfljpg_metadata
from mvfs_common.blur_condition import build_face_blur_condition_rgb
from mvfs_common.gaze_utils import append_gaze_to_landmarks_2d

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_image_rgb(path: str | Path, image_size: Optional[int] = None) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image_size is not None and (img.shape[0] != image_size or img.shape[1] != image_size):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous() * 2.0 - 1.0


def _list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_projected_3ddfa_landmarks_2d(path: str | Path) -> Optional[np.ndarray]:
    try:
        meta = read_dfljpg_metadata(path)
    except Exception:
        return None

    mvfs = meta.get("mvfs", {})
    if isinstance(mvfs, dict):
        td = mvfs.get("3ddfa_v3", {})
        if isinstance(td, dict) and td.get("landmarks") is not None:
            arr = np.asarray(td["landmarks"], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 68 and arr.shape[1] >= 2:
                return arr[:68, :2].astype(np.float32)

    lm = meta.get("landmarks", None)
    if lm is not None:
        arr = np.asarray(lm, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 68 and arr.shape[1] >= 2:
            return arr[:68, :2].astype(np.float32)

    return None


def load_face_mask_from_dfljpg(path: str | Path, image_size: Optional[int] = None) -> Optional[np.ndarray]:
    try:
        meta = read_dfljpg_metadata(path)
    except Exception:
        return None

    mvfs = meta.get("mvfs", {}) if isinstance(meta, dict) else {}
    face_seg = mvfs.get("face_seg", {}) if isinstance(mvfs, dict) else {}

    mask_path = face_seg.get("mask_path", None)
    if mask_path:
        img_path = Path(path)
        dataset_root = img_path.parent.parent.parent
        full_mask_path = dataset_root / mask_path
        if full_mask_path.exists():
            mask = cv2.imread(str(full_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if image_size is not None and (mask.shape[0] != image_size or mask.shape[1] != image_size):
                    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                return mask

    return None


def _draw_gaussian(heat: np.ndarray, x: float, y: float, sigma: float):
    h, w = heat.shape
    r = int(max(1, round(3 * sigma)))
    x0, y0 = int(round(x)), int(round(y))
    x1, x2 = max(0, x0 - r), min(w - 1, x0 + r)
    y1, y2 = max(0, y0 - r), min(h - 1, y0 + r)
    if x2 < x1 or y2 < y1:
        return
    yy, xx = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    heat[y1:y2 + 1, x1:x2 + 1] = np.maximum(heat[y1:y2 + 1, x1:x2 + 1], g)


def _draw_polyline_heat(heat: np.ndarray, pts: np.ndarray, thickness: int = 2):
    tmp = np.zeros_like(heat, dtype=np.uint8)
    pts_i = np.asarray(pts, dtype=np.int32)
    if len(pts_i) >= 2:
        cv2.polylines(tmp, [pts_i.reshape(-1, 1, 2)], False, 255, thickness, cv2.LINE_AA)
    heat[:] = np.maximum(heat, tmp.astype(np.float32) / 255.0)


def render_landmark_condition(
    landmarks: np.ndarray,
    image_size: int,
    sigma: float = 2.0,
    draw_lines: bool = True,
) -> torch.Tensor:
    """
    landmarks는 68점 3DDFA 뒤에 gaze iris 10점이 추가되어도 됨.
    선은 기존 68점 chain만 그리고, 모든 점에는 gaussian을 찍는다.
    따라서 gaze 10점은 선 연결 없이 점으로만 들어간다.
    """
    lm = np.asarray(landmarks, dtype=np.float32)
    heat = np.zeros((image_size, image_size), np.float32)

    chains = [
        list(range(0, 17)), list(range(17, 22)), list(range(22, 27)),
        list(range(27, 31)), list(range(31, 36)),
        list(range(36, 42)) + [36], list(range(42, 48)) + [42],
        list(range(48, 60)) + [48], list(range(60, 68)) + [60],
    ]

    if draw_lines and lm.shape[0] >= 68:
        for ch in chains:
            _draw_polyline_heat(heat, lm[ch, :2], thickness=max(1, int(round(sigma))))

    for x, y in lm[:, :2]:
        _draw_gaussian(heat, float(x), float(y), sigma)

    return torch.from_numpy(np.clip(heat, 0, 1)[None].astype(np.float32))


class TeacherBlurDataset(Dataset):
    """
    Teacher sample 구성:
      - clean        : A2 clean GT (-1~1)
      - blur_rgb     : 얼굴 영역만 blur 처리한 APPLE-style condition RGB (-1~1)
      - landmark_map : projected 2D 3DDFA landmark + gaze iris 5+5 heatmap (0~1, 1ch)
      - identity     : A1 identity image (-1~1)
      - face_mask    : debug / optional masked loss 용 (0~1, 1ch)
    """

    def __init__(
        self,
        index_path: str | Path,
        image_size: int = 512,
        random_identity_same_dir: bool = False,
        landmark_sigma: float = 2.0,
        landmark_draw_lines: bool = True,
        blur_downsample_size: int = 8,
        blur_gaussian_radius: float = 8.0,
        blur_feather_sigma: float = 0.0,
    ):
        self.index_path = Path(index_path)
        self.image_size = int(image_size)
        self.random_identity_same_dir = random_identity_same_dir
        self.landmark_sigma = float(landmark_sigma)
        self.landmark_draw_lines = landmark_draw_lines
        self.blur_downsample_size = int(blur_downsample_size)
        self.blur_gaussian_radius = float(blur_gaussian_radius)
        self.blur_feather_sigma = float(blur_feather_sigma)

        self.rows = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        if not self.rows:
            raise ValueError(f"Empty index: {self.index_path}")

    def __len__(self):
        return len(self.rows)

    def _infer_identity_path(self, row: Dict[str, Any]) -> str:
        if row.get("identity_path"):
            return row["identity_path"]

        clean_path = Path(row["clean_path"])
        a1_dir = clean_path.parent.parent / "A1"
        imgs = _list_images(a1_dir)
        if imgs:
            return str(random.choice(imgs) if self.random_identity_same_dir else imgs[0])

        return str(clean_path)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        clean_path = row["clean_path"]
        identity_path = self._infer_identity_path(row)

        clean_bgr = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if clean_bgr is None:
            raise FileNotFoundError(clean_path)

        orig_h, orig_w = clean_bgr.shape[:2]
        if (orig_h, orig_w) != (self.image_size, self.image_size):
            clean_bgr = cv2.resize(clean_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        face_mask = load_face_mask_from_dfljpg(clean_path, image_size=self.image_size)
        if face_mask is None:
            face_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        blur_bgr = build_face_blur_condition_rgb(
            clean_bgr,
            face_mask,
            downsample_size=self.blur_downsample_size,
            gaussian_radius=self.blur_gaussian_radius,
            feather_sigma=self.blur_feather_sigma,
        )

        clean = torch.from_numpy(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        clean = clean * 2.0 - 1.0
        blur_rgb = torch.from_numpy(cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        blur_rgb = blur_rgb * 2.0 - 1.0
        identity = _read_image_rgb(identity_path, self.image_size)

        lm = load_projected_3ddfa_landmarks_2d(clean_path)
        if lm is None:
            landmark_map = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            lm = lm.copy()

            sx = float(self.image_size) / float(orig_w)
            sy = float(self.image_size) / float(orig_h)

            lm[:, 0] *= sx
            lm[:, 1] *= sy

            # 핵심 변경점:
            # clean_path 옆의 .mvfs_gaze.json에서 iris 5+5점을 읽어
            # 3DDFA landmarks 뒤에 그대로 append한다.
            lm = append_gaze_to_landmarks_2d(
                image_path=clean_path,
                landmarks_2d=lm,
                scale_x=sx,
                scale_y=sy,
                target_w=self.image_size,
                target_h=self.image_size,
            )

            landmark_map = render_landmark_condition(
                lm,
                self.image_size,
                sigma=self.landmark_sigma,
                draw_lines=self.landmark_draw_lines,
            )

        face_mask_t = torch.from_numpy(face_mask.astype(np.float32) / 255.0)[None]

        sample = {
            "clean": clean,
            "blur_rgb": blur_rgb,
            "condition_rgb": blur_rgb,
            "landmark_map": landmark_map,
            "face_mask": face_mask_t,
            "identity": identity,
            "clean_path": clean_path,
            "identity_path": identity_path,
            "person_id": row.get("person_id", ""),
        }

        id_embed_path = row.get("id_embed_path")
        if id_embed_path:
            arr = np.load(id_embed_path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr["embedding"]
            sample["id_embed"] = torch.from_numpy(np.asarray(arr, dtype=np.float32))
            sample["id_embed_path"] = id_embed_path

        return sample
