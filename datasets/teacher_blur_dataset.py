# -*- coding: utf-8 -*-
from __future__ import annotations

import json, pickle, random
from pathlib import Path
from typing import Any, Dict, Optional, List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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

def read_dfljpg_metadata(path: str | Path) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return {}
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker == 0xDA:
            break
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            continue
        if i + 2 > len(data):
            break
        seg_len = int.from_bytes(data[i:i + 2], "big")
        if seg_len < 2 or i + seg_len > len(data):
            break
        seg_data = data[i + 2:i + seg_len]
        i += seg_len
        if marker == 0xEF:
            try:
                obj = pickle.loads(seg_data)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}

def load_projected_3ddfa_landmarks_2d(path: str | Path) -> Optional[np.ndarray]:
    """
    Current 3DDFA updater stores projected 2D landmarks here:
        meta["mvfs"]["3ddfa_v3"]["landmarks"]
    If top-level landmarks were replaced, fallback:
        meta["landmarks"]
    """
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
    mode: str = "single",
    sigma: float = 2.0,
    draw_lines: bool = True,
) -> torch.Tensor:
    lm = np.asarray(landmarks, dtype=np.float32)
    h = w = int(image_size)

    chains_all = [
        list(range(0, 17)), list(range(17, 22)), list(range(22, 27)),
        list(range(27, 31)), list(range(31, 36)),
        list(range(36, 42)) + [36], list(range(42, 48)) + [42],
        list(range(48, 60)) + [48], list(range(60, 68)) + [60],
    ]

    if mode == "single":
        heat = np.zeros((h, w), np.float32)
        if draw_lines:
            for ch in chains_all:
                _draw_polyline_heat(heat, lm[ch, :2], thickness=max(1, int(round(sigma))))
        for x, y in lm[:, :2]:
            _draw_gaussian(heat, float(x), float(y), sigma)
        return torch.from_numpy(np.clip(heat, 0, 1)[None].astype(np.float32))

    if mode == "parts":
        groups = [
            (list(range(0, 17)), [list(range(0, 17))]),
            (list(range(17, 27)) + list(range(36, 48)), [list(range(17, 22)), list(range(22, 27)), list(range(36, 42)) + [36], list(range(42, 48)) + [42]]),
            (list(range(27, 36)), [list(range(27, 31)), list(range(31, 36))]),
            (list(range(48, 68)), [list(range(48, 60)) + [48], list(range(60, 68)) + [60]]),
        ]
        maps = []
        for idx, chains in groups:
            heat = np.zeros((h, w), np.float32)
            if draw_lines:
                for ch in chains:
                    _draw_polyline_heat(heat, lm[ch, :2], thickness=max(1, int(round(sigma))))
            for x, y in lm[idx, :2]:
                _draw_gaussian(heat, float(x), float(y), sigma)
            maps.append(np.clip(heat, 0, 1))
        return torch.from_numpy(np.stack(maps, axis=0).astype(np.float32))

    raise ValueError(f"Unknown landmark_map_mode={mode}")

class TeacherBlurDataset(Dataset):
    """
    Returns:
      clean         : 3ch RGB [-1,1]
      condition     : blur RGB + projected 3DDFA landmark map
                      default 4ch = 3 + 1
      condition_rgb : 3ch blur RGB only, for debug
      landmark_map  : [0,1] map, for debug
      identity      : A1 image
      id_embed      : cached A1 average ID embedding
    """
    def __init__(
        self,
        index_path: str | Path,
        image_size: int = 512,
        random_identity_same_dir: bool = False,
        use_landmark_condition: bool = True,
        landmark_map_mode: str = "single",
        landmark_sigma: float = 2.0,
        landmark_draw_lines: bool = True,
    ):
        self.index_path = Path(index_path)
        self.image_size = int(image_size)
        self.random_identity_same_dir = random_identity_same_dir
        self.use_landmark_condition = use_landmark_condition
        self.landmark_map_mode = landmark_map_mode
        self.landmark_sigma = float(landmark_sigma)
        self.landmark_draw_lines = landmark_draw_lines

        self.rows = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"Empty index: {self.index_path}")

    def __len__(self): return len(self.rows)

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
        cond_path = row["condition_path"]
        identity_path = self._infer_identity_path(row)

        clean = _read_image_rgb(clean_path, self.image_size)
        condition_rgb = _read_image_rgb(cond_path, self.image_size)
        identity = _read_image_rgb(identity_path, self.image_size)

        if self.use_landmark_condition:
            lm = load_projected_3ddfa_landmarks_2d(clean_path)
            ch = 4 if self.landmark_map_mode == "parts" else 1
            if lm is None:
                landmark_map = torch.zeros((ch, self.image_size, self.image_size), dtype=torch.float32)
            else:
                img0 = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
                if img0 is not None:
                    h0, w0 = img0.shape[:2]
                    lm = lm.copy()
                    lm[:, 0] *= float(self.image_size) / float(w0)
                    lm[:, 1] *= float(self.image_size) / float(h0)
                landmark_map = render_landmark_condition(
                    lm,
                    image_size=self.image_size,
                    mode=self.landmark_map_mode,
                    sigma=self.landmark_sigma,
                    draw_lines=self.landmark_draw_lines,
                )
            condition = torch.cat([condition_rgb, landmark_map * 2.0 - 1.0], dim=0)
        else:
            landmark_map = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            condition = condition_rgb

        sample = {
            "clean": clean,
            "condition": condition,
            "condition_rgb": condition_rgb,
            "landmark_map": landmark_map,
            "identity": identity,
            "clean_path": clean_path,
            "condition_path": cond_path,
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
