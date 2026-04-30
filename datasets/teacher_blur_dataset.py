# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
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
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return img * 2.0 - 1.0

def _list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

class TeacherBlurDataset(Dataset):
    """Dataset for teacher deblurring/reconstruction.

    JSONL row example:
      {
        "person_id": "m_000001",
        "clean_path": ".../m_000001/A2/frame_000123.jpg",
        "condition_path": ".../conditions/blur_only/m_000001/frame_000123.png",
        "id_embed_path": ".../id_embeddings/m_000001.npy"
      }

    identity_path rule:
      1) if row["identity_path"] exists -> use it
      2) else try to infer sibling A1 directory from clean_path and take first image
      3) else fallback to clean_path
    """

    def __init__(self, index_path: str | Path, image_size: int = 512, random_identity_same_dir: bool = False):
        self.index_path = Path(index_path)
        self.image_size = image_size
        self.rows = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"Empty index: {self.index_path}")
        self.random_identity_same_dir = random_identity_same_dir

    def __len__(self) -> int:
        return len(self.rows)

    def _infer_identity_path(self, row: Dict[str, Any]) -> str:
        if row.get("identity_path"):
            return row["identity_path"]

        clean_path = Path(row["clean_path"])
        # expected .../<person_id>/A2/file.jpg
        a2_dir = clean_path.parent
        person_dir = a2_dir.parent
        a1_dir = person_dir / "A1"
        a1_images = _list_images(a1_dir)
        if a1_images:
            return str(random.choice(a1_images) if self.random_identity_same_dir else a1_images[0])

        # fallback
        return str(clean_path)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        clean_path = row["clean_path"]
        cond_path = row["condition_path"]
        identity_path = self._infer_identity_path(row)

        sample = {
            "clean": _read_image_rgb(clean_path, self.image_size),
            "condition": _read_image_rgb(cond_path, self.image_size),
            "identity": _read_image_rgb(identity_path, self.image_size),
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
