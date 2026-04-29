# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


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


class TeacherBlurDataset(Dataset):
    """Dataset for teacher deblurring/reconstruction.

    JSONL row format:
      {"clean_path": "...jpg", "condition_path": "...png", "identity_path": "...jpg"}

    Returns:
      clean: RGB tensor [-1,1]
      condition: RGB tensor [-1,1]
      identity: RGB tensor [-1,1]
      optional id_embed: if id embedding path exists in row
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
        self._clean_paths = [r["clean_path"] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        clean_path = row["clean_path"]
        cond_path = row["condition_path"]
        identity_path = row.get("identity_path", clean_path)
        if self.random_identity_same_dir:
            identity_path = random.choice(self._clean_paths)

        sample = {
            "clean": _read_image_rgb(clean_path, self.image_size),
            "condition": _read_image_rgb(cond_path, self.image_size),
            "identity": _read_image_rgb(identity_path, self.image_size),
            "clean_path": clean_path,
            "condition_path": cond_path,
            "identity_path": identity_path,
        }

        id_embed_path = row.get("id_embed_path")
        if id_embed_path:
            arr = np.load(id_embed_path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr["embedding"]
            sample["id_embed"] = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return sample
