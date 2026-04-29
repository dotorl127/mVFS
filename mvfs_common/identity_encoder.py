# -*- coding: utf-8 -*-
"""InsightFace identity embedding helper for MVFS.

Default choice for MVFS v0: buffalo_l.
- Easy auto-download through insightface.
- Includes SCRFD detector + ResNet50 recognition model.
- Lighter than antelopev2/Glint360K-R100 style setups.

Install example:
    pip install insightface onnxruntime-gpu
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class FaceEmbeddingResult:
    embedding: np.ndarray
    bbox: np.ndarray
    det_score: float
    image_path: Optional[str] = None


class InsightFaceIDEncoder:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Optional[Sequence[str]] = None,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
    ):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "InsightFace is required for InsightFaceIDEncoder. "
                "Install with: pip install insightface onnxruntime-gpu"
            ) from e

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, providers=list(providers))
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.model_name = model_name

    @staticmethod
    def _select_largest_face(faces):
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))

    def encode_image(self, image_bgr: np.ndarray, select: str = "largest") -> FaceEmbeddingResult:
        faces = self.app.get(image_bgr)
        if not faces:
            raise RuntimeError("No face detected by InsightFace.")
        if select == "largest":
            face = self._select_largest_face(faces)
        elif select == "first":
            face = faces[0]
        else:
            raise ValueError(f"Unknown select={select}")
        emb = np.asarray(face.embedding, dtype=np.float32)
        emb /= max(np.linalg.norm(emb), 1e-12)
        return FaceEmbeddingResult(
            embedding=emb,
            bbox=np.asarray(face.bbox, dtype=np.float32),
            det_score=float(getattr(face, "det_score", 0.0)),
        )

    def encode_file(self, path: str | Path, select: str = "largest") -> FaceEmbeddingResult:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        res = self.encode_image(img, select=select)
        res.image_path = str(path)
        return res


def build_mean_embedding(
    image_paths: Sequence[str | Path],
    encoder: InsightFaceIDEncoder,
    min_det_score: float = 0.0,
) -> dict:
    embs = []
    kept = []
    rejected = []
    for p in image_paths:
        try:
            res = encoder.encode_file(p)
            if res.det_score < min_det_score:
                rejected.append({"path": str(p), "reason": f"det_score<{min_det_score}", "score": res.det_score})
                continue
            embs.append(res.embedding)
            kept.append({"path": str(p), "det_score": res.det_score, "bbox": res.bbox.tolist()})
        except Exception as e:
            rejected.append({"path": str(p), "reason": str(e)})
    if not embs:
        raise RuntimeError("No valid embeddings.")
    arr = np.stack(embs).astype(np.float32)
    mean = arr.mean(axis=0)
    mean /= max(np.linalg.norm(mean), 1e-12)
    return {"embedding_mean": mean.astype(np.float32), "embeddings": arr, "kept": kept, "rejected": rejected}
