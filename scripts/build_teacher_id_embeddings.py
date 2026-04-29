# -*- coding: utf-8 -*-
"""
Build InsightFace buffalo_l ID embeddings for teacher blur reconstruction training
and attach id_embed_path to teacher_index.jsonl.

Input:
    workspace/person_x/teacher/teacher_index.jsonl

Expected index record fields, minimally:
    {
      "clean_path": "...",
      "condition_path": "...",
      ...
    }

This script:
    1. reads clean_path or identity_path from each record
    2. extracts InsightFace buffalo_l embedding
    3. saves one .npy per sample under teacher/id_embeddings/
    4. writes teacher_index_with_id.jsonl with id_embed_path
    5. writes id_embedding_mean.npz for this teacher set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(p: str, base: Path) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    return (base / q).resolve()


class InsightFaceBuffaloLEncoder:
    def __init__(self, ctx_id: int = 0, det_size=(640, 640), providers=None):
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
            except Exception as e:
                print(f"[WARN] onnxruntime preload_dlls failed: {e}")

        from insightface.app import FaceAnalysis

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def encode(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(image_bgr)
        if not faces:
            return None

        # Use largest detected face.
        def area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return float(max(0, x2 - x1) * max(0, y2 - y1))

        face = max(faces, key=area)
        emb = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm
        return emb.astype(np.float32)


def choose_identity_image_path(row: Dict[str, Any]) -> Optional[str]:
    # Priority: explicitly set identity_path, then clean_path, then image_path.
    for key in ["identity_path", "clean_path", "aligned_path", "image_path"]:
        v = row.get(key)
        if v:
            return str(v)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=str, help="Input teacher_index.jsonl")
    p.add_argument("--out-index", default=None, type=str, help="Output jsonl with id_embed_path")
    p.add_argument("--embed-dir", default=None, type=str, help="Directory to store per-sample .npy embeddings")
    p.add_argument("--mean-out", default=None, type=str, help="Output npz path for mean embedding")
    p.add_argument("--ctx-id", default=0, type=int, help="0 for first GPU, -1 for CPU")
    p.add_argument("--det-size", default=640, type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fallback-zero", action="store_true", help="Save zero embedding if detection fails")
    args = p.parse_args()

    index_path = Path(args.index).resolve()
    index_dir = index_path.parent
    rows = read_jsonl(index_path)

    out_index = Path(args.out_index).resolve() if args.out_index else index_dir / "teacher_index_with_id.jsonl"
    embed_dir = Path(args.embed_dir).resolve() if args.embed_dir else index_dir / "id_embeddings"
    mean_out = Path(args.mean_out).resolve() if args.mean_out else index_dir / "id_embedding_mean.npz"

    embed_dir.mkdir(parents=True, exist_ok=True)

    encoder = InsightFaceBuffaloLEncoder(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))

    new_rows: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    failed = 0

    for i, row in enumerate(rows):
        src = choose_identity_image_path(row)
        if not src:
            print(f"[SKIP] row {i}: no identity/clean image path")
            failed += 1
            continue

        img_path = resolve_path(src, index_dir)
        if not img_path.exists():
            print(f"[SKIP] row {i}: image not found: {img_path}")
            failed += 1
            continue

        out_name = Path(row.get("clean_path", img_path.name)).stem
        if not out_name:
            out_name = f"{i:08d}"
        emb_path = embed_dir / f"{out_name}.npy"

        emb: Optional[np.ndarray] = None
        if emb_path.exists() and not args.overwrite:
            emb = np.load(str(emb_path)).astype(np.float32)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[SKIP] row {i}: cannot read: {img_path}")
                failed += 1
                continue

            emb = encoder.encode(img)
            if emb is None:
                if args.fallback_zero:
                    emb = np.zeros((512,), dtype=np.float32)
                    print(f"[WARN] row {i}: no face for ID embedding, using zero: {img_path}")
                else:
                    print(f"[SKIP] row {i}: no face for ID embedding: {img_path}")
                    failed += 1
                    continue
            np.save(str(emb_path), emb.astype(np.float32))

        row2 = dict(row)
        row2["id_embed_path"] = str(emb_path)
        row2["id_encoder"] = "insightface_buffalo_l"
        new_rows.append(row2)
        embeddings.append(emb.astype(np.float32))

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(rows)}] processed")

    write_jsonl(out_index, new_rows)

    if embeddings:
        arr = np.stack(embeddings, axis=0).astype(np.float32)
        mean = arr.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        np.savez(str(mean_out), embedding_mean=mean.astype(np.float32), embeddings=arr, count=len(embeddings))
        print(f"[OK] mean embedding saved: {mean_out}")

    print(f"[DONE] input={len(rows)}, output={len(new_rows)}, failed={failed}")
    print(f"[OK] out index: {out_index}")


if __name__ == "__main__":
    main()
