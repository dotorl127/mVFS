# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

THIS = Path(__file__).resolve()
for p in [THIS.parent, *THIS.parents]:
    if (p / "mvfs_common").exists():
        sys.path.insert(0, str(p))
        break

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(path: Path, recursive: bool = False) -> List[Path]:
    if not path.exists():
        return []
    it = path.rglob("*") if recursive else path.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


class InsightFaceDirectEncoder:
    def __init__(self, ctx_id: int = 0):
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
            except Exception as e:
                print(f"[WARN] onnxruntime preload_dlls failed: {e}")

        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.rec_model = self.app.models.get("recognition", None)
        if self.rec_model is None:
            raise RuntimeError("InsightFace buffalo_l recognition model was not loaded.")

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        return x / (np.linalg.norm(x) + 1e-8)

    def encode_direct_aligned(self, img_bgr: np.ndarray):
        try:
            feat = self.rec_model.get_feat(img_bgr)
            if feat is None:
                return None
            return self.normalize(feat)
        except Exception as e:
            print(f"[WARN] ID encode failed: {e}")
            return None


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_mean_embedding(paths: List[Path], encoder: InsightFaceDirectEncoder, fallback_zero: bool = True):
    embs, failed = [], []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            failed.append(str(p))
            continue
        emb = encoder.encode_direct_aligned(img)
        if emb is None:
            failed.append(str(p))
            continue
        embs.append(emb)

    if not embs:
        if not fallback_zero:
            raise RuntimeError("No ID embeddings extracted. Use aligned A1 images or --fallback-zero-id.")
        mean = np.zeros((512,), dtype=np.float32)
    else:
        mean = np.stack(embs, 0).mean(0).astype(np.float32)
        mean = mean / (np.linalg.norm(mean) + 1e-8)

    return mean.astype(np.float32), {"num_input": len(paths), "num_success": len(embs), "num_failed": len(failed), "failed": failed[:20]}


def main():
    ap = argparse.ArgumentParser("Build teacher_index.jsonl and A1 mean ID embeddings from id/A1,A2 dataset")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--a1-name", default="A1")
    ap.add_argument("--a2-name", default="A2")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--ctx-id", type=int, default=0)
    ap.add_argument("--fallback-zero-id", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve()
    meta_root = out_root / "meta"
    emb_root = out_root / "id_embeddings"
    meta_root.mkdir(parents=True, exist_ok=True)
    emb_root.mkdir(parents=True, exist_ok=True)

    encoder = InsightFaceDirectEncoder(ctx_id=args.ctx_id)

    rows, reports, skipped = [], [], []
    id_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith("meta_")])

    for i, id_dir in enumerate(id_dirs):
        person_id = id_dir.name
        a1 = list_images(id_dir / args.a1_name, recursive=args.recursive)
        a2 = list_images(id_dir / args.a2_name, recursive=args.recursive)

        if not a1:
            skipped.append({"person_id": person_id, "reason": "no_A1"})
            continue
        if not a2:
            skipped.append({"person_id": person_id, "reason": "no_A2"})
            continue

        emb, report = build_mean_embedding(a1, encoder, fallback_zero=args.fallback_zero_id)
        emb_path = emb_root / f"{person_id}.npy"
        np.save(str(emb_path), emb)

        report["person_id"] = person_id
        report["id_embed_path"] = str(emb_path)
        reports.append(report)

        # First A1 image is used only as ID debug image and ID-loss target.
        identity_path = str(a1[0])

        for img in a2:
            rows.append({
                "person_id": person_id,
                "clean_path": str(img),
                "identity_path": identity_path,
                "id_embed_path": str(emb_path),
            })

        print(f"[{i+1}/{len(id_dirs)}] {person_id}: A1={len(a1)} A2={len(a2)} rows={len(a2)} id_ok={report['num_success']}")

    write_jsonl(meta_root / "teacher_index.jsonl", rows)
    write_jsonl(meta_root / "id_embedding_report.jsonl", reports)
    write_jsonl(out_root / "skipped.jsonl", skipped)

    print(f"[DONE] rows={len(rows)} skipped={len(skipped)}")
    print(f"[INDEX] {meta_root / 'teacher_index.jsonl'}")


if __name__ == "__main__":
    main()
