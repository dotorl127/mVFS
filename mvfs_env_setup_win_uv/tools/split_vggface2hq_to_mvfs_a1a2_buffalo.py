# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    person_id: str
    src_path: str
    filename: str
    ok: bool
    reason: str
    det_score: float = 0.0
    yaw: float = 999.0
    pitch: float = 999.0
    roll: float = 999.0
    blur_score: float = 0.0
    face_area_ratio: float = 0.0
    sim_to_mean: float = 0.0
    final_split: str = "trash"


def list_images(d: Path, recursive: bool = True) -> List[Path]:
    it = d.rglob("*") if recursive else d.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + eps)


def laplacian_blur_score(img_bgr: np.ndarray, bbox: Optional[np.ndarray] = None) -> float:
    if bbox is not None:
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            img_bgr = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_pose(face) -> Tuple[float, float, float]:
    pose = getattr(face, "pose", None)
    if pose is None:
        return 999.0, 999.0, 999.0
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size >= 3:
        return float(arr[0]), float(arr[1]), float(arr[2])
    return 999.0, 999.0, 999.0


def choose_main_face(faces, img_shape):
    if not faces:
        return None
    h, w = img_shape[:2]
    img_area = float(h * w)
    best = None
    best_score = -1.0
    for f in faces:
        bbox = np.asarray(f.bbox, dtype=np.float32)
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        det = float(getattr(f, "det_score", 0.0))
        score = det + 0.5 * (area / img_area)
        if score > best_score:
            best_score = score
            best = f
    return best


def safe_copy_or_move(src: Path, dst: Path, mode: str = "copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stem, suffix = dst.stem, dst.suffix
        k = 1
        while True:
            cand = dst.with_name(f"{stem}_{k:03d}{suffix}")
            if not cand.exists():
                dst = cand
                break
            k += 1

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def analyze_image(app, img_path: Path, person_id: str, args) -> Tuple[ImageRecord, Optional[np.ndarray]]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rec = ImageRecord(
        person_id=person_id,
        src_path=str(img_path),
        filename=img_path.name,
        ok=False,
        reason="init",
    )

    if img is None:
        rec.reason = "cannot_read"
        return rec, None

    faces = app.get(img)
    if len(faces) <= 0:
        rec.reason = "no_face"
        return rec, None

    if args.reject_multi_face and len(faces) > 1:
        rec.reason = f"multi_face_{len(faces)}"
        return rec, None

    face = choose_main_face(faces, img.shape)
    if face is None:
        rec.reason = "no_main_face"
        return rec, None

    bbox = np.asarray(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    face_area_ratio = float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(h * w))

    det_score = float(getattr(face, "det_score", 0.0))
    if det_score < args.min_det_score:
        rec.reason = f"low_det_{det_score:.3f}"
        rec.det_score = det_score
        return rec, None

    if face_area_ratio < args.min_face_area:
        rec.reason = f"small_face_{face_area_ratio:.3f}"
        rec.det_score = det_score
        rec.face_area_ratio = face_area_ratio
        return rec, None

    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        rec.reason = "no_embedding"
        return rec, None

    emb = l2norm(np.asarray(emb, dtype=np.float32))

    yaw, pitch, roll = get_pose(face)
    blur_score = laplacian_blur_score(img, bbox)

    rec.ok = True
    rec.reason = "ok"
    rec.det_score = det_score
    rec.yaw = yaw
    rec.pitch = pitch
    rec.roll = roll
    rec.blur_score = blur_score
    rec.face_area_ratio = face_area_ratio

    return rec, emb


def robust_mean_and_sims(embs: np.ndarray, outlier_sim: float):
    mean = l2norm(embs.mean(axis=0))
    sims = embs @ mean

    keep = sims >= outlier_sim
    if keep.sum() >= max(2, min(5, len(embs))):
        mean = l2norm(embs[keep].mean(axis=0))
        sims = embs @ mean

    return mean, sims


def is_frontal(rec: ImageRecord, args) -> bool:
    if rec.yaw == 999.0:
        return True
    return (
        abs(rec.yaw) <= args.frontal_yaw
        and abs(rec.pitch) <= args.frontal_pitch
        and abs(rec.roll) <= args.frontal_roll
    )


def write_records(dst_root: Path, person_id: str, records: List[ImageRecord]):
    meta_dir = dst_root / person_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "split_report.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def process_identity(app, person_dir: Path, dst_root: Path, args) -> dict:
    person_id = person_dir.name
    img_paths = list_images(person_dir, recursive=args.recursive)

    report = {
        "person_id": person_id,
        "num_images": len(img_paths),
        "num_valid_face": 0,
        "num_a1": 0,
        "num_a2": 0,
        "num_trash": 0,
        "status": "init",
        "reason": "",
    }

    if len(img_paths) < args.min_images_per_id:
        report["status"] = "skipped"
        report["reason"] = "too_few_images"
        return report

    records: List[ImageRecord] = []
    embs: List[np.ndarray] = []

    for p in img_paths:
        rec, emb = analyze_image(app, p, person_id, args)
        records.append(rec)
        if rec.ok and emb is not None:
            embs.append(emb)

    valid_records = [r for r in records if r.ok]
    report["num_valid_face"] = len(valid_records)

    if len(valid_records) < args.min_valid_per_id:
        for r in records:
            r.final_split = "trash"
            safe_copy_or_move(Path(r.src_path), dst_root / person_id / "trash" / r.filename, args.mode)
        report["status"] = "skipped"
        report["reason"] = "too_few_valid_faces"
        write_records(dst_root, person_id, records)
        return report

    embs_np = np.stack(embs, axis=0).astype(np.float32)
    mean_emb, sims = robust_mean_and_sims(embs_np, args.outlier_sim)

    k = 0
    for r in records:
        if r.ok:
            r.sim_to_mean = float(sims[k])
            k += 1

    candidates = []
    for r in records:
        if not r.ok:
            r.final_split = "trash"
            continue
        if r.sim_to_mean < args.outlier_sim:
            r.ok = False
            r.reason = f"identity_outlier_{r.sim_to_mean:.3f}"
            r.final_split = "trash"
            continue
        if r.blur_score < args.min_blur_score:
            r.ok = False
            r.reason = f"too_blurry_{r.blur_score:.1f}"
            r.final_split = "trash"
            continue
        candidates.append(r)

    if len(candidates) < args.min_valid_per_id:
        for r in records:
            r.final_split = "trash"
            safe_copy_or_move(Path(r.src_path), dst_root / person_id / "trash" / r.filename, args.mode)
        report["status"] = "skipped"
        report["reason"] = "too_few_candidates_after_filter"
        write_records(dst_root, person_id, records)
        return report

    frontal_candidates = [r for r in candidates if is_frontal(r, args)]
    if not frontal_candidates:
        frontal_candidates = candidates

    def a1_score(r: ImageRecord) -> float:
        pose_penalty = 0.0
        if r.yaw != 999.0:
            pose_penalty = (
                abs(r.yaw) / max(args.frontal_yaw, 1e-6)
                + abs(r.pitch) / max(args.frontal_pitch, 1e-6)
                + abs(r.roll) / max(args.frontal_roll, 1e-6)
            ) * 0.05
        blur_bonus = min(r.blur_score / 500.0, 1.0) * 0.05
        det_bonus = r.det_score * 0.03
        return r.sim_to_mean + blur_bonus + det_bonus - pose_penalty

    frontal_candidates = sorted(frontal_candidates, key=a1_score, reverse=True)
    a1_paths = set(r.src_path for r in frontal_candidates[: args.a1_count])

    for r in records:
        src = Path(r.src_path)
        if r.src_path in a1_paths:
            r.final_split = "A1"
            dst = dst_root / person_id / "A1" / r.filename
        elif r.ok:
            r.final_split = "A2"
            dst = dst_root / person_id / "A2" / r.filename
        else:
            r.final_split = "trash"
            dst = dst_root / person_id / "trash" / r.filename
        safe_copy_or_move(src, dst, args.mode)

    meta_dir = dst_root / person_id / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(meta_dir / "mean_identity.npy"), mean_emb.astype(np.float32))

    report["num_a1"] = sum(1 for r in records if r.final_split == "A1")
    report["num_a2"] = sum(1 for r in records if r.final_split == "A2")
    report["num_trash"] = sum(1 for r in records if r.final_split == "trash")
    report["status"] = "ok"
    report["reason"] = "ok"

    write_records(dst_root, person_id, records)
    return report


def main():
    ap = argparse.ArgumentParser("Split filtered VGGFace2-HQ ids into MVFS A1/A2/trash using InsightFace buffalo_l")
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--mode", default="copy", choices=["copy", "move", "hardlink"])
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--ctx-id", type=int, default=0)
    ap.add_argument("--det-size", type=int, default=640)

    ap.add_argument("--reject-multi-face", action="store_true")
    ap.add_argument("--min-det-score", type=float, default=0.60)
    ap.add_argument("--min-face-area", type=float, default=0.04)
    ap.add_argument("--min-blur-score", type=float, default=20.0)
    ap.add_argument("--min-images-per-id", type=int, default=4)
    ap.add_argument("--min-valid-per-id", type=int, default=3)

    ap.add_argument("--outlier-sim", type=float, default=0.35)
    ap.add_argument("--a1-count", type=int, default=1)
    ap.add_argument("--frontal-yaw", type=float, default=18.0)
    ap.add_argument("--frontal-pitch", type=float, default=18.0)
    ap.add_argument("--frontal-roll", type=float, default=15.0)
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except Exception as e:
            print("[WARN] onnxruntime preload_dlls failed:", e)

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))

    id_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])

    reports = []
    for person_dir in tqdm(id_dirs, desc="A1/A2 split"):
        reports.append(process_identity(app, person_dir, dst_root, args))

    with open(dst_root / "_summary.jsonl", "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in reports if r["status"] == "ok")
    print(f"[DONE] ok_id={ok}, skipped_id={len(reports) - ok}")
    print(f"[SUMMARY] {dst_root / '_summary.jsonl'}")


if __name__ == "__main__":
    main()