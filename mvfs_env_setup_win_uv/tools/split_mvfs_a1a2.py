# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + eps)


def list_images(id_dir: Path, recursive: bool = False):
    skip_dirs = {"A1", "A2", "A_trash", "trash", "meta"}
    iterator = id_dir.rglob("*") if recursive else id_dir.iterdir()
    return sorted([
        p for p in iterator
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not any(part in skip_dirs for part in p.parts)
    ])


def make_unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst

    stem, suffix = dst.stem, dst.suffix
    k = 1
    while True:
        cand = dst.with_name(f"{stem}_{k:03d}{suffix}")
        if not cand.exists():
            return cand
        k += 1


def copy_move_or_link(src: Path, dst: Path, mode: str):
    """
    mode:
      copy     : 원본 보존, dst로 복사
      move     : 원본에서 dst로 이동
      hardlink : 가능하면 하드링크, 실패하면 copy
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = make_unique_path(dst)

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def pad_image_bgr(img_bgr: np.ndarray, pad_ratio: float = 0.25) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    return cv2.copyMakeBorder(
        img_bgr,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def choose_main_face(faces):
    if not faces:
        return None

    return max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )


def get_face_data_with_padding(app: FaceAnalysis, img_path: Path, pad_ratio: float):
    raw_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if raw_img is None:
        return None, "cannot_read"

    padded_img = pad_image_bgr(raw_img, pad_ratio=pad_ratio)

    faces = app.get(padded_img)
    face = choose_main_face(faces)

    if face is None:
        return None, "no_face"

    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)

    if emb is None:
        return None, "no_embedding"

    pose = getattr(face, "pose", None)
    if pose is None:
        yaw, pitch, roll = 999.0, 999.0, 999.0
    else:
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        yaw = float(pose[0]) if len(pose) > 0 else 999.0
        pitch = float(pose[1]) if len(pose) > 1 else 999.0
        roll = float(pose[2]) if len(pose) > 2 else 999.0

    return {
        "path": img_path,
        "filename": img_path.name,
        "emb": l2norm(np.asarray(emb, dtype=np.float32)),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "det_score": float(getattr(face, "det_score", 0.0)),
    }, "ok"


def pose_distance(a: dict, b: dict, yaw_norm=60.0, pitch_norm=40.0, roll_norm=40.0) -> float:
    ay, ap, ar = a["yaw"], a["pitch"], a["roll"]
    by, bp, br = b["yaw"], b["pitch"], b["roll"]

    if 999.0 in {ay, ap, ar, by, bp, br}:
        return 0.0

    dy = abs(ay - by) / yaw_norm
    dp = abs(ap - bp) / pitch_norm
    dr = abs(ar - br) / roll_norm

    return float(np.sqrt((dy * dy + dp * dp + dr * dr) / 3.0))


def frontal_penalty(f: dict) -> float:
    yaw = f["yaw"]
    pitch = f["pitch"]
    roll = f["roll"]

    if 999.0 in {yaw, pitch, roll}:
        return 999.0

    return abs(yaw) + abs(pitch) + 0.5 * abs(roll)


def compute_hist_feature(img_path: Path, bins: int = 8) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.astype(np.float32).reshape(-1)
    hist /= hist.sum() + 1e-8
    return hist


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    bc = float(np.sum(np.sqrt(np.maximum(a, 0) * np.maximum(b, 0))))
    bc = max(0.0, min(1.0, bc))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def select_pose_diverse(candidates: list[dict], max_count: int) -> list[dict]:
    """
    각도 다양성을 우선으로 A2 후보를 고른다.
    max_count <= 0이면 전체 후보를 diversity rank 순서로만 정렬한다.
    """
    if not candidates:
        return []

    selected = []

    first = max(candidates, key=lambda x: x["similarity"])
    selected.append(first)

    remaining = [x for x in candidates if x is not first]
    target_count = len(candidates) if max_count <= 0 else min(max_count, len(candidates))

    while remaining and len(selected) < target_count:
        best = None
        best_score = -1e9

        for item in remaining:
            min_pose_dist = min(pose_distance(item, s) for s in selected)
            score = min_pose_dist + 0.05 * item["similarity"]

            if score > best_score:
                best_score = score
                best = item

        selected.append(best)
        remaining.remove(best)

    for i, item in enumerate(selected):
        item["a2_pose_rank"] = i

    return selected


def filter_hist_similar(selected: list[dict], hist_min_dist: float, bins: int) -> tuple[list[dict], list[dict]]:
    """
    pose diversity로 고른 후보 중 hist가 너무 비슷한 건 trash.
    selected 순서를 유지하면서 greedy로 통과시킴.
    """
    kept = []
    trashed = []

    for item in selected:
        hist = compute_hist_feature(item["path"], bins=bins)

        if hist is None:
            item["trash_reason"] = "hist_failed"
            trashed.append(item)
            continue

        item["hist"] = hist

        if not kept:
            item["hist_min_dist"] = 999.0
            kept.append(item)
            continue

        min_dist = min(hist_distance(hist, k["hist"]) for k in kept)
        item["hist_min_dist"] = float(min_dist)

        if hist_min_dist > 0 and min_dist < hist_min_dist:
            item["trash_reason"] = f"hist_too_similar_{min_dist:.4f}"
            trashed.append(item)
        else:
            kept.append(item)

    return kept, trashed


def get_output_dirs(src_id_dir: Path, root: Path, out_root: Optional[Path], person_id: str):
    """
    out_root가 None이면 in-place:
      root/person_id/A1,A2,A_trash,meta

    out_root가 있으면 separate output:
      out_root/person_id/A1,A2,A_trash,meta
    """
    base = src_id_dir if out_root is None else out_root / person_id
    return {
        "base": base,
        "A1": base / "A1",
        "A2": base / "A2",
        "A_trash": base / "A_trash",
        "meta": base / "meta",
    }


def process_id_dir(app: FaceAnalysis, id_dir: Path, root: Path, out_root: Optional[Path], args):
    person_id = id_dir.name
    dirs = get_output_dirs(id_dir, root, out_root, person_id)

    for key in ["A1", "A2", "A_trash", "meta"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    img_paths = list_images(id_dir, recursive=args.recursive)

    records = []
    valid_faces = []

    # 1. 모든 이미지 분석
    for img_path in img_paths:
        data, reason = get_face_data_with_padding(app, img_path, pad_ratio=args.pad_ratio)

        rec = {
            "filename": img_path.name,
            "path": str(img_path),
            "detect_reason": reason,
            "split": None,
            "similarity": None,
            "yaw": None,
            "pitch": None,
            "roll": None,
            "hist_min_dist": None,
            "a2_pose_rank": None,
            "trash_reason": None,
        }

        if data is None:
            rec["split"] = "A_trash"
            rec["trash_reason"] = reason
            records.append(rec)
            copy_move_or_link(img_path, dirs["A_trash"] / img_path.name, args.mode)
            continue

        rec["yaw"] = data["yaw"]
        rec["pitch"] = data["pitch"]
        rec["roll"] = data["roll"]
        records.append(rec)

        data["record"] = rec
        valid_faces.append(data)

    if not valid_faces:
        write_report(dirs["meta"], records)
        return {
            "id": person_id,
            "valid": 0,
            "A1": 0,
            "A2": 0,
            "trash": len(img_paths),
            "status": "no_valid_face",
        }

    # 2. 평균 embedding 계산
    all_embs = np.stack([f["emb"] for f in valid_faces], axis=0)
    mean_emb = l2norm(all_embs.mean(axis=0))
    np.save(str(dirs["meta"] / "mean_identity.npy"), mean_emb.astype(np.float32))

    # 3. 평균과 유사도 계산 및 identity filter
    final_candidates = []
    identity_trash = []

    for f in valid_faces:
        sim = float(np.dot(mean_emb, f["emb"]))
        f["similarity"] = sim
        f["record"]["similarity"] = sim

        if sim < args.identity_sim:
            f["record"]["split"] = "A_trash"
            f["record"]["trash_reason"] = f"low_mean_similarity_{sim:.4f}"
            identity_trash.append(f)
        else:
            final_candidates.append(f)

    # 모든 후보가 threshold 미달이면 평균과 가장 가까운 1장은 살림
    if not final_candidates:
        best = max(valid_faces, key=lambda x: x["similarity"])
        best["record"]["trash_reason"] = None
        final_candidates = [best]
        identity_trash = [f for f in valid_faces if f is not best]
        for f in identity_trash:
            f["record"]["split"] = "A_trash"
            f["record"]["trash_reason"] = f"low_mean_similarity_{f['similarity']:.4f}"

    # 4. A1 선정: 평균과 가깝고 정면
    final_candidates.sort(
        key=lambda x: x["similarity"] - args.a1_pose_weight * frontal_penalty(x),
        reverse=True,
    )
    a1 = final_candidates[0]

    a1["record"]["split"] = "A1"
    a1["record"]["trash_reason"] = None
    copy_move_or_link(a1["path"], dirs["A1"] / a1["filename"], args.mode)

    # 5. identity trash
    for f in identity_trash:
        if f is a1:
            continue
        copy_move_or_link(f["path"], dirs["A_trash"] / f["filename"], args.mode)

    # 6. 나머지 후보를 A2 후보로 두고 각도 다양성 우선 정렬/선택
    a2_candidates = [f for f in final_candidates[1:] if f not in identity_trash]
    pose_selected = select_pose_diverse(a2_candidates, max_count=args.a2_max_count)

    pose_selected_set = {f["filename"] for f in pose_selected}

    pose_trash = []
    for f in a2_candidates:
        if f["filename"] not in pose_selected_set:
            f["record"]["split"] = "A_trash"
            f["record"]["trash_reason"] = "not_selected_by_pose_diversity"
            pose_trash.append(f)

    # 7. hist가 너무 비슷한 샘플 제거
    hist_kept, hist_trash = filter_hist_similar(
        pose_selected,
        hist_min_dist=args.hist_min_dist,
        bins=args.hist_bins,
    )

    # 8. 최종 A2 이동/복사
    for f in hist_kept:
        f["record"]["split"] = "A2"
        f["record"]["trash_reason"] = None
        f["record"]["hist_min_dist"] = f.get("hist_min_dist", None)
        f["record"]["a2_pose_rank"] = f.get("a2_pose_rank", None)
        copy_move_or_link(f["path"], dirs["A2"] / f["filename"], args.mode)

    # 9. pose/hist trash
    for f in pose_trash + hist_trash:
        f["record"]["split"] = "A_trash"
        f["record"]["hist_min_dist"] = f.get("hist_min_dist", None)
        f["record"]["a2_pose_rank"] = f.get("a2_pose_rank", None)
        copy_move_or_link(f["path"], dirs["A_trash"] / f["filename"], args.mode)

    write_report(dirs["meta"], records)

    return {
        "id": person_id,
        "valid": len(valid_faces),
        "A1": 1,
        "A2": len(hist_kept),
        "trash": sum(1 for r in records if r.get("split") == "A_trash"),
        "status": "ok",
    }


def write_report(meta_dir: Path, records: list[dict]):
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "split_report.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser("VGGFace2-HQ padded InsightFace split: A1/A2/A_trash")
    ap.add_argument("--root", default="D:/MVFS/dataset/VGGFace2-HQ")
    ap.add_argument("--out-root", default="D:/MVFS/dataset/VGGFace2-HQ-split")
    ap.add_argument("--mode", default="copy", choices=["copy", "move", "hardlink"])
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--ctx-id", type=int, default=0)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--det-thresh", type=float, default=0.2)
    ap.add_argument("--pad-ratio", type=float, default=0.25)

    # identity filter
    ap.add_argument("--identity-sim", type=float, default=0.4)

    # A1 selection
    ap.add_argument("--a1-pose-weight", type=float, default=0.01)

    # A2 pose diversity + histogram filtering
    ap.add_argument("--a2-max-count", type=int, default=0, help="0이면 개수 제한 없음")
    ap.add_argument("--hist-min-dist", type=float, default=0.08, help="0이면 hist 유사 샘플 제거 안 함")
    ap.add_argument("--hist-bins", type=int, default=8)

    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "landmark_3d_68", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(
        ctx_id=args.ctx_id,
        det_size=(args.det_size, args.det_size),
        det_thresh=args.det_thresh,
    )

    id_dirs = [
        p for p in sorted(root.iterdir())
        if p.is_dir() and p.name not in {"A1", "A2", "A_trash", "trash", "meta"}
    ]

    summaries = []
    for id_dir in tqdm(id_dirs, desc="ID 정리 중"):
        summaries.append(process_id_dir(app, id_dir, root, out_root, args))

    summary_root = out_root if out_root is not None else root
    with open(summary_root / "_split_summary.jsonl", "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("[DONE]")
    print("[SUMMARY]", summary_root / "_split_summary.jsonl")


if __name__ == "__main__":
    main()