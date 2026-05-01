# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(id_dir: Path, recursive: bool = True) -> List[Path]:
    it = id_dir.rglob("*") if recursive else id_dir.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def normalize_gender(g: str) -> str:
    g = str(g).strip().lower()
    if g in {"woman", "female", "f"}:
        return "female"
    if g in {"man", "male", "m"}:
        return "male"
    return "unknown"


def normalize_race(r: str) -> str:
    r = str(r).strip().lower().replace(" ", "_").replace("-", "_")
    if r in {"black", "african", "african_american", "african_american_black"}:
        return "black"
    return r if r else "unknown"


def deepface_analyze_one(img_path: Path, detector_backend: str = "retinaface") -> Dict[str, str]:
    from deepface import DeepFace

    try:
        result = DeepFace.analyze(
            img_path=str(img_path),
            actions=["gender", "race"],
            detector_backend=detector_backend,
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            if len(result) == 0:
                return {"gender": "unknown", "race": "unknown"}
            result = result[0]

        gender = normalize_gender(result.get("dominant_gender", "unknown"))
        race = normalize_race(result.get("dominant_race", "unknown"))

        return {"gender": gender, "race": race}

    except Exception as e:
        return {"gender": "unknown", "race": "unknown", "error": str(e)}


def majority_label(labels: List[str]) -> tuple[str, int, int]:
    valid = [x for x in labels if x != "unknown"]
    if not valid:
        return "unknown", 0, len(labels)
    c = Counter(valid)
    label, count = c.most_common(1)[0]
    return label, count, len(labels)


def safe_move_dir(src: Path, dst_root: Path) -> Path:
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dst.exists():
        k = 1
        while True:
            cand = dst_root / f"{src.name}_{k:03d}"
            if not cand.exists():
                dst = cand
                break
            k += 1
    shutil.move(str(src), str(dst))
    return dst


def main():
    ap = argparse.ArgumentParser("Filter VGGFace2-HQ identities by DeepFace majority gender/race")
    ap.add_argument("--src-root", default="D:/MVFS/dataset/VGGFace2-HQ/1", help="VGGFace2-HQ root containing id directories")
    ap.add_argument("--kept-root", default="", help="Optional output root for kept ids. If empty, filter in-place.")
    ap.add_argument("--rejected-root", default="", help="Move rejected ids here. Recommended.")
    ap.add_argument("--delete-rejected", action="store_true", help="Actually delete rejected id directories. Dangerous.")
    ap.add_argument("--sample-count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--detector-backend", default="retinaface")
    ap.add_argument("--min-female-votes", type=int, default=3)
    ap.add_argument("--min-nonblack-votes", type=int, default=3)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    kept_root = Path(args.kept_root).resolve() if args.kept_root else None
    rejected_root = Path(args.rejected_root).resolve() if args.rejected_root else None

    if args.delete_rejected and rejected_root:
        raise ValueError("--delete-rejected and --rejected-root should not be used together.")

    if kept_root:
        kept_root.mkdir(parents=True, exist_ok=True)
    if rejected_root:
        rejected_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    id_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])

    report_path = Path(args.report).resolve() if args.report else src_root / "_deepface_filter_report.jsonl"
    kept_count = 0
    rejected_count = 0

    with report_path.open("w", encoding="utf-8") as rf:
        for id_dir in tqdm(id_dirs, desc="DeepFace id filtering"):
            images = list_images(id_dir, recursive=args.recursive)
            if len(images) == 0:
                decision = "reject"
                reason = "no_images"
                sample_paths = []
                analyses = []
            else:
                sample_paths = rng.sample(images, min(args.sample_count, len(images)))
                analyses = [deepface_analyze_one(p, args.detector_backend) for p in sample_paths]

                genders = [a.get("gender", "unknown") for a in analyses]
                races = [a.get("race", "unknown") for a in analyses]

                gender_major, gender_votes, total = majority_label(genders)

                nonblack_votes = sum(1 for r in races if r != "unknown" and r != "black")
                black_votes = sum(1 for r in races if r == "black")
                unknown_race = sum(1 for r in races if r == "unknown")

                female_ok = gender_major == "female" and gender_votes >= args.min_female_votes
                nonblack_ok = nonblack_votes >= args.min_nonblack_votes and black_votes == 0

                if female_ok and nonblack_ok:
                    decision = "keep"
                    reason = "female_nonblack_majority"
                else:
                    decision = "reject"
                    reason = (
                        f"gender={gender_major}:{gender_votes}/{total}, "
                        f"nonblack={nonblack_votes}, black={black_votes}, unknown_race={unknown_race}"
                    )

            rec = {
                "person_id": id_dir.name,
                "decision": decision,
                "reason": reason,
                "samples": [str(p) for p in sample_paths],
                "analyses": analyses,
            }
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if decision == "keep":
                kept_count += 1
                if kept_root:
                    dst = kept_root / id_dir.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(id_dir, dst)
            else:
                rejected_count += 1
                if args.delete_rejected:
                    shutil.rmtree(id_dir)
                elif rejected_root:
                    safe_move_dir(id_dir, rejected_root)

    print(f"[DONE] kept={kept_count}, rejected={rejected_count}")
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()