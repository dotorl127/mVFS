# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_race(r: str) -> str:
    r = str(r).strip().lower().replace("-", " ").replace("_", " ")
    r = " ".join(r.split())

    mapping = {
        "asian": "asian",
        "white": "white",
        "black": "black",
        "indian": "indian",
        "middle eastern": "middle eastern",
        "latino hispanic": "latino hispanic",
        "latino": "latino hispanic",
        "hispanic": "latino hispanic",
    }
    return mapping.get(r, r if r else "unknown")


def list_id_a2_images(root: Path):
    """
    return:
      {
        person_id: [A2 image paths...]
      }
    """
    by_id = {}

    for id_dir in sorted(root.iterdir()):
        if not id_dir.is_dir():
            continue

        a2_dir = id_dir / "A2"
        if not a2_dir.is_dir():
            continue

        imgs = [
            p for p in sorted(a2_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        if imgs:
            by_id[id_dir.name] = imgs

    return by_id


def pad_image_bgr(img_bgr, pad_ratio: float):
    if pad_ratio <= 0:
        return img_bgr

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


def analyze_race_with_padding(img_path: Path, detector_backend: str, pad_ratio: float):
    from deepface import DeepFace

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return {
            "race": "unknown",
            "reason": "cannot_read",
            "race_scores": {},
        }

    img = pad_image_bgr(img, pad_ratio)

    try:
        result = DeepFace.analyze(
            img_path=img,
            actions=["race"],
            detector_backend=detector_backend,
            enforce_detection=False,
            silent=True,
        )

        if isinstance(result, list):
            if not result:
                return {
                    "race": "unknown",
                    "reason": "empty_result",
                    "race_scores": {},
                }
            result = result[0]

        race = normalize_race(result.get("dominant_race", "unknown"))

        race_scores = result.get("race", {})
        if isinstance(race_scores, dict):
            race_scores = {normalize_race(k): float(v) for k, v in race_scores.items()}
        else:
            race_scores = {}

        return {
            "race": race,
            "reason": "ok",
            "race_scores": race_scores,
        }

    except Exception as e:
        return {
            "race": "unknown",
            "reason": f"error:{type(e).__name__}:{e}",
            "race_scores": {},
        }


def majority_race_from_samples(sample_results: list[dict], min_votes: int):
    races = [
        r["race"] for r in sample_results
        if r.get("race", "unknown") != "unknown"
    ]

    if not races:
        return "unknown", 0

    counter = Counter(races)
    race, votes = counter.most_common(1)[0]

    if votes < min_votes:
        return "unknown", votes

    return race, votes


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
        raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser("Sample 5 A2 images per ID, infer race, and write race-balanced selected A2 samples")
    ap.add_argument("--root", required=True, help="MVFS root: id/A1,A2,A_trash structure")
    ap.add_argument("--out-root", required=True, help="Output root for selected balanced samples")
    ap.add_argument("--mode", default="copy", choices=["copy", "move", "hardlink"])
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--sample-per-id", type=int, default=5, help="A2에서 ID당 race 판정에 사용할 샘플 수")
    ap.add_argument("--min-votes", type=int, default=2, help="sample-per-id 중 최소 득표 수. 5장 기준 2~3 추천")

    ap.add_argument("--reference-race", default="asian")
    ap.add_argument("--target-per-race", type=int, default=0, help="0이면 reference-race ID 개수를 기준으로 함")
    ap.add_argument("--include-races", default="asian,white,indian,middle eastern,latino hispanic,black")
    ap.add_argument("--include-unknown", action="store_true")

    ap.add_argument("--detector-backend", default="retinaface")
    ap.add_argument("--pad-ratio", type=float, default=0.25)

    ap.add_argument("--flat", action="store_true",
                    help="켜면 out-root/race/person_id_filename.jpg 형태로 저장. 기본은 out-root/person_id/A2/file.jpg")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    include_races = [normalize_race(x) for x in args.include_races.split(",")]
    include_races = [x for x in include_races if x]
    reference_race = normalize_race(args.reference_race)

    a2_by_id = list_id_a2_images(root)
    print(f"[INFO] found IDs with A2: {len(a2_by_id)}")

    id_records = []
    grouped = defaultdict(list)

    for person_id, imgs in tqdm(a2_by_id.items(), desc="DeepFace race on sampled A2"):
        samples = rng.sample(imgs, min(args.sample_per_id, len(imgs)))

        sample_results = []
        for p in samples:
            result = analyze_race_with_padding(
                p,
                detector_backend=args.detector_backend,
                pad_ratio=args.pad_ratio,
            )
            sample_results.append({
                "path": str(p),
                "race": result["race"],
                "reason": result["reason"],
                "race_scores": result["race_scores"],
            })

        race, votes = majority_race_from_samples(sample_results, args.min_votes)

        rec = {
            "person_id": person_id,
            "race": race,
            "votes": votes,
            "num_a2": len(imgs),
            "num_samples": len(samples),
            "sample_results": sample_results,
            "decision": "not_selected",
        }
        id_records.append(rec)

        if race in include_races:
            grouped[race].append((person_id, imgs, samples, rec))
        elif race == "unknown" and args.include_unknown:
            grouped["unknown"].append((person_id, imgs, samples, rec))

    before_counts = {race: len(grouped.get(race, [])) for race in include_races}
    if args.include_unknown:
        before_counts["unknown"] = len(grouped.get("unknown", []))

    ref_count = len(grouped.get(reference_race, []))
    target = int(args.target_per_race) if args.target_per_race > 0 else ref_count

    print("[INFO] before counts by ID race:")
    for race, count in before_counts.items():
        print(f"  {race}: {count}")
    print(f"[INFO] reference_race={reference_race}, ref_count={ref_count}, target_per_race={target}")

    selected = []
    selected_counts = Counter()

    select_races = list(include_races)
    if args.include_unknown:
        select_races.append("unknown")

    for race in select_races:
        arr = grouped.get(race, [])
        rng.shuffle(arr)
        chosen = arr[:target] if target > 0 else arr

        for person_id, imgs, samples, rec in chosen:
            rec["decision"] = "selected"
            selected.append((race, person_id, imgs, samples, rec))
            selected_counts[race] += 1

    print("[INFO] selected counts by ID race:")
    for race in select_races:
        print(f"  {race}: {selected_counts.get(race, 0)}")

    selected_file_records = []

    if not args.report_only:
        for race, person_id, imgs, samples, rec in tqdm(selected, desc=f"{args.mode} selected"):
            src_images = imgs

            for src in src_images:
                if args.flat:
                    safe_race = race.replace(" ", "_")
                    dst = out_root / safe_race / f"{person_id}_{src.name}"
                else:
                    dst = out_root / person_id / "A2" / src.name

                copy_move_or_link(src, dst, args.mode)

                selected_file_records.append({
                    "person_id": person_id,
                    "race": race,
                    "src_path": str(src),
                    "dst_path": str(dst),
                })

    summary = {
        "root": str(root),
        "out_root": str(out_root),
        "mode": args.mode,
        "flat": bool(args.flat),
        "report_only": bool(args.report_only),
        "sample_per_id": args.sample_per_id,
        "min_votes": args.min_votes,
        "reference_race": reference_race,
        "target_per_race": target,
        "before_counts_by_id": before_counts,
        "selected_counts_by_id": dict(selected_counts),
        "num_total_ids": len(a2_by_id),
        "num_selected_ids": len(selected),
        "num_selected_files": len(selected_file_records),
        "pad_ratio": args.pad_ratio,
        "detector_backend": args.detector_backend,
    }

    with open(out_root / "_a2_race_balance_id_report.jsonl", "w", encoding="utf-8") as f:
        for rec in id_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_root / "_a2_race_balance_selected_files.jsonl", "w", encoding="utf-8") as f:
        for rec in selected_file_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_root / "_a2_race_balance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print("[SUMMARY]", out_root / "_a2_race_balance_summary.json")


if __name__ == "__main__":
    main()
