# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

THIS = Path(__file__).resolve()
for p in [THIS.parent, *THIS.parents]:
    if (p / "mvfs_common").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Cannot find mvfs_common. Run from MVFS root or keep tools/ under MVFS root.")

from mvfs_common.gaze_utils import MediaPipeIrisExtractor, save_gaze_sidecar, sidecar_gaze_path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_a1a2_images(dataset_root: Path):
    for id_dir in sorted(dataset_root.iterdir()):
        if not id_dir.is_dir():
            continue

        for split in ("A1", "A2"):
            d = id_dir / split
            if not d.is_dir():
                continue

            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    yield id_dir.name, split, p


def main():
    ap = argparse.ArgumentParser("Extract iris 5-point gaze landmarks for MVFS A1/A2 images")
    ap.add_argument("--dataset-root", required=True, help="MVFS dataset root: id/A1,A2 structure")
    ap.add_argument("--pad-ratio", type=float, default=0.25)
    ap.add_argument("--min-det-conf", type=float, default=0.5)
    ap.add_argument("--min-track-conf", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug-dir", default="")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # TF 로그 조금 줄이기. mediapipe import 전 설정하는 게 좋음.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        extractor = MediaPipeIrisExtractor(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=args.min_det_conf,
            min_tracking_confidence=args.min_track_conf,
        )
    except ImportError as e:
        print("[ERROR] MediaPipe import failed.")
        print(str(e))
        print()
        print("protobuf runtime_version 에러면 보통 protobuf 버전 충돌임.")
        print("우선 아래를 mvfs venv에서 실행:")
        print("  python -m pip install -U protobuf")
        print()
        print("그래도 안 되면 mediapipe 전용 venv에서 이 gaze 추출만 돌리는 걸 추천.")
        raise

    total = ok = fail = skipped = 0
    records = []
    items = list(iter_a1a2_images(dataset_root))

    for person_id, split, img_path in tqdm(items, desc="extract iris5 gaze"):
        total += 1
        out_json = sidecar_gaze_path(img_path)

        if out_json.exists() and not args.overwrite:
            skipped += 1
            records.append({
                "person_id": person_id,
                "split": split,
                "image": str(img_path),
                "gaze_json": str(out_json),
                "status": "skipped_exists",
            })
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            fail += 1
            records.append({
                "person_id": person_id,
                "split": split,
                "image": str(img_path),
                "status": "cannot_read",
            })
            continue

        gaze = extractor.extract_iris5(img, pad_ratio=args.pad_ratio)

        if gaze is None:
            fail += 1
            bad = {
                "success": False,
                "reason": "no_face_or_no_iris",
                "pad_ratio": args.pad_ratio,
            }
            save_gaze_sidecar(img_path, bad)
            records.append({
                "person_id": person_id,
                "split": split,
                "image": str(img_path),
                "gaze_json": str(sidecar_gaze_path(img_path)),
                "status": "failed",
            })
            continue

        save_gaze_sidecar(img_path, gaze)
        ok += 1

        records.append({
            "person_id": person_id,
            "split": split,
            "image": str(img_path),
            "gaze_json": str(sidecar_gaze_path(img_path)),
            "status": "ok",
        })

        if debug_dir:
            vis = img.copy()
            for key in ("left_iris_5", "right_iris_5"):
                for x, y in gaze[key]:
                    cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 255), -1, cv2.LINE_AA)
            out_name = f"{person_id}_{split}_{img_path.stem}_iris5.jpg"
            cv2.imwrite(str(debug_dir / out_name), vis)

    report = {
        "dataset_root": str(dataset_root),
        "total": total,
        "ok": ok,
        "fail": fail,
        "skipped": skipped,
        "pad_ratio": args.pad_ratio,
        "min_det_conf": args.min_det_conf,
    }

    with open(dataset_root / "_gaze_extract_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(dataset_root / "_gaze_extract_report.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    extractor.close()

    print("[DONE]", report)
    print("[SUMMARY]", dataset_root / "_gaze_extract_summary.json")


if __name__ == "__main__":
    main()
