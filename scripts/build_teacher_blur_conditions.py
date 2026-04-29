# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvfs_common.dfljpg_io import DFLJPG
from mvfs_common.teacher_condition import apple_style_blur_condition

IMAGE_EXTS = {".jpg", ".jpeg"}


def build_conditions(args):
    aligned_dir = Path(args.aligned_dir)
    out_dir = Path(args.out_dir)
    cond_dir = out_dir / "conditions" / "blur_landmark"
    index_path = out_dir / "teacher_index.jsonl"
    debug_dir = out_dir / "debug" if args.debug else None

    cond_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in aligned_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    n_ok = 0
    n_skip = 0

    with open(index_path, "w", encoding="utf-8") as fidx:
        for p in files:
            try:
                dfl = DFLJPG.load(p)
                img = dfl.get_img()
                lm = dfl.get_landmarks()
                cond = apple_style_blur_condition(
                    img,
                    lm,
                    downsample=args.downsample,
                    expand=args.expand,
                    overlay_landmarks=not args.no_landmark_overlay,
                    landmark_thickness=args.landmark_thickness,
                )
                rel_name = p.relative_to(aligned_dir).with_suffix(".png")
                cond_path = cond_dir / rel_name
                cond_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cond_path), cond)

                if debug_dir:
                    dbg_path = debug_dir / rel_name
                    dbg_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dbg_path), cond)

                row = {
                    "clean_path": str(p),
                    "condition_path": str(cond_path),
                    "identity_path": str(p),
                    "stem": p.stem,
                }
                fidx.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                n_skip += 1
                print(f"[SKIP] {p}: {e}")

    print(f"[DONE] ok={n_ok}, skip={n_skip}, index={index_path}")


def build_parser():
    p = argparse.ArgumentParser("Build APPLE-style blur+landmark teacher conditions from DFLJPG aligned crops")
    p.add_argument("--aligned-dir", required=True, help="Directory containing DFLJPG aligned crops.")
    p.add_argument("--out-dir", required=True, help="Output directory, e.g. workspace/person_x/teacher")
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument("--expand", type=float, default=0.18)
    p.add_argument("--landmark-thickness", type=int, default=1)
    p.add_argument("--no-landmark-overlay", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


if __name__ == "__main__":
    build_conditions(build_parser().parse_args())
