# -*- coding: utf-8 -*-
"""
extract_images_dfljpg.py

MVFS / 3DDFA-V3 image-directory -> DFLJPG aligned crop extractor.

Place this file under:
    mvfs/3ddfa-v3/extract_images_dfljpg.py

Expected project layout:
    mvfs/
    ├─ mvfs_common/
    └─ 3ddfa-v3/
       ├─ extract_images_dfljpg.py
       ├─ mvfs_threeddfa_v3.py
       ├─ model/
       ├─ util/
       └─ face_box/

This is an additional process, not a replacement for extract_video_dfljpg.py.
It keeps the same extraction flow:
    image -> 3DDFA-V3 multi-face detection/fitting
          -> DFL-style aligned crop
          -> DFLJPG APP15 metadata
          -> optional debug images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np


# This script lives inside mvfs/3ddfa-v3.
# Add mvfs root to import mvfs_common.
_MVFS_ROOT = Path(__file__).resolve().parents[1]
if str(_MVFS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVFS_ROOT))

from mvfs_common.dfl_types import FaceType
from mvfs_common.dfl_save import save_aligned_dfljpg, save_debug_pair
from mvfs_common.dfljpg_io import DFLJPG
from mvfs_common.landmark_draw import draw_landmarks_68
from mvfs_threeddfa_v3 import ThreeDDFAExtractor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path, recursive: bool = False) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    it = images_dir.rglob("*") if recursive else images_dir.iterdir()
    paths = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def has_aligned_for_image(output_dir: Path, image_stem: str) -> bool:
    return any(output_dir.glob(f"{image_stem}_*.jpg"))


def next_face_index(output_dir: Path, image_stem: str) -> int:
    max_idx = -1
    for p in output_dir.glob(f"{image_stem}_*.jpg"):
        suffix = p.stem[len(image_stem) + 1:]
        try:
            max_idx = max(max_idx, int(suffix))
        except ValueError:
            pass
    return max_idx + 1


def make_unique_stem(image_path: Path, images_dir: Path, keep_subdirs: bool = False) -> str:
    """
    If keep_subdirs=False:
        images/a/b.png -> b
    If keep_subdirs=True:
        images/a/b.png -> a__b
    """
    if not keep_subdirs:
        return image_path.stem

    rel = image_path.relative_to(images_dir)
    parts = list(rel.parts)
    parts[-1] = Path(parts[-1]).stem
    return "__".join(parts)


def save_source_debug(
    debug_dir: Optional[Path],
    image_stem: str,
    image_bgr: np.ndarray,
    faces: list,
) -> None:
    if debug_dir is None:
        return

    dbg = image_bgr.copy()
    for i, face in enumerate(faces):
        lm = np.asarray(face["landmarks"], dtype=np.float32)
        bbox = face.get("bbox", face.get("rect", None))
        if bbox is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)
        draw_landmarks_68(dbg, lm.astype(np.int32), color=(0, 255, 0), thickness=1)
        if bbox is not None:
            cv2.putText(
                dbg,
                str(i),
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{image_stem}_src_lm_all.jpg"), dbg)


def extract_images_to_dfljpg(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    index_out = Path(args.index_out) if args.index_out else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if index_out:
        index_out.parent.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir, recursive=args.recursive)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    print(f"[INFO] images_dir={images_dir}")
    print(f"[INFO] images={len(images)}")
    print(f"[INFO] output={output_dir}")

    extractor = ThreeDDFAExtractor(
        device=args.device,
        detector=args.detector,
        backbone=args.backbone,
        max_faces=args.max_faces,
        sort_faces=args.sort_faces,
    )
    face_type = FaceType.from_string(args.face_type)

    index_f = open(index_out, "w", encoding="utf-8") if index_out else None
    total_faces = 0
    total_images_done = 0

    try:
        for image_i, image_path in enumerate(images):
            image_stem = make_unique_stem(image_path, images_dir, keep_subdirs=args.keep_subdirs_in_name)

            if args.skip_existing and has_aligned_for_image(output_dir, image_stem):
                print(f"[SKIP-EXIST] {image_stem}")
                continue

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[SKIP] cannot read: {image_path}")
                continue

            try:
                faces = extractor.get_faces(image_bgr)
            except Exception as e:
                print(f"[ERR] 3DDFA failed: {image_path} | {e}")
                continue

            if not faces:
                print(f"[NOFACE] {image_stem}")
                continue

            save_source_debug(debug_dir, image_stem, image_bgr, faces)

            face_idx = next_face_index(output_dir, image_stem)
            saved_for_image = 0

            for face in faces:
                try:
                    out_path = save_aligned_dfljpg(
                        image_bgr,
                        face["landmarks"],
                        image_stem,
                        face_idx,
                        output_dir,
                        args.image_size,
                        face_type,
                        args.jpeg_quality,
                    )

                    save_debug_pair(
                        debug_dir,
                        image_stem,
                        face_idx,
                        image_bgr,
                        out_path,
                        face["landmarks"],
                        face.get("bbox", face.get("rect")),
                    )

                    if index_f is not None:
                        rec = {
                            "image_path": str(image_path),
                            "image_stem": image_stem,
                            "face_idx": face_idx,
                            "aligned_path": str(out_path),
                            "source_bbox": face.get("bbox", face.get("rect")),
                        }
                        index_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    print(f"[OK] {out_path}")
                    face_idx += 1
                    saved_for_image += 1
                    total_faces += 1

                except Exception as e:
                    print(f"[ERR] save failed: {image_path} face_idx={face_idx} | {e}")

            total_images_done += 1
            print(f"[{image_i + 1}/{len(images)}] {image_stem}: saved={saved_for_image}")

    finally:
        if index_f is not None:
            index_f.close()

    print(f"[DONE] images_done={total_images_done}, total_faces={total_faces}")


def debug_draw_dfljpg(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.jpg"))
    for p in paths:
        dfl = DFLJPG.load(str(p))
        if dfl is None or not dfl.has_data():
            print(f"[SKIP] no DFL metadata: {p}")
            continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] cannot read image: {p}")
            continue

        lm = dfl.get_landmarks().astype(np.int32)
        if lm.shape[0] != 68:
            print(f"[SKIP] not 68 landmarks: {p}, shape={lm.shape}")
            continue

        dbg = img.copy()
        draw_landmarks_68(dbg, lm, color=(0, 255, 0), thickness=1)
        cv2.imwrite(str(output_dir / p.name), dbg)
        print(f"[OK] {output_dir / p.name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("MVFS 3DDFA-V3 image-directory DFLJPG extractor")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="Extract DFLJPG aligned crops from an image directory.")
    ex.add_argument("--images-dir", required=True, type=str)
    ex.add_argument("--output", required=True, type=str)
    ex.add_argument("--device", default="cuda", type=str)
    ex.add_argument("--detector", default="retinaface", choices=["retinaface", "mtcnn"])
    ex.add_argument("--backbone", default="resnet50", choices=["resnet50", "mbnetv3"])
    ex.add_argument("--image-size", default=512, type=int)
    ex.add_argument(
        "--face-type",
        default="whole_face",
        choices=["half", "mid_full", "full", "full_no_align", "whole_face", "head", "head_no_align", "mark_only"],
    )
    ex.add_argument("--jpeg-quality", default=95, type=int)
    ex.add_argument("--max-faces", default=0, type=int, help="0 means all detected faces.")
    ex.add_argument("--sort-faces", default="left_to_right", choices=["left_to_right", "area_desc", "none"])
    ex.add_argument("--recursive", action="store_true")
    ex.add_argument("--keep-subdirs-in-name", action="store_true")
    ex.add_argument("--skip-existing", action="store_true")
    ex.add_argument("--max-images", default=0, type=int)
    ex.add_argument("--debug-dir", default=None, type=str)
    ex.add_argument("--index-out", default=None, type=str)

    dbg = sub.add_parser("debug-draw", help="Draw DFLJPG landmarks from aligned jpg files.")
    dbg.add_argument("--input", required=True, type=str)
    dbg.add_argument("--output", required=True, type=str)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "extract":
        extract_images_to_dfljpg(args)
    elif args.cmd == "debug-draw":
        debug_draw_dfljpg(args)
    else:
        parser.error(f"unknown cmd: {args.cmd}")
