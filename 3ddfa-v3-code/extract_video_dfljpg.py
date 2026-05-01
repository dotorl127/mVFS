# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This script lives inside mvfs/3ddfa-v3. Add mvfs/ to sys.path.
_MVFS_ROOT = Path(__file__).resolve().parents[1]
if str(_MVFS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVFS_ROOT))

import cv2
import numpy as np

from mvfs_common.dfl_types import FaceType, draw_landmarks_68
from mvfs_common.dfljpg_io import read_dfljpg_metadata
from mvfs_common.dfl_save import debug_draw_dir, save_aligned_dfljpg
from mvfs_common.video_io import frame_name_for, iter_video_frames
from mvfs_threeddfa_v3 import ThreeDDFAExtractor


def extract_command(args):
    video_path = Path(args.video)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    face_type = FaceType.from_string(args.face_type)
    extractor = ThreeDDFAExtractor(
        device=args.device,
        detector=args.detector,
        backbone=args.backbone,
        max_faces=args.max_faces,
        sort_faces=args.sort_faces,
    )

    total = 0
    for frame_idx, frame in iter_video_frames(video_path, args.frame_step, args.max_frames):
        frame_name = frame_name_for(video_path, frame_idx)
        faces = extractor.get_faces(frame)
        if len(faces) == 0:
            print(f"[SKIP] {frame_name}: no faces")
            continue

        src_dbg = frame.copy() if debug_dir else None
        saved_this_frame = 0

        for face_idx, face in enumerate(faces):
            lm68 = face["landmarks"]
            try:
                out_path = save_aligned_dfljpg(
                    frame_bgr=frame,
                    lm68_src=lm68,
                    frame_name=frame_name,
                    face_idx=face_idx,
                    out_dir=out_dir,
                    image_size=args.image_size,
                    face_type=face_type,
                    jpeg_quality=args.jpeg_quality,
                )
                print(f"[OK] {out_path}")
                total += 1
                saved_this_frame += 1
            except Exception as e:
                print(f"[ERR] {frame_name} face#{face_idx}: {e}")
                continue

            if debug_dir:
                src_dbg = draw_landmarks_68(src_dbg, lm68, color=(0, 255, 0))
                aligned_img = cv2.imread(str(out_path))
                meta = read_dfljpg_metadata(out_path)
                if aligned_img is not None and meta is not None:
                    aligned_lm = np.asarray(meta["landmarks"], dtype=np.float32)
                    aligned_dbg = draw_landmarks_68(aligned_img, aligned_lm)
                    cv2.imwrite(str(debug_dir / f"{frame_name}_{face_idx:02d}_aligned_lm.jpg"), aligned_dbg)

        if debug_dir and src_dbg is not None:
            cv2.imwrite(str(debug_dir / f"{frame_name}_src_lm_all.jpg"), src_dbg)

        print(f"[FRAME] {frame_name}: detected={len(faces)}, saved={saved_this_frame}")

    print(f"Done. saved_faces={total}")


def debug_draw_command(args):
    debug_draw_dir(Path(args.input), Path(args.output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("3DDFA-V3 video -> DFLJPG aligned extractor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--video", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--detector", default="retinaface", choices=["retinaface", "mtcnn"])
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "mbnetv3"])
    p.add_argument("--image-size", default=512, type=int)
    p.add_argument("--face-type", default="whole_face", choices=["half", "mid_full", "full", "full_no_align", "whole_face", "head", "head_no_align", "mark_only"])
    p.add_argument("--jpeg-quality", default=95, type=int)
    p.add_argument("--frame-step", default=1, type=int)
    p.add_argument("--max-frames", default=0, type=int)
    p.add_argument("--max-faces", default=0, type=int, help="0 means save all detected faces per frame")
    p.add_argument("--sort-faces", default="left_to_right", choices=["left_to_right", "area_desc", "none"])
    p.add_argument("--debug-dir", default=None, type=str)
    p.set_defaults(func=extract_command)

    p = sub.add_parser("debug-draw")
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.set_defaults(func=debug_draw_command)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
