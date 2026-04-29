# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from mvfs_common.dfl_types import FaceType, draw_landmarks_68, get_transform_mat, rect_from_landmarks, transform_points
from mvfs_common.dfljpg_io import read_dfljpg_metadata, write_dfljpg_metadata


def save_aligned_dfljpg(
    frame_bgr: np.ndarray,
    lm68_src: np.ndarray,
    frame_name: str,
    face_idx: int,
    out_dir: Path,
    image_size: int,
    face_type: FaceType,
    jpeg_quality: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = get_transform_mat(lm68_src, image_size, face_type)
    aligned = cv2.warpAffine(frame_bgr, mat, (image_size, image_size), flags=cv2.INTER_LANCZOS4)
    lm68_aligned = transform_points(lm68_src, mat)

    out_path = out_dir / f"{frame_name}_{face_idx:02d}.jpg"
    ok = cv2.imwrite(str(out_path), aligned, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")

    meta = {
        "face_type": FaceType.to_string(face_type),
        "landmarks": lm68_aligned.astype(np.float32).tolist(),
        "source_filename": f"{frame_name}.png",
        "source_rect": rect_from_landmarks(lm68_src),
        "source_landmarks": lm68_src.astype(np.float32).tolist(),
        "image_to_face_mat": mat.astype(np.float32).tolist(),
    }
    write_dfljpg_metadata(out_path, meta)
    return out_path


def save_debug_pair(
    debug_dir: Optional[Path],
    frame_name: str,
    face_idx: int,
    frame_bgr: np.ndarray,
    out_path: Path,
    lm68_src: np.ndarray,
    bbox=None,
) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)

    src_dbg = draw_landmarks_68(frame_bgr.copy(), lm68_src)
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(src_dbg, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(str(debug_dir / f"{frame_name}_{face_idx:02d}_src_lm.jpg"), src_dbg)

    aligned = cv2.imread(str(out_path))
    meta = read_dfljpg_metadata(out_path)
    if aligned is not None and meta is not None and "landmarks" in meta:
        aligned_lm = np.asarray(meta["landmarks"], dtype=np.float32)
        cv2.imwrite(str(debug_dir / f"{frame_name}_{face_idx:02d}_aligned_lm.jpg"), draw_landmarks_68(aligned, aligned_lm))


def debug_draw_dir(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(input_dir.glob("*.jpg")):
        meta = read_dfljpg_metadata(p)
        if not meta or "landmarks" not in meta:
            print(f"[SKIP] no DFL metadata: {p}")
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"[SKIP] cannot read: {p}")
            continue
        lm = np.asarray(meta["landmarks"], dtype=np.float32)
        dbg = draw_landmarks_68(img, lm)
        cv2.imwrite(str(output_dir / p.name), dbg)
        print(f"[OK] {output_dir / p.name}")
