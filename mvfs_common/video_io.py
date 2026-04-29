# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


def frame_name_for(video_path: Path, frame_idx: int) -> str:
    return f"{video_path.stem}_{frame_idx:06d}"


def iter_video_frames(video_path: Path, frame_step: int = 1, max_frames: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    idx = 0
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_step == 0:
                yield idx, frame
                yielded += 1
                if max_frames > 0 and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def read_frame_at(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def frame_has_aligned(output_dir: Path, video_path: Path, frame_idx: int) -> bool:
    frame_name = frame_name_for(video_path, frame_idx)
    return any(output_dir.glob(f"{frame_name}_*.jpg"))


def collect_missing_frame_indices(video_path: Path, output_dir: Path, frame_step: int = 1, max_frames: int = 0) -> List[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    missing: List[int] = []
    sampled = 0
    for idx in range(total):
        if idx % frame_step != 0:
            continue
        if not frame_has_aligned(output_dir, video_path, idx):
            missing.append(idx)
        sampled += 1
        if max_frames > 0 and sampled >= max_frames:
            break
    return missing


def next_face_index(output_dir: Path, frame_name: str) -> int:
    max_idx = -1
    for p in output_dir.glob(f"{frame_name}_*.jpg"):
        suffix = p.stem.rsplit("_", 1)[-1]
        try:
            max_idx = max(max_idx, int(suffix))
        except ValueError:
            pass
    return max_idx + 1

# Frame-directory helpers for manual extraction from pre-extracted frames.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_frame_images(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")
    return sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def has_aligned_for_frame(output_dir: Path, frame_stem: str) -> bool:
    return any(output_dir.glob(f"{frame_stem}_*.jpg"))


def collect_missing_frame_paths(frames_dir: Path, output_dir: Path, frame_step: int = 1, max_frames: int = 0) -> List[Path]:
    frames = list_frame_images(frames_dir)
    if frame_step > 1:
        frames = frames[::frame_step]
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    return [p for p in frames if not has_aligned_for_frame(output_dir, p.stem)]
