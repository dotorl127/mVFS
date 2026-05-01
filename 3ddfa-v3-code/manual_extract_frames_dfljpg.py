# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# This script lives inside mvfs/3ddfa-v3.
# It imports shared utilities from mvfs/mvfs_common.
_MVFS_ROOT = Path(__file__).resolve().parents[1]
if str(_MVFS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVFS_ROOT))

import cv2
import numpy as np

from mvfs_common.dfl_types import FaceType
from mvfs_common.dfl_save import save_aligned_dfljpg, save_debug_pair
from mvfs_threeddfa_v3 import ThreeDDFAExtractor, get_face_from_manual_bbox


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

HELP_LINES = [
    "Move mouse near face | Wheel ROI size | L-click lock/unlock",
    "x: save current ROI and decide(frame done) | s: save ROI only | a: auto-save all faces",
    "z: previous missing frame | c: next missing frame | Space: toggle continuous skip",
    "Tab: vertical flip | r: reset | h: hide | q/Esc: quit",
]


def list_frame_images(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")
    return sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def has_aligned_for_frame(output_dir: Path, frame_stem: str) -> bool:
    # DFL-style aligned output: {frame_stem}_{face_idx:02d}.jpg
    return any(output_dir.glob(f"{frame_stem}_*.jpg"))


def collect_missing_frame_paths(frames_dir: Path, output_dir: Path, frame_step: int = 1, max_frames: int = 0) -> List[Path]:
    frames = list_frame_images(frames_dir)
    if frame_step > 1:
        frames = frames[::frame_step]
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    return [p for p in frames if not has_aligned_for_frame(output_dir, p.stem)]


def next_face_index(output_dir: Path, frame_stem: str) -> int:
    max_idx = -1
    for p in output_dir.glob(f"{frame_stem}_*.jpg"):
        suffix = p.stem[len(frame_stem) + 1:]
        try:
            max_idx = max(max_idx, int(suffix))
        except ValueError:
            pass
    return max_idx + 1


def flip_landmarks_vertical(lm: np.ndarray, height: int) -> np.ndarray:
    lm = np.asarray(lm, dtype=np.float32).copy()
    lm[:, 1] = (height - 1) - lm[:, 1]
    return lm


def flip_bbox_vertical(bbox: Tuple[float, float, float, float], height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1, (height - 1) - y2, x2, (height - 1) - y1


class DFLStylePointerUI:
    """DFL-like pointer-centered manual UI.

    - ROI follows mouse pointer.
    - Mouse wheel changes ROI size.
    - Left click locks/unlocks ROI.
    - Tab toggles vertical flip for upside-down faces.

    When vertical flip is ON, extraction is performed on the flipped view,
    but landmarks are mapped back to the original frame coordinate system.
    """

    def __init__(self, window_name: str = "3DDFA-V3 manual extract"):
        self.window_name = window_name
        self.view_scale = 1.0
        self.original: Optional[np.ndarray] = None
        self.base: Optional[np.ndarray] = None
        self.display: Optional[np.ndarray] = None
        self.cursor = (0, 0)
        self.rect_size = 120
        self.rect_locked = False
        self.hide_help = False
        self.flip_vertical = False
        self.max_width = 1368

    def set_image(self, img_bgr: np.ndarray, max_width: int = 1368, initial_rect_size: int = 120):
        self.original = img_bgr.copy()
        self.max_width = int(max_width)
        self.flip_vertical = False
        self.cursor = (img_bgr.shape[1] // 2, img_bgr.shape[0] // 2)
        self.rect_size = int(initial_rect_size)
        self.rect_locked = False
        self.hide_help = False
        self._rebuild_base_and_display()

    def _rebuild_base_and_display(self):
        assert self.original is not None
        self.base = cv2.flip(self.original, 0) if self.flip_vertical else self.original.copy()
        h, w = self.base.shape[:2]
        self.view_scale = min(1.0, float(self.max_width) / max(w, 1)) if self.max_width > 0 else 1.0
        if self.view_scale != 1.0:
            self.display = cv2.resize(self.base, (int(w * self.view_scale), int(h * self.view_scale)), interpolation=cv2.INTER_LINEAR)
        else:
            self.display = self.base.copy()
        cx, cy = self.cursor
        self.cursor = (int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1)))

    def toggle_vertical_flip(self):
        assert self.base is not None
        h = self.base.shape[0]
        cx, cy = self.cursor
        self.cursor = (cx, (h - 1) - cy)
        self.flip_vertical = not self.flip_vertical
        self.rect_locked = False
        self._rebuild_base_and_display()

    def _to_base(self, x: int, y: int) -> Tuple[int, int]:
        return int(x / self.view_scale), int(y / self.view_scale)

    def mouse_cb(self, event, x, y, flags, param):
        if self.base is None:
            return
        h, w = self.base.shape[:2]
        if event == cv2.EVENT_MOUSEMOVE and not self.rect_locked:
            ox, oy = self._to_base(x, y)
            self.cursor = (int(np.clip(ox, 0, w - 1)), int(np.clip(oy, 0, h - 1)))
        elif event == cv2.EVENT_MOUSEWHEEL and not self.rect_locked:
            delta = 1 if flags > 0 else -1
            step = max(4, int(self.rect_size * 0.08))
            self.rect_size = int(np.clip(self.rect_size + delta * step, 12, max(w, h)))
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not self.rect_locked:
                ox, oy = self._to_base(x, y)
                self.cursor = (int(np.clip(ox, 0, w - 1)), int(np.clip(oy, 0, h - 1)))
            self.rect_locked = not self.rect_locked

    def current_bbox_base(self) -> Tuple[int, int, int, int]:
        assert self.base is not None
        h, w = self.base.shape[:2]
        cx, cy = self.cursor
        rs = int(self.rect_size)
        x1 = int(np.clip(cx - rs, 0, w - 1))
        y1 = int(np.clip(cy - rs, 0, h - 1))
        x2 = int(np.clip(cx + rs, 0, w - 1))
        y2 = int(np.clip(cy + rs, 0, h - 1))
        return x1, y1, x2, y2

    def render(self, frame_name: str, saved_count: int, pos: int, total: int, continuous_skip: bool) -> np.ndarray:
        assert self.display is not None
        img = self.display.copy()
        s = self.view_scale
        x1, y1, x2, y2 = self.current_bbox_base()
        color = (0, 255, 255) if self.rect_locked else (0, 255, 0)
        cv2.rectangle(img, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), color, 2)
        cx, cy = self.cursor
        cv2.drawMarker(img, (int(cx * s), int(cy * s)), color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

        if self.flip_vertical:
            cv2.putText(img, "VERTICAL FLIP ON", (10, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, "VERTICAL FLIP ON", (10, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2, cv2.LINE_AA)

        if continuous_skip:
            cv2.putText(img, "CONTINUOUS SKIP ON - press Space to stop here", (10, img.shape[0] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, "CONTINUOUS SKIP ON - press Space to stop here", (10, img.shape[0] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

        if not self.hide_help:
            lines = [f"frame={frame_name}  missing={pos + 1}/{total}  saved_this_frame={saved_count}"] + HELP_LINES
            y = 24
            for line in lines:
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
                y += 24
        return img


def map_face_back_if_flipped(face: dict, ui: DFLStylePointerUI) -> dict:
    if not ui.flip_vertical:
        return face
    assert ui.base is not None
    h = ui.base.shape[0]
    f = dict(face)
    f["landmarks"] = flip_landmarks_vertical(f["landmarks"], h)
    if "bbox" in f and f["bbox"] is not None:
        f["bbox"] = flip_bbox_vertical(f["bbox"], h)
    if "rect" in f and f["rect"] is not None:
        f["rect"] = flip_bbox_vertical(f["rect"], h)
    return f


def extract_face_from_current_roi(extractor: ThreeDDFAExtractor, ui: DFLStylePointerUI):
    assert ui.base is not None
    bbox_base = ui.current_bbox_base()
    face = get_face_from_manual_bbox(extractor, ui.base, bbox_base)
    if face is None:
        return None, bbox_base
    face = map_face_back_if_flipped(face, ui)
    debug_bbox = face.get("bbox", face.get("rect", bbox_base))
    return face, debug_bbox


def get_auto_faces_current_view(extractor: ThreeDDFAExtractor, ui: DFLStylePointerUI):
    assert ui.base is not None
    faces = extractor.get_faces(ui.base)
    return [map_face_back_if_flipped(face, ui) for face in faces]


def save_face(frame: np.ndarray, face: dict, frame_stem: str, face_idx: int, out_dir: Path, args, face_type, debug_dir: Optional[Path], debug_bbox=None):
    out_path = save_aligned_dfljpg(
        frame,
        face["landmarks"],
        frame_stem,
        face_idx,
        out_dir,
        args.image_size,
        face_type,
        args.jpeg_quality,
    )
    save_debug_pair(debug_dir, frame_stem, face_idx, frame, out_path, face["landmarks"], debug_bbox or face.get("bbox"))
    return out_path


def manual_extract(args):
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    missing = collect_missing_frame_paths(frames_dir, out_dir, frame_step=args.frame_step, max_frames=args.max_frames)
    print(f"[INFO] frames_dir={frames_dir}")
    print(f"[INFO] missing frames without aligned crop: {len(missing)}")
    if not missing:
        return

    extractor = ThreeDDFAExtractor(
        device=args.device,
        detector=args.detector,
        backbone=args.backbone,
        max_faces=args.max_faces,
        sort_faces=args.sort_faces,
    )
    face_type = FaceType.from_string(args.face_type)

    ui = DFLStylePointerUI()
    cv2.namedWindow(ui.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ui.window_name, ui.mouse_cb)

    pos = 0
    continuous_skip = False
    while 0 <= pos < len(missing):
        frame_path = missing[pos]
        frame_stem = frame_path.stem
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[SKIP] cannot read frame: {frame_path}")
            pos += 1
            continue

        face_idx = next_face_index(out_dir, frame_stem)
        saved_this_frame = 0
        initial_size = args.rect_size if args.rect_size > 0 else max(48, min(frame.shape[:2]) // 8)
        ui.set_image(frame, max_width=args.manual_window_width, initial_rect_size=initial_size)

        while True:
            rendered = ui.render(frame_stem, saved_this_frame, pos, len(missing), continuous_skip)
            cv2.imshow(ui.window_name, rendered)
            wait_ms = args.continuous_skip_delay_ms if continuous_skip else 30
            key = cv2.waitKey(wait_ms)

            if continuous_skip:
                if key != -1 and (key & 0xFF) == ord(" "):
                    continuous_skip = False
                    print(f"[CONTINUOUS-SKIP OFF] stopped at {frame_stem}")
                    continue
                print(f"[CONTINUOUS-SKIP] {frame_stem}")
                pos += 1
                break

            if key == -1:
                continue
            key8 = key & 0xFF

            if key8 in (ord("q"), 27):
                cv2.destroyWindow(ui.window_name)
                print("[QUIT]")
                return
            if key8 == ord("h"):
                ui.hide_help = not ui.hide_help
                continue
            if key8 == ord("r"):
                ui.rect_locked = False
                continue
            if key8 == 9:  # Tab
                ui.toggle_vertical_flip()
                print(f"[FLIP] vertical_flip={ui.flip_vertical}")
                continue
            if key8 == ord(" "):
                continuous_skip = True
                print(f"[CONTINUOUS-SKIP ON] start after {frame_stem}")
                pos += 1
                break
            if key8 == ord("z"):
                pos = max(0, pos - 1)
                break
            if key8 == ord("c"):
                pos = min(len(missing) - 1, pos + 1)
                break

            if key8 == ord("a"):
                faces = get_auto_faces_current_view(extractor, ui)
                if not faces:
                    print(f"[AUTO] {frame_stem}: no faces")
                    continue
                for face in faces:
                    try:
                        out_path = save_face(frame, face, frame_stem, face_idx, out_dir, args, face_type, debug_dir)
                        print(f"[AUTO-OK] {out_path}")
                        face_idx += 1
                        saved_this_frame += 1
                    except Exception as e:
                        print(f"[AUTO-ERR] {frame_stem}: {e}")
                continue

            if key8 in (ord("s"), ord("x")):
                face, debug_bbox = extract_face_from_current_roi(extractor, ui)
                if face is None:
                    print(f"[ERR] manual 3DDFA failed for ROI={ui.current_bbox_base()} flip={ui.flip_vertical}")
                    continue
                try:
                    out_path = save_face(frame, face, frame_stem, face_idx, out_dir, args, face_type, debug_dir, debug_bbox=debug_bbox)
                    mode = "MANUAL-FLIP" if ui.flip_vertical else "MANUAL"
                    print(f"[{mode}-OK] {out_path}")
                    face_idx += 1
                    saved_this_frame += 1
                    ui.rect_locked = False
                    if key8 == ord("x"):
                        print(f"[DONE] {frame_stem}, saved={saved_this_frame}")
                        pos += 1
                        break
                except Exception as e:
                    print(f"[ERR] save failed: {e}")

    cv2.destroyWindow(ui.window_name)
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("3DDFA-V3 DFL-style manual extraction for existing frame images")
    p.add_argument("--frames-dir", required=True, type=str, help="Directory containing already extracted video frames.")
    p.add_argument("--output", required=True, type=str, help="Aligned DFLJPG output directory.")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--detector", default="retinaface", choices=["retinaface", "mtcnn"])
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "mbnetv3"])
    p.add_argument("--image-size", default=512, type=int)
    p.add_argument("--face-type", default="whole_face", choices=["half", "mid_full", "full", "full_no_align", "whole_face", "head", "head_no_align", "mark_only"])
    p.add_argument("--jpeg-quality", default=95, type=int)
    p.add_argument("--frame-step", default=1, type=int)
    p.add_argument("--max-frames", default=0, type=int)
    p.add_argument("--max-faces", default=0, type=int, help="Used by auto-save key 'a'. 0 means all detected faces.")
    p.add_argument("--sort-faces", default="left_to_right", choices=["left_to_right", "area_desc", "none"])
    p.add_argument("--manual-window-width", default=1368, type=int)
    p.add_argument("--rect-size", default=0, type=int, help="Initial half-size of pointer-centered ROI. 0 = auto.")
    p.add_argument("--continuous-skip-delay-ms", default=60, type=int)
    p.add_argument("--debug-dir", default=None, type=str)
    return p


if __name__ == "__main__":
    manual_extract(build_parser().parse_args())
