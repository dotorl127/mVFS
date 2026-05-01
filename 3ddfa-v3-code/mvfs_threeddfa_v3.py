# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import sys
from pathlib import Path
# This file lives inside mvfs/3ddfa-v3. Add mvfs/ to sys.path so
# shared modules under mvfs/mvfs_common can be imported.
_MVFS_ROOT = Path(__file__).resolve().parents[1]
if str(_MVFS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVFS_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from mvfs_common.dfl_types import rect_from_landmarks

# 3DDFA_V3 integration - multi-face version
# ----------------------------


def restore_ldm68_to_original(ldm68_crop: np.ndarray, trans_params: Optional[np.ndarray]) -> np.ndarray:
    """Match 3DDFA_V3 util.io.visualize_and_output() / back_resize_ldms style.

    3DDFA-V3 predicts landmarks in the aligned 224x224 crop coordinate system.
    The official visualizer flips y first, then maps them back with trans_params.
    """
    lm = np.asarray(ldm68_crop, dtype=np.float32).copy()
    lm[:, 1] = 224 - 1 - lm[:, 1]

    if trans_params is None:
        return lm

    w0, h0, s = trans_params[0], trans_params[1], trans_params[2]
    tx, ty = trans_params[3], trans_params[4]
    target_size = 224

    w = int(w0 * s)
    h = int(h0 * s)
    left = int(w / 2 - target_size / 2 + float((tx - w0 / 2) * s))
    up = int(h / 2 - target_size / 2 + float((h0 / 2 - ty) * s))

    lm[:, 0] = lm[:, 0] + left
    lm[:, 1] = lm[:, 1] + up
    lm[:, 0] = lm[:, 0] / max(w, 1) * w0
    lm[:, 1] = lm[:, 1] / max(h, 1) * h0
    return lm.astype(np.float32)


def _bbox_from_lm(lm: np.ndarray) -> Tuple[float, float, float, float]:
    lm = np.asarray(lm, dtype=np.float32)
    x0, y0 = lm[:, 0].min(), lm[:, 1].min()
    x1, y1 = lm[:, 0].max(), lm[:, 1].max()
    return float(x0), float(y0), float(x1), float(y1)


def _sort_faces(face_items: List[Dict[str, Any]], order: str = "left_to_right") -> List[Dict[str, Any]]:
    if order == "none":
        return face_items
    if order == "area_desc":
        return sorted(face_items, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
    if order == "left_to_right":
        return sorted(face_items, key=lambda f: f["bbox"][0])
    raise ValueError(f"Unknown face sort order: {order}")


class ThreeDDFAExtractor:
    """3DDFA-V3 extractor that processes all detected faces in a frame.

    The official face_box.retinaface.detector uses results_all[0] only.
    This class uses the same lower-level detector, but loops over every detected face,
    aligns every crop, batches them, and restores each predicted 68-point landmark
    back to the original frame coordinate system.
    """

    def __init__(
        self,
        device: str = "cuda",
        detector: str = "retinaface",
        backbone: str = "resnet50",
        max_faces: int = 0,
        sort_faces: str = "left_to_right",
    ):
        from model.recon import face_model
        from util.preprocess import load_lm3d, align_img

        self.args = SimpleNamespace(
            inputpath="",
            savepath="",
            device=device,
            iscrop=True,
            detector=detector,
            ldm68=True,
            ldm106=False,
            ldm106_2d=False,
            ldm134=False,
            seg=False,
            seg_visible=False,
            useTex=False,
            extractTex=False,
            backbone=backbone,
        )

        self.device = device
        self.detector_name = detector
        self.max_faces = max_faces
        self.sort_faces = sort_faces
        self.align_img = align_img
        self.lm3d_std = load_lm3d()
        self.recon_model = face_model(self.args)

        if detector == "retinaface":
            from face_box.facelandmark.large_model_infer import LargeModelInfer
            self.retina_model = LargeModelInfer("assets/large_base_net.pth", device=device)
            self.mtcnn_model = None
            print("use retinaface/large_model_infer for multi-face detection")
        elif detector == "mtcnn":
            from mtcnn import MTCNN
            self.mtcnn_model = MTCNN()
            self.retina_model = None
            print("use mtcnn for multi-face detection")
        else:
            raise ValueError("detector must be 'retinaface' or 'mtcnn'")

    def _detect_faces_retina(self, pil_img: Image.Image) -> List[Dict[str, Any]]:
        """Return all faces using 3DDFA-V3's large landmark model."""
        img_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        h = img_bgr.shape[0]
        _, results_all = self.retina_model.infer(img_bgr)

        faces = []
        for det_idx, results in enumerate(results_all):
            lm5_for_align = []
            lm5_for_bbox = []
            for idx in [74, 83, 54, 84, 90]:
                x, y = float(results[idx][0]), float(results[idx][1])
                lm5_for_bbox.append([x, y])
                lm5_for_align.append([x, h - 1 - y])  # same as official face_box.retinaface

            lm5_for_align = np.asarray(lm5_for_align, dtype=np.float32)
            lm5_for_bbox = np.asarray(lm5_for_bbox, dtype=np.float32)

            faces.append({
                "det_idx": det_idx,
                "lm5": lm5_for_align,
                "bbox": _bbox_from_lm(lm5_for_bbox),
            })

        faces = _sort_faces(faces, self.sort_faces)
        if self.max_faces and self.max_faces > 0:
            faces = faces[: self.max_faces]
        return faces

    def _detect_faces_mtcnn(self, pil_img: Image.Image) -> List[Dict[str, Any]]:
        img_rgb = np.asarray(pil_img)
        detections = self.mtcnn_model.detect_faces(img_rgb)

        faces = []
        for det_idx, d in enumerate(detections):
            if d.get("confidence", 0.0) <= 0.6:
                continue
            kp = d["keypoints"]
            lm5 = np.asarray([
                kp["left_eye"],
                kp["right_eye"],
                kp["nose"],
                kp["mouth_left"],
                kp["mouth_right"],
            ], dtype=np.float32)
            x, y, w, h = d["box"]
            faces.append({
                "det_idx": det_idx,
                "lm5": lm5,
                "bbox": (float(x), float(y), float(x + w), float(y + h)),
            })

        faces = _sort_faces(faces, self.sort_faces)
        if self.max_faces and self.max_faces > 0:
            faces = faces[: self.max_faces]
        return faces

    def _make_batch(self, pil_img: Image.Image, faces: List[Dict[str, Any]]):
        tensors = []
        valid_faces = []

        for f in faces:
            try:
                trans_params, crop_img, _lm, _ = self.align_img(pil_img, f["lm5"], self.lm3d_std)
                ten = torch.tensor(np.array(crop_img) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                f = dict(f)
                f["trans_params"] = trans_params
                valid_faces.append(f)
                tensors.append(ten)
            except Exception as e:
                print(f"[WARN] align_img failed face#{f.get('det_idx')}: {e}")

        if not tensors:
            return [], None

        return valid_faces, torch.cat(tensors, dim=0)

    @torch.no_grad()
    def get_faces(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        im_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(im_rgb)

        try:
            if self.detector_name == "retinaface":
                faces = self._detect_faces_retina(pil_img)
            else:
                faces = self._detect_faces_mtcnn(pil_img)
        except Exception as e:
            print(f"[WARN] detector failed: {e}")
            return []

        if len(faces) == 0:
            return []

        valid_faces, batch = self._make_batch(pil_img, faces)
        if batch is None:
            return []

        try:
            self.recon_model.input_img = batch.to(self.device)
            results = self.recon_model.forward()
        except Exception as e:
            print(f"[WARN] recon failed: {e}")
            return []

        if "ldm68" not in results:
            print(f"[WARN] result has no ldm68. keys={list(results.keys())}")
            return []

        ldm68 = results["ldm68"]
        if isinstance(ldm68, torch.Tensor):
            ldm68 = ldm68.detach().cpu().numpy()
        ldm68 = np.asarray(ldm68)

        if ldm68.ndim == 2:
            ldm68 = ldm68[None, ...]
        if ldm68.shape[1:] == (2, 68):
            ldm68 = np.transpose(ldm68, (0, 2, 1))
        if ldm68.shape[1:] != (68, 2):
            raise ValueError(f"Unexpected ldm68 batch shape: {ldm68.shape}")

        out = []
        for i, f in enumerate(valid_faces):
            lm68 = restore_ldm68_to_original(ldm68[i], f["trans_params"])
            out.append({
                "landmarks": lm68,
                "rect": rect_from_landmarks(lm68),
                "det_idx": f.get("det_idx", i),
                "bbox": f.get("bbox"),
            })
        return out


def bbox_to_lm5(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Approximate 5-point seed from a DFL-style manual rectangle.

    DFL manual mode lets the user place a square/rect near the face. 3DDFA-V3's
    align_img needs a coarse 5-point input, so this creates a weak seed from the
    rectangle. The final 68 landmarks are predicted by 3DDFA-V3.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    return np.asarray([
        [x1 + 0.34 * w, y1 + 0.38 * h],
        [x1 + 0.66 * w, y1 + 0.38 * h],
        [x1 + 0.50 * w, y1 + 0.55 * h],
        [x1 + 0.38 * w, y1 + 0.74 * h],
        [x1 + 0.62 * w, y1 + 0.74 * h],
    ], dtype=np.float32)


@torch.no_grad()
def get_face_from_manual_bbox(
    extractor: ThreeDDFAExtractor,
    frame_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Optional[Dict[str, Any]]:
    """Run 3DDFA-V3 using a user-provided pointer-centered rectangle."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = sorted([float(np.clip(x1, 0, w - 1)), float(np.clip(x2, 0, w - 1))])
    y1, y2 = sorted([float(np.clip(y1, 0, h - 1)), float(np.clip(y2, 0, h - 1))])
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    lm5 = bbox_to_lm5((x1, y1, x2, y2))
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    try:
        valid_faces, batch = extractor._make_batch(pil_img, [{"det_idx": 0, "lm5": lm5, "bbox": (x1, y1, x2, y2)}])
        if batch is None or len(valid_faces) == 0:
            return None
        extractor.recon_model.input_img = batch.to(extractor.device)
        results = extractor.recon_model.forward()
    except Exception as e:
        print(f"[WARN] manual 3DDFA failed: {e}")
        return None

    if "ldm68" not in results:
        print(f"[WARN] manual result has no ldm68. keys={list(results.keys())}")
        return None

    ldm68 = results["ldm68"]
    if isinstance(ldm68, torch.Tensor):
        ldm68 = ldm68.detach().cpu().numpy()
    ldm68 = np.asarray(ldm68)
    if ldm68.ndim == 2:
        ldm68 = ldm68[None, ...]
    if ldm68.shape[1:] == (2, 68):
        ldm68 = np.transpose(ldm68, (0, 2, 1))
    if ldm68.shape[1:] != (68, 2):
        print(f"[WARN] unexpected ldm68 shape: {ldm68.shape}")
        return None

    lm68 = restore_ldm68_to_original(ldm68[0], valid_faces[0]["trans_params"])
    return {"landmarks": lm68, "bbox": (x1, y1, x2, y2), "rect": (x1, y1, x2, y2)}
