# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def install_dummy_nvdiffrast_if_missing() -> None:
    try:
        import nvdiffrast.torch  # noqa
        return
    except Exception:
        pass
    pkg = types.ModuleType("nvdiffrast")
    torch_mod = types.ModuleType("nvdiffrast.torch")
    class _DummyRendererContext:
        def __init__(self, *args, **kwargs):
            pass
    def _disabled(*args, **kwargs):
        raise RuntimeError(
            "nvdiffrast is disabled. This aligned updater should not call renderer."
        )
    torch_mod.RasterizeCudaContext = _DummyRendererContext
    torch_mod.RasterizeGLContext = _DummyRendererContext
    torch_mod.rasterize = _disabled
    torch_mod.interpolate = _disabled
    torch_mod.texture = _disabled
    torch_mod.antialias = _disabled
    pkg.torch = torch_mod
    sys.modules["nvdiffrast"] = pkg
    sys.modules["nvdiffrast.torch"] = torch_mod

install_dummy_nvdiffrast_if_missing()

import torch  # noqa: E402
from model.recon import face_model  # noqa: E402


def read_dfljpg_metadata(path: Path) -> Dict[str, Any]:
    try:
        data = path.read_bytes()
    except Exception:
        return {}
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return {}
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker == 0xDA:
            break
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            continue
        if i + 2 > len(data):
            break
        seg_len = int.from_bytes(data[i:i + 2], "big")
        if seg_len < 2 or i + seg_len > len(data):
            break
        seg_data = data[i + 2:i + seg_len]
        i += seg_len
        if marker == 0xEF:
            try:
                obj = pickle.loads(seg_data)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}


def make_segment(marker: int, payload: bytes) -> bytes:
    seg_len = len(payload) + 2
    if seg_len > 65535:
        raise ValueError(f"APP segment too large: {seg_len} bytes")
    return b"\xff" + bytes([marker]) + seg_len.to_bytes(2, "big") + payload


def write_dfljpg_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError(f"Not a JPEG file: {path}")
    payload = pickle.dumps(metadata)
    app15 = make_segment(0xEF, payload)
    out = bytearray()
    out += data[:2]
    out += app15
    i = 2
    while i < len(data):
        if i + 1 >= len(data) or data[i] != 0xFF:
            out += data[i:]
            break
        marker = data[i + 1]
        if marker == 0xDA:
            out += data[i:]
            break
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            out += data[i:i+2]
            i += 2
            continue
        if i + 4 > len(data):
            out += data[i:]
            break
        seg_len = int.from_bytes(data[i+2:i+4], "big")
        if seg_len < 2 or i + 2 + seg_len > len(data):
            out += data[i:]
            break
        segment = data[i:i+2+seg_len]
        if marker != 0xEF:
            out += segment
        i += 2 + seg_len
    path.write_bytes(bytes(out))


def get_dfl_landmarks(meta: Dict[str, Any]) -> Optional[np.ndarray]:
    lm = meta.get("landmarks", None)
    if lm is None:
        return None
    arr = np.asarray(lm, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 68 and arr.shape[1] >= 2:
        return arr[:, :2]
    return None


class _Args:
    pass


class Aligned3DDFARecon:
    def __init__(self, device: str = "cuda", backbone: str = "resnet50"):
        args = _Args()
        args.device = device
        args.backbone = backbone
        args.ldm68 = True
        args.ldm106 = False
        args.ldm106_2d = False
        args.ldm134 = False
        args.seg = False
        args.seg_visible = False
        args.useTex = False
        args.extractTex = False
        self.device = device
        self.model = face_model(args)

    @staticmethod
    def image_to_tensor_224(img_bgr: np.ndarray, device: str) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        t = torch.tensor(img_224 / 255.0, dtype=torch.float32)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    @torch.no_grad()
    def infer(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = img_bgr.shape[:2]
        self.model.input_img = self.image_to_tensor_224(img_bgr, self.device)
        alpha = self.model.net_recon(self.model.input_img)
        alpha_dict = self.model.split_alpha(alpha)

        face_shape = self.model.compute_shape(alpha_dict["id"], alpha_dict["exp"])
        rotation = self.model.compute_rotation(alpha_dict["angle"])
        face_shape_transformed = self.model.transform(face_shape, rotation, alpha_dict["trans"])
        v3d = self.model.to_camera(face_shape_transformed)
        v2d = self.model.to_image(v3d)

        ldm68 = self.model.get_landmarks_68(v2d)[0].detach().cpu().numpy().astype(np.float32)
        ldm68[:, 1] = 223.0 - ldm68[:, 1]
        ldm68[:, 0] *= float(w) / 224.0
        ldm68[:, 1] *= float(h) / 224.0

        return {
            "landmarks": ldm68,
            "id_coeff": alpha_dict["id"][0].detach().cpu().numpy().astype(np.float32),
            "exp_coeff": alpha_dict["exp"][0].detach().cpu().numpy().astype(np.float32),
            "alb_coeff": alpha_dict["alb"][0].detach().cpu().numpy().astype(np.float32),
            "angle": alpha_dict["angle"][0].detach().cpu().numpy().astype(np.float32),
            "sh": alpha_dict["sh"][0].detach().cpu().numpy().astype(np.float32),
            "trans": alpha_dict["trans"][0].detach().cpu().numpy().astype(np.float32),
            "alpha": alpha[0].detach().cpu().numpy().astype(np.float32),
            "image_shape": np.asarray([h, w], dtype=np.int32),
        }


def list_images(d: Path, recursive: bool = False) -> List[Path]:
    if not d.exists():
        return []
    it = d.rglob("*") if recursive else d.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def iter_dataset_images(dataset_root: Path, splits: List[str], recursive: bool):
    id_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    for id_dir in id_dirs:
        person_id = id_dir.name
        for split in splits:
            split_dir = id_dir / split
            for img_path in list_images(split_dir, recursive):
                yield person_id, split, img_path


def write_sidecar_npz(path: Path, result: Dict[str, Any], dfl_lm: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "landmarks_3ddfa": result["landmarks"].astype(np.float32),
        "id_coeff": result["id_coeff"].astype(np.float32),
        "exp_coeff": result["exp_coeff"].astype(np.float32),
        "alb_coeff": result["alb_coeff"].astype(np.float32),
        "angle": result["angle"].astype(np.float32),
        "sh": result["sh"].astype(np.float32),
        "trans": result["trans"].astype(np.float32),
        "alpha": result["alpha"].astype(np.float32),
        "image_shape": result["image_shape"].astype(np.int32),
    }
    if dfl_lm is not None:
        arrays["landmarks_dfl"] = dfl_lm.astype(np.float32)
    np.savez(str(path), **arrays)


def update_one_image(img_path: Path, recon: Aligned3DDFARecon, args: argparse.Namespace, person_id: str, split_name: str):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, {"path": str(img_path), "reason": "cannot_read"}
    meta = read_dfljpg_metadata(img_path)
    dfl_lm = get_dfl_landmarks(meta)
    try:
        result = recon.infer(img)
    except Exception as e:
        return False, {"path": str(img_path), "reason": "3ddfa_infer_failed", "error": repr(e)}

    mvfs = meta.get("mvfs", {})
    if not isinstance(mvfs, dict):
        mvfs = {}

    landmarks_3ddfa_list = result["landmarks"].astype(np.float32).tolist()

    mvfs["3ddfa_v3"] = {
        "version": 3,
        "extractor": "3DDFA-V3",
        "mode": "aligned_no_detector_no_render",
        "person_id": person_id,
        "split": split_name,
        "landmarks": landmarks_3ddfa_list,
        "image_shape": result["image_shape"].astype(np.int32).tolist(),
    }

    # Optional MVFS migration:
    # Replace top-level DFLJPG landmarks with 3DDFA-V3 landmarks.
    # Keep the old DFL landmarks under mvfs.landmarks_dfl_backup so the operation is reversible.
    if args.replace_top_landmarks:
        if "landmarks" in meta and "landmarks_dfl_backup" not in mvfs:
            try:
                mvfs["landmarks_dfl_backup"] = np.asarray(meta["landmarks"], dtype=np.float32).tolist()
            except Exception:
                mvfs["landmarks_dfl_backup"] = meta.get("landmarks")
        meta["landmarks"] = landmarks_3ddfa_list
        mvfs["top_landmarks_source"] = "3ddfa_v3"
    mvfs["3dmm"] = {
        "id_coeff": result["id_coeff"].astype(np.float32).tolist(),
        "exp_coeff": result["exp_coeff"].astype(np.float32).tolist(),
        "alb_coeff": result["alb_coeff"].astype(np.float32).tolist(),
        "angle": result["angle"].astype(np.float32).tolist(),
        "pose": result["angle"].astype(np.float32).tolist(),
        "sh": result["sh"].astype(np.float32).tolist(),
        "trans": result["trans"].astype(np.float32).tolist(),
        "camera": result["trans"].astype(np.float32).tolist(),
        "alpha": result["alpha"].astype(np.float32).tolist(),
        "has_coeffs": True,
    }

    sidecar_path = None
    if args.write_npz:
        sidecar_path = img_path.parent.parent / args.meta_dir_name / split_name / f"{img_path.stem}.npz"
        write_sidecar_npz(sidecar_path, result, dfl_lm)
        mvfs["3dmm_npz"] = str(sidecar_path)

    meta["mvfs"] = mvfs
    if args.write_dfljpg:
        write_dfljpg_metadata(img_path, meta)

    return True, {"path": str(img_path), "person_id": person_id, "split": split_name, "sidecar_npz": str(sidecar_path) if sidecar_path else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--splits", default="A1,A2", type=str)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "mbnetv3"])
    parser.add_argument("--write-dfljpg", action="store_true")
    parser.add_argument("--write-npz", action="store_true")
    parser.add_argument("--meta-dir-name", default="meta_3ddfa", type=str)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--replace-top-landmarks", action="store_true", help="Replace top-level DFLJPG meta['landmarks'] with 3DDFA-V3 landmarks. Original DFL landmarks are backed up.")
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--report-out", default=None, type=str)
    args = parser.parse_args()

    if not args.write_dfljpg and not args.write_npz:
        args.write_dfljpg = True
        args.write_npz = True

    dataset_root = Path(args.dataset_root).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    report_path = Path(args.report_out).resolve() if args.report_out else dataset_root / "mvfs_3ddfa_update_report.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] splits={splits}")
    print(f"[INFO] mode=aligned_no_detector_no_render")
    print(f"[INFO] write_dfljpg={args.write_dfljpg}, write_npz={args.write_npz}")

    recon = Aligned3DDFARecon(device=args.device, backbone=args.backbone)

    total = ok = fail = skipped = 0
    with report_path.open("w", encoding="utf-8") as rf:
        for person_id, split, img_path in iter_dataset_images(dataset_root, splits, args.recursive):
            total += 1
            if args.max_images and total > args.max_images:
                break
            if args.skip_existing:
                meta = read_dfljpg_metadata(img_path)
                mvfs = meta.get("mvfs", {})
                if isinstance(mvfs, dict) and "3ddfa_v3" in mvfs and mvfs["3ddfa_v3"].get("mode") == "aligned_no_detector_no_render":
                    skipped += 1
                    continue
            success, rec = update_one_image(img_path, recon, args, person_id, split)
            rec["success"] = success
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if success:
                ok += 1
                print(f"[OK] {person_id}/{split}/{img_path.name}")
            else:
                fail += 1
                print(f"[FAIL] {person_id}/{split}/{img_path.name}: {rec.get('reason')} {rec.get('error','')}")
    print(f"[DONE] total={total}, ok={ok}, fail={fail}, skipped={skipped}")
    print(f"[REPORT] {report_path}")


if __name__ == "__main__":
    main()
