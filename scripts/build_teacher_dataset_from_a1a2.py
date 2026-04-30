# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        seg_len = int.from_bytes(data[i:i+2], "big")
        if seg_len < 2 or i + seg_len > len(data):
            break
        seg_data = data[i+2:i+seg_len]
        i += seg_len
        if marker == 0xEF:
            try:
                obj = pickle.loads(seg_data)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}


def get_lm_from_meta(meta: Dict[str, Any], source: str) -> Optional[np.ndarray]:
    source = source.lower()

    if source in ("3ddfa", "auto"):
        mvfs = meta.get("mvfs", {})
        if isinstance(mvfs, dict):
            td = mvfs.get("3ddfa_v3", {})
            if isinstance(td, dict) and td.get("landmarks") is not None:
                arr = np.asarray(td["landmarks"], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == 68 and arr.shape[1] >= 2:
                    return arr[:, :2]

    if source in ("dfl", "auto"):
        lm = meta.get("landmarks", None)
        if lm is not None:
            arr = np.asarray(lm, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == 68 and arr.shape[1] >= 2:
                return arr[:, :2]

    return None


def get_landmarks_for_condition(path: Path, source: str) -> Tuple[Optional[np.ndarray], str]:
    meta = read_dfljpg_metadata(path)

    if source.lower() == "3ddfa":
        lm = get_lm_from_meta(meta, "3ddfa")
        return lm, "3ddfa" if lm is not None else "missing_3ddfa"

    if source.lower() == "dfl":
        lm = get_lm_from_meta(meta, "dfl")
        return lm, "dfl" if lm is not None else "missing_dfl"

    lm = get_lm_from_meta(meta, "3ddfa")
    if lm is not None:
        return lm, "3ddfa"
    lm = get_lm_from_meta(meta, "dfl")
    if lm is not None:
        return lm, "dfl"
    return None, "missing"


def draw_landmarks_68(img_bgr: np.ndarray, lm: np.ndarray, color=(0, 255, 0), thickness: int = 1) -> np.ndarray:
    out = img_bgr.copy()
    lm = np.asarray(lm, dtype=np.int32)
    chains = [
        range(0, 17), range(17, 22), range(22, 27), range(27, 31), range(31, 36),
        list(range(36, 42)) + [36], list(range(42, 48)) + [42],
        list(range(48, 60)) + [48], list(range(60, 68)) + [60],
    ]
    for chain in chains:
        pts = [tuple(lm[i, :2]) for i in chain if 0 <= i < len(lm)]
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
    for p in lm:
        cv2.circle(out, tuple(p[:2]), 1, color, -1, cv2.LINE_AA)
    return out


def make_face_hull_mask(image_shape, landmarks: np.ndarray, expand: float = 0.20) -> np.ndarray:
    """
    얼굴 외곽 + 눈썹 + 코 상단 일부를 포함한 convex hull mask.
    blur 영역을 APPLE처럼 얼굴 쪽에만 걸기 위한 근사 mask.
    """
    h, w = image_shape[:2]
    lm = np.asarray(landmarks, dtype=np.float32)

    use_idx = list(range(0, 17)) + list(range(17, 27)) + [27, 28, 29, 30]
    pts = lm[use_idx, :2].copy()

    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    pts[:, 0] = cx + (pts[:, 0] - cx) * (1.0 + expand)
    pts[:, 1] = cy + (pts[:, 1] - cy) * (1.0 + expand)

    pts = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def feather_mask(mask: np.ndarray, feather_sigma: float = 8.0) -> np.ndarray:
    mask_f = mask.astype(np.float32) / 255.0
    if feather_sigma > 0:
        mask_f = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    return np.clip(mask_f, 0.0, 1.0)


def make_soft_blur_condition(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    downsample: int = 32,
    expand: float = 0.20,
    blur_sigma: float = 2.0,
    feather_sigma: float = 8.0,
    blur_whole_image: bool = False,
):
    """
    APPLE 의도에 더 가까운 teacher blur condition.
    - 얼굴 영역 mask를 만들고
    - 전체 이미지를 저해상도화했다가 bicubic으로 복원하고
    - Gaussian blur를 더한 뒤
    - 얼굴 mask 영역에만 feather blend
    """
    h, w = image_bgr.shape[:2]

    if blur_whole_image:
        mask_bin = np.ones((h, w), dtype=np.uint8) * 255
        bbox = (0, 0, w - 1, h - 1)
    else:
        mask_bin = make_face_hull_mask(image_bgr.shape, landmarks, expand=expand)
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            bbox = (0, 0, w - 1, h - 1)
        else:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    small = cv2.resize(image_bgr, (downsample, downsample), interpolation=cv2.INTER_AREA)
    blurred_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    if blur_sigma > 0:
        blurred_full = cv2.GaussianBlur(blurred_full, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    mask_f = feather_mask(mask_bin, feather_sigma=feather_sigma)
    out = image_bgr.astype(np.float32) * (1.0 - mask_f[..., None]) \
        + blurred_full.astype(np.float32) * mask_f[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, bbox, (mask_f * 255).astype(np.uint8)


def make_debug_image(condition_bgr: np.ndarray, landmarks: np.ndarray, bbox, lm_source: str, mask=None):
    dbg = condition_bgr.copy()
    x1, y1, x2, y2 = bbox

    if mask is not None:
        color_mask = np.zeros_like(dbg)
        color_mask[..., 1] = mask
        dbg = cv2.addWeighted(dbg, 1.0, color_mask, 0.25, 0)

    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)
    dbg = draw_landmarks_68(dbg, landmarks)

    txt = f"DEBUG ONLY | lm_source={lm_source} | teacher cond = soft blur"
    cv2.putText(dbg, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(dbg, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return dbg


class InsightFaceBuffaloLEncoder:
    def __init__(self, ctx_id: int = 0, det_size=(640, 640), providers=None):
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
            except Exception as e:
                print(f"[WARN] onnxruntime preload_dlls failed: {e}")

        from insightface.app import FaceAnalysis
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.rec_model = self.app.models.get("recognition", None)
        if self.rec_model is None:
            raise RuntimeError("InsightFace buffalo_l recognition model was not loaded.")

    @staticmethod
    def _norm(emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-8)

    def encode_direct_aligned(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            feat = self.rec_model.get_feat(image_bgr)
            if feat is None:
                return None
            return self._norm(feat)
        except Exception as e:
            print(f"[WARN] direct ID embedding failed: {e}")
            return None

    def encode_detect(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(image_bgr)
        if not faces:
            return None

        def area(face):
            x1, y1, x2, y2 = face.bbox
            return float(max(0, x2 - x1) * max(0, y2 - y1))

        face = max(faces, key=area)
        return self._norm(face.embedding)

    def encode(self, image_bgr: np.ndarray, mode: str = "direct") -> Optional[np.ndarray]:
        mode = mode.lower()
        if mode == "direct":
            return self.encode_direct_aligned(image_bgr)
        if mode == "detect":
            return self.encode_detect(image_bgr)

        emb = self.encode_direct_aligned(image_bgr)
        if emb is not None:
            return emb
        return self.encode_detect(image_bgr)


def list_images(path: Path, recursive: bool = False) -> List[Path]:
    if not path.exists():
        return []
    it = path.rglob("*") if recursive else path.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_mean_id_embedding(
    image_paths: Sequence[Path],
    encoder: InsightFaceBuffaloLEncoder,
    mode: str = "direct",
    fallback_zero: bool = True,
):
    embs, failed = [], []

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            failed.append(str(p))
            continue
        emb = encoder.encode(img, mode=mode)
        if emb is None:
            failed.append(str(p))
            continue
        embs.append(emb.astype(np.float32))

    if not embs:
        if not fallback_zero:
            raise RuntimeError(
                "No ID embeddings could be extracted. "
                "Use --id-embed-mode direct for aligned crops or add --fallback-zero-id."
            )
        mean = np.zeros((512,), dtype=np.float32)
    else:
        arr = np.stack(embs, 0).astype(np.float32)
        mean = arr.mean(0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)

    return mean.astype(np.float32), {
        "num_input": len(image_paths),
        "num_success": len(embs),
        "num_failed": len(failed),
        "failed": failed[:20],
        "id_embed_mode": mode,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_identity_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def build_teacher_dataset(args):
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve()

    cond_root = out_root / "conditions" / "blur_only"
    debug_root = out_root / "conditions" / "debug_blur_landmark"
    emb_root = out_root / "id_embeddings"
    meta_root = out_root / "meta"

    for d in [cond_root, debug_root, emb_root, meta_root]:
        d.mkdir(parents=True, exist_ok=True)

    encoder = None if args.no_id else InsightFaceBuffaloLEncoder(
        ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size)
    )

    rows, skipped, reports = [], [], []
    lm_counts = {"3ddfa": 0, "dfl": 0, "missing": 0, "missing_3ddfa": 0, "missing_dfl": 0}

    id_dirs = iter_identity_dirs(dataset_root)
    if args.max_ids > 0:
        id_dirs = id_dirs[:args.max_ids]

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] identities={len(id_dirs)}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] landmark_source={args.landmark_source}")
    print(f"[INFO] id_embed_mode={args.id_embed_mode}")
    print(f"[INFO] blur_mode=soft_blur_face_region")

    for id_i, id_dir in enumerate(id_dirs):
        person_id = id_dir.name
        a1_dir, a2_dir = id_dir / args.a1_name, id_dir / args.a2_name

        a1_images = list_images(a1_dir, recursive=args.recursive)
        a2_images = list_images(a2_dir, recursive=args.recursive)

        if not a2_images:
            skipped.append({"person_id": person_id, "reason": "no_A2_images", "a2_dir": str(a2_dir)})
            continue

        id_embed_path = emb_root / f"{person_id}.npy"
        if args.no_id:
            np.save(str(id_embed_path), np.zeros((512,), dtype=np.float32))
            report = {"person_id": person_id, "no_id": True}
        else:
            src = a1_images
            if args.id_source.lower() == "a1a2":
                src = a1_images + a2_images
            elif args.id_source.lower() == "a2":
                src = a2_images

            if not src:
                skipped.append({"person_id": person_id, "reason": "no_id_source_images"})
                continue

            emb, report = build_mean_id_embedding(src, encoder, args.id_embed_mode, args.fallback_zero_id)
            report["person_id"] = person_id
            report["id_source"] = args.id_source
            np.save(str(id_embed_path), emb.astype(np.float32))

        reports.append(report)

        if args.max_images_per_id > 0:
            a2_images = a2_images[:args.max_images_per_id]

        saved = 0
        for img_path in a2_images:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                skipped.append({"person_id": person_id, "path": str(img_path), "reason": "cannot_read"})
                continue

            lm, lm_source = get_landmarks_for_condition(img_path, args.landmark_source)
            lm_counts[lm_source] = lm_counts.get(lm_source, 0) + 1
            if lm is None:
                skipped.append({"person_id": person_id, "path": str(img_path), "reason": f"no_{args.landmark_source}_landmarks"})
                continue

            condition, bbox, cond_mask = make_soft_blur_condition(
                image_bgr=img,
                landmarks=lm,
                downsample=args.downsample,
                expand=args.expand,
                blur_sigma=args.blur_sigma,
                feather_sigma=args.feather_sigma,
                blur_whole_image=args.blur_whole_image,
            )

            cond_dir = cond_root / person_id
            dbg_dir = debug_root / person_id
            cond_dir.mkdir(parents=True, exist_ok=True)
            dbg_dir.mkdir(parents=True, exist_ok=True)

            cond_path = cond_dir / f"{img_path.stem}.png"
            debug_path = dbg_dir / f"{img_path.stem}.jpg"

            cv2.imwrite(str(cond_path), condition)
            if args.debug:
                dbg = make_debug_image(condition, lm, bbox, lm_source, cond_mask)
                cv2.imwrite(str(debug_path), dbg)

            rows.append({
                "person_id": person_id,
                "clean_path": str(img_path),
                "condition_path": str(cond_path),
                "condition_type": "soft_blur_no_landmark",
                "id_embed_path": str(id_embed_path),
                "id_source": args.id_source,
                "id_embed_mode": args.id_embed_mode,
                "landmark_source": lm_source,
                "requested_landmark_source": args.landmark_source,
                "debug_path": str(debug_path) if args.debug else None,
                "blur_downsample": args.downsample,
                "blur_expand": args.expand,
                "blur_sigma": args.blur_sigma,
                "feather_sigma": args.feather_sigma,
                "blur_bbox": [int(v) for v in bbox],
            })
            saved += 1

        print(f"[{id_i+1}/{len(id_dirs)}] {person_id}: A1={len(a1_images)} A2={len(a2_images)} saved={saved}")

    write_jsonl(meta_root / "teacher_index.jsonl", rows)
    write_jsonl(out_root / "skipped.jsonl", skipped)
    write_jsonl(meta_root / "id_embedding_report.jsonl", reports)
    (meta_root / "landmark_source_counts.json").write_text(json.dumps(lm_counts, indent=2), encoding="utf-8")

    print(f"[DONE] rows={len(rows)} skipped={len(skipped)}")
    print(f"[LANDMARK_COUNTS] {lm_counts}")
    print(f"[OK] teacher index: {meta_root / 'teacher_index.jsonl'}")


def build_parser():
    p = argparse.ArgumentParser("Build MVFS teacher soft-blur dataset from id/A1,A2 DFLJPG dataset")
    p.add_argument("--dataset-root", required=True, type=str)
    p.add_argument("--out-root", required=True, type=str)

    p.add_argument("--a1-name", default="A1", type=str)
    p.add_argument("--a2-name", default="A2", type=str)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--downsample", default=32, type=int)
    p.add_argument("--expand", default=0.20, type=float)
    p.add_argument("--blur-sigma", default=2.0, type=float)
    p.add_argument("--feather-sigma", default=8.0, type=float)
    p.add_argument("--blur-whole-image", action="store_true")
    p.add_argument("--debug", action="store_true")

    p.add_argument("--id-source", default="A1", choices=["A1", "A2", "A1A2"])
    p.add_argument("--id-embed-mode", default="direct", choices=["direct", "detect", "auto"],
                   help="direct uses buffalo_l recognition directly on aligned crop; detect runs face detector first.")
    p.add_argument("--landmark-source", default="3ddfa", choices=["3ddfa", "dfl", "auto"])

    p.add_argument("--ctx-id", default=0, type=int)
    p.add_argument("--det-size", default=640, type=int)
    p.add_argument("--fallback-zero-id", action="store_true")
    p.add_argument("--no-id", action="store_true")

    p.add_argument("--max-ids", default=0, type=int)
    p.add_argument("--max-images-per-id", default=0, type=int)
    return p


if __name__ == "__main__":
    build_teacher_dataset(build_parser().parse_args())
