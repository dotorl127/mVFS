# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, *_THIS.parents]:
    if (_p / "mvfs_common").exists():
        sys.path.insert(0, str(_p))
        break
else:
    raise RuntimeError("Cannot find mvfs_common. Expected a parent directory containing mvfs_common.")

from mvfs_common.dfljpg_utils import read_dfljpg_metadata, write_dfljpg_metadata, ensure_mvfs_meta
from mvfs_common.face_parsing_bisenet import FaceSegExtractor

IMAGE_EXTS = {".jpg", ".jpeg"}


def iter_images(dataset_root: Path, splits=("A1", "A2")):
    for person_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        for split in splits:
            d = person_dir / split
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    yield person_dir.name, split, p


def rel_to_dataset(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def main():
    ap = argparse.ArgumentParser("Write face segmentation mask path into DFLJPG metadata and save mask as sidecar PNG")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["A1", "A2"])
    ap.add_argument("--mask-dir-name", default="meta_faceseg")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    mask_root = dataset_root / args.mask_dir_name
    mask_root.mkdir(parents=True, exist_ok=True)

    seg = FaceSegExtractor(device=args.device)

    ok = fail = skip = 0
    for person_id, split, img_path in iter_images(dataset_root, args.splits):
        try:
            meta = read_dfljpg_metadata(img_path)
            mvfs = ensure_mvfs_meta(meta)
            face_seg = mvfs.setdefault("face_seg", {})

            mask_path = mask_root / person_id / split / f"{img_path.stem}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            # New sidecar path key.
            existing_path = face_seg.get("mask_path")
            if args.skip_existing and existing_path and (dataset_root / existing_path).exists():
                skip += 1
                continue
            if args.skip_existing and mask_path.exists():
                face_seg["mask_path"] = rel_to_dataset(mask_path, dataset_root)
                face_seg["source"] = "facexlib_bisenet"
                face_seg["include_ids"] = seg.include_ids
                # Remove old inline mask if it exists. It can exceed JPEG APP segment size.
                face_seg.pop("mask_png", None)
                write_dfljpg_metadata(img_path, meta)
                skip += 1
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[FAIL] {img_path}: cannot_read")
                fail += 1
                continue

            mask = seg.predict_mask(img)
            cv2.imwrite(str(mask_path), mask)

            face_seg["mask_path"] = rel_to_dataset(mask_path, dataset_root)
            face_seg["source"] = "facexlib_bisenet"
            face_seg["include_ids"] = seg.include_ids

            # Important:
            # Do NOT store PNG bytes directly in DFLJPG metadata.
            # JPEG APP segments are limited to 65535 bytes, and pickle payload can exceed it.
            face_seg.pop("mask_png", None)

            write_dfljpg_metadata(img_path, meta)
            ok += 1

            if ok % 50 == 0:
                print(f"[INFO] ok={ok} skip={skip} fail={fail}")

        except Exception as e:
            print(f"[FAIL] {img_path}: {e}")
            fail += 1

    print(f"[DONE] ok={ok} skip={skip} fail={fail}")
    print(f"[MASK_ROOT] {mask_root}")


if __name__ == "__main__":
    main()
