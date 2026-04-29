# -*- coding: utf-8 -*-
"""
Attach 3DMM metadata paths/values from DFLJPG metadata into teacher_index.jsonl.

This does not require 3DDFA-V3 at training time. It only reads already-saved
DFLJPG APP15 metadata fields.

Expected DFLJPG optional metadata:
    meta["mvfs"]["3dmm"] = {
        "id_coeff": [...],
        "exp_coeff": [...],
        "pose": [...],
        "camera": [...],
        "quality": float
    }

If found, this script saves compact per-sample npz files and adds:
    "3dmm_path": ".../3dmm/<stem>.npz"

If not found, it keeps the row and optionally marks has_3dmm=false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

_MVFS_ROOT = Path(__file__).resolve().parents[1]
if str(_MVFS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVFS_ROOT))

from mvfs_common.dfljpg_io import DFLJPG


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(p: str, base: Path) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    return (base / q).resolve()


def find_3dmm(meta: Dict[str, Any]):
    if "mvfs" in meta and isinstance(meta["mvfs"], dict):
        mv = meta["mvfs"]
        if "3dmm" in mv and isinstance(mv["3dmm"], dict):
            return mv["3dmm"]
    if "3dmm" in meta and isinstance(meta["3dmm"], dict):
        return meta["3dmm"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=str)
    p.add_argument("--out-index", default=None, type=str)
    p.add_argument("--out-dir", default=None, type=str)
    p.add_argument("--require-3dmm", action="store_true")
    args = p.parse_args()

    index_path = Path(args.index).resolve()
    index_dir = index_path.parent
    rows = read_jsonl(index_path)

    out_index = Path(args.out_index).resolve() if args.out_index else index_dir / "teacher_index_with_3dmm.jsonl"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else index_dir / "3dmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    found = 0
    missing = 0

    for i, row in enumerate(rows):
        clean = row.get("clean_path") or row.get("aligned_path") or row.get("image_path")
        if not clean:
            missing += 1
            if args.require_3dmm:
                continue
            row2 = dict(row)
            row2["has_3dmm"] = False
            new_rows.append(row2)
            continue

        img_path = resolve_path(str(clean), index_dir)
        dfl = DFLJPG.load(str(img_path))
        if dfl is None or not dfl.has_data():
            missing += 1
            if args.require_3dmm:
                continue
            row2 = dict(row)
            row2["has_3dmm"] = False
            new_rows.append(row2)
            continue

        meta = dfl.get_dict()
        m3 = find_3dmm(meta)
        if m3 is None:
            missing += 1
            if args.require_3dmm:
                continue
            row2 = dict(row)
            row2["has_3dmm"] = False
            new_rows.append(row2)
            continue

        stem = Path(clean).stem
        npz_path = out_dir / f"{stem}.npz"

        save_kwargs = {}
        for key in ["id_coeff", "exp_coeff", "pose", "camera"]:
            if key in m3 and m3[key] is not None:
                save_kwargs[key] = np.asarray(m3[key], dtype=np.float32)
        if "quality" in m3:
            save_kwargs["quality"] = np.asarray([float(m3["quality"])], dtype=np.float32)

        if save_kwargs:
            np.savez(str(npz_path), **save_kwargs)

        row2 = dict(row)
        row2["has_3dmm"] = True
        row2["3dmm_path"] = str(npz_path)
        new_rows.append(row2)
        found += 1

    write_jsonl(out_index, new_rows)
    print(f"[DONE] found={found}, missing={missing}, output_rows={len(new_rows)}")
    print(f"[OK] out index: {out_index}")


if __name__ == "__main__":
    main()
