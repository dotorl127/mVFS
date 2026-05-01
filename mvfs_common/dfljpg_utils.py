# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

APP15_MARKER = b"\xFF\xEF"


def read_dfljpg_metadata(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
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

        if marker == 0xDA:  # SOS
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


def write_dfljpg_metadata(path: str | Path, meta: Dict[str, Any]) -> None:
    """
    Write DFLJPG metadata into a JPEG APP15 segment.

    Important:
    - JPEG APP segment length is 2 bytes, so payload must remain < ~64KB.
    - Large objects such as segmentation masks must be stored as sidecar files,
      with only a small relative path saved in metadata.
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError(f"Not a JPEG file: {path}")

    payload = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) + 2 > 65535:
        raise ValueError(
            f"DFLJPG metadata too large: {len(payload)} bytes. "
            "Store large arrays/images as sidecar files and keep only paths in metadata."
        )

    new_seg = APP15_MARKER + (len(payload) + 2).to_bytes(2, "big") + payload

    out = bytearray()
    out += data[:2]
    i = 2
    inserted = False

    while i + 4 <= len(data):
        if data[i] != 0xFF:
            if not inserted:
                out += new_seg
                inserted = True
            out += data[i:]
            i = len(data)
            break

        marker = data[i + 1]

        if marker == 0xDA:  # SOS
            if not inserted:
                out += new_seg
                inserted = True
            out += data[i:]
            i = len(data)
            break

        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            out += data[i:i + 2]
            i += 2
            continue

        if i + 4 > len(data):
            break

        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        seg_end = i + 2 + seg_len
        if seg_len < 2 or seg_end > len(data):
            break

        if marker == 0xEF and not inserted:
            out += new_seg
            inserted = True
        elif marker == 0xEF:
            # Drop duplicate APP15 metadata segment.
            pass
        else:
            out += data[i:seg_end]

        i = seg_end

    if not inserted:
        out = bytearray(data[:2]) + bytearray(new_seg) + bytearray(data[2:])

    path.write_bytes(bytes(out))


def ensure_mvfs_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    mvfs = meta.setdefault("mvfs", {})
    if not isinstance(mvfs, dict):
        meta["mvfs"] = {}
        mvfs = meta["mvfs"]
    return mvfs
