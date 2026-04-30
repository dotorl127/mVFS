
# -*- coding: utf-8 -*-
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict

APP15_MARKER = b"\xFF\xEF"

def read_dfljpg_metadata(path: str | Path) -> Dict[str, Any]:
    data = Path(path).read_bytes()
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
        seg_len = int.from_bytes(data[i:i + 2], 'big')
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
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError(f'Not a JPEG file: {path}')

    payload = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    seg = APP15_MARKER + (len(payload) + 2).to_bytes(2, 'big') + payload

    out = bytearray()
    out += data[:2]  # SOI
    i = 2
    inserted = False
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            out += data[i:]
            i = len(data)
            break
        marker = data[i + 1]
        if marker == 0xDA:  # SOS
            if not inserted:
                out += seg
                inserted = True
            out += data[i:]
            i = len(data)
            break
        out += data[i:i+2]
        i += 2
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            continue
        seg_len = int.from_bytes(data[i:i+2], 'big')
        current_seg = data[i-2:i+seg_len]
        if marker == 0xEF and not inserted:
            # Skip old APP15 metadata segment
            out = out[:-2]
            out += seg
            inserted = True
        else:
            out += data[i:i+seg_len]
        i += seg_len
    if not inserted:
        out = bytearray(data[:2]) + bytearray(seg) + bytearray(data[2:])
    path.write_bytes(bytes(out))

def ensure_mvfs_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    mvfs = meta.setdefault('mvfs', {})
    if not isinstance(mvfs, dict):
        meta['mvfs'] = {}
        mvfs = meta['mvfs']
    return mvfs
