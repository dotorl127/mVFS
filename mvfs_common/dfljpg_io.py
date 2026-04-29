# -*- coding: utf-8 -*-
"""DFLJPG-compatible JPEG APP15 metadata reader/writer.

Standalone replacement for DeepFaceLab's DFLJPG helper.
It preserves DFL-style keys:
    face_type, landmarks, source_filename, source_rect,
    source_landmarks, image_to_face_mat
and allows mvfs-specific extension keys under `mvfs`.
"""
from __future__ import annotations

import pickle
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _parse_jpeg_chunks(data: bytes) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    pos = 0
    n = len(data)

    while pos < n:
        if pos + 2 > n:
            break
        marker_l, marker_h = struct.unpack("BB", data[pos:pos + 2])
        pos += 2
        if marker_l != 0xFF:
            raise ValueError("Invalid JPEG marker")

        chunk_name = None
        chunk_size = None
        chunk_data = None
        chunk_ex_data = None

        if marker_h & 0xF0 == 0xD0:
            rst = marker_h & 0x0F
            if rst <= 7:
                chunk_name = f"RST{rst}"
                chunk_size = 0
            elif rst == 0x8:
                chunk_name = "SOI"
                chunk_size = 0
            elif rst == 0x9:
                chunk_name = "EOI"
                chunk_size = 0
            elif rst == 0xA:
                chunk_name = "SOS"
            elif rst == 0xB:
                chunk_name = "DQT"
            elif rst == 0xD:
                chunk_name = "DRI"
                chunk_size = 2
        elif marker_h & 0xF0 == 0xC0:
            sub = marker_h & 0x0F
            if sub == 0:
                chunk_name = "SOF0"
            elif sub == 2:
                chunk_name = "SOF2"
            elif sub == 4:
                chunk_name = "DHT"
        elif marker_h & 0xF0 == 0xE0:
            chunk_name = f"APP{marker_h & 0x0F}"

        if chunk_size is None:
            if pos + 2 > n:
                break
            chunk_size = struct.unpack(">H", data[pos:pos + 2])[0] - 2
            pos += 2

        if chunk_size > 0:
            chunk_data = data[pos:pos + chunk_size]
            pos += chunk_size

        if chunk_name == "SOS":
            c = pos
            while c + 1 < n and not (data[c] == 0xFF and data[c + 1] == 0xD9):
                c += 1
            chunk_ex_data = data[pos:c]
            pos = c

        chunks.append({"name": chunk_name, "m_h": marker_h, "data": chunk_data, "ex_data": chunk_ex_data})
        if chunk_name == "EOI":
            break

    return chunks


def _dump_jpeg_chunks(chunks: List[Dict[str, Any]]) -> bytes:
    out = bytearray()
    for c in chunks:
        out += struct.pack("BB", 0xFF, c["m_h"])
        if c.get("data") is not None:
            out += struct.pack(">H", len(c["data"]) + 2)
            out += c["data"]
        if c.get("ex_data") is not None:
            out += c["ex_data"]
    return bytes(out)


class DFLJPG:
    def __init__(self, filename: str | Path):
        self.filename = str(filename)
        self.data: bytes = b""
        self.chunks: List[Dict[str, Any]] = []
        self.dfl_dict: Dict[str, Any] = {}
        self.shape: Optional[tuple[int, int, int]] = None

    @staticmethod
    def load(filename: str | Path) -> "DFLJPG":
        inst = DFLJPG(filename)
        with open(filename, "rb") as f:
            inst.data = f.read()
        inst._parse()
        return inst

    def _parse(self) -> None:
        self.chunks = _parse_jpeg_chunks(self.data)
        self.dfl_dict = {}
        self.shape = None
        for c in self.chunks:
            if c["name"] in ("SOF0", "SOF2") and c.get("data") is not None:
                try:
                    _, h, w = struct.unpack(">BHH", c["data"][:5])
                    self.shape = (h, w, 3)
                except Exception:
                    pass
            if c["name"] == "APP15" and isinstance(c.get("data"), bytes):
                try:
                    self.dfl_dict = pickle.loads(c["data"])
                except Exception:
                    self.dfl_dict = {}

    def dump(self) -> bytes:
        meta = dict(self.dfl_dict)
        for k in list(meta.keys()):
            if meta[k] is None:
                meta.pop(k)

        chunks = [c for c in self.chunks if c["name"] != "APP15"]
        last_app_idx = 0
        for i, c in enumerate(chunks):
            if c["m_h"] & 0xF0 == 0xE0:
                last_app_idx = i
        chunks.insert(last_app_idx + 1, {"name": "APP15", "m_h": 0xEF, "data": pickle.dumps(meta), "ex_data": None})
        return _dump_jpeg_chunks(chunks)

    def save(self) -> None:
        with open(self.filename, "wb") as f:
            f.write(self.dump())

    def get_dict(self) -> Dict[str, Any]:
        return self.dfl_dict

    def set_dict(self, value: Dict[str, Any]) -> None:
        self.dfl_dict = value

    def get_img(self) -> np.ndarray:
        img = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.filename)
        return img

    def get_landmarks(self) -> np.ndarray:
        return np.asarray(self.dfl_dict["landmarks"], dtype=np.float32)

    def set_landmarks(self, landmarks: Any) -> None:
        self.dfl_dict["landmarks"] = np.asarray(landmarks, dtype=np.float32).tolist()

    def get_source_landmarks(self) -> Optional[np.ndarray]:
        v = self.dfl_dict.get("source_landmarks")
        return None if v is None else np.asarray(v, dtype=np.float32)

    def set_source_landmarks(self, landmarks: Any) -> None:
        self.dfl_dict["source_landmarks"] = np.asarray(landmarks, dtype=np.float32).tolist()

    def get_image_to_face_mat(self) -> Optional[np.ndarray]:
        v = self.dfl_dict.get("image_to_face_mat")
        return None if v is None else np.asarray(v, dtype=np.float32)

    def set_image_to_face_mat(self, mat: Any) -> None:
        self.dfl_dict["image_to_face_mat"] = np.asarray(mat, dtype=np.float32).tolist()

    def set_mvfs_meta(self, key: str, value: Any) -> None:
        self.dfl_dict.setdefault("mvfs", {})[key] = value

    def get_mvfs_meta(self, key: str, default: Any = None) -> Any:
        return self.dfl_dict.get("mvfs", {}).get(key, default)


# Backward-compatible functional API used by older mvfs scripts.
def write_dfljpg_metadata(jpg_path: Path | str, meta: Dict[str, Any]) -> None:
    jpg_path = Path(jpg_path)
    dfl = DFLJPG.load(jpg_path)
    dfl.set_dict(meta)
    dfl.save()


def read_dfljpg_metadata(jpg_path: Path | str) -> Optional[Dict[str, Any]]:
    try:
        return DFLJPG.load(jpg_path).get_dict()
    except Exception:
        return None


def update_dfljpg_metadata(jpg_path: Path | str, updates: Dict[str, Any]) -> None:
    dfl = DFLJPG.load(jpg_path)
    meta = dfl.get_dict()
    meta.update(updates)
    dfl.set_dict(meta)
    dfl.save()
