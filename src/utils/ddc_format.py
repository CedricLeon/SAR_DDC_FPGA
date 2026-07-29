"""DDC downlink-product container format (v1) — reference codec.

A ``.ddc`` file is the compressed SAR tile as it would be queued for downlink: a fixed HEADER
(how to decode), a BODY of per-patch rANS bitstreams, and an optional TRAILER offset table for
random access. Little-endian, positional (no field names in the file). See docs/onboard_pipeline.md §7.

Layout
------
HEADER: 46 fixed bytes + two length-prefixed UTF-8 strings::

    magic 4s "DDC1" | flags B | arch_id B | N H | M H | patch H | stride H |
    scene_H I | scene_W I | grid_r H | grid_a H | amp_min f | amp_max f | eps f | params_sha 8s
    | tile_id_len H | tile_id... | model_id_len H | model_id...

BODY, x (grid_r*grid_a) in row-major order::

    len_z I | z_bytes (len_z) | len_y I | y_bytes (len_y)      # FP: len_z = 0

TRAILER (only if flags bit0 set): (grid_r*grid_a) x u64 = byte offset of each patch record from
file start. Its location is DERIVED, not stored: ``table_start = filesize - n*8`` (n known from the
header), so the reader seeks straight to it with no pointer.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

MAGIC = b"DDC1"
_FIXED = "<4sBBHHHHIIHHfff8s"
_FIXED_SIZE = struct.calcsize(_FIXED)  # 46
_OFFSET_BYTES = 8  # u64 per trailer entry

ARCH_IDS = {"FP": 0, "ResFP": 1, "SHyp": 2, "ResSHyp": 3}
ARCH_NAMES = {v: k for k, v in ARCH_IDS.items()}


@dataclass
class DDCHeader:
    """Everything the decoder needs before touching the body."""

    arch_id: int
    N: int
    M: int
    patch: int
    stride: int
    scene_H: int
    scene_W: int
    grid_r: int
    grid_a: int
    amp_min: float
    amp_max: float
    eps: float
    params_sha: bytes  # 8 bytes — guard: must match the ground station's CDF tables
    tile_id: str
    model_id: str
    has_index: bool = True

    @property
    def n_patches(self) -> int:
        """Number of patches in the body (grid_r * grid_a)."""
        return self.grid_r * self.grid_a


def write_ddc(path, header: DDCHeader, records: list[tuple[bytes, bytes]]) -> list[int]:
    """Serialize a .ddc. ``records`` = [(z_bytes, y_bytes), ...], row-major, len == n_patches.

    Returns the list of per-patch byte offsets (also written as the trailer if has_index).
    """
    n = header.n_patches
    if len(records) != n:
        raise ValueError(f"{len(records)} records but grid implies {n}")
    if len(header.params_sha) != 8:
        raise ValueError("params_sha must be exactly 8 bytes")
    tile_b = header.tile_id.encode("utf-8")
    model_b = header.model_id.encode("utf-8")
    fixed = struct.pack(
        _FIXED,
        MAGIC,
        1 if header.has_index else 0,
        header.arch_id,
        header.N,
        header.M,
        header.patch,
        header.stride,
        header.scene_H,
        header.scene_W,
        header.grid_r,
        header.grid_a,
        header.amp_min,
        header.amp_max,
        header.eps,
        header.params_sha,
    )
    offsets: list[int] = []
    with open(path, "wb") as f:
        f.write(fixed)
        f.write(struct.pack("<H", len(tile_b)))
        f.write(tile_b)
        f.write(struct.pack("<H", len(model_b)))
        f.write(model_b)
        for z, y in records:
            offsets.append(f.tell())  # record start = current byte position
            f.write(struct.pack("<I", len(z)))
            f.write(z)
            f.write(struct.pack("<I", len(y)))
            f.write(y)
        if header.has_index:
            f.write(struct.pack(f"<{n}Q", *offsets))
    return offsets


def _read_header(f):
    """Read the fixed header + tile_id/model_id strings, return DDCHeader and body offset."""
    vals = struct.unpack(_FIXED, f.read(_FIXED_SIZE))
    if vals[0] != MAGIC:
        raise ValueError(f"not a DDC file (magic={vals[0]!r})")
    (
        _,
        flags,
        arch_id,
        N,
        M,
        patch,
        stride,
        scene_H,
        scene_W,
        grid_r,
        grid_a,
        amp_min,
        amp_max,
        eps,
        params_sha,
    ) = vals
    tlen = struct.unpack("<H", f.read(2))[0]
    tile_id = f.read(tlen).decode("utf-8")
    mlen = struct.unpack("<H", f.read(2))[0]
    model_id = f.read(mlen).decode("utf-8")
    header = DDCHeader(
        arch_id,
        N,
        M,
        patch,
        stride,
        scene_H,
        scene_W,
        grid_r,
        grid_a,
        amp_min,
        amp_max,
        eps,
        params_sha,
        tile_id,
        model_id,
        bool(flags & 1),
    )
    return header, f.tell()  # second value = byte offset where the body starts


def read_header(path) -> DDCHeader:
    """Read the fixed header + tile_id/model_id strings, return DDCHeader."""
    with open(path, "rb") as f:
        header, _ = _read_header(f)
    return header


def read_ddc(path) -> tuple[DDCHeader, list[tuple[bytes, bytes]]]:
    """Sequential decode — walks n patch records back-to-back (each self-delimits via its
    lengths)."""
    with open(path, "rb") as f:
        header, _ = _read_header(f)
        records: list[tuple[bytes, bytes]] = []
        for _ in range(header.n_patches):
            lz = struct.unpack("<I", f.read(4))[0]
            z = f.read(lz)
            ly = struct.unpack("<I", f.read(4))[0]
            y = f.read(ly)
            records.append((z, y))
    return header, records


def read_offset_table(path, header: DDCHeader) -> tuple[list[int], int]:
    """Locate + read the trailer.

    Location is derived: ``table_start = filesize - n*8``.
    """
    if not header.has_index:
        raise ValueError("file has no offset index (flags bit0 = 0)")
    n = header.n_patches
    table_start = os.path.getsize(path) - n * _OFFSET_BYTES
    with open(path, "rb") as f:
        f.seek(table_start)
        offsets = struct.unpack(f"<{n}Q", f.read(n * _OFFSET_BYTES))
    return list(offsets), table_start


def read_patch(path, offset: int) -> tuple[bytes, bytes]:
    """Random-access a single patch record given its byte offset (from the trailer)."""
    with open(path, "rb") as f:
        f.seek(offset)
        lz = struct.unpack("<I", f.read(4))[0]
        z = f.read(lz)
        ly = struct.unpack("<I", f.read(4))[0]
        y = f.read(ly)
    return z, y
