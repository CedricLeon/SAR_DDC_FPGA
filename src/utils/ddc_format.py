"""DDC downlink-product container format (v1) — reference codec.

A ``.ddc`` file is the compressed SAR tile as it would be queued for downlink: a fixed HEADER
(how to decode), a BODY of per-patch rANS bitstreams, and an optional TRAILER offset table for
random access. Little-endian, positional (no field names in the file). See docs/onboard_pipeline.md §6.

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
from pathlib import Path

MAGIC = b"DDC1"
_FIXED = "<4sBBHHHHIIHHfff8s"
_FIXED_SIZE = struct.calcsize(_FIXED)  # 46
_OFFSET_BYTES = 8  # u64 per trailer entry

ARCH_IDS = {"FP": 0, "ResFP": 1, "SHyp": 2, "ResSHyp": 3}
ARCH_NAMES = {v: k for k, v in ARCH_IDS.items()}


# --- params_sha guard (FNV-1a-64 over the entropy CDF tables) ---
# The .ddc header carries an 8-byte guard so a decoder can refuse a file whose entropy CDF tables
# differ from the ones it holds. CANONICAL ALGORITHM = FNV-1a-64 over the concatenated bytes of the
# sorted entropy_params/*.npy files, emitted little-endian. This MUST stay byte-identical to the
# on-board writer inference_cpp/src/stream/stream_pipeline.cpp::fnv1a_params (same offset basis /
# prime / LE output) — the board stamps every real .ddc this way, so a Python verifier that used a
# different hash (an earlier draft used SHA-256[:8]) would reject every genuine downlink product.
# Run `python src/utils/ddc_format.py` for the known-answer check.
_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_U64_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash of ``data`` (standard offset basis / prime)."""
    h = _FNV64_OFFSET
    for b in data:
        h = ((h ^ b) * _FNV64_PRIME) & _U64_MASK
    return h


def fnv1a_64_bytes(data: bytes) -> bytes:
    """FNV-1a-64 of ``data`` as 8 little-endian bytes (the on-disk params_sha form)."""
    return fnv1a_64(data).to_bytes(8, "little")


def params_guard(entropy_params_dir) -> bytes:
    """Canonical 8-byte params_sha: FNV-1a-64 over the sorted entropy_params/*.npy bytes, LE.

    Streams each file's bytes through one running hash in sorted-path order — matches the C++
    fnv1a_params, which sorts the std::filesystem::path list and hashes each file's bytes in turn.
    """
    h = _FNV64_OFFSET
    for fp in sorted(Path(entropy_params_dir).glob("*.npy")):
        for b in fp.read_bytes():
            h = ((h ^ b) * _FNV64_PRIME) & _U64_MASK
    return h.to_bytes(8, "little")


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


def _read_exact(f, n: int, what: str) -> bytes:
    """Read exactly ``n`` bytes or raise — never silently truncate (errors over fallbacks)."""
    b = f.read(n)
    if len(b) != n:
        raise ValueError(f"truncated .ddc: wanted {n} bytes for {what}, got {len(b)}")
    return b


def _read_header(f):
    """Read the fixed header + tile_id/model_id strings, return DDCHeader and body offset."""
    vals = struct.unpack(_FIXED, _read_exact(f, _FIXED_SIZE, "fixed header"))
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
    tlen = struct.unpack("<H", _read_exact(f, 2, "tile_id length"))[0]
    tile_id = _read_exact(f, tlen, "tile_id").decode("utf-8")
    mlen = struct.unpack("<H", _read_exact(f, 2, "model_id length"))[0]
    model_id = _read_exact(f, mlen, "model_id").decode("utf-8")
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
            lz = struct.unpack("<I", _read_exact(f, 4, "len_z"))[0]
            z = _read_exact(f, lz, "z stream")
            ly = struct.unpack("<I", _read_exact(f, 4, "len_y"))[0]
            y = _read_exact(f, ly, "y stream")
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
    if table_start < _FIXED_SIZE:
        raise ValueError(f"corrupt .ddc: trailer start {table_start} precedes the header")
    with open(path, "rb") as f:
        f.seek(table_start)
        offsets = struct.unpack(f"<{n}Q", _read_exact(f, n * _OFFSET_BYTES, "offset table"))
    return list(offsets), table_start


def read_patch(path, offset: int) -> tuple[bytes, bytes]:
    """Random-access a single patch record given its byte offset (from the trailer)."""
    with open(path, "rb") as f:
        f.seek(offset)
        lz = struct.unpack("<I", _read_exact(f, 4, "len_z"))[0]
        z = _read_exact(f, lz, "z stream")
        ly = struct.unpack("<I", _read_exact(f, 4, "len_y"))[0]
        y = _read_exact(f, ly, "y stream")
    return z, y


def _selfcheck() -> None:
    """Known-answer + guard smoke test — run with ``python src/utils/ddc_format.py``."""
    import io

    # Canonical FNV-1a-64 test vectors — proves parity with the C++ fnv1a_params constants
    # (offset 0xcbf29ce484222325, prime 0x100000001b3, little-endian output).
    assert fnv1a_64(b"") == 0xCBF29CE484222325
    assert fnv1a_64(b"a") == 0xAF63DC4C8601EC8C
    assert fnv1a_64(b"foobar") == 0x85944171F73967E8
    assert fnv1a_64_bytes(b"a") == bytes.fromhex("8cec01864cdc63af")

    # Malformed input must raise, not silently truncate.
    try:
        _read_exact(io.BytesIO(b"ab"), 4, "probe")
    except ValueError:
        pass
    else:
        raise AssertionError("_read_exact did not raise on a short read")

    print("ddc_format self-check: PASS (FNV-1a-64 KAT + read guard)")


if __name__ == "__main__":
    _selfcheck()
