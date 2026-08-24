"""Core sequential compress/decompress loop — the Python replica of `stream_pipeline.cpp`'s `seq`
(default, non-optimized) behavior: load the full tile, patchify at a fixed overlap (no
symmetrization), normalize, run one patch through the model at a time, entropy-code, accumulate
`.ddc` records. See `docs/tmp_jetson_orin_overnight.md` for the design trail and
`inference_edge/README.md` for scope notes.

Deliberately excluded (per the "much simpler" scope): CPU threading/worker pools, independent-
schedule flags, DPU-style subgraph/lane placement (no DPU here), windowed/streamed tile reads (the
whole tile is loaded at once — trivial on Jetson-class RAM, unlike the ZCU102's 3 GB budget).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from src.utils.constants import AMP_MAX, AMP_MIN, EPS
from src.utils.tiling import make_offsets

from .timing import StageTimer

PATCH = 256


def normalize_input(x_lin: Tensor) -> Tensor:
    """Raw linear-amplitude [B,2,H,W] -> log-scale [0,1].

    Identical formula to scripts/evaluation/benchmark_gpu.py / the training-time
    LightningModule.forward normalization — single source of truth is src/utils/constants.py's
    AMP_MIN/AMP_MAX/EPS.
    """
    return (torch.log(torch.square(x_lin) + EPS) - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)


def denormalize_output(x_norm: Tensor) -> Tensor:
    """Inverse of `normalize_input`'s log-scale mapping, back to linear amplitude (matches
    `postprocess`/`denorm` in the FPGA-aligned schema)."""
    log_r = 2 * (x_norm * (AMP_MAX - AMP_MIN) + AMP_MIN)
    return torch.sqrt(torch.clamp(torch.exp(log_r) - EPS, min=0.0))


def load_tile(path: Path) -> np.ndarray:
    """Load a [H, W, 2] float32 raw-amplitude (real, imag) tile from `.cos` (TerraSAR-X CoSAR) or a
    pre-extracted `.npy`.

    No symmetrization — matches the current onboard pipeline's default (E1).
    """
    if path.suffix == ".cos":
        from src.utils.sar_utils import load_cosar

        tile = load_cosar(path)
        if tile is None:
            raise FileNotFoundError(f"could not read CoSAR file: {path}")
        return tile.astype(np.float32)
    tile = np.load(path)
    if tile.ndim != 3 or tile.shape[-1] != 2:
        raise ValueError(f"expected a [H,W,2] tile, got {tile.shape} from {path}")
    return tile.astype(np.float32)


def make_grid(H: int, W: int, overlap: int, max_rows: int = -1) -> tuple[list[int], list[int]]:
    """Azimuth (row) + range (col) patch-offset grid — byte-identical rule to the C++ `make_grid`
    (`src/utils/tiling.make_offsets`, already pinned against the board-verified counts by that
    module's own self-test).

    `max_rows` caps azimuth patch-rows, matching `--max-rows` in
    `main_stream.cpp`, for quick tests before committing to a full-scene run.
    """
    if overlap < 0 or overlap >= PATCH:
        raise ValueError(f"overlap must be in [0, {PATCH}), got {overlap}")
    stride = PATCH - overlap
    row_offs = make_offsets(H, PATCH, stride)
    col_offs = make_offsets(W, PATCH, stride)
    if max_rows >= 0:
        row_offs = row_offs[:max_rows]
    return row_offs, col_offs


def _count_bytes(strings: Any) -> int:
    """Count the total number of bytes in a nested structure of strings, lists, or tuples."""
    total = 0
    for s in strings:
        total += sum(len(x) for x in s) if isinstance(s, (list, tuple)) else len(s)
    return total


def latent_shapes(
    net: torch.nn.Module, device: torch.device
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Spatial (H',W') shapes of `y` (always) and `z` (hyperprior archs only) for a PATCH x PATCH
    input, via one dummy forward pass on zeros.

    Both are fixed by (patch size, architecture) alone, so
    they need computing once, not stored per-patch — same trick the `.ddc` format itself relies on
    (docs/onboard_pipeline.md §9: "latent shapes are derived from patch+arch, not stored"). Returns
    `(y_shape, None)` for factorized-prior archs (no hyperprior, no `z`).
    """
    with torch.inference_mode():
        x = torch.zeros(1, 1, PATCH, PATCH, device=device)
        y = net.g_a(x)
        y_shape = (y.shape[-2], y.shape[-1])
        if not hasattr(net, "h_a"):
            return y_shape, None
        z = net.h_a(torch.abs(torch.cat([y, y], dim=1)))
        return y_shape, (z.shape[-2], z.shape[-1])


def compress_tile(
    net: torch.nn.Module,
    tile: np.ndarray,
    overlap: int,
    device: torch.device,
    timer: StageTimer,
    max_rows: int = -1,
) -> tuple[list[tuple[bytes, bytes]], int, int]:
    """Compress a [H,W,2] tile into a list of `(z_bytes, y_bytes)` `.ddc` records, row-major
    (azimuth-outer, range-inner — identical grid order to `stream_pipeline.cpp::process_block` / the
    `.ddc` body layout). Returns `(records, grid_r, grid_a)` (range-cols, azimuth-rows — same naming as
    the C++ `StreamResult`).
    """
    H, W, C = tile.shape
    if C != 2:
        raise ValueError(f"expected tile[...,2], got last dim {C}")
    row_offs, col_offs = make_grid(H, W, overlap, max_rows)
    has_hyper = hasattr(net, "h_a")
    records: list[tuple[bytes, bytes]] = []

    with torch.inference_mode():
        for ro in row_offs:
            for co in col_offs:
                with timer.time("patchify"):
                    patch = tile[ro : ro + PATCH, co : co + PATCH, :]
                    x = (
                        torch.from_numpy(np.ascontiguousarray(patch))
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                    )

                with timer.time("normalize"):
                    x = normalize_input(x)
                    x_real, x_imag = x[:, :1], x[:, 1:]

                with timer.time("g_a"):
                    y_real = net.g_a(x_real)
                    y_imag = net.g_a(x_imag)
                y = torch.cat((y_real, y_imag), dim=1)

                if has_hyper:
                    with timer.time("h_a"):
                        z = net.h_a(torch.abs(y))
                    with timer.time("eb_compress"):
                        z_strings = net.entropy_bottleneck.compress(z)
                    with timer.time("eb_decompress"):
                        z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
                    with timer.time("h_s"):
                        scales = net.h_s(z_hat)
                    with timer.time("gc_compress"):
                        indexes = net.gaussian_conditional.build_indexes(scales)
                        y_strings = net.gaussian_conditional.compress(y, indexes)
                    records.append((z_strings[0], y_strings[0]))
                else:
                    with timer.time("eb_compress"):
                        y_strings = net.entropy_bottleneck.compress(y)
                    records.append((b"", y_strings[0]))

    return records, len(col_offs), len(row_offs)


def decompress_record(
    net: torch.nn.Module,
    z_bytes: bytes,
    y_bytes: bytes,
    device: torch.device,
    y_shape: tuple[int, int],
    z_shape: tuple[int, int] | None,
) -> np.ndarray:
    """Decode one `.ddc`-style record back to a `[PATCH, PATCH]` linear-amplitude reconstruction
    (real+imag averaged — the MERLIN convention, per CLAUDE.md: "average real + imag predictions
    when computing intensity for metrics"). Mirrors `run_scenario_decompress`'s per-patch sequence
    in `benchmark_gpu.py`. `y_shape`/`z_shape` come from `latent_shapes` (computed once, not per-
    patch); `z_shape` is `None` for factorized-prior archs (no hyperprior, no `z`).

    Raises if the reconstruction contains non-finite values: for hyperprior archs, `scale` must land in
    the exact same `gc_scale_table` bucket at decode as at encode, and `h_s`'s FP32 output is not
    bit-reproducible across GPU architectures — decoding on a different GPU than the one that compressed
    can desync the entropy decode for individual elements and produce silently-wrong (not just
    imprecise) reconstructions. Caught the hard way once already (see project_jetson_edge_pipeline
    memory / docs/tmp_jetson_orin_overnight.md) — this is the errors-over-fallbacks guard so it fails
    loudly instead of quietly poisoning a PSNR/SSIM average next time.
    """
    has_hyper = hasattr(net, "h_a")
    n = net.nb_channels_main
    with torch.inference_mode():
        if has_hyper:
            assert z_shape is not None
            z_hat = net.entropy_bottleneck.decompress([z_bytes], z_shape)
            scales = net.h_s(z_hat)
            indexes = net.gaussian_conditional.build_indexes(scales)
            y_hat = net.gaussian_conditional.decompress([y_bytes], indexes)
        else:
            y_hat = net.entropy_bottleneck.decompress([y_bytes], y_shape)

        recon_real = net.g_s(y_hat[:, :n, :, :])
        recon_imag = net.g_s(y_hat[:, n:, :, :])
        recon_norm = 0.5 * (recon_real + recon_imag)  # MERLIN convention: average re+im
        recon_lin = denormalize_output(recon_norm)
    out = recon_lin.squeeze().cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError(
            "decompress_record: non-finite reconstruction — almost certainly decoding on a different "
            "GPU than the one that compressed this .ddc (FP32 h_s desyncs the entropy decode across "
            "architectures for hyperprior archs). Decode on the same device that compressed."
        )
    return out
