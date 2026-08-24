"""Verification: decode a `.ddc` this pipeline produced and score it against a reference.

Deliberately narrow scope for tonight: verification `.ddc` runs use `overlap=0` (the plain snap grid,
`src/utils/tiling.make_offsets` with `stride=patch`) so patch index `k` lines up 1:1, row-major, with
the pre-computed MERLIN full-scene ground-truth patch stacks already cached for the E1 symmetrization
study (`data/cache/symstudy/*_merlin_gt.npy`, `[n, 256, 256]`) — no need to regenerate MERLIN GT or
implement index-remapping for an overlapping grid. The *production* run (the numbers that actually get
reported) uses `overlap=2` per the current pipeline's default and is not what this module scores —
production quality is a separate, coarser sanity check (bpp/compression-ratio only) against the
existing FPGA table in `docs/onboard_pipeline.md` §10, done by hand/notebook, not by this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.utils.metrics import mse, psnr, ssim

from .pipeline import decompress_record, latent_shapes


def score_ddc_against_gt(
    net: torch.nn.Module,
    records: list[tuple[bytes, bytes]],
    gt_patches: np.ndarray,
    device: torch.device,
    sample: int = -1,
) -> dict[str, Any]:
    """Decode up to `sample` records (all, if -1) and score each against `gt_patches[idx]` (linear-
    amplitude, `[n, 256, 256]`) using the current, post-fix metric basis (`src/utils/metrics.py`:
    AMP_LIN_99-clipped MSE/PSNR/SSIM — the `f8b0fd5` fix, see `project_jetson_benchmark_scope`
    memory).

    Returns per-patch values + summary stats.
    """
    if len(records) != gt_patches.shape[0]:
        raise ValueError(
            f"{len(records)} records but {gt_patches.shape[0]} GT patches — this scorer assumes an "
            f"overlap=0 .ddc matching the cached GT grid exactly; index k must mean the same patch in "
            f"both. Re-run compress with --overlap 0 against the same tile."
        )
    y_shape, z_shape = latent_shapes(net, device)

    n = len(records) if sample < 0 else min(sample, len(records))
    idxs = np.linspace(0, len(records) - 1, n, dtype=int) if sample > 0 else range(len(records))

    per_patch: list[dict[str, float]] = []
    for idx in idxs:
        z_bytes, y_bytes = records[idx]
        recon = decompress_record(net, z_bytes, y_bytes, device, y_shape, z_shape)
        gt = gt_patches[idx]
        recon_t = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        mse_v = mse(recon_t, gt_t)
        per_patch.append(
            {
                "idx": int(idx),
                "mse": mse_v,
                "psnr": psnr(recon_t, gt_t, mse_v),
                "ssim": ssim(recon_t, gt_t),
            }
        )

    psnrs = [p["psnr"] for p in per_patch]
    ssims = [p["ssim"] for p in per_patch]
    return {
        "n_scored": len(per_patch),
        "n_total_records": len(records),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "per_patch": per_patch,
    }
