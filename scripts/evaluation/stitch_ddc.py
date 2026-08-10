#!/usr/bin/env python3
"""stitch_ddc.py — reconstruct a full despeckled tile from a streaming ``.ddc`` and its board-
decoded per-patch amplitudes, by ramp-blending on the ``.ddc`` grid (overlap study, pass 3).

The board compresses with ``stream_pipeline --overlap N`` and decodes with ``stream_pipeline
--decode`` (real INT8 h_s + g_s), giving a per-patch linear-amplitude stack ``[n, P, P]`` in record
order (azimuth row outer, range col inner). This script reproduces the exact patch offsets from the
header (``src.utils.tiling.make_offsets``, the same snap rule the board used) and feather-blends the
patches into the full ``[scene_H, scene_W]`` tile — assembled in host RAM, so the board never holds
the whole f32 canvas. See docs/onboard_pipeline.md §10.

    python scripts/evaluation/stitch_ddc.py --ddc o8.ddc --patches o8_recon.npy --out recon_o8.npy
    #   + optional scoring against the project MERLIN full-tile GT:
    python scripts/evaluation/stitch_ddc.py --ddc o8.ddc --patches o8_recon.npy --out recon_o8.npy \
        --gt data/.../linA_MERLIN_full_Hamburg.npy --metrics recon_o8_metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ddc_format import read_header  # noqa: E402
from src.utils.tiling import blend_patches, make_offsets  # noqa: E402


def stitch(ddc_path: str, patches_path: str):
    """Blend the decoded per-patch stack into the full tile.

    Returns (recon [H,W], header).
    """
    h = read_header(ddc_path)
    overlap = h.patch - h.stride
    if overlap < 0:
        raise ValueError(f"{ddc_path}: stride {h.stride} > patch {h.patch} (corrupt header?)")

    patches = np.load(patches_path)
    if patches.ndim != 3 or patches.shape[1:] != (h.patch, h.patch):
        raise ValueError(f"patches {patches.shape} != (n, {h.patch}, {h.patch})")

    # The header must describe the same grid make_offsets rebuilds — else board/host disagree.
    row_offs = make_offsets(h.scene_H, h.patch, h.stride)
    col_offs = make_offsets(h.scene_W, h.patch, h.stride)
    if (len(row_offs), len(col_offs)) != (h.grid_a, h.grid_r):
        raise ValueError(
            f"grid mismatch: header {h.grid_a}x{h.grid_r} vs make_offsets "
            f"{len(row_offs)}x{len(col_offs)} (patch={h.patch}, stride={h.stride})"
        )
    if patches.shape[0] != h.grid_a * h.grid_r:
        raise ValueError(f"{patches.shape[0]} patches but header grid is {h.grid_a * h.grid_r}")

    print(
        f"[stitch] {h.scene_H}x{h.scene_W}  patch={h.patch} overlap={overlap}  "
        f"grid {h.grid_a}x{h.grid_r} = {patches.shape[0]} patches"
    )
    recon = blend_patches(patches, h.scene_H, h.scene_W, h.patch, overlap)
    return recon, h


def _ssim_map(x: np.ndarray, y: np.ndarray, data_range: float, sigma: float = 1.5) -> np.ndarray:
    """Per-pixel Gaussian-window SSIM map (scipy, O(N) memory).

    The project's torchmetrics SSIM allocates ~80x the image (154 GB on the full scene, OOM); this
    matches it within ~3e-4 on small inputs. Standard Wang SSIM (k1=0.01, k2=0.03, Gaussian
    sigma=1.5). Returns the map so callers can average it over sub-regions (seam / interior).
    """
    from scipy.ndimage import gaussian_filter

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    vx = gaussian_filter(x * x, sigma) - mx2
    vy = gaussian_filter(y * y, sigma) - my2
    vxy = gaussian_filter(x * y, sigma) - mxy
    return ((2 * mxy + c1) * (2 * vxy + c2)) / ((mx2 + my2 + c1) * (vx + vy + c2))


def _seam_mask(shape: tuple, patch: int = 256, band: int = 3) -> np.ndarray:
    """Boolean mask of pixels within ±band of the non-overlap patch-grid interior boundaries —
    where independent-patch seams appear (≈4.6% of pixels at 256-grid ±3).

    The fixed reference grid, so all overlaps are scored at the same locations.
    """
    hgt, wid = shape
    m = np.zeros(shape, dtype=bool)
    for b in make_offsets(hgt, patch, patch)[1:]:
        m[max(0, b - band) : b + band, :] = True
    for b in make_offsets(wid, patch, patch)[1:]:
        m[:, max(0, b - band) : b + band] = True
    return m


def score_arrays(recon: np.ndarray, gt: np.ndarray) -> dict:
    """Coherent metrics of the stitched tile vs an in-memory full-tile GT — all clipped to
    ``AMP_LIN_99`` (consistent with the project PSNR/MSE, and so the INT8 bright-scatterer cap
    doesn't dominate).

    Reports full-tile MSE/PSNR/SSIM/EPD, plus PSNR/SSIM split into the **seam band** (±3px of the
    patch grid, where independent-patch seams live) and the **interior**. Overlap fixes seams,
    whose effect the full-tile mean dilutes ~21x, so the seam split is the sensitive number (§10).
    MS-SSIM omitted at full-scene scale (torchmetrics OOMs). Takes ``gt`` as an array so batch
    callers (rescore_overlap_tiles.py) load the 1.9 GB GT once.
    """
    import math

    import torch

    from src.utils.constants import AMP_LIN_99
    from src.utils.metrics import epd, mse, psnr

    if gt.shape != recon.shape:
        raise ValueError(f"GT {gt.shape} != recon {recon.shape}")
    lim = float(AMP_LIN_99)
    rc = np.minimum(recon.astype(np.float32), lim)  # clip both to AMP_LIN_99, like mse()/psnr()
    gc = np.minimum(gt, lim)
    seam = _seam_mask(gt.shape)
    smap = _ssim_map(rc, gc, lim)
    rt, gtt = torch.from_numpy(rc), torch.from_numpy(gc)

    def _psnr(mask: np.ndarray) -> float:
        m = float(((rc[mask] - gc[mask]) ** 2).mean())
        return 20 * math.log10(lim) - 10 * math.log10(m) if m > 0 else float("inf")

    return {
        "psnr": float(psnr(rt, gtt)),
        "ssim": float(smap.mean()),
        "mse": float(mse(rt, gtt)),
        "epd": float(epd(rt, gtt)),
        "psnr_seam": _psnr(seam),
        "ssim_seam": float(smap[seam].mean()),
        "psnr_interior": _psnr(~seam),
        "ssim_interior": float(smap[~seam].mean()),
        "seam_frac": float(seam.mean()),
    }


def score(recon: np.ndarray, gt_path: str) -> dict:
    """Convenience wrapper: load the GT from ``gt_path`` and score (see ``score_arrays``)."""
    return score_arrays(recon, np.load(gt_path).astype(np.float32))


def main() -> None:
    """CLI: stitch a .ddc + decoded patches into the full tile; optional GT scoring."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ddc", required=True, help=".ddc from stream_pipeline --overlap N")
    ap.add_argument("--patches", required=True, help="[n,P,P] linA from stream_pipeline --decode")
    ap.add_argument("--out", required=True, help="output stitched [H,W] linA .npy")
    ap.add_argument("--gt", help="MERLIN full-tile GT [H,W] linA (optional scoring)")
    ap.add_argument("--metrics", help="write scores JSON here (requires --gt)")
    args = ap.parse_args()

    recon, h = stitch(args.ddc, args.patches)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, recon)
    print(
        f"[stitch] wrote {args.out}  shape={recon.shape}  "
        f"range=[{float(recon.min()):.4g}, {float(recon.max()):.4g}]  "
        f"nan={int(np.isnan(recon).sum())}"
    )

    if args.gt:
        overlap = h.patch - h.stride
        m = score(recon, args.gt)
        m.update({"overlap": overlap, "n_patches": int(h.grid_a * h.grid_r)})
        print(
            "[stitch] vs GT:  "
            + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()
            )
        )
        if args.metrics:
            Path(args.metrics).write_text(json.dumps(m, indent=2))
            print(f"[stitch] wrote {args.metrics}")


if __name__ == "__main__":
    main()
