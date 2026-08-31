#!/usr/bin/env python3
"""jetson_vs_fpga_crop.py — visual verification: noisy / MERLIN GT / FPGA / Jetson reconstructions
of the same crop, side by side. Reuses the project's dB display convention from
scripts/evaluation/compare_recon.py (20*log10, gray cmap, vmin/vmax = 1/99th percentile of the GT
panel) rather than inventing a new one. One-off script for the Jetson N2 baseline visual check
(docs/tmp_jetson_orin_overnight.md) — not part of any pipeline.

python scripts/evaluation/jetson_vs_fpga_crop.py
"""

from __future__ import annotations

import matplotlib
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import get_all_distortion_metrics
from src.utils.tiling import make_offsets

# The crop the FPGA side already has a saved reconstruction for (results/fpga/active_model/results/) —
# reusing it means an apples-to-apples region with zero new FPGA inference needed.
ROW0, ROW1 = 11000, 12024
COL0, COL1 = 8500, 9524
P = 256

RAW_TILE = REPO_ROOT / (
    "data/cache/symstudy/Hamburg_TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_"
    "IMAGE_HH_SRA_strip_004_full_raw.npy"
)
MERLIN_GT_PATCHES = REPO_ROOT / (
    "data/cache/symstudy/Hamburg_TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_"
    "IMAGE_HH_SRA_strip_004_full_merlin_gt.npy"
)
FPGA_RECON = (
    REPO_ROOT / "results/fpga/active_model/results/Hamburg_[11000:12024-8500:9524]_recon_linA.npy"
)
# Pre-decoded ON Orin (same hardware that compressed the .ddc) — decoding this same file locally on
# a different GPU produced inf/garbage in ~1/3 of these patches: FP32 h_s is not bit-reproducible
# across GPU architectures, and GaussianConditional's entropy decode needs every element's scale in
# the exact same gc_scale_table bucket as at encode time or it desyncs. Same root cause the project
# already flagged in onboard_pipeline.md's on-ground-decoder note — see docs/tmp_jetson_orin_overnight.md.
JETSON_CROP_RECON = REPO_ROOT / "results/benchmark_jetson/orin/jetson_crop_recon.npy"
OUT_PNG = REPO_ROOT / "results/benchmark_jetson/orin/jetson_vs_fpga_vs_merlin_crop.png"


def to_db(lin_a: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Linear amplitude -> log-intensity in dB — same convention as compare_recon.py."""
    return 20.0 * np.log10(np.maximum(lin_a, eps))


def stitch_crop_from_patch_stack(
    patches: np.ndarray, row_offs: list[int], col_offs: list[int]
) -> np.ndarray:
    """Stitch just the patches covering [ROW0:ROW1, COL0:COL1] from a row-major [n,P,P] stack
    (overlap=0 grid, gapless — direct placement, no blending needed) and crop exactly."""
    r_idx = [i for i, ro in enumerate(row_offs) if ro < ROW1 and ro + P > ROW0]
    c_idx = [i for i, co in enumerate(col_offs) if co < COL1 and co + P > COL0]
    canvas_r0, canvas_c0 = row_offs[r_idx[0]], col_offs[c_idx[0]]
    canvas_h = row_offs[r_idx[-1]] + P - canvas_r0
    canvas_w = col_offs[c_idx[-1]] + P - canvas_c0
    canvas = np.zeros((canvas_h, canvas_w), np.float32)
    for ri in r_idx:
        for ci in c_idx:
            idx = ri * len(col_offs) + ci
            ro, co = row_offs[ri] - canvas_r0, col_offs[ci] - canvas_c0
            canvas[ro : ro + P, co : co + P] = patches[idx]
    return canvas[ROW0 - canvas_r0 : ROW1 - canvas_r0, COL0 - canvas_c0 : COL1 - canvas_c0]


def main() -> None:
    """Entry point."""
    print("loading Jetson crop (pre-decoded on Orin), MERLIN GT crop, FPGA recon, noisy input...")
    jetson = np.load(JETSON_CROP_RECON).astype(np.float32)
    assert jetson.shape == (
        ROW1 - ROW0,
        COL1 - COL0,
    ), f"Jetson crop shape {jetson.shape} unexpected"
    assert np.isfinite(
        jetson
    ).all(), "non-finite values in the Jetson crop — decode issue, do not plot"

    row_offs = make_offsets(32901, P, P)  # overlap=0 grid used by verify_overlap0.ddc
    col_offs = make_offsets(14686, P, P)
    merlin_patches = np.load(MERLIN_GT_PATCHES, mmap_mode="r")
    merlin = stitch_crop_from_patch_stack(merlin_patches, row_offs, col_offs)

    fpga = np.load(FPGA_RECON).astype(np.float32)
    assert fpga.shape == (ROW1 - ROW0, COL1 - COL0), f"FPGA recon shape {fpga.shape} unexpected"

    raw = np.load(RAW_TILE, mmap_mode="r")[ROW0:ROW1, COL0:COL1, :].astype(np.float32)
    noisy = np.sqrt(0.5 * (raw[..., 0] ** 2 + raw[..., 1] ** 2))  # MERLIN reflectivity convention

    print("\n== distortion metrics (linA, AMP_LIN_99 basis) vs MERLIN GT ==")
    for name, im in [("FPGA (INT8)", fpga), ("Jetson (FP32)", jetson)]:
        m = get_all_distortion_metrics(im, merlin)
        print(
            f"  {name:<16} PSNR={m['psnr']:.2f} dB  SSIM={m['ssim']:.4f}  MS-SSIM={m['ms_ssim']:.4f}"
        )
    m = get_all_distortion_metrics(jetson, fpga)
    print(
        f"  {'Jetson vs FPGA':<16} PSNR={m['psnr']:.2f} dB  SSIM={m['ssim']:.4f}  MS-SSIM={m['ms_ssim']:.4f}"
    )

    print(f"\nplotting -> {OUT_PNG}")
    ref_db = to_db(merlin)
    vmin, vmax = np.percentile(ref_db, 1), np.percentile(ref_db, 99)
    cols = [
        ("Noisy input", noisy),
        ("MERLIN (GT)", merlin),
        ("FPGA · INT8", fpga),
        ("Jetson Orin · FP32", jetson),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (name, im) in zip(axes, cols):
        ax.imshow(to_db(im), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=12)
    fig.suptitle(
        f"Hamburg [{ROW0}:{ROW1}, {COL0}:{COL1}] — SHyp-relu_s0_L20_pt — log-intensity (dB)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print("done.")


if __name__ == "__main__":
    main()
