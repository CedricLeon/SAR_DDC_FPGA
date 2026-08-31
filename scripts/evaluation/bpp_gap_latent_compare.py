#!/usr/bin/env python3
"""bpp_gap_latent_compare.py — INT8 (FPGA) vs FP32 (Jetson) pre-entropy-coding latent comparison.

Investigates the ~3.9% bpp gap between the FPGA (INT8) and ddc-edge (FP32) production runs
(docs/onboard_pipeline.md §12) by comparing the actual y/z latent VALUES the two pipelines hand to
the entropy coder, for the identical 16 patches (a 4x4 non-overlapping grid over the same
1024x1024 crop, r11000:12024, c8500:9524 — same crop already used for the visual comparison).

FPGA side: dumped on the real ZCU102 board via the new `dump_latents` tool
(inference_cpp/src/tools/dump_latents.cpp) — real DPU inference, not a simulation. See
docs/tmp_jetson_orin_overnight.md for why a host-side fake-quantization simulation was rejected in
favor of this more rigorous (but more work) route.

    python scripts/evaluation/bpp_gap_latent_compare.py
"""

from __future__ import annotations

import matplotlib
import numpy as np
import rootutils
import torch

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ddc_edge.model_loading import (
    load_model_from_checkpoint,
    resolve_checkpoint_from_model_dir,
)
from ddc_edge.pipeline import normalize_input

FPGA_DUMP_DIR = REPO_ROOT / "results/benchmark_jetson/bpp_investigation/fpga_latents"
CROP_TILE = REPO_ROOT / (
    "data/cache/symstudy/Hamburg_TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_"
    "IMAGE_HH_SRA_strip_004_r11000c8500h1024w1024_raw.npy"
)
MODEL_DIR = (
    REPO_ROOT / "results/fpga/active_model"
)  # local repo convention (Orin's copy is renamed, this isn't)
OUT_PNG = (
    REPO_ROOT / "results/benchmark_jetson/bpp_investigation/int8_vs_fp32_latent_histograms.png"
)

P = 256
OFFSETS = [
    0,
    256,
    512,
    768,
]  # local to the 1024x1024 crop — matches the dump_latents --patches call


def jetson_latents(net: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run the FP32 forward pass (g_a, h_a) for the same 16 patches, return flat (y, z) arrays."""
    tile = np.load(CROP_TILE).astype(np.float32)  # [1024,1024,2], same file used on Jetson/FPGA
    ys, zs = [], []
    with torch.inference_mode():
        for r in OFFSETS:
            for c in OFFSETS:
                patch = tile[r : r + P, c : c + P, :]
                x = (
                    torch.from_numpy(np.ascontiguousarray(patch))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                x = normalize_input(x)
                y_real = net.g_a(x[:, :1])
                y_imag = net.g_a(x[:, 1:])
                y = torch.cat((y_real, y_imag), dim=1)
                z = net.h_a(torch.abs(y))
                ys.append(y.flatten().cpu().numpy())
                zs.append(z.flatten().cpu().numpy())
    return np.concatenate(ys), np.concatenate(zs)


def fpga_latents() -> tuple[np.ndarray, np.ndarray]:
    """Load the board-dumped y/z .npy files for the same 16 patches, return flat arrays."""
    ys, zs = [], []
    for r in OFFSETS:
        for c in OFFSETS:
            ys.append(np.load(FPGA_DUMP_DIR / f"y_r{r}_c{c}.npy").astype(np.float32))
            zs.append(np.load(FPGA_DUMP_DIR / f"z_r{r}_c{c}.npy").astype(np.float32))
    return np.concatenate(ys), np.concatenate(zs)


def describe(name: str, arr: np.ndarray) -> None:
    """Print a summary of the given array's statistics, including mean, std, and unique rounded
    symbols."""
    rounded = np.round(arr)
    print(
        f"  {name:<22} n={arr.size:>7}  mean={arr.mean():+8.4f}  std={arr.std():7.4f}  "
        f"|mean_round|={np.abs(rounded).mean():7.4f}  unique_symbols={np.unique(rounded).size}"
    )


def symbol_entropy_bits(arr: np.ndarray) -> float:
    """Empirical entropy (bits/symbol) of the rounded values — a direct, model-free proxy for what
    the bpp gap would trace back to if the *distribution* (not the entropy coder) differs."""
    vals, counts = np.unique(np.round(arr), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    """Entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_path, manifest = resolve_checkpoint_from_model_dir(MODEL_DIR, REPO_ROOT)
    print(f"model: {manifest['model_name']} ({ckpt_path})")
    net = load_model_from_checkpoint(ckpt_path).to(device)

    print("running FP32 forward pass on Jetson-side code path (local GPU)...")
    y_fp32, z_fp32 = jetson_latents(net, device)
    print("loading FPGA (INT8, real board) dumped latents...")
    y_int8, z_int8 = fpga_latents()

    print(f"\n== y latent (n={y_fp32.size} FP32 vs {y_int8.size} INT8-DPU) ==")
    describe("FP32 (Jetson)", y_fp32)
    describe("INT8 (FPGA)", y_int8)
    print(
        f"  entropy(round(y)) FP32={symbol_entropy_bits(y_fp32):.4f} bits  "
        f"INT8={symbol_entropy_bits(y_int8):.4f} bits  "
        f"(FPGA bpp is measured HIGHER — if entropy(INT8) > entropy(FP32) here too, that's "
        f"consistent evidence, not proof, that quantization noise explains the gap)"
    )

    print(f"\n== z latent (n={z_fp32.size} FP32 vs {z_int8.size} INT8-DPU) ==")
    describe("FP32 (Jetson)", z_fp32)
    describe("INT8 (FPGA)", z_int8)
    print(
        f"  entropy(round(z)) FP32={symbol_entropy_bits(z_fp32):.4f} bits  "
        f"INT8={symbol_entropy_bits(z_int8):.4f} bits"
    )

    # ---- plot: raw-value histograms (top row) + rounded-symbol histograms (bottom row) ----
    color_fp32 = "#3B7DD8"  # categorical pair, fixed assignment (not cycled)
    color_int8 = "#D8763B"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for ax, (name, fp32, int8) in zip(
        axes[0], [("y (raw)", y_fp32, y_int8), ("z (raw)", z_fp32, z_int8)]
    ):
        lo, hi = np.percentile(np.concatenate([fp32, int8]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 120)
        ax.hist(fp32, bins=bins, color=color_fp32, alpha=0.6, label="Jetson (FP32)", density=True)
        ax.hist(int8, bins=bins, color=color_int8, alpha=0.6, label="FPGA (INT8)", density=True)
        ax.set_title(name, fontsize=11)
        ax.set_yticks([])
        ax.legend(fontsize=8, frameon=False)

    for ax, (name, fp32, int8) in zip(
        axes[1],
        [("y (rounded to symbol)", y_fp32, y_int8), ("z (rounded to symbol)", z_fp32, z_int8)],
    ):
        r_fp32, r_int8 = np.round(fp32), np.round(int8)
        lo, hi = np.percentile(np.concatenate([r_fp32, r_int8]), [0.5, 99.5])
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        ax.hist(
            r_fp32, bins=bins, color=color_fp32, alpha=0.6, label="Jetson (FP32)", density=True
        )
        ax.hist(r_int8, bins=bins, color=color_int8, alpha=0.6, label="FPGA (INT8)", density=True)
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "SHyp-relu_s0_L20_pt — pre-entropy-coding latents, same 16 patches\n"
        "(Hamburg [11000:12024, 8500:9524], 4x4 grid) — FPGA dump is real DPU inference, not simulated",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
