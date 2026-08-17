#!/usr/bin/env python3
"""fanout_lane_plot.py — fan-out lane-scaling figure (throughput + energy vs lane count, per arch).

Reads the warm fan-out result JSONs
(``results/benchmark_stream/<model>/p0_t{N}_fo_pf_neon_warm.json``) and plots, per architecture,
throughput (patch/s, mean ± 1σ over the timed iterations) on the left axis and energy (J/patch) on the
right, over a log2 lane axis. Legend = the four arch colours; the solid/dashed metric encoding is in
the suptitle. Missing lane counts are skipped (e.g. only FP/SHyp go past 32).

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/fanout_lane_plot.py
"""

import json

import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STREAM = REPO_ROOT / "results" / "benchmark_stream"
LANES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 64, 128]
COLORS = {"FP": "#1f77b4", "SHyp": "#d62728", "ResFP": "#2ca02c", "ResSHyp": "#9467bd"}


def series(arch: str, seed: int = 0, lam: int = 20):
    """Return (lanes, mean patch/s, std patch/s, J/patch) arrays for the warm fan-out runs."""
    xs, mean, std, jp = [], [], [], []
    for ln in LANES:
        f = STREAM / f"{arch}-relu_s{seed}_L{lam}_pt" / f"p0_t{ln}_fo_pf_neon_warm.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        totals = j.get("totals_s") or [j["median_total_s"]]
        pit = np.array([j["n_patches"] / t for t in totals])
        xs.append(ln)
        mean.append(pit.mean())
        std.append(pit.std())
        jp.append(j["j_per_patch"])
    return map(np.array, (xs, mean, std, jp))


def main():
    """Entry point."""
    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    ax2 = ax1.twinx()
    for arch, color in COLORS.items():
        xs, mean, std, jp = series(arch)
        if len(xs) == 0:
            continue
        ax1.plot(xs, mean, "-o", color=color, label=arch, markersize=4)
        ax1.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18, lw=0)
        ax2.plot(xs, jp, "--", color=color, alpha=0.45, lw=1.2)
    for x in (3, 4):  # 3 DPU cores, 4 A53 cores
        ax1.axvline(x, color="gray", ls=":", lw=1)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(LANES)
    ax1.set_xticklabels(LANES, fontsize=8)
    ax1.set_xlabel("fan-out lanes (= worker threads · log2)")
    ax1.set_ylabel("throughput (patch/s)")
    ax2.set_ylabel("energy (J/patch)")
    fig.suptitle(
        "Fan-out lane scaling — warm, λ=20  ·  solid = patch/s (L, mean±1σ) · dashed = J/patch (R)",
        fontsize=10.5,
    )
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(title="arch", loc="center left", fontsize=9)
    fig.tight_layout()
    out = STREAM / "fanout_lane_scaling_4arch.png"
    fig.savefig(out, dpi=140)
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
