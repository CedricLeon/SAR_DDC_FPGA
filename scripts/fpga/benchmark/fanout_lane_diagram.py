#!/usr/bin/env python3
"""fanout_lane_diagram.py — per-lane per-stage timing diagram for the N1 fan-out (3 vs 4 lanes).

Reads the fan-out result JSONs (stream_fanout_sweep.py) and draws, for each arch, stacked per-lane
bars of the per-patch stage budget (normalize + g_a + h_a + h_s + entropy) at 3 and 4 DPU lanes —
so the DPU collision (a lane's g_a / h_a / h_s ballooning when >3 lanes share 3 cores) is visible
next to the throughput it costs. Stage compute is cache-independent, so bars use the COLD run; both
COLD and WARM throughput are annotated (FP's win is warm). Palette = Okabe–Ito (CVD-safe); CPU
stages warm, DPU stages cool, so a DPU collision reads as growing cool segments.

conda activate DDC_FPGA python scripts/fpga/benchmark/fanout_lane_diagram.py --archs ResSHyp,FP
--lanes 3,4
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rootutils
from matplotlib.patches import Patch

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
STREAM_DIR = REPO_ROOT / "results" / "benchmark_stream"

# Pipeline-ordered stages (bottom→top of stack) with Okabe–Ito colours. CPU=warm, DPU=cool.
STAGES = [
    ("norm_ms_patch", "normalize (CPU)", "#E69F00"),  # orange
    ("ga_ms_patch", "g_a (DPU)", "#0072B2"),  # blue
    ("ha_ms_patch", "h_a (DPU)", "#56B4E9"),  # sky blue
    ("hs_ms_patch", "h_s (DPU)", "#009E73"),  # bluish green
    ("entropy_ms_patch", "entropy (CPU)", "#D55E00"),  # vermilion
]


def load(model: str, lanes: int, warm: bool) -> dict:
    """One fan-out result dict (cold/warm) for a model + lane count, or None if absent."""
    tag = "warm" if warm else "cold"
    f = STREAM_DIR / model / f"p0_t{lanes}_fo_pf_neon_{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--archs", default="ResSHyp,FP")
    ap.add_argument("--lam", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lanes", default="3,4", help="lane counts to compare (columns)")
    ap.add_argument("--out", default=str(STREAM_DIR / "fanout_lane_timings.png"))
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    lane_counts = [int(x) for x in args.lanes.split(",")]
    nrows, ncols = len(archs), len(lane_counts)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(
        "DPU fan-out — per-lane per-patch stage budget (cold; 3 vs 4 lanes on 3 cores)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    for r, arch in enumerate(archs):
        model = f"{arch}-relu_s{args.seed}_L{args.lam}_pt"
        # shared y per row so 3-vs-4 heights compare directly
        row_max = 0.0
        clean = None  # 1-lane-equivalent clean per-patch time (from the smallest lane count)
        for lanes in lane_counts:
            d = load(model, lanes, warm=False)
            if not d:
                continue
            tot = [sum(L[k] for k, _, _ in STAGES) for L in d["lanes"] if L["patches"] > 0]
            row_max = max(row_max, max(tot))
            if clean is None:
                clean = min(tot)
        for c, lanes in enumerate(lane_counts):
            ax = axes[r][c]
            cold = load(model, lanes, warm=False)
            warm = load(model, lanes, warm=True)
            if not cold:
                ax.set_visible(False)
                continue
            lanes_data = [L for L in cold["lanes"] if L["patches"] > 0]
            x = list(range(len(lanes_data)))
            bottoms = [0.0] * len(lanes_data)
            for key, _label, color in STAGES:
                vals = [L[key] for L in lanes_data]
                ax.bar(
                    x,
                    vals,
                    bottom=bottoms,
                    color=color,
                    width=0.72,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            # clean-placement reference line (the 3-lane / uncontended per-patch time)
            if clean:
                ax.axhline(clean, color="#555555", lw=1.0, ls="--", zorder=2)
            # per-lane total labels
            for xi, tot in zip(x, bottoms):
                ax.text(
                    xi,
                    tot + row_max * 0.015,
                    f"{tot:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )
            cps = cold["median_patch_s"]
            wps = warm["median_patch_s"] if warm else float("nan")
            ax.set_title(
                f"{arch} · {lanes} lanes\n{cps:.0f} patch/s cold · {wps:.0f} warm", fontsize=10
            )
            ax.set_xticks(x)
            ax.set_xticklabels([f"lane {i}" for i in x], fontsize=8)
            ax.set_ylim(0, row_max * 1.15)
            ax.set_ylabel("ms / patch" if c == 0 else "")
            ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    handles = [Patch(facecolor=col, edgecolor="white", label=lab) for _, lab, col in STAGES]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#555555",
            lw=1.0,
            ls="--",
            label="clean per-patch time (no core contention)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
