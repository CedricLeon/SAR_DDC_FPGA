#!/usr/bin/env python3
"""stream_roofline.py — DPU-kernel roofline (Williams) for the compress/decompress subgraphs.

x = operational intensity = WL / (LdWB + LdFM + StFM)  [OP/byte, from vaitrace]
y = achieved performance   = WL / HW_RT                 [GOP/s, from vaitrace]
roofs: compute = B4096 @ 300 MHz = 4096*0.3 = 1229 GOP/s per core (2x for s1 g_a);
       memory  = PS DDR4-2133 64-bit = 17.06 GB/s (ZCU102 UG1182).
Data are the measured vaitrace --txt_summary means (s0 L1000); a point below the compute roof by
(1 - Effic) is DPU overhead. FP and ResSHyp g_a/g_s are distinct (residual connections in the Res
variant), so both are plotted.
"""

import argparse

import matplotlib
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPU_PEAK = 4096 * 0.300  # GOP/s per B4096 core @ 300 MHz
DDR_BW = 17.06  # GB/s == GOP/s per (OP/byte) on the diagonal
# (model, subgraph, WL_GOP, HW_RT_ms, LdWB_MB, LdFM_MB, StFM_MB) — vaitrace --txt_summary, s0 L1000.
# FP and ResSHyp g_a/g_s differ (residual connections in the Res variant), so no dedupe.
SUBGRAPHS = [
    ("ResSHyp", "g_a", 39.755, 33.508, 3.520, 10.766, 7.531),
    ("ResSHyp", "g_s", 38.132, 32.972, 3.097, 10.859, 7.812),
    ("ResSHyp", "h_a", 0.275, 0.829, 4.688, 0.062, 0.0),
    ("ResSHyp", "h_s", 0.176, 0.517, 3.001, 0.021, 0.082),
    ("FP", "g_a", 4.512, 4.103, 1.175, 2.891, 2.531),
    ("FP", "g_s", 2.888, 3.377, 0.752, 2.734, 2.688),
]
MODEL_COL = {"ResSHyp": "#4C78A8", "FP": "#F58518"}


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "benchmark_stream" / "roofline_dpu.png")
    )
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ridge = DPU_PEAK / DDR_BW
    xs = np.logspace(0, 4, 200)
    ax.plot(xs, np.minimum(DPU_PEAK, DDR_BW * xs), "k-", lw=1.5)
    ax.text(
        1.2,
        DPU_PEAK * 1.06,
        f"B4096@300MHz compute roof = {DPU_PEAK:.0f} GOP/s",
        fontsize=8,
        color="#333333",
    )
    ax.text(
        ridge * 0.9,
        DPU_PEAK * 1.25,
        f"DDR {DDR_BW} GB/s roof\nridge = {ridge:.0f} OP/byte",
        fontsize=8,
        color="#54A24B",
        ha="right",
    )

    seen = set()
    for model, name, wl, rt, ldwb, ldfm, stfm in SUBGRAPHS:
        perf = wl / (rt / 1000.0)
        inten = wl * 1e9 / ((ldwb + ldfm + stfm) * 1e6)
        eff = 100 * perf / DPU_PEAK
        lbl = model if model not in seen else None
        seen.add(model)
        ax.plot(inten, perf, "o", ms=9, color=MODEL_COL[model], label=lbl)
        ax.annotate(
            f"{model} {name}: {eff:.0f}%",
            (inten, perf),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=7,
            color=MODEL_COL[model],
        )
    ax.legend(loc="lower right", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operational intensity  (OP / byte of DDR traffic)")
    ax.set_ylabel("performance  (GOP/s)")
    ax.set_title("DPU roofline — FP + ResSHyp subgraphs (vaitrace-measured, L1000)")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(50, 4000)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
