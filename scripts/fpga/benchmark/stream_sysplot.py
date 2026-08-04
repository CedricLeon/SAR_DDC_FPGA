#!/usr/bin/env python3
"""stream_sysplot.py — two system-level views of the ablation (sketches for discussion).

(i) throughput vs the optimization ladder, with the SD-read ceiling + per-model warm lines. (ii) an
OP/byte "system roofline": x = compress DPU OPs per SLC byte (fixed per model), y = achieved DPU
OP/s; roofs = SD read (diagonal) + DPU compute (B4096, 2 cores). FP (low intensity) rides the SD
roof (read-bound); ResSHyp (high intensity) sits near the compute roof (compute-bound); the warm
star breaks the SD roof.

Numbers are hardcoded from the L1000 cold sweep + vaitrace — a sketch, not final.
"""

import argparse

import matplotlib
import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIGS = ["seq", "+s1", "+p0", "+prefetch", "+neon"]
PS = {
    "FP": [26.4, 29.8, 53.3, 85.9, 85.5],
    "ResSHyp": [9.2, 13.8, 17.7, 22.1, 22.8],
}  # patch/s cold
WARM_PS = {"FP": 138.5, "ResSHyp": 22.9}
SLC_B = 256 * 256 * 4  # bytes of int16 SLC per patch
SD_MBS = 23.6  # SD sequential-read ceiling (measured)
OPS = {"FP": 2 * 4.512, "ResSHyp": 2 * 39.755 + 0.275 + 0.176}  # compress DPU GOP/patch (vaitrace)
DPU_PEAK = 4096 * 0.300  # GOP/s per B4096 core @ 300 MHz
COL = {"FP": "#F58518", "ResSHyp": "#4C78A8"}


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "benchmark_stream" / "sysplot.png")
    )
    args = ap.parse_args()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- (i) throughput vs ladder ----
    axL.axhline(SD_MBS, color="#54A24B", ls="--", lw=1.3)
    axL.text(0, SD_MBS + 0.6, f"SD read ceiling {SD_MBS} MB/s", color="#54A24B", fontsize=9)
    for m in ("FP", "ResSHyp"):
        mb = [p * SLC_B / 1e6 for p in PS[m]]
        axL.plot(CONFIGS, mb, "o-", color=COL[m], label=m)
        warm = WARM_PS[m] * SLC_B / 1e6
        axL.axhline(warm, color=COL[m], ls=":", lw=1.0, alpha=0.7)
        axL.text(4.05, warm, f"{m} warm {warm:.0f}", color=COL[m], fontsize=8, va="center")
    axL.set_ylabel("SLC throughput  (MB/s)")
    axL.set_title("(i) throughput vs optimization — cold, full scene, L1000")
    axL.legend(loc="upper left")
    axL.grid(axis="y", ls=":", alpha=0.4)

    # ---- (ii) OP/byte system roofline ----
    xs = np.logspace(4, 6, 100)
    axR.plot(xs, np.minimum(2 * DPU_PEAK, SD_MBS * 1e6 * xs / 1e9), "k-", lw=1.5)
    axR.axhline(2 * DPU_PEAK, color="#333", ls=":", alpha=0.5)
    axR.text(
        1.1e4,
        2 * DPU_PEAK * 1.04,
        f"DPU compute roof (s1, 2 cores) = {2 * DPU_PEAK:.0f} GOP/s",
        fontsize=8,
    )
    axR.text(1.1e4, 300, f"SD read roof ({SD_MBS} MB/s x intensity)", color="#54A24B", fontsize=8)
    for m in ("FP", "ResSHyp"):
        inten = OPS[m] * 1e9 / SLC_B
        for cfg, p in zip(CONFIGS, PS[m]):
            axR.plot(inten, p * OPS[m], "o", color=COL[m], ms=7)
        axR.plot(inten, WARM_PS[m] * OPS[m], "*", color=COL[m], ms=15)
        axR.annotate(
            f"{m}\n({'read' if m == 'FP' else 'compute'}-bound)",
            (inten, PS[m][0] * OPS[m]),
            textcoords="offset points",
            xytext=(6, -30),
            fontsize=8,
            color=COL[m],
        )
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("compress DPU OPs per SLC byte")
    axR.set_ylabel("achieved DPU OP/s  (GOP/s)")
    axR.set_title("(ii) OP/byte roofline — * = warm; cold ladder climbs each stack")
    axR.grid(True, which="both", ls=":", alpha=0.3)
    axL.set_xlabel("optimization")

    fig.tight_layout()
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
