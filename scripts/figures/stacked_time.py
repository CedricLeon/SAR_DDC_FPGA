#!/usr/bin/env python3
"""Stacked time-per-patch figure (DATE'27) — makes the bottleneck migration visible.

s0 serial per-patch decomposition, absolute ms, for ALL FOUR architectures
(FP, SHyp, ResFP, ResSHyp) so the two independent axes are disentangled:
  - residual axis  : g_a 10 -> 73 ms  (FP->ResFP, SHyp->ResSHyp)
  - hyperprior axis: adds h_a/h_s (DPU) + a heavier Gaussian-conditional entropy
                     (SHyp/ResSHyp use gc_compress ~11 ms vs FP/ResFP eb ~6.6 ms)

Slices are coloured by compute resource (blue = ARM CPU, orange = DPU); the DPU
block is split g_a vs hyperprior side-network. Message: FP/SHyp bars are balanced
(normalize + entropy roughly match a small g_a) -> CPU-bound; ResFP/ResSHyp are
dominated by the residual g_a block -> DPU-bound. The binding resource migrates
with topology; the roofline (separate figure) then shows g_a is the decider.

Numbers are read from the repo so the figure is reproducible:
  stages  <- results/date27/s0/<arch>/s0_compress_entoff.json  (E3, entropy OFF — the
             true sequential baseline; L20; per-stage compute is lambda-independent)

P1.0 migration: repointed to results/date27/; the SD-read segment is dropped
(F1 — the warm-read basis is the only one now, and s0/ carries no read stage).
Visual design otherwise unchanged.

Run:  conda activate DDC_FPGA && python scripts/figures/stacked_time.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/stacked_time.{pdf,png}
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import (
    ARCHS,
    DISPLAY,
    PALETTE,
    PALETTE_FILL,
    blend,
    load_s0_stages,
    save_figure,
)

# stage (category) -> which s0 stage keys sum into it; compress scenario only
CAT_STAGES = {
    "normalize": ["normalize"],
    "g_a": ["g_a"],  # DPU main encoder, run x2 (re,im)
    "hyperprior": ["h_a", "h_s"],  # DPU side-network (0 for FP/ResFP)
    "entropy": ["eb_compress", "gc_compress"],  # CPU entropy (factorized EB / Gaussian GC)
}

# draw order bottom->top (temporal), colour, legend label. normalize/g_a are their
# loc_cpu/loc_dpu fill blended toward the edge; entropy/hyperprior use the edge directly.
# (Same fill/edge pairs as fig:system_dataflow — see _figutils.PALETTE / .PALETTE_FILL.)
ORDER = [
    ("normalize", blend(PALETTE_FILL["cpu"], PALETTE["cpu"]), "normalize (CPU)"),
    ("g_a", blend(PALETTE_FILL["dpu"], PALETTE["dpu"]), r"$g_a$(real), $g_a$(imag) (DPU)"),
    ("hyperprior", PALETTE["dpu"], r"$h_a, h_s$ (DPU)"),
    ("entropy", PALETTE["cpu"], "entropy rANS (CPU)"),
]


def load_arch(arch):
    """Per-category ms/patch for ``arch`` (each CAT_STAGES group summed over its s0 stages)."""
    st = load_s0_stages(arch)
    return {
        cat: sum(st[k]["mean_ms"] for k in keys if k in st) for cat, keys in CAT_STAGES.items()
    }


data = {a: load_arch(a) for a in ARCHS}

fig, ax = plt.subplots(figsize=(6.4, 3.6))
xpos = {a: i for i, a in enumerate(ARCHS)}
BW = 0.66
for a in ARCHS:
    bottom = 0.0
    for cat, color, _ in ORDER:
        v = data[a][cat]
        if v <= 0:
            continue
        ax.bar(
            xpos[a], v, BW, bottom=bottom, color=color, edgecolor="white", linewidth=0.8, zorder=2
        )
        if v >= 5.0:  # label only segments with room
            ax.text(
                xpos[a],
                bottom + v / 2,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#1a1a1a",
                zorder=3,
            )
        bottom += v
    ax.text(xpos[a], bottom + 2.0, f"{bottom:.0f} ms", ha="center", va="bottom", fontsize=9.5)

ax.set_xticks(list(xpos.values()))
ax.set_xticklabels([DISPLAY[a] for a in ARCHS], fontsize=9.5)
ax.set_ylabel("latency [ms]")
ax.set_ylim(0, 105)  # hand-tuned headroom above the tallest bar (ResSHyp ~95 ms, no read)
ax.spines[["top", "right"]].set_visible(False)

# legend top->bottom of stack
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white") for _, c, _ in reversed(ORDER)
]
labels = [lab for _, _, lab in reversed(ORDER)]
ax.legend(
    handles,
    labels,
    fontsize=8.5,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    handlelength=1.1,
)

save_figure(fig, "stacked_time")
for a in ARCHS:
    print(
        f"  {a:8s}: "
        + ", ".join(f"{k}={v:.1f}" for k, v in data[a].items())
        + f"  total={sum(data[a].values()):.1f} ms"
    )
