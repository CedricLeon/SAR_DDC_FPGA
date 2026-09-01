#!/usr/bin/env python3
"""F1 — per-patch sequential breakdown (DATE'27): the bottleneck migrates with topology.

s0 serial per-patch decomposition, absolute ms, all four architectures. Two
independent topology axes are on display:
  - residual axis  : g_a 10 -> 73 ms  (FP->ResFP, SHyp->ResSHyp)
  - hyperprior axis : adds the h_a/h_s side-network (DPU) + a heavier Gaussian-
                      conditional entropy stage (gc_compress ~11 ms vs eb ~6.6 ms)

Design (F1, P1.1): slices are coloured strictly by the compute resource that runs
them -- CPU (blue) vs DPU (orange), the palette every DATE'27 figure shares
(_figutils.PALETTE). The stack is resource-grouped, not temporal: both CPU stages
(normalize, entropy) at the bottom, both DPU stages (g_a, the h_a/h_s side-network)
on top, so the blue/orange boundary in each bar *is* the CPU/DPU split and the
orange fraction is the DPU share -- printed on every bar. FP/SH read blue-dominant
(CPU-bound), ResFP/ResSH orange-dominant (DPU-bound); the two pairs are set apart
on the x-axis and bracket-labelled. The reader should reach "the binding resource
follows the topology" from the picture alone.

Numbers (lambda=20, entropy-opt OFF -- the true sequential baseline; the ladder's
r7 is where the entropy optimisation's payoff shows). Per-stage compute is
lambda-independent.
  stages  <- results/date27/s0/<arch>/s0_compress_entoff.json   (E3, via _figutils)

Run:  conda activate DDC_FPGA && python scripts/figures/stacked_time.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/stacked_time.{pdf,png}
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import (
    ARCHS,
    CPU_BOUND,
    DISPLAY,
    PALETTE,
    PALETTE_FILL,
    blend,
    load_s0_stages,
    save_figure,
)
from matplotlib.patches import Patch

# category -> (s0 stage keys summed into it, resource, legend label). compress scenario only.
# Draw order is the list order, bottom -> top: CPU stages first, then DPU stages, so the
# blue|orange seam in every bar is exactly the CPU/DPU boundary.
CATS = [
    ("normalize", ["normalize"], "cpu", "normalize (CPU)"),
    ("entropy", ["eb_compress", "gc_compress"], "cpu", "entropy rANS (CPU)"),
    ("g_a", ["g_a"], "dpu", r"$g_a$(real)$+g_a$(imag) (DPU)"),
    ("hyperprior", ["h_a", "h_s"], "dpu", r"$h_a{+}h_s$ side-network (DPU)"),
]
RES = {c: res for c, _, res, _ in CATS}
# within a resource the primary stage takes the saturated tone, the secondary a lighter
# tint of the SAME hue (fill blended a third of the way back to the saturated tone -- the
# bare fill is too near white to read) -- so each bar still shows one CPU + one DPU region.
FACE = {
    "normalize": PALETTE["cpu"],
    "entropy": blend(PALETTE_FILL["cpu"], PALETTE["cpu"], 0.33),
    "g_a": PALETTE["dpu"],
    "hyperprior": blend(PALETTE_FILL["dpu"], PALETTE["dpu"], 0.33),
}


def load_arch(arch):
    """(per-category ms/patch, DPU fraction of the per-patch total, total ms) for ``arch``."""
    st = load_s0_stages(arch)
    vals = {c: sum(st[k]["mean_ms"] for k in keys if k in st) for c, keys, _, _ in CATS}
    total = sum(vals.values())
    dpu = sum(v for c, v in vals.items() if RES[c] == "dpu")
    return vals, dpu / total, total


data = {a: load_arch(a) for a in ARCHS}

# two visually separated pairs on x: FP,SH | ResFP,ResSH
xpos = {a: (i if a in CPU_BOUND else i + 0.6) for i, a in enumerate(ARCHS)}
BW = 0.62

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for a in ARCHS:
    vals, dpu_share, total = data[a]
    bottom = 0.0
    for cat, _, _, _ in CATS:
        v = vals[cat]
        if v <= 0:
            continue
        ax.bar(
            xpos[a],
            v,
            BW,
            bottom=bottom,
            color=FACE[cat],
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
        )
        if v >= 6.0:
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
    # CPU|DPU seam = sum of the CPU stages: solid tick across the bar so the DPU-share
    # headline has an unambiguous referent.
    cpu_ms = sum(vals[c] for c in vals if RES[c] == "cpu")
    ax.plot(
        [xpos[a] - BW / 2, xpos[a] + BW / 2], [cpu_ms, cpu_ms], color="#1a1a1a", lw=1.2, zorder=4
    )
    # headline per bar: DPU share (bold, DPU colour) over the absolute total (small, grey)
    ax.text(
        xpos[a],
        total + 6.0,
        f"DPU {dpu_share * 100:.0f}%",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["dpu"],
    )
    ax.text(
        xpos[a],
        total + 1.5,
        f"{total:.0f} ms",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )

# bracket + label under each pair
ax.set_ylim(0, 116)
DPU_BOUND = [a for a in ARCHS if a not in CPU_BOUND]
for members, label in ((CPU_BOUND, "CPU-bound"), (DPU_BOUND, "DPU-bound")):
    x0, x1 = xpos[members[0]], xpos[members[-1]]
    ax.plot([x0, x1], [-13, -13], color="#555555", lw=1.2, clip_on=False)
    for xe in (x0, x1):
        ax.plot([xe, xe], [-13, -10.5], color="#555555", lw=1.2, clip_on=False)
    ax.text(
        (x0 + x1) / 2,
        -17,
        label,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#333333",
        clip_on=False,
    )

ax.set_xticks([xpos[a] for a in ARCHS])
ax.set_xticklabels([DISPLAY[a] for a in ARCHS], fontsize=10)
ax.set_ylabel("per-patch latency [ms]")
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.tick_params(axis="x", length=0)

handles = [Patch(facecolor=FACE[c], edgecolor="white", label=lab) for c, _, _, lab in CATS]
ax.legend(
    handles=handles,
    fontsize=8.5,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    handlelength=1.2,
    labelspacing=0.35,
)

save_figure(fig, "stacked_time")
for a in ARCHS:
    vals, dpu_share, total = data[a]
    print(
        f"  {a:8s}: "
        + ", ".join(f"{k}={v:.1f}" for k, v in vals.items())
        + f"  total={total:.1f} ms  DPU={dpu_share * 100:.1f}%"
    )
