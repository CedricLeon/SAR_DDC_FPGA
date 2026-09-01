#!/usr/bin/env python3
"""Optimization ladder (DATE'27) — cumulative throughput up the rungs r0–r7.

One panel, all four architectures, the single cumulative ladder the ledger settled
on (``docs/DATE27_paper_plan.md`` §4.0):

    r0 seq · r1 mt · r2 fo3 · r3 fo3p · r4 knee · r5 +neon · r6 +dbuf · r7 +ent

Each rung = the rung before it + one more flag (never ``--s1``). r1–r4 are the
scheduling rungs, r5–r7 the CPU-kernel rungs; the mechanism shows as ResSH's
r2→r3 jump (naive round-robin → pinned subgraph→core placement, ~2.15× at 3 lanes)
and the CPU-bound archs' continued climb through r5–r7 where the DPU-bound ones
have already plateaued. The thin dashed line is the SD cold-read ceiling — the
storage-bound floor every warm rung sits above only because the tile is page-cached.

Values straight from ``results/date27/ladder/<arch>/r{0..7}_*_warm.json``
(``median_patch_s``), via ``_figutils``.

P1.0 migration: the pre-migration figure was two side-by-side panels with
*different* rung sets per binding resource; the date27 campaign measured ONE
cumulative r0–r7 sequence for every arch, so this is now the single panel §4.0
specifies. Bars keep the per-arch palette and value labels; the two shaded bands
and cumulative-× annotations are F2's redesign (batch P1.2), not done here.

Run:  conda activate DDC_FPGA && python scripts/figures/optimization_ladder.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/optimization_ladder.{pdf,png}
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import (
    ARCH_COLORS,
    ARCHS,
    DISPLAY,
    PALETTE,
    RUNG_LABELS,
    RUNGS,
    cold_read_ceiling,
    ladder_series,
    save_figure,
)

series = {a: ladder_series(a) for a in ARCHS}
ceiling = float(np.mean([cold_read_ceiling(a) for a in ARCHS]))  # ~92 patch/s, arch-independent

fig, ax = plt.subplots(figsize=(9.6, 3.6))
x = np.arange(len(RUNGS))
bw = 0.20
ymax = max(max(v) for v in series.values())

for j, a in enumerate(ARCHS):
    ys = series[a]
    xs = x + (j - 1.5) * bw
    bars = ax.bar(
        xs,
        ys,
        bw,
        color=ARCH_COLORS[a],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        label=DISPLAY[a],
    )
    for b, v in zip(bars, ys):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + ymax * 0.012,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="black",
            rotation=90,
            zorder=4,
        )

ax.axhline(ceiling, ls="--", lw=0.7, color=PALETTE["ceiling"], zorder=2)
ax.text(
    len(RUNGS) - 0.5,
    ceiling + ymax * 0.015,
    "SD cold-read ceiling",
    ha="right",
    va="bottom",
    fontsize=7.5,
    color=PALETTE["ceiling"],
)

ax.set_xticks(x)
ax.set_xticklabels([f"r{r}\n{RUNG_LABELS[r]}" for r in RUNGS], fontsize=8.5)
ax.set_ylabel("throughput [patch/s]")
ax.set_ylim(0, ymax * 1.22)
ax.spines[["top", "right"]].set_visible(False)
for s in ax.spines.values():  # bars' zorder=3 would otherwise paint over the spines
    s.set_zorder(4)
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=4)

fig.tight_layout()
save_figure(fig, "optimization_ladder")
print(f"  cold-read ceiling: {ceiling:.1f} patch/s")
for a in ARCHS:
    ys = series[a]
    print(
        f"  {a:8s} "
        + " -> ".join(f"{v:.1f}" for v in ys)
        + f"   (cumulative r0->r7 {ys[-1] / ys[0]:.1f}x)"
    )
