#!/usr/bin/env python3
"""Energy ladder (DATE'27) — J/patch across the same rungs as the throughput ladder.

Companion to ``optimization_ladder.py``: same single cumulative ladder r0–r7, all
four archs, but the bar height is energy per patch (J/patch, MPSoC PS+PL INA226,
cooldown-gated). Message (``docs/onboard_pipeline.md`` §10): parallelism *costs
power but saves energy* — the mean board power (annotated at r0 and r7) rises up
each ladder, yet J/patch falls, because throughput rises faster than draw. The
cross-arch spread at r7 is ~7× (FP ~0.08 vs ResSH ~0.55 J/patch).

Values from ``results/date27/ladder/<arch>/r{0..7}_*_warm.json``
(``j_per_patch``, ``avg_power_w``), via ``_figutils``.

P1.0 migration: was two side-by-side panels with per-binding-resource rung sets;
the date27 campaign measured one cumulative r0–r7 sequence per arch, so this is now
a single panel (matching the throughput ladder). F5 owns the final design decision
(bar ladder vs table).

Run:  conda activate DDC_FPGA && python scripts/figures/energy_ladder.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/energy_ladder.{pdf,png}
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import (
    ARCH_COLORS,
    ARCHS,
    DISPLAY,
    RUNG_LABELS,
    RUNGS,
    ladder_series,
    save_figure,
)

jpp = {a: ladder_series(a, "j_per_patch") for a in ARCHS}
watt = {a: ladder_series(a, "avg_power_w") for a in ARCHS}

fig, ax = plt.subplots(figsize=(9.6, 3.6))
x = np.arange(len(RUNGS))
bw = 0.20
ymax = max(max(v) for v in jpp.values())

for j, a in enumerate(ARCHS):
    ys, ws = jpp[a], watt[a]
    xs = x + (j - 1.5) * bw
    ax.bar(
        xs,
        ys,
        bw,
        color=ARCH_COLORS[a],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        label=DISPLAY[a],
    )
    # annotate mean board power at the first and last rung
    for idx in (0, len(RUNGS) - 1):
        ax.annotate(
            f"{ws[idx]:.0f} W",
            (xs[idx], ys[idx]),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=ARCH_COLORS[a],
            rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels([f"r{r}\n{RUNG_LABELS[r]}" for r in RUNGS], fontsize=8.5)
ax.set_ylabel("energy per patch [J]")
ax.set_ylim(0, ymax * 1.20)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=4)
ax.text(
    0.01,
    0.02,
    "labels = mean board power (W); J/patch falls as power rises",
    transform=ax.transAxes,
    fontsize=6.6,
    color="#888888",
    style="italic",
)

fig.tight_layout()
save_figure(fig, "energy_ladder")
for a in ARCHS:
    ys = jpp[a]
    print(
        f"  {a:8s} "
        + " -> ".join(f"{v:.3f}" for v in ys)
        + f"  ({ys[0] / ys[-1]:.1f}x less energy r0->r7)"
    )
