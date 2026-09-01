#!/usr/bin/env python3
"""fp_cpu_stack.py — FP-only A53 CPU occupancy vs fan-out lanes (DATE'27).

Stacked area over the same log2 lane axis as fig:lane: %busy -> +%idle = 100 % of
the 4 A53 cores. "%busy" is the mean fraction of the four cores inside a
``kind=cpu`` trace span over the steady-state window (``fanout_occupancy`` /
``_figutils.cpu_occupancy_series``).

P1.0 migration: the old ``results/benchmark_stream/cpu_probe/mpL_<L>.log`` mpstat
source was deleted in P0.C. Rebuilt from the E2 trace CSVs. **The %usr / %sys split
is gone** — a trace marks a core busy, not *why* — so this is now a 2-band
busy/idle stack, and the colours come from the shared palette (CPU blue).
F6 is a candidate cut; if kept, its redesign decides whether it earns its space
next to the occupancy panel of fig:lane, which now carries the same series.

Run:  conda activate DDC_FPGA && python scripts/figures/fp_cpu_stack.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/fp_cpu_stack.{pdf,png}
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import PALETTE, blend, cpu_occupancy_series, save_figure

TICKS = [1, 2, 4, 8, 16, 32, 64]  # powers of 2 within the grid -> evenly spaced on log2

C_BUSY, C_IDLE = PALETTE["cpu"], "#D3D3D3"
C_BUSY_DOT = blend(PALETTE["cpu"], "#000000", 0.35)  # darker same-hue boundary marker

xs, busy = cpu_occupancy_series("FP")
idle = 100.0 - busy

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.stackplot(
    xs, busy, idle, colors=[C_BUSY, C_IDLE], zorder=1, labels=["%busy (CPU work)", "%idle"]
)
ax.plot(xs, busy, "-o", color=C_BUSY_DOT, markersize=3, lw=1, zorder=3)

# The stack is opaque everywhere, so the grid must sit on top (not below, matplotlib's
# default) to be visible at all -- white reads against both fills.
ax.grid(alpha=0.3, which="both", zorder=2, color="white")
ax.set_xscale("log", base=2)
ax.set_xticks(TICKS)
ax.set_xticklabels(TICKS, fontsize=8)
ax.set_xlabel("CPU worker threads (log2 scale)")
ax.set_ylabel("A53 cores occupancy [%]")
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(0, 100)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=7.2, frameon=True)

save_figure(fig, "fp_cpu_stack")
print("  FP %busy: " + " ".join(f"{L}:{b:.0f}" for L, b in zip(xs, busy)))
