#!/usr/bin/env python3
"""fanout_lane_plot.py — fan-out lane-scaling figure (throughput / occupancy / energy vs lanes).
Output: ``lane_scaling.{pdf,png}``.

Three stacked panels over a shared log2 lane axis, one line per architecture
(distinguished by colour; uniform circle markers):

* **throughput** — patch/s, mean ± 1σ over the timed iterations, from the warm
  fan-out result JSONs (``results/date27/lanes/<arch>/t{N}_fo_neon_pf_ent_warm.json``,
  entropy-ON for all four archs).
* **occupancy** — mean per-core DPU busy % (solid) plus mean 4-core CPU busy %
  (dashed), both from ``fanout_occupancy`` (re-exported through ``_figutils``): the
  direct read of the binding resource — DPU-bound archs saturate the 3 cores
  (~100 %) while CPU-bound ones plateau a third idle with CPU higher.
* **energy** — J/patch, from the same JSONs.

A star marks each arch's **knee** (§4.0 gate review: FP 12 / SH 21 / ResFP 6 /
ResSH 15) on every panel, annotated with that panel's value. Missing lane counts
are skipped.

P1.0 migration: repointed to results/date27/; the ``sys.path`` hack into
``scripts/fpga/benchmark/`` is gone (``fanout_occupancy`` now lives beside this
file); FP knee 32→12; the 128-lane point and the XRT-wall annotation are out of the
grid; the dashed CPU series is derived from the trace CSVs' ``kind=cpu`` spans, not
the deleted ``cpu_probe/`` mpstat logs. Visual design otherwise unchanged (per-arch
label nudges below are the old hand-tuned values — F4 re-tunes them).

    conda activate DDC_FPGA
    python scripts/figures/fanout_lane_plot.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import (
    ARCH_COLORS,
    DISPLAY,
    KNEE,
    LANE_GRID,
    cpu_occupancy_series,
    lane_counts,
    load_lane,
    occupancy_series,
    save_figure,
)
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

LANES = LANE_GRID
TICKS = LANE_GRID  # tick + label every measured lane count (not just powers of 2)
COLORS = ARCH_COLORS
MARKERS = {a: "o" for a in COLORS}  # color-only per arch (uniform marker)
OP = KNEE  # operating point = the per-arch knee (§4.0)


def series(arch: str):
    """Return (lanes, mean patch/s, std patch/s, J/patch) arrays for the warm fan-out runs."""
    xs, mean, std, jp = [], [], [], []
    for ln in lane_counts(arch):
        j = load_lane(arch, ln)
        totals = j.get("totals_s") or [j["median_total_s"]]
        pit = np.array([j["n_patches"] / t for t in totals])
        xs.append(ln)
        mean.append(pit.mean())
        std.append(pit.std())
        jp.append(j["j_per_patch"])
    return map(np.array, (xs, mean, std, jp))


def _peak(ax, arch, xs, ys, fmt, dy):
    """Star + value label at the arch's knee lane, if that lane has data."""
    if not len(xs):
        return
    hit = np.where(xs == OP[arch])[0]
    if not len(hit):
        return
    x, y = OP[arch], ys[hit[0]]
    ax.plot(x, y, marker="*", markersize=9, color=COLORS[arch], mec="black", mew=0.6, zorder=6)
    ax.annotate(
        fmt.format(y),
        (x, y),
        textcoords="offset points",
        xytext=(0, dy),
        ha="center",
        va="bottom" if dy >= 0 else "top",
        fontsize=9.5,
        color="black",
        zorder=7,
    )


def main():
    """Entry point."""
    fig, (ax_t, ax_o) = plt.subplots(
        2, 1, sharex=True, figsize=(9.2, 7.2), gridspec_kw={"height_ratios": [2.6, 2.0]}
    )
    # Per-arch knee-label offset (points) where the default (+7, above) collides with a
    # neighbour; every arch not listed keeps +7. Hand-tuned for the date27 grid.
    DPU_OCC_DY = {"SHyp": -7, "FP": 7}  # FP DPU busy sits just above its own CPU-busy label
    CPU_OCC_DY = {"ResFP": -7, "FP": -8}  # FP CPU busy label goes below the FP DPU one

    for arch, color in COLORS.items():
        mk = MARKERS[arch]
        xs, mean, std, _jp = series(arch)
        if len(xs):
            ax_t.plot(xs, mean, "-", marker=mk, color=color, label=DISPLAY[arch], markersize=4)
            ax_t.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18, lw=0)
            _peak(ax_t, arch, xs, mean, "{:.0f}", 7)
        ox, om, _, _ = occupancy_series(arch)
        if len(ox):
            ax_o.plot(ox, om, "-", marker=mk, color=color, markersize=4)
            _peak(ax_o, arch, ox, om, "{:.0f}%", DPU_OCC_DY.get(arch, 7))
        cx, cu = cpu_occupancy_series(arch)  # mean 4-core CPU busy %, dashed
        if len(cx):
            ax_o.plot(cx, cu, "--", color=color, lw=1.5, alpha=0.85)
            _peak(ax_o, arch, cx, cu, "{:.0f}%", CPU_OCC_DY.get(arch, 7))

    for ax in (ax_t, ax_o):
        ax.grid(alpha=0.22, which="both")
        # a darker vertical rule at each per-arch knee
        for m in sorted(set(KNEE.values())):
            ax.axvline(m, color="#6f6f6f", lw=0.7, alpha=0.5, zorder=0)
    ax_o.axhline(100, color="gray", ls="--", lw=1, alpha=0.7)  # full DPU saturation

    ax_t.set_xscale("log", base=2)
    ax_o.set_xticks(TICKS)
    ax_o.set_xticklabels(TICKS, fontsize=8)
    ax_o.xaxis.set_minor_locator(plt.NullLocator())  # no extra unlabelled log2 minor ticks
    # nudge the labels of near-touching tick pairs apart (points; +right / -left)
    _LABEL_DX = {15: -2.0, 16: 2.0, 20: -2.0, 21: 2.0, 30: -1.0, 32: 1.0}
    for lab in ax_o.get_xticklabels():
        dx = _LABEL_DX.get(int(lab.get_text()))
        if dx:
            lab.set_transform(
                lab.get_transform() + ScaledTranslation(dx / 72.0, 0, fig.dpi_scale_trans)
            )
    ax_o.set_xlabel("CPU worker threads (log2 scale)", fontsize=11)
    ax_t.set_ylabel("throughput [patch/s]", fontsize=11)
    ax_o.set_ylabel("occupancy [%]", fontsize=11)
    ax_o.set_ylim(0, 116)
    ax_o.legend(
        handles=[
            Line2D([0], [0], color="#555", ls="-", lw=1.8, label="DPU busy"),
            Line2D([0], [0], color="#555", ls="--", lw=1.8, label="CPU busy"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.83),  # top of box just under the 100% line, no overlap
        borderaxespad=0,
        fontsize=10,
        framealpha=0.9,
        handlelength=2.2,
    )
    ax_t.set_ylim(top=ax_t.get_ylim()[1] * 1.08)  # headroom so the peak label clears the top

    handles, labels = ax_t.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="*",
            markersize=10,
            markerfacecolor="#555",
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
    )
    labels.append("knee")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    # keep the pre-migration save behaviour (png dpi 140, no tight bbox — the bottom
    # fig.legend is placed manually and a tight crop would clip it)
    save_figure(fig, "lane_scaling", dpi=140, bbox_inches=None)


if __name__ == "__main__":
    main()
