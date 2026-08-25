#!/usr/bin/env python3
"""fanout_lane_plot.py — fan-out lane-scaling figure (throughput / DPU occupancy / energy vs
lanes).

Three stacked panels over a shared log2 lane axis, one line per architecture (distinguished by colour;
uniform circle markers):

* **throughput** — patch/s, mean ± 1σ over the timed iterations, from the warm fan-out result JSONs
  (``results/benchmark_stream/<model>/p0_t{N}_fo_pf_neon_warm.json``).
* **occupancy** — mean per-core DPU busy % (solid, e=median band from e=min/max) plus CPU %usr of the 4
  A53 cores (dashed), both from ``fanout_occupancy``: the direct read of the binding resource — DPU-bound
  archs saturate the 3 cores (~100 %) while CPU-bound ones plateau a third idle with CPU %usr higher.
* **energy** — J/patch, from the same JSONs.

A star marks each arch's **operating point** (throughput peak, §10) on every panel, annotated with that
panel's value. Missing lane counts are skipped.

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/fanout_lane_plot.py
"""

import json

import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fanout_occupancy import cpu_occupancy_series, occupancy_series
from matplotlib.lines import Line2D

STREAM = REPO_ROOT / "results" / "benchmark_stream"
LANES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128]
TICKS = [1, 2, 3, 4, 6, 9, 12, 16, 24, 32, 48, 64, 96, 128]
COLORS = {"FP": "#1f77b4", "SHyp": "#d62728", "ResFP": "#2ca02c", "ResSHyp": "#9467bd"}
MARKERS = {
    a: "o" for a in ("FP", "SHyp", "ResFP", "ResSHyp")
}  # color-only per arch (uniform marker)
OP = {
    "FP": 32,
    "SHyp": 24,
    "ResFP": 6,
    "ResSHyp": 20,
}  # operating point = KNEE (smallest lane within 1% of the warm peak; DATE'27 basis, §5)


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


def _peak(ax, arch, xs, ys, fmt, dy):
    """Star + value label at the arch's operating-point lane, if that lane has data."""
    if not len(xs):
        return
    hit = np.where(xs == OP[arch])[0]
    if not len(hit):
        return
    x, y = OP[arch], ys[hit[0]]
    ax.plot(x, y, marker="*", markersize=12, color=COLORS[arch], mec="black", mew=0.6, zorder=6)
    ax.annotate(
        fmt.format(y),
        (x, y),
        textcoords="offset points",
        xytext=(0, dy),
        ha="center",
        fontsize=7,
        color="black",
        zorder=7,
    )


def main():
    """Entry point."""
    fig, (ax_t, ax_o, ax_e) = plt.subplots(
        3, 1, sharex=True, figsize=(9.2, 9.6), gridspec_kw={"height_ratios": [2.6, 2.0, 1.5]}
    )
    for arch, color in COLORS.items():
        mk = MARKERS[arch]
        xs, mean, std, jp = series(arch)
        if len(xs):
            ax_t.plot(xs, mean, "-", marker=mk, color=color, label=arch, markersize=4)
            ax_t.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18, lw=0)
            ax_e.plot(xs, jp, "-", marker=mk, color=color, markersize=4)
            _peak(ax_t, arch, xs, mean, "{:.0f}", 12)
            _peak(ax_e, arch, xs, jp, "{:.2f}", 11)
        ox, om, olo, ohi = occupancy_series(arch)
        if len(ox):
            ax_o.plot(ox, om, "-", marker=mk, color=color, markersize=4)
            ax_o.fill_between(ox, olo, ohi, color=color, alpha=0.18, lw=0)
            _peak(ax_o, arch, ox, om, "{:.0f}%", 11)
        cx, cu = cpu_occupancy_series(arch)  # CPU %usr (of 4 A53), dashed
        if len(cx):
            ax_o.plot(cx, cu, "--", color=color, lw=1.5, alpha=0.85)
            _peak(ax_o, arch, cx, cu, "{:.0f}%", 11)

    for ax in (ax_t, ax_o, ax_e):
        ax.grid(alpha=0.3, which="both")
    ax_o.axhline(100, color="gray", ls="--", lw=1, alpha=0.7)  # full DPU saturation

    ax_t.set_xscale("log", base=2)
    ax_e.set_xticks(TICKS)
    ax_e.set_xticklabels(TICKS, fontsize=8)
    ax_e.set_xlabel("CPU worker threads, log2 scale")
    ax_t.set_ylabel("throughput [patch/s]")
    ax_o.set_ylabel("occupancy [%]")
    ax_o.set_ylim(0, 116)
    ax_o.legend(
        handles=[
            Line2D([0], [0], color="#555", ls="-", lw=1.8, label="DPU busy"),
            Line2D([0], [0], color="#555", ls="--", lw=1.8, label="CPU %usr"),
        ],
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        handlelength=2.2,
    )
    ax_e.set_ylabel("energy [J/patch]")
    ax_t.set_ylim(top=ax_t.get_ylim()[1] * 1.08)  # headroom so the peak label clears the top

    handles, labels = ax_t.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="*",
            markersize=13,
            markerfacecolor="#555",
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
    )
    labels.append("operating point (knee)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
    )
    out = STREAM / "fanout_lane_scaling_4arch.png"
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"-> {out.relative_to(REPO_ROOT)} (+ .pdf)")


if __name__ == "__main__":
    main()
