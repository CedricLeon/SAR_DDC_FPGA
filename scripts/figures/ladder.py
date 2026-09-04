#!/usr/bin/env python3
"""ladder.py — the cumulative optimization ladder seq…+ent, as TWO independent figures.

One rung axis (seq…+ent × 4 archs), one set of ``results/date27/ladder/<arch>/``
loaders, one band/label convention — but two separate floats in the manuscript, so
this script emits two standalone figures rather than one two-panel figure:

* ``--throughput`` → **F2** (Section III): bar height = cumulative speedup (each arch
  ÷ its own ``seq``). Two shaded bands (mt–knee "scheduling", +neon–+ent
  "CPU kernels"); ``seq`` and the knee→+neon boundary are set off by an unshaded gap.
  The first/last rung group is labelled with its absolute patch/s.
* ``--energy``     → **F5** (Section IV): bar height = J/patch (PS+PL INA226,
  cooldown-gated). Same rungs, same bands. Message: parallelism costs power but
  saves energy — mean board power (seq/+ent labels) rises while J/patch falls.

With neither flag both are produced. Merged from ``optimization_ladder.py`` +
``energy_ladder.py`` (P1.2) — they shared every bit of rung/label/band/colour logic.

Sanity check on load (``--throughput``): cumulative seq→+ent must land on
x5.5 / x5.0 / x3.7 / x3.7 for FP / SH / ResFP / ResSH (§4.0 gate review). A mismatch
aborts.

Run:  conda activate DDC_FPGA && python scripts/figures/ladder.py [--throughput|--energy]
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/{optimization_ladder,energy_ladder}.{pdf,png}
"""

from __future__ import annotations

import argparse

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

# Expected cumulative seq→+ent throughput speedup — the §4.0 gate-review anchor.
EXPECTED_CUM = {"FP": 5.5, "SHyp": 5.0, "ResFP": 3.7, "ResSHyp": 3.7}

BW = 0.20  # bar width (4 archs per rung group)
GAP = 0.35  # extra spacing inserted after seq (r0) and after knee (r4)
HALFW = 4 * BW / 2 + 0.06  # rung-group half-width, for band extents


def _positions() -> np.ndarray:
    """X centre of each rung group, with an unshaded gap after seq and after knee."""
    xs, cur = [], 0.0
    for r in RUNGS:
        xs.append(cur)
        cur += 1.0 + (GAP if r in (0, 4) else 0.0)
    return np.array(xs)


def _bands(ax, pos, ytop, *, log=False):
    """Shade the 'scheduling' (mt–knee) and 'CPU kernels' (+neon–+ent) bands."""
    y_label = ytop / 1.12 if log else ytop * 0.985
    for name, lo, hi in (("scheduling", 1, 4), ("CPU kernels", 5, 7)):
        x0, x1 = pos[lo] - HALFW, pos[hi] + HALFW
        ax.axvspan(x0, x1, color="#000000", alpha=0.05, lw=0, zorder=0)
        ax.text(
            (x0 + x1) / 2,
            y_label,
            name,
            ha="center",
            va="top",
            fontsize=6.5,
            color="#808080",
        )


def _draw(series: dict, ylabel: str, *, annotate, head: float = 1.18, log: bool = False) -> tuple:
    """Grouped bar ladder from ``series`` (arch → per-rung values).

    ``annotate(ax, arch, xs, ys)``
    adds the per-arch overlay. ``head`` = top-of-axis headroom factor; ``log`` = log y (bars are
    clipped at the axis bottom rather than drawn from 0). Returns ``(fig, ax, pos)``.
    """
    pos = _positions()
    ymax = max(v.max() for v in series.values())
    ymin = min(v.min() for v in series.values())
    ytop = ymax * head
    ybot = ymin / 1.8 if log else 0.0

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _bands(ax, pos, ytop, log=log)

    for j, a in enumerate(ARCHS):
        xs = pos + (j - 1.5) * BW
        ax.bar(
            xs,
            series[a],
            BW,
            color=ARCH_COLORS[a],
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
            label=DISPLAY[a],
        )
        annotate(ax, a, xs, series[a])

    if log:
        ax.set_yscale("log")
        # gridlines only at the (few) labelled ticks — see do_energy; minor ticks stay
        # as bare marks
        ax.grid(axis="y", which="major", alpha=0.30, lw=0.5, color="#9a9a9a", zorder=0.5)
        ax.tick_params(axis="y", which="minor", length=1.5)
    else:
        ax.grid(axis="y", which="major", alpha=0.28, lw=0.5, color="#9a9a9a", zorder=0.5)

    ax.set_xticks(pos)
    ax.set_xticklabels([RUNG_LABELS[r] for r in RUNGS], fontsize=6.5)
    ax.tick_params(axis="both", length=2)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(ybot, ytop)
    ax.set_xlim(pos[0] - 0.55, pos[-1] + 0.55)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_zorder(4)

    fig.tight_layout()
    return fig, ax, pos


def _power_range_labels(ax, pos, watt, series):
    """Board-power span across the 4 archs — one range label just above the tallest bar of the seq
    and +ent columns (per-arch W in a 0.6-wide group would collide).

    The 'power rises' half of the F5 message; the reader associates it by x-position. The gap is
    multiplicative so it looks right on a log axis.
    """
    log = ax.get_yscale() == "log"
    for r in (0, len(RUNGS) - 1):
        lo, hi = min(watt[a][r] for a in ARCHS), max(watt[a][r] for a in ARCHS)
        coltop = max(series[a][r] for a in ARCHS)
        y = coltop * 1.08 if log else coltop + 0.015 * ax.get_ylim()[1]
        ax.text(
            pos[r],
            y,
            f"{lo:.0f}–{hi:.0f} W",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color="black",
        )


def do_throughput(absolute: bool = False):
    """F2 — the optimization ladder.

    ``absolute`` → bar height = patch/s; otherwise
    cumulative speedup (each arch ÷ its own seq), with seq/+ent patch/s labelled for
    the fastest (FP) and slowest (ResSH) archs.
    """
    absol = {a: np.array(ladder_series(a, "median_patch_s")) for a in ARCHS}
    cum = {a: absol[a][-1] / absol[a][0] for a in ARCHS}

    bad = {a: round(cum[a], 1) for a in ARCHS if round(cum[a], 1) != EXPECTED_CUM[a]}
    if bad:
        raise SystemExit(
            f"cumulative seq→+ent speedup mismatch vs §4.0 gate review:\n"
            f"  got      {{{', '.join(f'{a}:{cum[a]:.2f}' for a in ARCHS)}}}\n"
            f"  expected {EXPECTED_CUM}\n"
            f"  offending: {bad}"
        )

    series = absol if absolute else {a: absol[a] / absol[a][0] for a in ARCHS}

    def annotate(ax, a, xs, ys):
        if absolute or a not in ("FP", "ResSHyp"):
            return
        for r in (0, 7):  # seq/+ent patch/s for the extremes (inner bars collide)
            ax.annotate(
                f"{absol[a][r]:.0f}",
                (xs[r], ys[r]),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                fontsize=6.0,
                color="black",
            )

    ylabel = "throughput  [patch/s]" if absolute else "speedup  [× over seq]"
    fig, ax, _ = _draw(series, ylabel, annotate=annotate, head=1.16)
    if not absolute:
        ax.axhline(1.0, color="#999999", lw=0.6, ls=":", zorder=2)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=1.0,
        fontsize=6.0,
        loc="upper left",
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.4,
        borderaxespad=0.35,
    ).set_zorder(5)
    fig.tight_layout()
    save_figure(fig, "optimization_ladder")

    print(f"F2 throughput ladder ({'absolute patch/s' if absolute else '× over seq'}):")
    for a in ARCHS:
        print(
            f"  {a:8s} "
            + " -> ".join(f"{v:.2f}" for v in series[a])
            + f"   (seq {absol[a][0]:.1f} -> +ent {absol[a][-1]:.1f} p/s, x{cum[a]:.2f})"
        )


def do_energy(absolute: bool = False):
    """F5 — the energy ladder (mirror of F2).

    ``absolute`` → bar height = J/patch;
    otherwise J/patch ÷ its own seq. Board-power range labelled above seq/+ent
    either way (the 'power rises while energy falls' message).
    """
    jpp = {a: np.array(ladder_series(a, "j_per_patch")) for a in ARCHS}
    watt = {a: np.array(ladder_series(a, "avg_power_w")) for a in ARCHS}

    series = jpp if absolute else {a: jpp[a] / jpp[a][0] for a in ARCHS}
    ylabel = "energy per patch  [J]" if absolute else "energy per patch  [× of seq]"

    # Absolute J/patch spans ~15× (0.08–1.15) → log y; the ratio view is <1 decade → linear.
    fig, ax, pos = _draw(
        series, ylabel, annotate=lambda *a: None, head=1.35 if absolute else 1.22, log=absolute
    )
    if absolute:
        # keep every log tick mark; label only round decade/half-decade values
        from matplotlib.ticker import (
            FixedFormatter,
            FixedLocator,
            LogLocator,
            NullFormatter,
        )

        lab = [t for t in (0.05, 0.1, 0.2, 0.5, 1.0) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs="all", numticks=200))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(lab))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in lab]))
        ax.tick_params(axis="y", which="major", labelsize=5.5)
    else:
        ax.axhline(1.0, color="#999999", lw=0.6, ls=":", zorder=2)
    _power_range_labels(ax, pos, watt, series)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=1.0,
        fontsize=6.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),  # its own box, one row, just below the x labels
        ncol=len(ARCHS),
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.4,
    )
    fig.tight_layout()
    save_figure(fig, "energy_ladder")

    print(f"F5 energy ladder ({'absolute J/patch' if absolute else '× of seq'}):")
    for a in ARCHS:
        print(
            f"  {a:8s} "
            + " -> ".join(f"{v:.3f}" for v in series[a])
            + f"   (seq {jpp[a][0]:.3f} -> +ent {jpp[a][-1]:.3f} J,"
            + f" {jpp[a][0] / jpp[a][-1]:.1f}x less;  {watt[a][0]:.1f}->{watt[a][-1]:.1f} W)"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--throughput", action="store_true", help="emit F2 only")
    g.add_argument("--energy", action="store_true", help="emit F5 only")
    m = p.add_mutually_exclusive_group()
    m.add_argument(
        "--absolute",
        action="store_true",
        help="force raw units (patch/s, J/patch) for whatever is drawn",
    )
    m.add_argument(
        "--ratio",
        action="store_true",
        help="force per-arch ratio (× over seq) for whatever is drawn",
    )
    args = p.parse_args()

    # Manuscript defaults: F2 = ratio (the mechanism contribution per rung), F5 =
    # absolute (J/patch — the ~10× cross-arch spread is itself a result). --absolute
    # / --ratio override for exploration.
    if not args.energy:
        do_throughput(absolute=args.absolute)
    if not args.throughput:
        do_energy(absolute=args.absolute or not args.ratio)
