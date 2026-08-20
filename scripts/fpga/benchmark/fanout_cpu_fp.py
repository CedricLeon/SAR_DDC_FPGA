#!/usr/bin/env python3
"""fanout_cpu_fp.py — FP CPU-binding figure: why the CPU-bound arch roofs below its DPU.

Three stacked panels vs fan-out lanes (log2), FP only:

* **throughput** (patch/s) — climbs, plateaus ~200, dips past 64L.
* **occupancy** — CPU compute (`%usr` of 4 A53 cores) rises to ~79 % (the binding resource), DPU
  (mean of 3 cores, from ``fanout_occupancy``) plateaus ~66 % (never the limiter), and CPU **idle**
  bottoms ~17 % and holds — the balanced-pipeline signature, not a thread shortage.
* **per-patch CPU cost** — user-CPU/patch inflates from the clean-seq 12.1 ms toward ~16 ms as more
  threads contend (cache/DDR); this is what makes >64 lanes counter-productive.

CPU data: ``results/benchmark_stream/cpu_probe/{mpL,spL}_<L>.log`` (mpstat + the run's patch/s).

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/fanout_cpu_fp.py
"""

import re

import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fanout_occupancy import occupancy_series

CPU = REPO_ROOT / "results" / "benchmark_stream" / "cpu_probe"
LANES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128]
TICKS = [1, 2, 3, 4, 6, 9, 12, 16, 24, 32, 48, 64, 96, 128]
SEQ_CPU_MS = 12.1  # clean single-thread user-CPU per patch (no contention)
C_CPU, C_DPU, C_IDLE, C_TPUT, C_COST = "#d62728", "#2ca02c", "#7f7f7f", "#1f77b4", "#9467bd"


def _steady(fn):
    """Return the mean (usr, sys, idle) % over the steady-state portion of an mpstat log."""
    a = []
    for line in open(fn):
        p = line.split()
        if len(p) >= 12 and re.match(r"\d\d:\d\d:\d\d", p[0]) and p[1] == "all":
            a.append((float(p[2]), float(p[4]), float(p[11])))  # usr, sys, idle
    w = a[5:-3] if len(a) > 10 else a
    n = len(w)
    return sum(x[0] for x in w) / n, sum(x[1] for x in w) / n, sum(x[2] for x in w) / n


def _tput(L):
    """Return the patch/s from the single-thread run's log."""
    for line in open(CPU / f"spL_{L}.log"):
        m = re.search(r"([\d.]+) patch/s", line)
        if m:
            return float(m.group(1))


def cpu_series():
    """Return (lanes, patch/s, %usr, %idle, ms/patch) arrays for the CPU probe runs."""
    xs, tput, usr, idle, cost = [], [], [], [], []
    for L in LANES:
        f = CPU / f"mpL_{L}.log"
        if not f.exists():
            continue
        u, _s, i = _steady(f)
        t = _tput(L)
        xs.append(L)
        tput.append(t)
        usr.append(u)
        idle.append(i)
        cost.append(u / 100 * 4 * 1000 / t)
    return map(np.array, (xs, tput, usr, idle, cost))


def main():
    """Entry point."""
    xs, tput, usr, idle, cost = cpu_series()
    dx, dmean, dlo, dhi = occupancy_series("FP")

    fig, (ax_t, ax_o, ax_c) = plt.subplots(
        3, 1, sharex=True, figsize=(8.8, 9.0), gridspec_kw={"height_ratios": [2.0, 2.4, 1.8]}
    )

    ax_t.plot(xs, tput, "-o", color=C_TPUT, markersize=4)
    ax_t.set_ylabel("throughput\n[patch/s]")

    ax_o.plot(xs, usr, "-o", color=C_CPU, markersize=4, label="CPU compute (%usr, 4 A53)")
    ax_o.plot(dx, dmean, "-^", color=C_DPU, markersize=4, label="DPU busy (3 cores)")
    ax_o.fill_between(dx, dlo, dhi, color=C_DPU, alpha=0.15, lw=0)
    ax_o.plot(xs, idle, "-o", color=C_IDLE, markersize=3, label="CPU idle")
    ax_o.axhline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax_o.set_ylabel("occupancy [%]")
    ax_o.set_ylim(0, 122)
    ax_o.legend(fontsize=8, loc="upper center", ncol=3, frameon=True, columnspacing=1.2)

    ax_c.plot(xs, cost, "-o", color=C_COST, markersize=4, label="user-CPU / patch")
    ax_c.axhline(SEQ_CPU_MS, color="gray", ls="--", lw=1, alpha=0.8)
    ax_c.annotate(
        "clean seq 12.1 ms",
        (xs[0], SEQ_CPU_MS),
        textcoords="offset points",
        xytext=(2, 3),
        fontsize=7.5,
        color="gray",
    )
    ax_c.set_ylabel("CPU cost\n[ms/patch]")

    for ax in (ax_t, ax_o, ax_c):
        ax.grid(alpha=0.3, which="both")
    ax_t.set_xscale("log", base=2)
    ax_c.set_xticks(TICKS)
    ax_c.set_xticklabels(TICKS, fontsize=8)
    ax_c.set_xlabel("CPU worker threads, log2 scale")
    fig.suptitle(
        "FP is CPU-bound: cores fill to ~79% (DPU only ~66%), ~17% idle holds, cost inflates",
        fontsize=10.5,
    )
    fig.tight_layout()
    out = REPO_ROOT / "results" / "benchmark_stream" / "fp_cpu_binding.png"
    fig.savefig(out, dpi=140)
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
