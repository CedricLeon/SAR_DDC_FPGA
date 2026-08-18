#!/usr/bin/env python3
"""fanout_occupancy.py — per-core DPU occupancy from the fan-out ``--trace`` CSVs (offline, no board).

For each ``results/benchmark_stream/traces/<arch>_<L>lane_L20.csv`` we reconstruct, per DPU call, the
compute interval ``[t1 - min(span, e), t1]`` (``span`` includes the synchronous ``execute_async``
queue-wait; ``e`` = the per-kernel **median** 1-lane exec time, recovered from the arch's 1-lane trace),
place it on its core (creation-order round-robin ``lane -> core``, validated against the ``coreid`` log),
and take the union of intervals per core over the **steady-state window** = ``[last lane to start,
first lane to finish]`` (trims pipeline fill + drain). ``busy_core = union / window``.

The min/median/max of the 1-lane distribution give a robust ``[lo, hi]`` band on the occupancy (the
distributions are tight, so the band is narrow). ``occupancy_series`` feeds the fan-out lane-scaling
figure (``fanout_lane_plot.py``); ``__main__`` prints the per-core table.

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/fanout_occupancy.py
"""

import csv
import glob
import os
import re
import statistics as st
from typing import Dict, List, Optional, Tuple

import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

TR = REPO_ROOT / "results" / "benchmark_stream" / "traces"
KERN = ("g_a", "h_a", "h_s")
CORE_RE = re.compile(r"device_core_id[_= ]+(\d+)")
SUB_RE = re.compile(r"\[DPUSubgraphRunner\]\s+L(\d+)_(\w+)")


def dpu_events(path) -> List[Tuple[int, str, float, float]]:
    """Return ``(lane, stage, t0_ms, t1_ms)`` for every DPU-kind trace row."""
    ev = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "dpu":
                ev.append((int(r["lane"]), r["stage"], float(r["t0_ms"]), float(r["t1_ms"])))
    return ev


def e_table(arch: str, stat: str) -> Dict[str, float]:
    """Per-kernel exec-time proxy (``min``|``median``|``max``) from the arch's 1-lane trace."""
    ev = dpu_events(TR / f"{arch}_1lane_L20.csv")
    out = {}
    for k in KERN:
        d = [t1 - t0 for _, sg, t0, t1 in ev if sg == k]
        if d:
            out[k] = {"min": min(d), "median": st.median(d), "max": max(d)}[stat]
    return out


def lane_core(stage: str, lane: int, n_lanes: int) -> int:
    """Core a lane's kernel lands on: subgraph-major creation order, VART round-robin over 3 cores."""
    base = {"g_a": 0, "h_a": n_lanes, "h_s": 2 * n_lanes}[stage]
    return (base + lane) % 3


def validate_mapping(arch: str, n_lanes: int) -> str:
    """Cross-check ``lane_core`` against the captured ``device_core_id`` log, if present."""
    log = TR / "coreid" / f"{arch}_{n_lanes}lane_L20.coreid.log"
    if not log.exists():
        return "no-log"
    cur, bad, n = None, 0, 0
    for line in open(log, errors="ignore"):
        m = CORE_RE.search(line)
        if m:
            cur = int(m.group(1))
            continue
        s = SUB_RE.search(line)
        if s:
            n += 1
            if lane_core(s.group(2), int(s.group(1)), n_lanes) != cur:
                bad += 1
    return "OK" if bad == 0 else f"MISMATCH {bad}/{n}"


def _union_len(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total, cs, ce = 0.0, *intervals[0]
    for a, b in intervals[1:]:
        if a > ce:
            total += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return total + (ce - cs)


def occupancy(arch: str, n_lanes: int, stat: str = "median") -> Tuple[Optional[Dict[int, float]], float, int]:
    """Per-core busy fraction over the steady-state window. Returns ``(busy_by_core, window_ms, n_lanes)``.

    ``busy_by_core`` is ``None`` when the all-lanes-active window collapses (too few patches per lane).
    """
    ev = dpu_events(TR / f"{arch}_{n_lanes}lane_L20.csv")
    e = e_table(arch, stat)
    first, last = {}, {}
    for ln, _, t0, t1 in ev:
        first[ln] = min(first.get(ln, 1e18), t0)
        last[ln] = max(last.get(ln, 0.0), t1)
    w0 = max(first.values())  # last lane to start -> fill done
    w1 = min(last.values())  # first lane to finish -> drain begins
    wlen = w1 - w0
    if wlen <= 0:
        return None, wlen, len(first)
    per_core: Dict[int, List[Tuple[float, float]]] = {0: [], 1: [], 2: []}
    for ln, sg, t0, t1 in ev:
        comp = min(t1 - t0, e.get(sg, t1 - t0))  # clamp: compute = min(span, e), never negative
        a, b = max(t1 - comp, w0), min(t1, w1)  # clip to window
        if b > a:
            per_core[lane_core(sg, ln, n_lanes)].append((a, b))
    return {c: _union_len(v) / wlen for c, v in per_core.items()}, wlen, len(first)


def lanes_available(arch: str) -> List[int]:
    """Lane counts with a trace on disk for this arch."""
    out = []
    for p in glob.glob(str(TR / f"{arch}_*lane_L20.csv")):
        m = re.match(rf"{arch}_(\d+)lane_L20", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def occupancy_series(arch: str):
    """``(lanes, mean%, lo%, hi%)`` mean-of-3-cores occupancy for the figure; ``lo/hi`` from e=min/max."""
    xs, mean, lo, hi = [], [], [], []
    for L in lanes_available(arch):
        bmed = occupancy(arch, L, "median")[0]
        if bmed is None:
            continue
        bmin = occupancy(arch, L, "min")[0]
        bmax = occupancy(arch, L, "max")[0]
        xs.append(L)
        mean.append(sum(bmed.values()) / 3 * 100)
        lo.append(sum(bmin.values()) / 3 * 100)
        hi.append(sum(bmax.values()) / 3 * 100)
    return map(np.array, (xs, mean, lo, hi))


def main():
    """Print the per-core occupancy table across every arch x lane trace."""
    op = {"FP": 64, "SHyp": 48, "ResFP": 6, "ResSHyp": 32}  # 32L stands in for ResSHyp 24L (no 24L trace)
    hdr = f"{'arch':8} {'L':>4} {'win_ms':>7} {'c0':>5} {'c1':>5} {'c2':>5} {'mean%':>6} {'[lo':>5} {'hi]':>5} {'map':>6}"
    print(hdr)
    for arch in ("FP", "SHyp", "ResFP", "ResSHyp"):
        for L in lanes_available(arch):
            bmed, wlen, nl = occupancy(arch, L, "median")
            if bmed is None:
                print(f"{arch:8} {L:>4} {wlen:>7.0f}  window collapsed ({nl} lanes)")
                continue
            lo = sum(occupancy(arch, L, "min")[0].values()) / 3 * 100
            hi = sum(occupancy(arch, L, "max")[0].values()) / 3 * 100
            mean = sum(bmed.values()) / 3 * 100
            star = " *op" if L == op[arch] else ""
            print(f"{arch:8} {L:>4} {wlen:>7.0f} {bmed[0]*100:>5.1f} {bmed[1]*100:>5.1f} {bmed[2]*100:>5.1f} "
                  f"{mean:>6.1f} {lo:>5.1f} {hi:>5.1f} {validate_mapping(arch, L):>6}{star}")


if __name__ == "__main__":
    main()
