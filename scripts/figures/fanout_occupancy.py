#!/usr/bin/env python3
"""fanout_occupancy.py — per-core DPU occupancy + 4-core CPU occupancy from the fan-out ``--trace``
CSVs (offline, no board).

Moved from ``scripts/fpga/benchmark/`` into ``scripts/figures/`` in P1.0 — it is
figure-only infrastructure now, and ``_figutils`` re-exports its public functions
so the figure scripts import from one place.

**DPU per-core.** For each ``results/date27/lanes/<arch>/t<N>_occtrace.csv`` we
reconstruct, per DPU call, the compute interval ``[t1 - min(span, e), t1]``
(``span`` includes the synchronous ``execute_async`` queue-wait; ``e`` = the
kernel's **median** 1-lane exec time, recovered from the arch's 1-lane trace),
place it on its core (creation-order round-robin ``lane -> core``, subgraph-major:
lane *k*'s ``g_a`` on core *k* mod 3), and take the union of intervals per core
over the **steady-state window** = ``[last lane to start, first lane to finish]``
(trims pipeline fill + drain). ``busy_core = union / window``, asserted <= 100 % on
load. Method + derivation: ``docs/onboard_pipeline.md`` §6.

**CPU 4-core.** The old ``cpu_probe/`` mpstat source was deleted in P0.C, so CPU
occupancy comes from the trace's own ``kind=cpu`` spans: at each instant ``k``
worker threads are inside a CPU span, so ``min(k, 4)`` of the four A53 cores are
busy; integrate ``min(k, 4)`` over the steady-state window and divide by
``4 * window`` -> mean fraction of the four cores doing CPU work. Bounded <= 100 %
by construction. Caveat: under heavy oversubscription a CPU span also covers
scheduler preemption, so this is a mild over-estimate of true busy time — the best
available without per-core scheduler accounting.

Trace CSV columns: ``lane,patch,stage,kind,t0_ms,t1_ms`` with
``kind in {dpu, cpu, read}``; ``lane = -1`` for the shared reader.

    conda activate DDC_FPGA
    python scripts/figures/fanout_occupancy.py
"""

import csv
import glob
import os
import re
import statistics as st
from typing import Dict, List, Optional, Tuple

import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)

TR = REPO_ROOT / "results" / "date27" / "lanes"
KERN = ("g_a", "h_a", "h_s")
CORE_RE = re.compile(r"device_core_id[_= ]+(\d+)")
SUB_RE = re.compile(r"\[DPUSubgraphRunner\]\s+L(\d+)_(\w+)")
N_A53 = 4  # ARM Cortex-A53 cores on the ZCU102


def _trace(arch: str, n_lanes: int):
    """Path to the subsampled occupancy trace for ``arch`` at ``n_lanes`` lanes."""
    return TR / arch / f"t{n_lanes}_occtrace.csv"


def dpu_events(path) -> List[Tuple[int, str, float, float]]:
    """Return ``(lane, stage, t0_ms, t1_ms)`` for every DPU-kind trace row."""
    ev = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "dpu":
                ev.append((int(r["lane"]), r["stage"], float(r["t0_ms"]), float(r["t1_ms"])))
    return ev


def cpu_spans(path) -> List[Tuple[float, float]]:
    """Return ``(t0_ms, t1_ms)`` for every CPU-kind trace row (any worker lane)."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cpu":
                out.append((float(r["t0_ms"]), float(r["t1_ms"])))
    return out


def e_table(arch: str, stat: str) -> Dict[str, float]:
    """Per-kernel exec-time proxy (``min``|``median``|``max``) from the arch's 1-lane trace."""
    ev = dpu_events(_trace(arch, 1))
    out = {}
    for k in KERN:
        d = [t1 - t0 for _, sg, t0, t1 in ev if sg == k]
        if d:
            out[k] = {"min": min(d), "median": st.median(d), "max": max(d)}[stat]
    return out


def lane_core(stage: str, lane: int, n_lanes: int) -> int:
    """Core a lane's kernel lands on: subgraph-major creation order, VART round-robin over 3 DPU
    cores (``docs/onboard_pipeline.md`` §6)."""
    base = {"g_a": 0, "h_a": n_lanes, "h_s": 2 * n_lanes}[stage]
    return (base + lane) % 3


def validate_mapping(arch: str, n_lanes: int) -> str:
    """Cross-check ``lane_core`` against a captured ``device_core_id`` log, if present.

    (The date27 traces ship without the ``coreid/`` logs, so this returns ``no-log``
    there — kept for when a re-capture includes them.)
    """
    log = TR / arch / "coreid" / f"t{n_lanes}_occtrace.coreid.log"
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
    """Total length of the union of the intervals."""
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


def _steady_window(ev: List[Tuple[int, str, float, float]]) -> Tuple[float, float, int]:
    """``(w0, w1, n_lanes)`` — last DPU lane to start, first DPU lane to finish."""
    first, last = {}, {}
    for ln, _, t0, t1 in ev:
        first[ln] = min(first.get(ln, 1e18), t0)
        last[ln] = max(last.get(ln, 0.0), t1)
    return max(first.values()), min(last.values()), len(first)


def occupancy(
    arch: str, n_lanes: int, stat: str = "median"
) -> Tuple[Optional[Dict[int, float]], float, int]:
    """Per-core DPU busy fraction over the steady-state window. Returns ``(busy_by_core, window_ms,
    n_lanes)``.

    ``busy_by_core`` is ``None`` when the all-lanes-active window collapses (too few
    patches per lane). Raises if any core exceeds 100 % — that would mean the
    ``[t1 - e, t1]`` attribution is broken for the trace.
    """
    ev = dpu_events(_trace(arch, n_lanes))
    e = e_table(arch, stat)
    w0, w1, nl = _steady_window(ev)
    wlen = w1 - w0
    if wlen <= 0:
        return None, wlen, nl
    per_core: Dict[int, List[Tuple[float, float]]] = {0: [], 1: [], 2: []}
    for ln, sg, t0, t1 in ev:
        comp = min(t1 - t0, e.get(sg, t1 - t0))  # compute = min(span, e), never negative
        a, b = max(t1 - comp, w0), min(t1, w1)  # clip to window
        if b > a:
            per_core[lane_core(sg, ln, n_lanes)].append((a, b))
    busy = {c: _union_len(v) / wlen for c, v in per_core.items()}
    for c, frac in busy.items():
        if frac > 1.0 + 1e-6:
            raise ValueError(
                f"{arch} {n_lanes}L core {c}: reconstructed DPU busy = {frac * 100:.1f}% "
                f"> 100% — the [t1 - e, t1] attribution is broken for this trace"
            )
    return busy, wlen, nl


def _cpu_core_busy(arch: str, n_lanes: int) -> Optional[float]:
    """Mean % of the 4 A53 cores busy with CPU work over the steady-state window (integral of
    ``min(active_cpu_spans, 4)`` / ``4 * window``)."""
    spans = cpu_spans(_trace(arch, n_lanes))
    w0, w1, _ = _steady_window(dpu_events(_trace(arch, n_lanes)))
    win = w1 - w0
    if win <= 0 or not spans:
        return None
    edges = []
    for t0, t1 in spans:
        a, b = max(t0, w0), min(t1, w1)
        if b > a:
            edges.append((a, 1))
            edges.append((b, -1))
    edges.sort()
    active, prev, area = 0, w0, 0.0
    for t, d in edges:
        area += min(active, N_A53) * (t - prev)
        active += d
        prev = t
    return area / (N_A53 * win) * 100.0


def lanes_available(arch: str) -> List[int]:
    """Lane counts with a trace on disk for this arch."""
    out = []
    for p in glob.glob(str(TR / arch / "t*_occtrace.csv")):
        m = re.match(r"t(\d+)_occtrace", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def cpu_occupancy_series(arch: str):
    """``(lanes, mean % of the 4 A53 cores busy with CPU work)`` from the trace CSVs.

    The CPU counterpart to ``occupancy_series``; see the module docstring for the
    method and its oversubscription caveat.
    """
    xs, occ = [], []
    for L in lanes_available(arch):
        v = _cpu_core_busy(arch, L)
        if v is not None:
            xs.append(L)
            occ.append(v)
    return map(np.array, (xs, occ))


def occupancy_series(arch: str):
    """``(lanes, mean%, lo%, hi%)`` mean-of-3-cores DPU occupancy for the figure; ``lo/hi`` from
    ``e`` = min/max of the 1-lane exec distribution."""
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
    """Print the per-core DPU + 4-core CPU occupancy table across every arch x lane trace."""
    knee = {"FP": 12, "SHyp": 24, "ResFP": 6, "ResSHyp": 20}  # == _figutils.KNEE (§4.0)
    hdr = (
        f"{'arch':8} {'L':>4} {'win_ms':>7} {'c0':>5} {'c1':>5} {'c2':>5} "
        f"{'dpu%':>6} {'[lo':>5} {'hi]':>5} {'cpu4%':>6} {'map':>6}"
    )
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
            cpu4 = _cpu_core_busy(arch, L)
            star = " *knee" if L == knee[arch] else ""
            print(
                f"{arch:8} {L:>4} {wlen:>7.0f} {bmed[0] * 100:>5.1f} {bmed[1] * 100:>5.1f} "
                f"{bmed[2] * 100:>5.1f} {mean:>6.1f} {lo:>5.1f} {hi:>5.1f} "
                f"{cpu4:>6.1f} {validate_mapping(arch, L):>6}{star}"
            )


if __name__ == "__main__":
    main()
