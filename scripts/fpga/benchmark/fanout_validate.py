#!/usr/bin/env python3
"""fanout_validate.py — validate DPU fan-out placement from a ``--trace`` CSV, by measuring how
many lanes run ``g_a`` *concurrently* (= physical DPU cores actually in use) — with no core id
needed.

VART exposes no runner→core API, so we cannot read which core a lane lands on. But we can measure how
many lanes execute ``g_a`` at the same instant, which is a lower bound on the cores in use. Each g_a
call's execution window is ``[t_done − e, t_done]`` where ``e`` is the pure g_a exec (the shortest g_a
span in the trace); ``execute_async`` is synchronous on this board (Phase 0), so ``t_done`` is the exact
completion instant. Sweeping these windows across lanes gives the concurrency over time. Its maximum is
the number of cores running g_a at once:

  * pinned (deterministic) reaches **3** — the 3 lanes genuinely run on 3 distinct cores (placement works);
  * ``--lane-major`` stays at **1** — all g_a serialized on one core (the collision);
  * 4 lanes caps at **3** — oversubscription, never a 4th core.

The max never exceeding the 3 physical cores is the consistency check: an impossible >3 would mean the
reconstruction (or the placement model) is wrong. A small overlap tolerance (``--tol``, default 1 ms >
the ~0.6 ms int8 quant/dequant that pads each window inside ``run()``) keeps call-handoff slivers from
counting as concurrency.

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/fanout_validate.py t_pin.csv t_lm.csv t_t4.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

CORES = 3  # ZCU102 = 3 real B4096 DPU cores (xdputil reports a 4th empty slot; it is not usable)


def load(path: Path) -> list:
    """Read a v2 trace CSV (lane,patch,stage,kind,t0_ms,t1_ms) into event dicts."""
    with open(path) as f:
        return [
            {
                "lane": int(r["lane"]),
                "stage": r["stage"],
                "kind": r["kind"],
                "t0": float(r["t0_ms"]),
                "t1": float(r["t1_ms"]),
            }
            for r in csv.DictReader(f)
        ]


def concurrency(events: list, tol: float):
    """Max concurrent g_a executions + the fraction of g_a-execution time spent at each level.

    tol shrinks each execution window by tol/2 per side so sub-tol handoff overlaps (the
    quant/dequant slop that pads the run() span) are not counted as real concurrency.
    """
    ga = [e for e in events if e["stage"] == "g_a" and e["kind"] == "dpu"]
    if not ga:
        return None
    exec_ms = min(e["t1"] - e["t0"] for e in ga)  # pure g_a exec (shortest = least-queued call)
    edges = []
    for e in ga:
        a, b = e["t1"] - exec_ms + tol / 2, e["t1"] - tol / 2
        if b > a:
            edges.append((a, +1))
            edges.append((b, -1))
    edges.sort(
        key=lambda x: (x[0], x[1])
    )  # end (-1) before start (+1) at ties => touching != overlap
    cur = mx = 0
    prev = None
    time_at: dict = defaultdict(float)
    for t, d in edges:
        if prev is not None and cur > 0:
            time_at[cur] += t - prev
        cur += d
        mx = max(mx, cur)
        prev = t
    total = sum(time_at.values())
    dist = {k: 100.0 * v / total for k, v in time_at.items()} if total else {}
    return {"exec_ms": exec_ms, "max_conc": mx, "n": len(ga), "dist": dist}


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", nargs="+", help="one or more --trace CSVs")
    ap.add_argument(
        "--tol", type=float, default=1.0, help="overlap tolerance ms (quant/dequant slop)"
    )
    args = ap.parse_args()

    print(f"{'trace':28} {'g_a exec':>9} {'max cores':>10}  time-at-concurrency")
    worst_ok = True
    for path in args.csv:
        r = concurrency(load(Path(path)), args.tol)
        if r is None:
            print(f"{Path(path).stem:28} (no g_a events)")
            continue
        ok = r["max_conc"] <= CORES
        worst_ok &= ok
        flag = "" if ok else f"  ⚠ > {CORES} physical cores!"
        dist = " ".join(f"{k}:{r['dist'][k]:.0f}%" for k in sorted(r["dist"]))
        print(f"{Path(path).stem:28} {r['exec_ms']:>7.1f}ms {r['max_conc']:>10}  {dist}{flag}")
    if not worst_ok:
        raise SystemExit("validation FAILED: concurrency exceeds the physical core count")


if __name__ == "__main__":
    main()
