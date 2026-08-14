#!/usr/bin/env python3
"""fanout_gantt.py — execution-timeline (Gantt) of the DPU fan-out from a --trace CSV.

``stream_pipeline --fanout --trace t.csv`` writes one row per sub-step (schema v2):
``lane,patch,stage,kind,t0_ms,t1_ms`` (lane -1 = the prefetch reader). ``stage`` is per-call —
``g_a`` appears twice (real, imag), its CPU glue is ``g_a_cpu``, and EB compress/decompress are
``eb_enc`` / ``eb_dec``. ``kind`` is ``dpu`` | ``cpu`` | ``read``. This draws a swim-lane timeline,
one track per DPU lane plus the reader, over a short window of steady-state patches.

**How a DPU bar is drawn (honest about measured vs inferred).** On this board ``execute_async`` is
synchronous (Phase 0: it blocks for the whole job, ``wait()`` is a no-op), so a DPU event's end time
is the *exact* job-completion instant. We split the bar into the **compute** part (solid, its
uncontended duration ``e``) and the **wait** part (white, hatched): a call that ended at ``t1`` and
takes ``e`` uncontended must have executed during ``[t1-e, t1]`` and queued during ``[t0, t1-e]``, so
``wait = measured span − e`` and the hatch sits before the solid. A call is drawn as waiting only when
it runs longer than **any clean call of that kernel** (the max span in ``--solo-csv``) — a data-derived
threshold (no hardcoded floor), so a clean trace is wait-free by construction and the tiny h_a/h_s
kernels don't paint jitter as waits. The wait is therefore **inferred** (``measured − solo min``), not a
directly measured queue time. ``e`` (the *solo* exec) is the minimum span of that stage in ``--solo-csv``
(default: this file); self is correct for a clean run, but for a contended trace pass a clean 1-lane /
pinned trace (else the contended max hides the waits).
A DPU span also includes the in-``run()`` int8 quantize/dequantize (~0.2–0.7 ms), which cancels in the
subtraction. CPU and read bars are drawn as their measured span.

    python scripts/fpga/benchmark/fanout_gantt.py --csv rsh_t4.csv --solo-csv rsh_t3.csv \
        --title "ResSHyp · 4 lanes" --patches 3
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# stage -> (colour, label). DPU cool, CPU warm, read grey. Order = pipeline order (for the legend).
STAGES = {
    "read": ("#999999", "read (producer)"),
    "normalize": ("#E69F00", "normalize (CPU)"),
    "g_a_cpu": ("#F0E442", "g_a split/interleave (CPU)"),
    "g_a": ("#0072B2", "g_a (DPU)"),
    "h_a": ("#56B4E9", "h_a (DPU)"),
    "h_s": ("#009E73", "h_s (DPU)"),
    "eb_enc": ("#D55E00", "entropy: EB enc (CPU)"),
    "eb_dec": ("#E8845E", "entropy: EB dec (CPU)"),
    "gc": ("#CC79A7", "entropy: GC (CPU)"),
}


def load(path: Path) -> list:
    """Read a v2 trace CSV into event dicts."""
    with open(path) as f:
        return [
            {
                "lane": int(r["lane"]),
                "patch": int(r["patch"]),
                "stage": r["stage"],
                "kind": r["kind"],
                "t0": float(r["t0_ms"]),
                "t1": float(r["t1_ms"]),
            }
            for r in csv.DictReader(f)
        ]


def solo_durations(events: list) -> dict:
    """Per DPU stage, the (min, max) uncontended span from the solo source.

    min = pure exec, used as the wait magnitude (wait = measured span − min). max = the clean-
    jitter ceiling, used as the wait *threshold*: a call is drawn as 'waiting' only if it runs
    longer than any clean call of that kernel ever did. Data-derived (no hardcoded floor), and it
    makes a clean trace wait-free by construction — the tiny h_a/h_s kernels no longer paint
    spurious waits from ordinary jitter, while the real queue-behind-g_a waits (tens of ms) sit far
    above the ceiling. Only DPU events are reconstructed; CPU/read stages are drawn as measured.
    """
    d = defaultdict(list)
    for e in events:
        if e["kind"] == "dpu":
            d[e["stage"]].append(e["t1"] - e["t0"])
    return {s: (min(v), max(v)) for s, v in d.items()}


def pick_window(events: list, n_patches: int, skip: int):
    """A steady-state window: n_patches of the busiest lane, skipping the first `skip` (cold) ones."""
    per_lane = defaultdict(list)
    seen = defaultdict(set)
    for e in sorted(events, key=lambda e: e["t0"]):
        if e["lane"] < 0 or e["patch"] in seen[e["lane"]]:
            continue
        seen[e["lane"]].add(e["patch"])
        per_lane[e["lane"]].append((e["patch"], e["t0"]))
    ref = max(per_lane, key=lambda ln: len(per_lane[ln]))
    ref_patches = per_lane[ref]
    i = min(skip, max(0, len(ref_patches) - n_patches))
    t_start = ref_patches[i][1]
    last_patch = ref_patches[min(i + n_patches, len(ref_patches)) - 1][0]
    t_end = max(e["t1"] for e in events if e["lane"] == ref and e["patch"] == last_patch)
    return t_start, t_end


def consistency_check(events: list, solo: dict):
    """Cheap sanity report: a DPU event should never be shorter than its solo exec (that would mean
    the solo estimate is too high). Print any violations so the reconstruction is not trusted blindly.
    """
    bad = 0
    for e in events:
        if e["kind"] == "dpu" and e["stage"] in solo:
            lo, _hi = solo[e["stage"]]
            if (e["t1"] - e["t0"]) < lo * 0.98:
                bad += 1
    if bad:
        print(f"  ⚠ {bad} DPU events shorter than their solo exec — solo may be over-estimated")


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--solo-csv", default=None, help="clean trace for uncontended solo exec (default: self)"
    )
    ap.add_argument("--title", default="")
    ap.add_argument("--patches", type=int, default=3)
    ap.add_argument("--skip", type=int, default=3, help="skip this many cold patches first")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = load(Path(args.csv))
    solo = solo_durations(load(Path(args.solo_csv)) if args.solo_csv else events)
    consistency_check(events, solo)
    t_start, t_end = pick_window(events, args.patches, args.skip)
    span = t_end - t_start
    pad = 0.02 * span
    win = [e for e in events if e["t1"] > t_start - pad and e["t0"] < t_end + pad]

    lanes = [-1] + sorted({e["lane"] for e in win if e["lane"] >= 0})  # reader always present
    y_of = {ln: i for i, ln in enumerate(lanes)}
    fig, ax = plt.subplots(figsize=(11, 0.85 * len(lanes) + 1.7))

    def bar(x0, w, y, **kw):
        x0 = max(x0, 0.0)
        w = min(x0 + w, span) - x0
        if w > 0:
            ax.add_patch(Rectangle((x0, y - 0.34), w, 0.68, linewidth=0.6, zorder=3, **kw))

    for e in win:
        y = y_of[e["lane"]]
        x0, dur = e["t0"] - t_start, e["t1"] - e["t0"]
        color = STAGES.get(e["stage"], ("#333", ""))[0]
        lo, hi = solo.get(e["stage"], (dur, dur))
        if e["kind"] == "dpu" and dur > hi:
            # ran longer than any clean call of this kernel => it queued. Synchronous execute_async: it
            # ended at t1, so it ran during [t1-lo, t1] (lo = pure exec) and queued during [t0, t1-lo]:
            # draw the inferred wait (hatch) then the compute (solid).
            bar(x0, dur - lo, y, facecolor="white", hatch="////", edgecolor="#8a8a8a")
            bar(x0 + (dur - lo), lo, y, facecolor=color, edgecolor="white")
        else:
            bar(x0, dur, y, facecolor=color, edgecolor="white")

    if not any(e["lane"] < 0 for e in win):  # reads are prefetched up-front, off the steady window
        ax.text(
            0.01 * span,
            y_of[-1],
            "  reads prefetched up-front — none active in this steady window",
            va="center",
            ha="left",
            fontsize=8,
            color="#777",
            style="italic",
        )

    ax.set_xlim(0, span)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels(["reader" if ln < 0 else f"lane {ln}" for ln in lanes])
    ax.invert_yaxis()
    ax.set_xlabel(
        "time (ms) — each timeline starts at 0; DPU wait is inferred (measured − solo exec)"
    )
    ax.grid(axis="x", color="#ececec", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        f"DPU fan-out execution timeline — {args.title or Path(args.csv).stem}",
        fontsize=12,
        fontweight="bold",
    )

    present = [s for s in STAGES if any(e["stage"] == s for e in win)]
    handles = [
        Patch(facecolor=STAGES[s][0], edgecolor="white", label=STAGES[s][1]) for s in present
    ]
    handles.append(
        Patch(
            facecolor="white",
            hatch="////",
            edgecolor="#8a8a8a",
            label="inferred wait (queued for a DPU core)",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(4, len(handles)),
        fontsize=9,
        frameon=False,
    )

    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
