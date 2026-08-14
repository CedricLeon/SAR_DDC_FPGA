#!/usr/bin/env python3
"""fanout_gantt.py — execution-timeline (Gantt) of the DPU fan-out from a --trace CSV.

`stream_pipeline --fanout --trace t.csv` writes one row per stage: ``lane,patch,stage,t0_ms,t1_ms``
(lane -1 = the prefetch reader thread). This draws a swim-lane timeline — one track per DPU lane, plus
the reader on top — over a short window of `--patches` steady-state patches.

Each **DPU** stage bar is split into the compute part (solid, its uncontended duration) and the **wait**
part (white, hatched) — the time the call spent queued because another lane held its DPU core. So a
collision reads directly as a hatched block, not a longer solid bar. The uncontended duration per stage
is taken as the minimum seen in ``--solo-csv`` (a 1- or 3-lane trace; defaults to this file).

The window is anchored on a real read event so the reader track is always shown, and every timeline is
**shifted to start at t=0** for easy comparison across figures.

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

# stage -> (colour, label). DPU cool, CPU warm, read grey.
STAGES = {
    "read": ("#999999", "read (producer)"),
    "normalize": ("#E69F00", "normalize (CPU)"),
    "g_a": ("#0072B2", "g_a (DPU)"),
    "h_a": ("#56B4E9", "h_a (DPU)"),
    "h_s": ("#009E73", "h_s (DPU)"),
    "eb": ("#D55E00", "entropy: EB (CPU)"),
    "gc": ("#CC79A7", "entropy: GC (CPU)"),
}
DPU_STAGES = {"g_a", "h_a", "h_s"}  # these can wait on a shared core


def load(path: Path) -> list:
    """Read a trace CSV into event dicts."""
    with open(path) as f:
        return [
            {
                "lane": int(r["lane"]),
                "patch": int(r["patch"]),
                "stage": r["stage"],
                "t0": float(r["t0_ms"]),
                "t1": float(r["t1_ms"]),
            }
            for r in csv.DictReader(f)
        ]


def solo_durations(events: list) -> dict:
    """Uncontended duration per stage = the minimum observed (the cleanest instance)."""
    d = defaultdict(list)
    for e in events:
        d[e["stage"]].append(e["t1"] - e["t0"])
    return {s: min(v) for s, v in d.items()}


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


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--solo-csv", default=None, help="clean trace for uncontended baselines (default: self)"
    )
    ap.add_argument("--title", default="")
    ap.add_argument("--patches", type=int, default=3)
    ap.add_argument("--skip", type=int, default=3, help="skip this many cold patches first")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = load(Path(args.csv))
    solo = solo_durations(load(Path(args.solo_csv)) if args.solo_csv else events)
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
        base = solo.get(e["stage"], dur)
        if (
            e["stage"] in DPU_STAGES and dur > base * 1.15
        ):  # split: wait (hatch) then compute (solid)
            bar(x0, dur - base, y, facecolor="white", hatch="////", edgecolor="#8a8a8a")
            bar(x0 + (dur - base), base, y, facecolor=color, edgecolor="white")
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
    ax.set_xlabel("time (ms) — each timeline starts at 0")
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
            label="wait (queued for a DPU core)",
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
