#!/usr/bin/env python3
"""stream_gantt.py — schedule timeline (Gantt) diagrams for the streaming compressor.

Panels (top->bottom): seq, s1, p0(+s1), and a coarse row-block streaming view
(read‖compute‖write). Per-patch stage durations are the measured means from the
benchmark_hardware s0/s1 compress runs; the row-block read/compute come from the
benchmark_stream sweep. The overlap/wait layout is illustrative.

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/stream_gantt.py --model ResSHyp-relu_s0_L1000_pt --detail merged
    python scripts/fpga/benchmark/stream_gantt.py --model FP-relu_s0_L1000_pt --detail full

--detail merged : the tiny DPU sub-stages (h_a, h_s) fold into one block (legible).
--detail full   : every stage drawn + labelled (accurate, busier).
p0 shows 3 workers for legibility; the board runs up to 4 (--workers).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BH = REPO_ROOT / "results" / "benchmark_hardware"
BS = REPO_ROOT / "results" / "benchmark_stream"
CANON = ["normalize", "g_a", "h_a", "eb_compress", "eb_decompress", "h_s", "gc_compress"]
DPU = {"g_a", "h_a", "h_s"}
LABEL = {
    "normalize": "norm",
    "g_a": "g_a",
    "h_a": "h_a",
    "eb_compress": "eb",
    "eb_decompress": "eb'",
    "h_s": "h_s",
    "gc_compress": "gc",
}
COL = {
    "DPU": "#4C78A8",
    "CPU": "#F58518",
    "read": "#54A24B",
    "write": "#B279A2",
    "wait": "#D6D6D6",
}


def stage_ms(model, cfg):
    """[(stage, mean_ms)] from benchmark_hardware/<model>/<cfg>_compress.json (canonical order)."""
    d = json.loads((BH / model / f"{cfg}_compress.json").read_text())
    st = {k: v["mean_ms"] for k, v in d["stages"].items()}
    return [(k, st[k]) for k in CANON if k in st]


def stream_block_ms(model):
    """(read_ms, compute_ms) per row-block from the benchmark_stream sweep."""
    cold = json.loads((BS / model / "seq_cold.json").read_text())
    warm = json.loads((BS / model / "p0_s1_t4_pf_neon_warm.json").read_text())
    ga = cold["grid"][0]
    return cold["read_ms"] / ga, warm["median_total_s"] * 1000 / ga


def bar(ax, row, t0, dur, label, res):
    """One stage bar (res keys COL); label it if wide enough and a label is given."""
    ax.broken_barh([(t0, dur)], (row - 0.4, 0.8), facecolors=COL[res], edgecolor="white", lw=0.5)
    if dur > 4.0 and label:
        ax.text(t0 + dur / 2, row, label, ha="center", va="center", fontsize=7, color="white")


def draw_linear(ax, cfg, s1_ga, split):
    """Seq / s1 panel: one patch, fully serialized.

    split=True puts g_a on DPU0‖DPU1.
    """
    cpu_row = 2 if split else 1
    t = 0.0
    for name, dur in cfg:
        if name == "g_a" and split:
            bar(ax, 0, t, s1_ga, "g_a·re", "DPU")
            bar(ax, 1, t, s1_ga, "g_a·im", "DPU")
            t += s1_ga
        elif name in DPU:
            bar(ax, 0, t, dur, LABEL[name], "DPU")
            t += dur
        else:
            bar(ax, cpu_row, t, dur, LABEL[name], "CPU")
            t += dur
    ax.set_yticks([0, 1, 2] if split else [0, 1])
    ax.set_yticklabels(["DPU0", "DPU1", "CPU"] if split else ["DPU", "CPU"])
    return t


def draw_p0(ax, cfg, s1_ga, detail, workers):
    """Honest p0(+s1): workers normalize early, then WAIT for the DPU mutex; the DPU burst (g_a on
    DPU0‖DPU1, then h_a/h_s) is serialized back-to-back.

    Rows: DPU0, DPU1, W0..W(workers-1).
    """
    d = dict(cfg)
    norm = d.get("normalize", 0.0)
    ha, hs = d.get("h_a", 0.0), d.get("h_s", 0.0)
    dpu = s1_ga + ha + hs
    post = [(k, d[k]) for k in CANON if k in d and k not in DPU and k != "normalize"]  # eb(') + gc
    for k in range(workers):
        w = 2 + (k % workers)
        start = norm + k * dpu
        bar(ax, w, 0.0, norm, "norm", "CPU")  # normalize early (all workers in parallel)
        if k > 0:
            bar(ax, w, norm, start - norm, "wait", "wait")  # queued on the DPU mutex
        bar(ax, 0, start, s1_ga, "g_a·re", "DPU")  # DPU0
        bar(ax, 1, start, s1_ga, "g_a·im", "DPU")  # DPU1
        if detail == "full":
            if ha:
                bar(ax, 0, start + s1_ga, ha, "h_a", "DPU")
            if hs:
                bar(ax, 0, start + s1_ga + ha, hs, "h_s", "DPU")
        elif ha + hs > 0:
            bar(ax, 0, start + s1_ga, ha + hs, "", "DPU")
        tp = start + dpu  # post-DPU CPU stages on the worker row
        for name, dur in post:
            bar(ax, w, tp, dur, LABEL[name], "CPU")
            tp += dur
    ax.set_yticks([0, 1] + [2 + i for i in range(workers)])
    ax.set_yticklabels(["DPU0", "DPU1"] + [f"W{i}" for i in range(workers)])
    return norm + workers * dpu + sum(dur for _, dur in post)


def draw_stream(ax, rb, cb):
    """Coarse row-block view: producer reads block N+1 while workers compute block N; write at end.
    Compute >> read (ResSHyp) -> read hidden; read >= compute (FP) -> read-bound."""
    n, step = 3, max(rb, cb)
    for k in range(n):
        bar(ax, 2, k * step, rb, "read", "read")
        bar(ax, 1, rb + k * step, cb, "compute block", "CPU" if cb < rb else "DPU")
    bar(ax, 0, rb + n * step, max(8, step * 0.02), "write", "write")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["writer", "compute", "reader"])
    bound = "read-bound" if rb > cb else "compute-bound"
    ax.set_title(
        f"streaming (row-block) — read {rb:.0f} ms ‖ compute {cb:.0f} ms/block → {bound}",
        fontsize=10,
        loc="left",
    )
    return rb + n * step + 12


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="ResSHyp-relu_s0_L1000_pt")
    ap.add_argument("--detail", choices=["merged", "full"], default="merged")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s0 = stage_ms(args.model, "s0")
    s1_ga = dict(stage_ms(args.model, "s1"))["g_a"]
    rb, cb = stream_block_ms(args.model)

    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    pre = dict(s0)["normalize"]
    m_seq = draw_linear(axes[0], s0, s1_ga, split=False)
    m_s1 = draw_linear(axes[1], s0, s1_ga, split=True)
    m_p0 = draw_p0(axes[2], s0, s1_ga, args.detail, args.workers)
    m_st = draw_stream(axes[3], rb, cb)
    axes[0].set_title(f"seq — fully serialized ({m_seq:.0f} ms/patch)", fontsize=10, loc="left")
    axes[1].set_title(
        f"s1 — g_a(re)‖g_a(im) on 2 DPU cores ({m_s1:.0f} ms/patch)", fontsize=10, loc="left"
    )
    axes[2].set_title(
        f"p0(+s1) — {args.workers} workers normalize early then WAIT for the DPU "
        "mutex (board runs up to 4)",
        fontsize=10,
        loc="left",
    )
    for ax, mx, lo in zip(axes, [m_seq, m_s1, m_p0, m_st], [-pre - 2, -pre - 2, 0, 0]):
        ax.set_xlim(lo, mx + 3)
        ax.grid(axis="x", ls=":", alpha=0.4)
    axes[3].set_xlabel("time (ms)")
    handles = [
        Patch(fc=COL["DPU"], label="DPU"),
        Patch(fc=COL["CPU"], label="CPU"),
        Patch(fc=COL["read"], label="SD read"),
        Patch(fc=COL["wait"], label="wait"),
        Patch(fc=COL["write"], label="write"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(f"Streaming schedules — {args.model}  [{args.detail}]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or str(BS / f"gantt_{args.model.split('-')[0]}_{args.detail}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
