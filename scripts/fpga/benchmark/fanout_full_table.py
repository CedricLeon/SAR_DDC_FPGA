#!/usr/bin/env python3
"""fanout_full_table.py — one merged fan-out metrics table (all archs, all lanes).

Reads the fan-out result JSONs (stream_fanout_sweep.py) and emits a single markdown table: rows are
(arch × metric) — g_a ms/call, per-patch compute latency (normalize+DPU+entropy), throughput,
avg power, J/patch — columns are lane counts 1..4, and each cell is ``cold / warm``. Latency excludes
the (prefetched) SD read. Written to results/benchmark_stream/fanout_full_table.md.

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/fanout_full_table.py
"""

import argparse
import json
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
STREAM = REPO_ROOT / "results" / "benchmark_stream"

# (arch, short topology tag). Order light→heavy DPU.
ARCHS = [("FP", "factorized"), ("SHyp", "hyperprior"), ("ResFP", "factorized + residual"),
         ("ResSHyp", "hyperprior + residual")]
LANES = [1, 2, 3, 4]


def _load(model: str, lanes: int, warm: bool):
    tag = "warm" if warm else "cold"
    f = STREAM / model / f"p0_t{lanes}_fo_pf_neon_{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _metrics(j: dict) -> dict:
    """g_a (mean ms/call over lanes), latency (mean per-patch compute ms), tput, power, J/patch."""
    ls = [L for L in j["lanes"] if L["patches"] > 0]
    ga = sum(L["ga_ms_call"] for L in ls) / len(ls)
    lat = sum(L["norm_ms_patch"] + L["ga_ms_patch"] + L["ha_ms_patch"] + L["hs_ms_patch"]
              + L["entropy_ms_patch"] for L in ls) / len(ls)
    return {"g_a [ms]": ga, "latency [ms]": lat, "throughput [patch/s]": j["median_patch_s"],
            "avg power [W]": j["avg_power_w"], "J/patch": j["j_per_patch"]}


ROWS = [("g_a [ms]", "{:.1f}"), ("latency [ms]", "{:.0f}"), ("throughput [patch/s]", "{:.1f}"),
        ("avg power [W]", "{:.1f}"), ("J/patch", "{:.3f}")]


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lam", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(STREAM / "fanout_full_table.md"))
    args = ap.parse_args()

    lines = [
        "# Fan-out metrics — all archs (full scene, λ%d)" % args.lam,
        "",
        "Each cell is **cold / warm**. Latency = per-patch compute time (normalize + DPU + entropy); "
        "the prefetched SD read is not included. Lanes 1–3 map to the 3 DPU cores; 4 oversubscribes.",
        "",
        "| arch (topology) | metric | 1 lane | 2 lanes | 3 lanes | 4 lanes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arch, tag in ARCHS:
        model = f"{arch}-relu_s{args.seed}_L{args.lam}_pt"
        cold = {L: (_metrics(j) if (j := _load(model, L, False)) else None) for L in LANES}
        warm = {L: (_metrics(j) if (j := _load(model, L, True)) else None) for L in LANES}
        for ri, (metric, fmt) in enumerate(ROWS):
            head = f"**{arch}** ({tag})" if ri == 0 else ""
            cells = []
            for L in LANES:
                c = fmt.format(cold[L][metric]) if cold[L] else "–"
                w = fmt.format(warm[L][metric]) if warm[L] else "–"
                cells.append(f"{c} / {w}")
            lines.append(f"| {head} | {metric} | " + " | ".join(cells) + " |")

    doc = "\n".join(lines) + "\n"
    Path(args.out).write_text(doc)
    print(doc)
    print(f"-> {Path(args.out).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
