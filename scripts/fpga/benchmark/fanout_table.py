#!/usr/bin/env python3
"""fanout_table.py — scaling table for the N1 DPU fan-out lane sweep.

Reads the fan-out result JSONs written by stream_fanout_sweep.py (results/benchmark_stream/<model>/,
label carries ``fo`` + ``tL``) and builds, per model, the lane-scaling table: throughput and its ratio
vs 1 lane (cold + warm), energy/patch, and the per-lane g_a ms/call spread (the placement proxy — a
wide min…max means lanes landed on contended cores, not 3 clean ones). It also pulls two references
that already exist on disk: the non-fan-out p0+s1 best (same model dir) and the pure-DPU ``nn_only``
ceiling (results/benchmark_hardware/<model>/nn_only_compress_dpu{1,2,3}.json).

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/fanout_table.py                 # all fan-out models found
    python scripts/fpga/benchmark/fanout_table.py --models ResSHyp-relu_s0_L1000_pt
"""

import argparse
import json
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
STREAM_DIR = REPO_ROOT / "results" / "benchmark_stream"
HW_DIR = REPO_ROOT / "results" / "benchmark_hardware"


def load_fanout_runs(model_dir: Path) -> list:
    """All fan-out result dicts in a model dir (fanout==True), each tagged with its source file."""
    runs = []
    for f in sorted(model_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("fanout"):
            runs.append(d)
    return runs


def p0s1_best(model_dir: Path):
    """The best non-fan-out p0+s1 run (highest cold patch/s) in the dir, or None — the reference
    the fan-out must beat.

    Identified by s1==True and fanout not set.
    """
    best = None
    for f in sorted(model_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("s1") and not d.get("fanout") and d.get("cold", True):
            if best is None or d["median_patch_s"] > best["median_patch_s"]:
                best = d
    return best


def nn_only_ceiling(model_name: str) -> dict:
    """Pure-DPU nn_only fps at dpu1/2/3 for this model (compress), or {} if absent."""
    out = {}
    for n in (1, 2, 3):
        f = HW_DIR / model_name / f"nn_only_compress_dpu{n}.json"
        if f.exists():
            out[n] = json.loads(f.read_text()).get("throughput_fps")
    return out


def ga_spread(run: dict) -> str:
    """Min…max g_a ms/call across the run's lanes (the placement-collision proxy)."""
    calls = [L["ga_ms_call"] for L in run.get("lanes", []) if L.get("patches", 0) > 0]
    if not calls:
        return "—"
    lo, hi = min(calls), max(calls)
    return f"{lo:.1f}" if abs(hi - lo) < 0.5 else f"{lo:.1f}…{hi:.1f}"


def fmt_fps(run: dict, base_fps: float) -> str:
    """Pipe-free 'patch/s (×ratio)' cell, ratio vs base_fps (1-lane, same mode)."""
    if run is None:
        return "—"
    fps = run["median_patch_s"]
    ratio = fps / base_fps if base_fps else 0.0
    return f"{fps:.1f} ({ratio:.2f}×)"


def fmt_energy(run: dict) -> str:
    """Pipe-free 'J/patch @ W' cell."""
    if run is None:
        return "—"
    jp, w = run.get("j_per_patch"), run.get("avg_power_w")
    return f"{jp:.3f} @ {w:.1f} W" if jp and w else "—"


def build_model_table(model_dir: Path) -> str:
    """Markdown block for one model: lane-scaling table (cold+warm) + references."""
    model = model_dir.name
    runs = load_fanout_runs(model_dir)
    if not runs:
        return ""
    # index by (lanes, cold)
    by = {}
    for r in runs:
        by[(r["threads"], bool(r["cold"]))] = r
    lanes = sorted({r["threads"] for r in runs})
    n_patches = runs[0].get("n_patches")
    bpp = runs[0].get("bpp")

    cold1 = by.get((1, True))
    warm1 = by.get((1, False))
    base_cold = cold1["median_patch_s"] if cold1 else (runs[0]["median_patch_s"])
    base_warm = warm1["median_patch_s"] if warm1 else base_cold

    lines = [
        f"### {model} — DPU fan-out lane scaling",
        "",
        f"Full scene (n={n_patches}, bpp={bpp:.3f}); ratio vs 1 lane (same mode). "
        "g_a ms/call spread across lanes = placement proxy (wide ⇒ contended cores).",
        "",
        "| lanes | cold patch/s (×) | warm patch/s (×) | cold J/patch @ W | cold g_a ms/call |",
        "| --- | --- | --- | --- | --- |",
    ]
    for L in lanes:
        c = by.get((L, True))
        w = by.get((L, False))
        note = "" if L <= 3 else " ⚠oversub"
        lines.append(
            f"| {L}{note} | {fmt_fps(c, base_cold)} | {fmt_fps(w, base_warm)} | "
            f"{fmt_energy(c)} | {ga_spread(c) if c else '—'} |"
        )

    # References
    ref = []
    best = p0s1_best(model_dir)
    if best:
        ref.append(f"p0+s1 best (cold): **{best['median_patch_s']:.1f} patch/s**")
    nn = nn_only_ceiling(model)
    if nn:
        ceil = " / ".join(f"dpu{k}={v:.1f}" for k, v in sorted(nn.items()))
        ref.append(f"nn_only DPU ceiling (fps): {ceil}")
    if ref:
        lines += ["", "> " + "  ·  ".join(ref)]
    return "\n".join(lines) + "\n"


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models", nargs="*", help="specific model dir names (default: all with fan-out runs)"
    )
    ap.add_argument("--out", default=str(STREAM_DIR / "fanout_table.md"))
    args = ap.parse_args()

    if args.models:
        dirs = [STREAM_DIR / m for m in args.models]
    else:
        dirs = [d for d in sorted(STREAM_DIR.iterdir()) if d.is_dir() and load_fanout_runs(d)]

    blocks = [b for d in dirs if (b := build_model_table(d))]
    if not blocks:
        raise SystemExit("no fan-out runs found under results/benchmark_stream/")
    doc = "# DPU fan-out lane scaling (N1)\n\n" + "\n".join(blocks)
    Path(args.out).write_text(doc)
    print(doc)
    print(f"\n-> {Path(args.out).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
