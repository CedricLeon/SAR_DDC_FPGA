#!/usr/bin/env python3
"""jetson_batch_precision_table.py — build markdown tables from jetson_power_arch_sweep.py's
batch/precision/fuse sweep JSONs.

Reads results/benchmark_jetson/orin/<subdir>/*.json (each carries arch, power mode via hw_info.nvpmodel,
batch_size, precision, fuse_reim, throughput, power, bpp) and prints, per power mode, a table with:

- batch / precision / fuse           — the config, read from the JSON fields (not the filename)
- patch/s                            — throughput_patch_s
- latency ms                         — 1000/patch_s (amortized per-patch at batch>1, NOT single-patch)
- W (total) / W (compute)            — whole-board / on-chip-compute average power
- mJ/patch                           — energy_j / n_patches * 1000
- bpp                                — sanity (should be ~constant per arch)

The best-throughput row within each (mode, arch) block is **bold**. `--best` instead prints, per arch,
the single highest-throughput config across all modes (used to pick the operating point for stage B).

    python scripts/evaluation/jetson_batch_precision_table.py --subdir batch_precision_sweep
    python scripts/evaluation/jetson_batch_precision_table.py --subdir batch_precision_sweep --best
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
MODE_ORDER = ["MAXN", "MODE_50W", "MODE_30W", "MODE_15W"]
ARCH_ORDER = ["FP", "ResFP", "SHyp", "ResSHyp"]
_MODE_RE = re.compile(r"NV Power Mode:\s*(\S+)")


def mode_of(d: dict) -> str:
    """Power mode from the recorded nvpmodel -q string; '?' if absent."""
    m = _MODE_RE.search((d.get("hw_info") or {}).get("nvpmodel", ""))
    return m.group(1) if m else "?"


def load_rows(subdir: Path) -> list[dict]:
    """Load all JSONs in subdir and return a list of dicts with the fields we want for the
    table."""
    rows = []
    for p in sorted(subdir.glob("*.json")):
        d = json.loads(p.read_text())
        n = d["n_patches"]
        power = d.get("power") or {}
        ps = d["throughput_patch_s"]
        ej = power.get("energy_j")
        rows.append(
            {
                "arch": d["arch"],
                "mode": mode_of(d),
                "batch": d.get("batch_size", 1),
                "prec": d.get("precision", "fp32"),
                "fuse": bool(d.get("fuse_reim", False)),
                "patch_s": ps,
                "latency_ms": 1000.0 / ps if ps else float("nan"),
                "w_total": power.get("avg_power_w"),
                "w_compute": power.get("avg_power_w_compute_only"),
                "mj_patch": (ej / n * 1000.0) if ej is not None else None,
                "bpp": d.get("bpp"),
                "file": p.name,
            }
        )
    return rows


def _f(v, spec):
    """Simple formatting."""
    return format(v, spec) if v is not None else "—"


def print_mode_tables(rows: list[dict]) -> None:
    """Print a table per power mode, with the best-throughput row in bold."""
    modes = sorted(
        {r["mode"] for r in rows}, key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else 99
    )
    for mode in modes:
        print(f"\n### {mode}\n")
        print(
            "| arch | batch | prec | fuse | patch/s | latency ms | W (total) | W (compute) | mJ/patch | bpp |"
        )
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for arch in [
            a for a in ARCH_ORDER if any(r["arch"] == a and r["mode"] == mode for r in rows)
        ]:
            block = [r for r in rows if r["arch"] == arch and r["mode"] == mode]
            best = max(block, key=lambda r: r["patch_s"])
            for r in sorted(block, key=lambda r: (r["batch"], r["prec"], r["fuse"])):
                b = r is best
                ps = f"**{r['patch_s']:.2f}**" if b else f"{r['patch_s']:.2f}"
                print(
                    f"| {r['arch']} | {r['batch']} | {r['prec']} | {'yes' if r['fuse'] else 'no'} | "
                    f"{ps} | {_f(r['latency_ms'], '.2f')} | {_f(r['w_total'], '.2f')} | "
                    f"{_f(r['w_compute'], '.2f')} | {_f(r['mj_patch'], '.1f')} | {_f(r['bpp'], '.4f')} |"
                )


def print_best(rows: list[dict]) -> None:
    """Print a single best-throughput row per architecture, across all modes present."""
    print("\n### Best-throughput config per architecture (across all modes present)\n")
    print("| arch | mode | batch | prec | fuse | patch/s | mJ/patch |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for arch in [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]:
        block = [r for r in rows if r["arch"] == arch]
        best = max(block, key=lambda r: r["patch_s"])
        print(
            f"| {arch} | {best['mode']} | {best['batch']} | {best['prec']} | "
            f"{'yes' if best['fuse'] else 'no'} | **{best['patch_s']:.2f}** | {_f(best['mj_patch'], '.1f')} |"
        )


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subdir", default="batch_precision_sweep")
    ap.add_argument(
        "--best", action="store_true", help="print best config per arch instead of full tables"
    )
    args = ap.parse_args()
    d = REPO_ROOT / "results" / "benchmark_jetson" / "orin" / args.subdir
    if not d.is_dir():
        raise SystemExit(f"no results dir: {d}")
    rows = load_rows(d)
    if not rows:
        raise SystemExit(f"no JSONs in {d} -- run jetson_power_arch_sweep.py first")
    print(f"[table] {len(rows)} runs from {d}")
    if args.best:
        print_best(rows)
    else:
        print_mode_tables(rows)


if __name__ == "__main__":
    main()
