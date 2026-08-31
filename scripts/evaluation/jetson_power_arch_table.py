#!/usr/bin/env python3
"""jetson_power_arch_table.py — build the results table from jetson_power_arch_sweep.py's JSONs.

Reads results/benchmark_jetson/orin/power_sweep/<arch>_<mode>.json (16 files for the full 4x4 sweep,
fewer for a partial --archs/--modes run) and prints a markdown table with, per (mode, arch):

- g_a ms/patch — the pipeline's single "g_a" timer scope, which covers BOTH the real and imag forward
  calls per patch (see pipeline.py compress_tile: one `with timer.time("g_a")` wraps both) — not a
  single-call number.
- patch/s (throughput) and the derived full per-patch latency (1000/throughput ms) — valid as "the"
  per-patch latency because ddc-edge is strictly sequential (no pipelining/overlap between patches), so
  there is no distinction between throughput-implied and wall-clock per-patch latency.
- Average power: whole-board total (avg_power_w, the headline figure per the 2026-08-24 project
  decision — see power.py) and the scope-matched compute-only figure (avg_power_w_compute_only,
  excludes the board-I/O/DRAM rail) side by side.
- Energy/patch in mJ, computed from the whole-board total (power['energy_j'] / n_patches), per the same
  2026-08-24 decision to use the total as the basis for energy reporting.

    python scripts/evaluation/jetson_power_arch_table.py
    python scripts/evaluation/jetson_power_arch_table.py --csv results/benchmark_jetson/orin/power_sweep/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

RESULTS_DIR = REPO_ROOT / "results" / "benchmark_jetson" / "orin" / "power_sweep"
MODES = ["MAXN", "MODE_50W", "MODE_30W", "MODE_15W"]
ARCHS = ["FP", "ResFP", "SHyp", "ResSHyp"]

COLUMNS = [
    ("mode", "Mode", "<", 9),
    ("arch", "Arch", "<", 8),
    ("ga_ms_patch", "g_a ms/patch", ">", 12, ".3f"),
    ("patch_s", "patch/s", ">", 8, ".2f"),
    ("latency_ms", "latency ms", ">", 10, ".2f"),
    ("avg_w_total", "W (total)", ">", 9, ".3f"),
    ("avg_w_compute", "W (compute)", ">", 11, ".3f"),
    ("mj_patch", "mJ/patch", ">", 9, ".2f"),
]


def load_row(arch: str, mode: str) -> dict | None:
    """Load one (arch, mode) JSON and derive the table's metrics from it; None if not yet run."""
    path = RESULTS_DIR / f"{arch}_{mode}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    n = d["n_patches"]
    power = d.get("power") or {}
    patch_s = d["throughput_patch_s"]
    energy_j = power.get("energy_j")
    return {
        "arch": arch,
        "mode": mode,
        "n_patches": n,
        "ga_ms_patch": d["stages_ms"]["g_a"] / n,
        "patch_s": patch_s,
        "latency_ms": 1000.0 / patch_s if patch_s > 0 else float("nan"),
        "avg_w_total": power.get("avg_power_w"),
        "avg_w_compute": power.get("avg_power_w_compute_only"),
        "mj_patch": (energy_j / n * 1000.0) if energy_j is not None else None,
    }


def fmt(row: dict, key: str, spec: str) -> str:
    v = row.get(key)
    return format(v, spec) if v is not None else "—"


def print_table(rows: list[dict]) -> None:
    """Print a markdown table (fixed `| --- |` separator, matching repo convention)."""
    header_cells = [c[1] for c in COLUMNS]
    print("| " + " | ".join(header_cells) + " |")
    print("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for r in rows:
        cells = []
        for col in COLUMNS:
            key, _, align = col[0], col[1], col[2]
            if len(col) == 4:
                cells.append(f"{r[key]:{align}{col[3]}}")
            else:
                width, spec = col[3], col[4]
                cells.append(f"{fmt(r, key, spec):{align}{width}}")
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="also write a CSV to this path")
    args = parser.parse_args()

    rows, missing = [], []
    for mode in MODES:
        for arch in ARCHS:
            row = load_row(arch, mode)
            (rows if row is not None else missing).append(
                row if row is not None else f"{arch}_{mode}"
            )

    if missing:
        print(f"[table] missing (not yet run or fetched): {missing}\n")
    if not rows:
        raise SystemExit("no results found -- run jetson_power_arch_sweep.py first")

    print_table(rows)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[table] csv -> {csv_path}")


if __name__ == "__main__":
    main()
