#!/usr/bin/env python3
"""run_unified_benchmark.py — one command to benchmark the SAR-DDC pipeline across HW platforms.

Drives the host GPU/CPU benchmark locally AND the FPGA benchmark over SSH, into a single
results tree, so the cross-platform notebook can compare them on one schema.

    conda activate DDC_FPGA
    python scripts/benchmark/run_unified_benchmark.py --model-dir results/fpga/active_model/ --power
    python scripts/benchmark/run_unified_benchmark.py --model-dir results/fpga/active_model/ --no-fpga   # host only
    python scripts/benchmark/run_unified_benchmark.py --model-dir results/fpga/active_model/ --no-gpu --no-cpu  # FPGA only

Results:
    host  → results/benchmark_unified/<model>/baseline_<scenario>_<platform>.json   (benchmark_gpu.py)
    fpga  → results/benchmark_hardware/<model>/<config>_<scenario>.json             (benchmark_sweep.py, SSH)
Both share the aligned schema; the unified loader (notebooks/_benchmark_loader.py) reads both.

TGRS-era tool. `results/benchmark_hardware/` was archived to `/mnt/vitisAI/DDC_results_archive/2026-08-31/`
after the DATE'27 campaign; the FPGA backend still re-populates it if run, but DATE'27's cross-platform
story uses `results/date27/` (FPGA) + `results/benchmark_jetson/` (edge). Kept for TGRS-revision
reproducibility — the `--no-fpga` host path is unaffected.

Extensible to new hardware: a backend is a callable registered in BACKENDS. To add e.g. a Jetson
edge GPU, write a backend that runs its own benchmark (locally or over SSH) emitting the aligned
schema into results/benchmark_unified/, register it, and add a `--no-jetson` toggle.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BENCHMARK_GPU = ROOT / "scripts" / "evaluation" / "benchmark_gpu.py"
BENCHMARK_SWEEP = ROOT / "scripts" / "fpga" / "benchmark" / "benchmark_sweep.py"
UNIFIED_DIR = ROOT / "results" / "benchmark_unified"
HARDWARE_DIR = ROOT / "results" / "benchmark_hardware"


def _run(cmd: list, **kw) -> int:
    """Run a subprocess, streaming output; return its exit code (no raise)."""
    print(f"[unified] $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], **kw).returncode


# ---------------------------------------------------------------------------
# Backends — each takes (ctx, args) and returns True on success, False on skip/fail.
# `ctx` carries the resolved model context (model_dir, model_name).
# ---------------------------------------------------------------------------
def backend_host(ctx: dict, args: argparse.Namespace) -> bool:
    """GPU + CPU host benchmark via benchmark_gpu.py (one call per scenario; --no-gpu/--no-cpu
    honoured)."""
    if args.no_gpu and args.no_cpu:
        print("[unified] host backend: both --no-gpu and --no-cpu → skipping host.")
        return False
    ok = True
    for scenario in ctx["scenarios"]:
        cmd = [
            sys.executable,
            BENCHMARK_GPU,
            "--model-dir",
            ctx["model_dir"],
            "--scenario",
            scenario,
            "--warmup",
            args.warmup,
            "--iters",
            args.iters,
            "--subset",
            args.subset,
            "--data",
            args.data,
        ]
        if args.power:
            cmd += ["--power", "--idle-baseline", args.idle_baseline]
        if args.no_gpu:
            cmd.append("--no-gpu")
        if args.no_cpu:
            cmd.append("--no-cpu")
        rc = _run(cmd)
        if rc != 0:
            print(f"[unified] host backend failed on scenario={scenario} (rc={rc}).")
            ok = False
    return ok


def backend_fpga(ctx: dict, args: argparse.Namespace) -> bool:
    """FPGA benchmark over SSH via benchmark_sweep.py (deploy + s0/s1 × compress/full + fetch).

    Re-runs the board sweep; relies on the ZCU102 being reachable. Skip with --no-fpga to reuse the
    FPGA results already in results/benchmark_hardware/.
    """
    if not HARDWARE_DIR.exists():
        print(
            f"[unified] NOTE: {HARDWARE_DIR.relative_to(ROOT)} does not exist — it was archived to "
            "/mnt/vitisAI/DDC_results_archive/2026-08-31/ after the DATE'27 campaign. This backend "
            "will re-create it (TGRS-era schema). For DATE'27 numbers use results/date27/. "
            "Pass --no-fpga to run host-only."
        )
    cmd = [
        sys.executable,
        BENCHMARK_SWEEP,
        "--models",
        ctx["model_name"],
        "--skip-ceiling",
        "--skip-roofline",
        "--iters",
        args.fpga_iters,
        "--subset",
        args.subset,
    ]
    if args.rebuild_cpp:
        cmd.append("--rebuild-cpp")
    rc = _run(cmd)
    if rc != 0:
        print(f"[unified] FPGA backend failed (rc={rc}). Is the ZCU102 reachable? Continuing.")
        return False
    return True


# platform-key → (backend callable, enabled-predicate). Add new HW here.
BACKENDS = {
    "host": (backend_host, lambda a: not (a.no_gpu and a.no_cpu)),
    "fpga": (backend_fpga, lambda a: not a.no_fpga),
}


def resolve_model(args: argparse.Namespace) -> dict:
    """Resolve the model context (model_dir, model_name) from --model-dir's manifest.json."""
    import json

    model_dir = Path(args.model_dir).resolve()
    manifest = model_dir / "manifest.json"
    if not manifest.exists():
        sys.exit(f"manifest.json not found in {model_dir} (expected a compiled model dir).")
    model_name = json.load(open(manifest)).get("model_name")
    if not model_name:
        sys.exit(f"'model_name' missing from {manifest}.")
    return {
        "model_dir": str(model_dir),
        "model_name": model_name,
        "scenarios": [s.strip() for s in args.scenarios.split(",") if s.strip()],
    }


def main() -> None:
    """Entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Compiled FPGA model dir with manifest.json (e.g. results/fpga/active_model/).",
    )
    p.add_argument(
        "--scenarios", default="compress,full", help="Comma-separated (default: compress,full)."
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100, help="Host timed iterations (default: 100).")
    p.add_argument(
        "--fpga-iters", type=int, default=50, help="FPGA timed iterations (default: 50)."
    )
    p.add_argument(
        "--subset", type=int, default=20, help="Patches cycled, all platforms (default: 20)."
    )
    p.add_argument(
        "--data", default="data/processed_hdf5/TSX_spatial_splits_5_256x256/test_sub500_seed42.npy"
    )
    p.add_argument("--power", action="store_true", help="Enable power sampling on all platforms.")
    p.add_argument("--idle-baseline", type=float, default=10.0)
    p.add_argument(
        "--rebuild-cpp",
        action="store_true",
        help="Rebuild the board C++ binary before the FPGA sweep.",
    )
    p.add_argument("--no-gpu", action="store_true", help="Skip the host GPU benchmark.")
    p.add_argument("--no-cpu", action="store_true", help="Skip the host CPU benchmark.")
    p.add_argument(
        "--no-fpga", action="store_true", help="Skip the FPGA sweep (reuse existing results)."
    )
    args = p.parse_args()

    ctx = resolve_model(args)
    print("=" * 64)
    print(" run_unified_benchmark")
    print(f"  model      : {ctx['model_name']}")
    print(f"  scenarios  : {ctx['scenarios']}")
    print("  platforms  : " + ", ".join(k for k, (_, en) in BACKENDS.items() if en(args)))
    print(f"  power      : {args.power}")
    print("=" * 64)

    results = {}
    for key, (backend, enabled) in BACKENDS.items():
        if not enabled(args):
            print(f"\n[unified] {key}: disabled, skipping.")
            continue
        print(f"\n{'=' * 64}\n  Backend: {key}\n{'=' * 64}")
        results[key] = backend(ctx, args)

    print("\n" + "=" * 64)
    print(" Summary")
    for key, ok in results.items():
        print(f"  {key:6s}: {'OK' if ok else 'skipped/failed'}")
    print(f"  host results → {UNIFIED_DIR / ctx['model_name']}")
    print(f"  fpga results → {HARDWARE_DIR / ctx['model_name']}")
    print("  Analyse with notebooks/benchmark_cross_platform_analysis.ipynb")
    print("=" * 64)


if __name__ == "__main__":
    main()
