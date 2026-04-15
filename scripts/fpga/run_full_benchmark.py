#!/usr/bin/env python3
"""Full benchmark orchestrator — GPU/CPU (host) + FPGA (ZCU102) for all scenarios.

Runs the complete benchmark suite for one compiled model across all three platforms:

  Phase 1 — GPU + CPU  : ``benchmark_gpu.py`` × 5 scenarios (this machine)
  Phase 2 — FPGA setup : copy + scp compiled model dir → ZCU102:active_model/
  Phase 3 — FPGA run   : ``benchmark_fpga.py`` × 5 scenarios (SSH to ZCU102)
  Phase 4 — Fetch      : scp JSON results back → ``results/benchmark/<model_name>/``

All results land in ``results/benchmark/<model_name>/``:
  benchmark_gpu_<scenario>.json    ← GPU results
  benchmark_cpu_<scenario>.json    ← CPU results
  benchmark_fpga_<scenario>.json   ← FPGA results

Usage
-----
    # Recommended: point at a compiled model directory
    python scripts/fpga/run_full_benchmark.py \\
        --model-dir results/fpga/active_model/ \\
        --power --idle-baseline 10

    # Specific compiled model
    python scripts/fpga/run_full_benchmark.py \\
        --model-dir results/fpga/compiled_models/ResSHyp-relu_s1_L1000_pt/ \\
        --warmup 20 --iters 100

    # GPU + CPU only (no board access required)
    python scripts/fpga/run_full_benchmark.py \\
        --model-dir results/fpga/active_model/ \\
        --no-fpga

    # FPGA only (GPU+CPU already done, skip transfer if model already on board)
    python scripts/fpga/run_full_benchmark.py \\
        --model-dir results/fpga/active_model/ \\
        --no-gpu --no-cpu --skip-transfer

Estimated runtime (RTX A4000 + ZCU102, --warmup 20 --iters 100)
----------------------------------------------------------------
                        No power   --power --idle-baseline 10
  GPU + CPU (5 scen.)   ~5 min         ~8 min
  FPGA      (5 scen.)   ~4 min         ~5 min
  Transfer  (scp)       ~1 min         ~1 min
  ─────────────────────────────────────────────
  Total                ~10 min        ~14 min
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- Import constants from deploy.py (same directory) ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy import FPGA_BASE_DIR, FPGA_HOST, PROJECT_ROOT

# ---- Script locations ----
BENCHMARK_GPU_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_gpu.py"
BENCHMARK_FPGA_SCRIPT = Path(__file__).resolve().parent / "benchmark_fpga.py"

# GPU/CPU benchmark scenarios (from benchmark_gpu.py)
GPU_SCENARIOS: list[str] = ["full", "compress", "decompress", "nn_only", "entropy_only"]
# FPGA benchmark scenarios (from benchmark_fpga.py)
FPGA_SCENARIOS: list[str] = ["full", "compress", "decompress", "dpu_only", "entropy_only"]

# Rough wall-time estimates per scenario (seconds, 100 iters + 20 warmup baseline).
# Used only for ETA display — not authoritative.
# GPU/CPU times are dominated by entropy coding (~280ms/iter); nn_only is much faster.
# FPGA times are from measured ZCU102 results.
_ETA_S: dict[str, dict[str, int]] = {
    "gpu": {"full": 35, "compress": 25, "decompress": 30, "nn_only": 5, "entropy_only": 35},
    "cpu": {"full": 55, "compress": 38, "decompress": 42, "nn_only": 25, "entropy_only": 35},
    "fpga": {"full": 55, "compress": 30, "decompress": 32, "dpu_only": 24, "entropy_only": 38},
    # Note: all FPGA scenarios now run real‖imag in parallel by default (--no-parallel for baseline)
}


# ============================================================
# Data structure
# ============================================================


@dataclass
class ScenarioResult:
    platform: str  # "gpu" | "cpu" | "fpga"
    scenario: str
    status: str = "pending"  # "ok" | "failed" | "skipped"
    wall_s: float = 0.0  # subprocess wall time (combined GPU+CPU if run together)
    latency_ms: float | None = None  # latency_total_mean_ms from JSON


# ============================================================
# Utilities
# ============================================================


def _hms(s: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, sec = divmod(int(max(s, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _scale_eta(
    platform: str, scenarios: list[str], iters: int, warmup: int, power: bool, idle_baseline: float
) -> str:
    """Return a human-readable ETA string for a set of scenarios on one platform."""
    scale = (iters + warmup) / 120.0
    base = sum(_ETA_S.get(platform, {}).get(s, 35) for s in scenarios)
    idle = idle_baseline * len(scenarios) if power else 0.0
    return _hms(base * scale + idle)


def _hdr(msg: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {msg}")
    print(f"{'=' * 66}")


def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Print then run a command."""
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kwargs)


def _latency_from_json(path: Path) -> float | None:
    """Return ``latency_total_mean_ms`` from a benchmark JSON file, or None."""
    try:
        with open(path) as f:
            return json.load(f).get("latency_total_mean_ms")
    except Exception:
        return None


# ============================================================
# Phase 1 — GPU + CPU
# ============================================================


def phase_gpu_cpu(
    model_dir: Path,
    out_dir: Path,
    warmup: int,
    iters: int,
    power: bool,
    power_hz: float,
    idle_baseline: float,
    no_gpu: bool,
    no_cpu: bool,
) -> list[ScenarioResult]:
    """Run benchmark_gpu.py for every GPU scenario.

    One subprocess per scenario.
    """
    results: list[ScenarioResult] = []
    scale = (iters + warmup) / 120.0

    for scenario in GPU_SCENARIOS:
        gpu_s = 0.0 if no_gpu else _ETA_S["gpu"].get(scenario, 35) * scale
        cpu_s = 0.0 if no_cpu else _ETA_S["cpu"].get(scenario, 35) * scale
        idle_s = idle_baseline * (int(not no_gpu) + int(not no_cpu)) if power else 0.0
        print(f"\n  → GPU/CPU  '{scenario}'  (est. {_hms(gpu_s + cpu_s + idle_s)})")

        cmd = [
            sys.executable,
            str(BENCHMARK_GPU_SCRIPT),
            "--model-dir",
            str(model_dir),
            "--scenario",
            scenario,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--output-dir",
            str(out_dir),
        ]
        if power:
            cmd += ["--power", "--power-hz", str(power_hz)]
        if idle_baseline > 0 and power:
            cmd += ["--idle-baseline", str(idle_baseline)]
        if no_gpu:
            cmd.append("--no-gpu")
        if no_cpu:
            cmd.append("--no-cpu")

        t0 = time.monotonic()
        proc = _run(cmd)
        wall = time.monotonic() - t0
        ok = proc.returncode == 0

        if not no_gpu:
            jpath = out_dir / f"benchmark_gpu_{scenario}.json"
            results.append(
                ScenarioResult(
                    "gpu",
                    scenario,
                    "ok" if ok else "failed",
                    wall,
                    _latency_from_json(jpath),
                )
            )
        if not no_cpu:
            jpath = out_dir / f"benchmark_cpu_{scenario}.json"
            results.append(
                ScenarioResult(
                    "cpu",
                    scenario,
                    "ok" if ok else "failed",
                    wall,
                    _latency_from_json(jpath),
                )
            )
    return results


# ============================================================
# Phase 2 — FPGA setup (copy script + scp)
# ============================================================


def phase_fpga_setup(model_dir: Path, skip_transfer: bool) -> None:
    """Ensure benchmark_fpga.py is current, then scp model dir to ZCU102:active_model/.

    The transfer mirrors deploy.py's phase_transfer: wipe the remote active_model/,
    then scp -r the local model directory.  ``scp -r`` follows symlinks, so pointing
    at ``results/fpga/active_model`` (a symlink) works correctly.
    """
    # Ensure the latest helper scripts are present in the model dir before transfer.
    # Copy each script from its corresponding source file in this scripts/fpga/ folder.
    SCRIPTS_DIR = Path(__file__).resolve().parent
    scripts_src = {
        "benchmark_fpga.py": BENCHMARK_FPGA_SCRIPT,
        "inference_hybrid.py": SCRIPTS_DIR / "inference_hybrid.py",
        "inference_utils.py": SCRIPTS_DIR / "inference_utils.py",
    }

    for script, src in scripts_src.items():
        local_script = model_dir / script
        if not local_script.exists():
            print(f"  {script} not in model dir — copying from {src}")
        else:
            print(f"  Updating {script} to current version in {model_dir}")
        shutil.copy2(src, local_script)

    if skip_transfer:
        print("  --skip-transfer: skipping scp (assuming ZCU102 already has this model).")
        return

    remote_active = f"{FPGA_BASE_DIR}/active_model"
    print(f"  Wiping {FPGA_HOST}:{remote_active} ...")
    _run(["ssh", FPGA_HOST, f"rm -rf {remote_active}"], check=True)

    print(f"  scp -r {model_dir} → {FPGA_HOST}:{FPGA_BASE_DIR}/")
    # scp -r on a symlink follows it, so we always ship the real directory.
    _run(["scp", "-r", str(model_dir), f"{FPGA_HOST}:{FPGA_BASE_DIR}/"], check=True)

    # Rename the uploaded directory to "active_model" if it was not already named that.
    # (model_dir.name is e.g. "ResSHyp-relu_s1_L1000_pt"; deploy.py transfers the
    # resolved symlink target which is a named directory.)
    uploaded_name = model_dir.resolve().name  # real dir name after symlink resolution
    if uploaded_name != "active_model":
        print(f"  Renaming {FPGA_BASE_DIR}/{uploaded_name} → {remote_active} on board ...")
        _run(
            ["ssh", FPGA_HOST, f"mv {FPGA_BASE_DIR}/{uploaded_name} {remote_active}"],
            check=True,
        )
    print("  Transfer complete.")


# ============================================================
# Phase 3 — FPGA benchmarks (SSH)
# ============================================================


def phase_fpga_run(
    warmup: int,
    iters: int,
    power: bool,
    power_hz: float,
    idle_baseline: float,
) -> list[ScenarioResult]:
    """SSH into ZCU102 and run benchmark_fpga.py for every FPGA scenario."""
    results: list[ScenarioResult] = []
    remote_dir = f"{FPGA_BASE_DIR}/active_model"
    scale = (iters + warmup) / 120.0

    for scenario in FPGA_SCENARIOS:
        est_s = _ETA_S["fpga"].get(scenario, 35) * scale + (idle_baseline if power else 0.0)
        print(f"\n  → FPGA  '{scenario}'  (est. {_hms(est_s)})")

        cmd_parts = [
            f"export PYTHONPATH=$PYTHONPATH:{FPGA_BASE_DIR} &&",
            f"cd {remote_dir} &&",
            "python3 benchmark_fpga.py",
            "--xmodel",
            "./*.xmodel",
            "--scenario",
            scenario,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--collect-hw-meta",
        ]
        if power:
            cmd_parts += ["--power", "--power-hz", str(int(power_hz))]
        if idle_baseline > 0 and power:
            cmd_parts += ["--idle-baseline", str(int(idle_baseline))]

        t0 = time.monotonic()
        proc = _run(["ssh", FPGA_HOST, " ".join(cmd_parts)])
        wall = time.monotonic() - t0

        results.append(
            ScenarioResult(
                "fpga",
                scenario,
                "ok" if proc.returncode == 0 else "failed",
                wall,
            )
        )
    return results


# ============================================================
# Phase 4 — Fetch FPGA results
# ============================================================


def phase_fetch(out_dir: Path) -> None:
    """SCP all benchmark JSON files from ZCU102:active_model/results/ to out_dir."""
    remote_glob = f"{FPGA_HOST}:{FPGA_BASE_DIR}/active_model/results/benchmark_*.json"
    print(f"  Fetching: {remote_glob}")
    print(f"  Into    : {out_dir}/")
    _run(["scp", remote_glob, f"{out_dir}/"])


# ============================================================
# Summary
# ============================================================


def print_summary(
    model_name: str,
    out_dir: Path,
    results: list[ScenarioResult],
    total_start: float,
) -> None:
    _hdr(f"BENCHMARK COMPLETE — {model_name}")

    by_platform: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_platform.setdefault(r.platform, []).append(r)

    for platform, plist in by_platform.items():
        print(f"\n  {platform.upper()}:")
        for r in plist:
            # Try to read latency from JSON if not already populated
            lat = r.latency_ms
            if lat is None:
                if r.platform == "fpga":
                    p = out_dir / f"benchmark_fpga_{r.scenario}.json"
                else:
                    p = out_dir / f"benchmark_{r.platform}_{r.scenario}.json"
                lat = _latency_from_json(p)

            icon = {"ok": "✓", "failed": "✗", "skipped": "—"}.get(r.status, "?")
            lat_str = f"  {lat:8.1f} ms/patch" if lat is not None else "           —"
            t_str = f"  ({_hms(r.wall_s)})" if r.wall_s > 0 else ""
            c = (
                "\033[32m"
                if r.status == "ok"
                else ("\033[33m" if r.status == "skipped" else "\033[31m")
            )
            print(f"    {c}{icon}\033[0m  {r.scenario:<16}{lat_str}{t_str}")

    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"\n  \033[31m{len(failed)} scenario(s) failed:\033[0m")
        for r in failed:
            print(f"    {r.platform}:{r.scenario}")

    total = time.monotonic() - total_start
    print(f"\n  Results : {out_dir}")
    print(f"  Runtime : {_hms(total)}")
    print(f"{'=' * 66}")


# ============================================================
# Entrypoint
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Path to a compiled FPGA model directory (must contain manifest.json "
            "and a .xmodel file). Accepts the active_model symlink directly."
        ),
    )

    g = parser.add_argument_group("measurement settings")
    g.add_argument(
        "--warmup", type=int, default=20, help="Warmup iterations per scenario (default: 20)."
    )
    g.add_argument(
        "--iters", type=int, default=100, help="Measured iterations per scenario (default: 100)."
    )
    g.add_argument(
        "--power",
        action="store_true",
        help=(
            "Enable power measurement. " "GPU: nvidia-smi + Intel RAPL.  FPGA: INA226 via sysfs."
        ),
    )
    g.add_argument(
        "--idle-baseline",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Capture idle power for N seconds before each scenario (requires --power). "
            "Recommended: 10."
        ),
    )
    g.add_argument(
        "--power-hz-gpu",
        type=float,
        default=10.0,
        help="GPU/CPU power polling frequency in Hz (default: 10).",
    )
    g.add_argument(
        "--power-hz-fpga",
        type=float,
        default=50.0,
        help="FPGA INA226 polling frequency in Hz (default: 50).",
    )

    g2 = parser.add_argument_group("platform / phase skips")
    g2.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmark.")
    g2.add_argument("--no-cpu", action="store_true", help="Skip CPU benchmark.")
    g2.add_argument("--no-fpga", action="store_true", help="Skip FPGA benchmark.")
    g2.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Skip scp transfer to ZCU102 (model already up to date on board).",
    )

    args = parser.parse_args()
    total_start = time.monotonic()

    # ---- Validate model dir ----
    model_dir = Path(args.model_dir).resolve()
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {model_dir}")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest: dict[str, Any] = json.load(f)
    model_name: str = manifest["model_name"]

    if not args.no_fpga:
        xmodel_files = list(model_dir.glob("*.xmodel"))
        if not xmodel_files:
            print(f"Error: no .xmodel file found in {model_dir}")
            sys.exit(1)

    # ---- Output dir ----
    out_dir = PROJECT_ROOT / "results" / "benchmark" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ETA estimates ----
    do_gpu_cpu = not (args.no_gpu and args.no_cpu)
    do_fpga = not args.no_fpga

    gpu_eta = (
        ""
        if args.no_gpu
        else _scale_eta(
            "gpu", GPU_SCENARIOS, args.iters, args.warmup, args.power, args.idle_baseline
        )
    )
    cpu_eta = (
        ""
        if args.no_cpu
        else _scale_eta(
            "cpu", GPU_SCENARIOS, args.iters, args.warmup, args.power, args.idle_baseline
        )
    )
    fpga_eta = (
        ""
        if args.no_fpga
        else _scale_eta(
            "fpga", FPGA_SCENARIOS, args.iters, args.warmup, args.power, args.idle_baseline
        )
    )

    combined_gpu_cpu = (
        _hms(
            sum(
                _ETA_S.get(p, {}).get(s, 35)
                for p in ([] if args.no_gpu else ["gpu"]) + ([] if args.no_cpu else ["cpu"])
                for s in GPU_SCENARIOS
            )
            * (args.iters + args.warmup)
            / 120.0
            + (
                args.idle_baseline
                * len(GPU_SCENARIOS)
                * (int(not args.no_gpu) + int(not args.no_cpu))
                if args.power
                else 0
            )
        )
        if do_gpu_cpu
        else ""
    )

    platforms = [
        p
        for p, skip in [("GPU", args.no_gpu), ("CPU", args.no_cpu), ("FPGA", args.no_fpga)]
        if not skip
    ]

    # ---- Print run plan ----
    print(f"\n{'#' * 66}")
    print("  run_full_benchmark.py")
    print(f"  Model     : {model_name}")
    print(f"  Model dir : {model_dir}")
    print(f"  Output    : {out_dir}")
    print(f"  Warmup    : {args.warmup}   Iters  : {args.iters}")
    print(
        f"  Power     : {'ON  (GPU %.0fHz, FPGA %.0fHz)' % (args.power_hz_gpu, args.power_hz_fpga) if args.power else 'OFF'}"
    )
    if args.idle_baseline > 0 and args.power:
        print(f"  Idle base : {args.idle_baseline}s per scenario")
    print(f"  Platforms : {', '.join(platforms) if platforms else '(none — nothing to do)'}")
    if not platforms:
        print("  Nothing to do. Exiting.")
        sys.exit(0)
    if combined_gpu_cpu:
        print(f"  Est. GPU+CPU : {combined_gpu_cpu}")
    if fpga_eta:
        print(f"  Est. FPGA    : {fpga_eta}")
    print(f"{'#' * 66}")

    all_results: list[ScenarioResult] = []

    # ============================================================
    # Phase 1 — GPU + CPU
    # ============================================================
    if do_gpu_cpu:
        _hdr(f"Phase 1 — GPU + CPU  ({len(GPU_SCENARIOS)} scenarios, est. {combined_gpu_cpu})")
        r = phase_gpu_cpu(
            model_dir=model_dir,
            out_dir=out_dir,
            warmup=args.warmup,
            iters=args.iters,
            power=args.power,
            power_hz=args.power_hz_gpu,
            idle_baseline=args.idle_baseline,
            no_gpu=args.no_gpu,
            no_cpu=args.no_cpu,
        )
        all_results.extend(r)
    else:
        print("\n  Phase 1 skipped (--no-gpu and --no-cpu).")

    # ============================================================
    # Phase 2 — FPGA setup
    # ============================================================
    if do_fpga:
        _hdr(f"Phase 2 — FPGA setup  (→ {FPGA_HOST}:{FPGA_BASE_DIR}/active_model/)")
        phase_fpga_setup(model_dir, args.skip_transfer)

        # ============================================================
        # Phase 3 — FPGA benchmarks
        # ============================================================
        _hdr(f"Phase 3 — FPGA benchmark  ({len(FPGA_SCENARIOS)} scenarios, est. {fpga_eta})")
        r = phase_fpga_run(
            warmup=args.warmup,
            iters=args.iters,
            power=args.power,
            power_hz=args.power_hz_fpga,
            idle_baseline=args.idle_baseline,
        )
        all_results.extend(r)

        # ============================================================
        # Phase 4 — Fetch results
        # ============================================================
        _hdr("Phase 4 — Fetch FPGA results")
        phase_fetch(out_dir)
    else:
        print("\n  Phases 2–4 skipped (--no-fpga).")

    # ============================================================
    # Summary
    # ============================================================
    print_summary(model_name, out_dir, all_results, total_start)


if __name__ == "__main__":
    main()
