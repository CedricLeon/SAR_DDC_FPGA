#!/usr/bin/env python3
"""Batch FPGA deployment: fetch W&B runs matching filters and deploy each via deploy.py.

    python scripts/fpga/batch_deploy.py --tag <label> [options]
    python scripts/fpga/batch_deploy.py --tag <label> --dry-run
    python scripts/fpga/batch_deploy.py --tag <label> --run-ids abc123 def456 ...

Runs are fetched from W&B and filtered by FILTERS_CONFIG (see USER CONFIGURATION).
For each run:
  - If compiled model already exists in compiled_models/ → skip compile (default).
  - Use --force-recompile to override and always recompile.

Timing is recorded per run. A summary table and a copy-pasteable list of failed IDs
are printed at the end.

Logging:
  Batch coordination log (headers, per-run status, final summary):
      results/fpga/batch_deploy/<tag>_<timestamp>.log
  Individual compile.log / deploy.log per model are still written by deploy.py as usual.
  Note: deploy.py subprocess output appears on the terminal but is NOT duplicated in the
  batch log; it is already captured in each model's own compile.log / deploy.log.
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import wandb
from omegaconf import OmegaConf

# ---- Import constants and helpers from deploy.py (same directory) ----
# We add scripts/fpga/ to sys.path so `import deploy` works without __init__.py files.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy import (
    ARCH_JSON_LOOKUP,
    FPGA_DATASET_SIZE,
    PROJECT_ROOT,
    VITIS_AI_ROOT,
    Tee,
    _make_compiled_model_name,
)

# ============================================================
# USER CONFIGURATION
# ============================================================

ENTITY = "cedric-leonard"
PROJECT = "SAR_DDC_FPGA"

# Filters applied when fetching all runs from W&B.
# Format: list of (dotted_config_key, operator, value) tuples.
# Supported operators: ==, !=, in, not in, is_none, exists, >, <,
#                      is_before, is_after, is_none_or_is_before, is_none_or_is_after
#
# Alternative: W&B supports server-side MongoDB-style filtering via the `filters` kwarg:
#   api.runs(f"{ENTITY}/{PROJECT}", filters={
#       "$and": [
#           {"config.model.net.activation": {"$eq": "relu"}},
#           {"config.model.net.no_output_padding": {"$eq": True}},
#       ]
#   })
# This avoids fetching the full project list, but is unreliable for deeply nested keys
# in W&B's internal flat config storage. Also, complex ops (is_none_or_X, date ranges)
# cannot be expressed server-side and still need client-side logic anyway.
# The tuple system below is consistent with update_wandb_runs.py and gives full
# flexibility at negligible cost for a project of ~hundreds of runs.
FILTERS_CONFIG: List[Tuple] = [
    # Default: only DPU-compatible architectures (relu activation, no output_padding issue)
    ("model.net.activation", "==", "relu"),
    ("model.net.no_output_padding", "==", True),
    ("model.net.no_residual_blocks", "==", True),
    ("seed", "in", [0, 1, 2, 3, 4, 5]),
    ("model.criterion.lmbda", "in", [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]),
]

# ============================================================
# CONSTANTS
# ============================================================

DEPLOY_SCRIPT = Path(__file__).resolve().parent / "deploy.py"
COMPILED_MODELS_DIR = PROJECT_ROOT / "results" / "fpga" / "compiled_models"
BATCH_LOG_DIR = PROJECT_ROOT / "results" / "fpga" / "batch_deploy"

# ============================================================
# FILTER HELPERS
# ============================================================


def check_single_condition(value: Any, op: str, test_value: Any) -> bool:
    """Check a single filter condition (mirrors update_wandb_runs.py for consistency)."""
    if op == "==":
        return value == test_value
    elif op == "!=":
        return value != test_value
    elif op == "in":
        return value in test_value
    elif op == "not in":
        return value not in test_value
    elif op == "is_none":
        return value is None
    elif op == "exists":
        return value is not None
    elif op == ">":
        return value is not None and float(value) > float(test_value)
    elif op == "<":
        return value is not None and float(value) < float(test_value)
    elif op == "is_before":
        if value is None:
            return False
        return datetime.fromisoformat(str(value)) < test_value
    elif op == "is_after":
        if value is None:
            return False
        return datetime.fromisoformat(str(value)) > test_value
    elif op == "is_none_or_is_before":
        if value is None:
            return True
        return datetime.fromisoformat(str(value)) < test_value
    elif op == "is_none_or_is_after":
        if value is None:
            return True
        return datetime.fromisoformat(str(value)) > test_value
    else:
        raise ValueError(f"Unsupported filter op: '{op}'")


def run_matches_filters(run_cfg: dict) -> bool:
    """Return True if the W&B run config satisfies all FILTERS_CONFIG conditions."""
    cfg = OmegaConf.create(run_cfg)
    for key, op, test_value in FILTERS_CONFIG:
        value = OmegaConf.select(cfg, key)
        if not check_single_condition(value, op, test_value):
            return False
    return True


# ============================================================
# HELPERS
# ============================================================


@dataclass
class RunResult:
    """Outcome of a single deploy.py invocation."""

    run_id: str
    wandb_name: str
    model_name: str
    compile_skipped: bool
    duration_s: float = 0.0
    status: str = "pending"  # "success" | "failed"
    error: str = ""


def _run_dir_from_wandb_run(run: Any) -> str:
    """Convert absolute output_dir from W&B config to VITIS_AI_ROOT-relative path.

    W&B stores paths.output_dir as an absolute host path (e.g. /home/.../DDC_FPGA/logs/...).
    deploy.py expects RUN_DIR relative to VITIS_AI_ROOT  (e.g. DDC_FPGA/logs/...).
    """
    abs_path = Path(run.config["paths"]["output_dir"])
    return str(abs_path.relative_to(VITIS_AI_ROOT))


def _compiled_name_from_wandb_run(run: Any) -> str:
    """Derive the expected compiled model name from a W&B run's config."""
    cfg = OmegaConf.create(run.config)
    return _make_compiled_model_name(cfg)


def _is_compiled(model_name: str) -> bool:
    """Return True if compiled_models/<model_name>/ already exists on disk."""
    return (COMPILED_MODELS_DIR / model_name).exists()


def _print_run_table(runs_info: List[Tuple]) -> None:
    """Print a formatted pre-deployment table of all matched runs."""
    header = (
        f"  {'#':<4} {'Run ID':<10}  {'W&B Name':<50}  "
        f"{'λ':<7}  {'seed':<5}  {'Created':<10}  {'Compiled Model':<44}  Status"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, (run, model_name, compiled) in enumerate(runs_info):
        lmbda = run.config.get("lambda", "?")
        seed = run.config.get("seed", "?")
        created_date = str(run.created_at)[:10]  # "YYYY-MM-DD" from ISO timestamp
        status_str = "\033[32m\u2713 compiled\033[0m" if compiled else "\u00b7 to compile"
        print(
            f"  {i + 1:<4} {run.id:<10}  {run.name:<50}  "
            f"{str(lmbda):<7}  {str(seed):<5}  {created_date:<10}  {model_name:<44}  {status_str}"
        )


def _print_summary(results: List[RunResult]) -> None:
    """Print the final batch summary table, timings, and failed run IDs."""
    print(f"\n{'=' * 80}")
    print("  BATCH SUMMARY")
    print(f"{'=' * 80}")
    header = f"  {'#':<4} {'Run ID':<10}  {'Model Name':<50}  {'Skip':<5}  {'Time':>7}  Status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(results):
        skip = "yes" if r.compile_skipped else "no"
        duration = f"{r.duration_s:.0f}s" if r.duration_s > 0 else "-"
        status_str = "\033[32msuccess\033[0m" if r.status == "success" else "\033[31mFAILED\033[0m"
        print(
            f"  {i + 1:<4} {r.run_id:<10}  {r.model_name:<50}  "
            f"{skip:<5}  {duration:>7}  {status_str}"
        )

    total = len(results)
    succeeded = sum(1 for r in results if r.status == "success")
    failed = [r for r in results if r.status == "failed"]
    compiled_count = sum(1 for r in results if not r.compile_skipped and r.status == "success")
    skipped_count = sum(1 for r in results if r.compile_skipped and r.status == "success")

    print(
        f"\n  {succeeded}/{total} runs succeeded  "
        f"({compiled_count} compiled, {skipped_count} skipped compile)"
    )

    durations = [r.duration_s for r in results if r.duration_s > 0]
    if durations:
        total_time = sum(durations)
        mean_time = total_time / len(durations)
        compile_times = [
            r.duration_s for r in results if not r.compile_skipped and r.duration_s > 0
        ]
        skip_times = [r.duration_s for r in results if r.compile_skipped and r.duration_s > 0]
        print(f"  Total time         : {total_time / 60:.1f} min")
        print(f"  Mean/run           : {mean_time:.0f}s")
        if compile_times:
            print(f"  Mean (compile)     : {sum(compile_times) / len(compile_times):.0f}s")
        if skip_times:
            print(f"  Mean (skip-compile): {sum(skip_times) / len(skip_times):.0f}s")

    if failed:
        ids_str = " ".join(r.run_id for r in failed)
        print(f"\n  \033[31mFailed run IDs ({len(failed)}):\033[0m")
        print(f"    {ids_str}")
        print("  \u2192 Redeploy with:")
        print(f"    python scripts/fpga/batch_deploy.py --tag <tag> --run-ids {ids_str}")


# ============================================================
# ENTRYPOINT
# ============================================================


def main() -> None:
    """Main entry point for batch deployment."""
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Batch identity
    parser.add_argument(
        "--tag",
        required=True,
        metavar="TAG",
        help="Short label identifying this batch (used in the log filename, e.g. 'relu_seed0-2').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched runs and exit without deploying.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        metavar="ID",
        default=None,
        help=(
            "Deploy specific W&B run IDs directly, bypassing FILTERS_CONFIG. "
            "Each ID is fetched individually from W&B."
        ),
    )
    parser.add_argument(
        "--force-recompile",
        action="store_true",
        help="Recompile even if compiled model already exists in compiled_models/.",
    )

    # Phase skips — applied globally to every run in the batch
    g_skip = parser.add_argument_group("phase skips (applied to all runs)")
    g_skip.add_argument("--skip-transfer", action="store_true", help="Skip Phase 2 (SCP to FPGA).")
    g_skip.add_argument("--skip-infer", action="store_true", help="Skip Phase 3 (inference).")
    g_skip.add_argument("--skip-fetch", action="store_true", help="Skip Phase 4 (fetch results).")

    # Compile options — applied globally to every run in the batch
    g_compile = parser.add_argument_group("compile options (applied to all runs)")
    g_compile.add_argument(
        "--arch",
        default="ZCU102",
        choices=list(ARCH_JSON_LOOKUP.keys()),
        help="Target DPU architecture (default: ZCU102).",
    )
    g_compile.add_argument("--inspect", action="store_true")
    g_compile.add_argument("--eval-float", action="store_true")
    g_compile.add_argument("--eval-quant", action="store_true")
    g_compile.add_argument("--fast-finetune", action="store_true")
    g_compile.add_argument("--image-graph", action="store_true")

    # Inference options
    g_infer = parser.add_argument_group("inference options")
    g_infer.add_argument(
        "--subset",
        type=int,
        default=FPGA_DATASET_SIZE,
        help=f"Number of test samples for FPGA inference (default: {FPGA_DATASET_SIZE}).",
    )

    args = parser.parse_args()

    # ---- Set up batch log directory ----
    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = BATCH_LOG_DIR / f"{args.tag}_{timestamp}.log"

    # ---- Fetch runs from W&B ----
    api = wandb.Api()
    if args.run_ids:
        print(f"Fetching {len(args.run_ids)} runs by ID from {ENTITY}/{PROJECT}...")
        runs = []
        for rid in args.run_ids:
            try:
                runs.append(api.run(f"{ENTITY}/{PROJECT}/{rid}"))
                print(f"  Fetched {rid}")
            except Exception as exc:
                print(f"  WARNING: Could not fetch run '{rid}': {exc}")
    else:
        print(f"Fetching all runs from {ENTITY}/{PROJECT}...")
        all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
        print(f"  {len(all_runs)} total runs. Applying FILTERS_CONFIG: {FILTERS_CONFIG}")
        runs = [r for r in all_runs if run_matches_filters(r.config)]
        print(f"  \u2192 {len(runs)} runs match.")

    if not runs:
        print("No runs to deploy. Exiting.")
        sys.exit(0)

    # ---- Compute model names + check compiled status ----
    runs_info: List[Tuple] = []
    for run in runs:
        try:
            model_name = _compiled_name_from_wandb_run(run)
            compiled = _is_compiled(model_name)
            runs_info.append((run, model_name, compiled))
        except Exception as exc:
            print(f"  WARNING: Skipping run {run.id} — could not compute model name: {exc}")

    if not runs_info:
        print("No deployable runs after model name resolution. Exiting.")
        sys.exit(0)

    # ---- Print pre-deployment table ----
    print(f"\n{'#' * 70}")
    print(f"  batch_deploy.py  —  tag: {args.tag}  —  {len(runs_info)} runs")
    print(
        f"  arch: {args.arch}  |  subset: {args.subset}  |  force-recompile: {args.force_recompile}"
    )
    skip_str = ", ".join(
        [
            f"transfer={'SKIP' if args.skip_transfer else 'yes'}",
            f"infer={'SKIP' if args.skip_infer else 'yes'}",
            f"fetch={'SKIP' if args.skip_fetch else 'yes'}",
        ]
    )
    print(f"  phases: {skip_str}")
    print(f"{'#' * 70}")
    _print_run_table(runs_info)

    if args.dry_run:
        print("\n[dry-run] Exiting without deploying.")
        sys.exit(0)

    print(f"\nBatch log: {log_path}")
    print("Starting deployment loop...\n")

    # ---- Deploy loop ----
    results: List[RunResult] = []
    with Tee(log_path, verbose=True):
        try:
            for i, (run, model_name, compiled) in enumerate(runs_info):
                skip_compile = compiled and not args.force_recompile

                print(f"\n{'#' * 70}")
                print(f"  Run {i + 1}/{len(runs_info)}: {run.id} ({run.name})")
                print(f"  Model  : {model_name}")
                print(
                    f"  Compile: {'SKIP (already compiled)' if skip_compile else 'YES — will compile'}"
                )
                print(f"{'#' * 70}")

                result = RunResult(
                    run_id=run.id,
                    wandb_name=run.name,
                    model_name=model_name,
                    compile_skipped=skip_compile,
                )

                # Build deploy.py command
                cmd = [
                    sys.executable,
                    str(DEPLOY_SCRIPT),
                    "--run-dir",
                    _run_dir_from_wandb_run(run),
                    "--arch",
                    args.arch,
                    "--subset",
                    str(args.subset),
                ]
                if skip_compile:
                    cmd += ["--skip-compile", "--model-name", model_name]
                else:
                    cmd += ["--wandb-run-id", run.id]
                if args.skip_transfer:
                    cmd.append("--skip-transfer")
                if args.skip_infer:
                    cmd.append("--skip-infer")
                if args.skip_fetch:
                    cmd.append("--skip-fetch")
                if args.eval_float:
                    cmd.append("--eval-float")
                if args.eval_quant:
                    cmd.append("--eval-quant")
                if args.fast_finetune:
                    cmd.append("--fast-finetune")
                if args.inspect:
                    cmd.append("--inspect")
                if args.image_graph:
                    cmd.append("--image-graph")

                print(f"[CMD] {' '.join(cmd)}")

                t0 = time.monotonic()
                try:
                    subprocess.run(cmd, check=True)
                    result.duration_s = time.monotonic() - t0
                    result.status = "success"
                    print(f"\n  \033[32m✓ {run.id}: success in {result.duration_s:.0f}s\033[0m")
                except subprocess.CalledProcessError as exc:
                    result.duration_s = time.monotonic() - t0
                    result.status = "failed"
                    result.error = str(exc)
                    print(
                        f"\n  \033[31m✗ {run.id}: FAILED (exit {exc.returncode}) "
                        f"in {result.duration_s:.0f}s\033[0m"
                    )

                results.append(result)

        finally:
            # Always print summary — even if the batch is interrupted (Ctrl+C)
            if results:
                _print_summary(results)

    if any(r.status == "failed" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
