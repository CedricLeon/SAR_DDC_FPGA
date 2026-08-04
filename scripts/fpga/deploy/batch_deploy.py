#!/usr/bin/env python3
"""Batch FPGA deployment: fetch W&B runs matching filters and deploy each via deploy.py.

    python scripts/fpga/deploy/batch_deploy.py --config scripts/fpga/deploy/batch_deploy_configs/<cfg>.yaml --tag <label> [options]
    python scripts/fpga/deploy/batch_deploy.py --config scripts/fpga/deploy/batch_deploy_configs/<cfg>.yaml --tag <label> --dry-run
    python scripts/fpga/deploy/batch_deploy.py --tag <label> --run-ids abc123 def456 ...

Pass --config with a path to a YAML filter config (see scripts/fpga/deploy/batch_deploy_configs/).
Either --config or --run-ids must be provided.
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
from typing import Any, Dict, List, Optional, Tuple

import wandb
import yaml
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
    ensure_cpp_binary,
)

# ============================================================
# W&B SETUP
# ============================================================

ENTITY = "cedric-leonard"
PROJECT = "SAR_DDC_FPGA"

# ============================================================
# FILTER CONFIG LOADER
# ============================================================


def load_filters_config(config_path: str) -> Tuple[str, List[Tuple]]:
    """Load a YAML filter config and return (description, filters_as_tuples).

    ``config_path`` must be a path to a YAML file (absolute or relative to CWD).
    Filter configs live in scripts/fpga/deploy/batch_deploy_configs/.

    Each YAML filter entry has keys ``field``, ``op``, ``value`` and is
    converted to the tuple ``(field, op, value)`` expected by
    ``run_matches_filters``.

    Supported operators: ==, !=, in, not in, is_none, exists, >, <,
                         contains, not contains,
                         is_before, is_after, is_none_or_is_before,
                         is_none_or_is_after.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Filter config not found: '{config_path}'")

    with open(p) as f:
        raw = yaml.safe_load(f)

    description = raw.get("description", p.stem)
    filters: List[Tuple] = [
        (entry["field"], entry["op"], entry["value"]) for entry in raw.get("filters", [])
    ]
    return description, filters


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
        if isinstance(test_value, (int, float)):
            try:
                return float(value) == float(test_value)
            except (TypeError, ValueError):
                pass
        return value == test_value
    elif op == "!=":
        if isinstance(test_value, (int, float)):
            try:
                return float(value) != float(test_value)
            except (TypeError, ValueError):
                pass
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
    elif op == "contains":
        return value is not None and str(test_value) in str(value)
    elif op == "not contains":
        return value is None or str(test_value) not in str(value)
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


def run_matches_filters(run: Any, filters: List[Tuple]) -> bool:
    """Return True if the W&B run satisfies all filter conditions.

    Keys prefixed with ``run.`` are resolved from the W&B run object's attributes
    (e.g. ``run.name``, ``run.id``).  All other keys are resolved from ``run.config``
    via OmegaConf.
    """
    cfg = OmegaConf.create(run.config)
    for key, op, test_value in filters:
        if key.startswith("run."):
            attr = key[len("run.") :]
            value = getattr(run, attr, None)
        else:
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
        print(f"    python scripts/fpga/deploy/batch_deploy.py --tag <tag> --run-ids {ids_str}")


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
        "--config",
        metavar="NAME_OR_PATH",
        default=None,
        help=(
            "Path to a YAML filter config (absolute or relative to CWD). "
            "Configs live in scripts/fpga/deploy/batch_deploy_configs/. "
            "Required unless --run-ids is given."
        ),
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
            "Deploy specific W&B run IDs directly, bypassing filter config. "
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
    g_infer.add_argument(
        "--skip-test-set",
        action="store_true",
        help=(
            "Pass --skip-test-set to each deploy.py call: skip the test-subset phase "
            "and run only the Hamburg tile evaluation. Useful for fast tile re-evaluation "
            "after a binary fix."
        ),
    )
    g_infer.add_argument(
        "--save-recons",
        action="store_true",
        help=(
            "Pass --save-recons to each deploy.py call: store every test-subset "
            "reconstruction so future metric changes can be re-scored off-board "
            "(~69 MB compressed per model)."
        ),
    )

    args = parser.parse_args()

    # ---- Validate: need --config or --run-ids ----
    if not args.run_ids and not args.config:
        parser.error("Either --config <name> or --run-ids <id ...> is required.")

    # ---- Load filter config (not needed when --run-ids is used) ----
    filters: List[Tuple] = []
    config_description = ""
    if args.config:
        config_description, filters = load_filters_config(args.config)
        print(f"Filter config: {args.config!r} \u2014 {config_description}")

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
        print(f"  {len(all_runs)} total runs \u2014 applying {len(filters)} filter(s)...")
        runs = [r for r in all_runs if run_matches_filters(r, filters)]
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
    if config_description:
        print(f"  config: {args.config!r}  ({config_description})")
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

    print("\nEnsuring C++ inference binary is current on board...")
    ensure_cpp_binary()

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
                if args.skip_test_set:
                    cmd.append("--skip-test-set")
                if args.save_recons:
                    cmd.append("--save-recons")

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
