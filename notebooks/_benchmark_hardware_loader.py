"""_benchmark_hardware_loader.py — load C++ benchmark_hardware JSON results into DataFrames.

Canonical storage layout expected by these functions:
    results/benchmark_hardware/<model_name>/<config>_<scenario>[_dpuN][_entN].json

Three public loader functions + one join utility:
    load_runs(results_dir)             -> DataFrame (one row per benchmark run)
    load_stage_breakdowns(results_dir) -> DataFrame (long format: one row per run x stage)
    load_quality_metrics(compiled_dir) -> DataFrame (one row per compiled model)
    join_hw_quality(runs_df, quality_df) -> DataFrame (left join on model_name)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Identity columns — the composite key that must be unique in runs_df.
# Never drop these in groupby; the hygiene check enforces their presence.
# ---------------------------------------------------------------------------
IDENTITY_COLS = ["model_name", "config", "scenario", "dpu_cores", "entropy_threads"]


def load_runs(results_dir: Path | str) -> pd.DataFrame:
    """Return one row per benchmark JSON found under results_dir/<model>/*.json.

    Raises ValueError if two rows share the same composite key (IDENTITY_COLS).
    """
    results_dir = Path(results_dir)
    rows = []
    for json_path in sorted(results_dir.glob("*/*.json")):
        if json_path.parent.name.startswith("_"):
            continue  # skip _roofline and other meta subdirs
        model_name = json_path.parent.name
        row = _parse_run(json_path, model_name)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    _check_uniqueness(df, json_paths={r["_path"]: r for r in rows})
    df = df.drop(columns=["_path"])
    return df.reset_index(drop=True)


def load_stage_breakdowns(results_dir: Path | str) -> pd.DataFrame:
    """Return long-format DataFrame: one row per (run, stage) pair.

    Columns: IDENTITY_COLS + model_name, arch + stage, mean_ms, std_ms,
             median_ms, p95_ms, min_ms, max_ms, n
    """
    results_dir = Path(results_dir)
    rows = []
    for json_path in sorted(results_dir.glob("*/*.json")):
        if json_path.parent.name.startswith("_"):
            continue
        model_name = json_path.parent.name
        with open(json_path) as f:
            d = json.load(f)
        identity = _identity_from_json(d, model_name)
        for stage_name, stats in d.get("stages", {}).items():
            rows.append(
                {
                    **identity,
                    "stage": stage_name,
                    "mean_ms": stats.get("mean_ms"),
                    "std_ms": stats.get("std_ms"),
                    "median_ms": stats.get("median_ms"),
                    "p95_ms": stats.get("p95_ms"),
                    "min_ms": stats.get("min_ms"),
                    "max_ms": stats.get("max_ms"),
                    "n": stats.get("n"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


# Architecture label map: xmodel filename stem → canonical arch name.
# The arch field in JSON only encodes the base class (SHyp / FP), not
# whether residual blocks are present, so we derive arch from model_name.
_ARCH_FROM_MODEL = re.compile(r"^(ResSHyp|SHyp|ResFP|FP)")


def load_quality_metrics(compiled_dir: Path | str) -> pd.DataFrame:
    """Return one row per compiled model that has a results/metrics.json.

    compiled_dir should point to results/fpga/compiled_models/.
    """
    compiled_dir = Path(compiled_dir)
    rows = []
    for metrics_path in sorted(compiled_dir.glob("*/results/metrics.json")):
        model_name = metrics_path.parent.parent.name
        m = _ARCH_FROM_MODEL.match(model_name)
        arch = m.group(1) if m else "unknown"
        lam_match = re.search(r"_L(\d+)_", model_name)
        seed_match = re.search(r"_(s\d+)_", model_name)
        with open(metrics_path) as f:
            d = json.load(f)
        merlin = d.get("MERLIN", {})
        adam = d.get("ADAM", {})
        recon = d.get("recon", {})
        rows.append(
            {
                "model_name": model_name,
                "arch": arch,
                "lambda": int(lam_match.group(1)) if lam_match else None,
                "seed": seed_match.group(1) if seed_match else None,
                "bpp": merlin.get("bpp"),
                "psnr_MERLIN": merlin.get("psnr"),
                "ssim_MERLIN": merlin.get("ssim"),
                "psnr_ADAM": adam.get("psnr"),
                "ssim_ADAM": adam.get("ssim"),
                "enl_recon": recon.get("enl"),
                "ratio_enl_recon": recon.get("ratio_enl"),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def join_hw_quality(runs_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join runs_df with quality_df on model_name.

    Warns about benchmarked models with no quality data and vice versa.
    """
    bench_models = set(runs_df["model_name"].unique())
    quality_models = set(quality_df["model_name"].unique())

    missing_quality = bench_models - quality_models
    if missing_quality:
        print(f"[warn] benchmarked models without quality data: {sorted(missing_quality)}")

    extra_quality = quality_models - bench_models
    if extra_quality:
        print(f"[info] quality data available but not yet benchmarked: {sorted(extra_quality)}")

    return runs_df.merge(quality_df, on="model_name", how="left", suffixes=("", "_quality"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_from_json(d: dict, model_name: str) -> dict:
    """Extract identity columns from a benchmark JSON dict."""
    m = _ARCH_FROM_MODEL.match(model_name)
    arch = m.group(1) if m else d.get("arch", "unknown")
    return {
        "model_name": model_name,
        "arch": arch,
        "config": d.get("config"),
        "scenario": d.get("scenario"),
        "dpu_cores": int(d.get("dpu_cores", 1)),
        "entropy_threads": int(d.get("entropy_threads", 1)),
    }


def _parse_run(json_path: Path, model_name: str) -> dict:
    """Parse a single benchmark JSON file into a flat dict of metrics, prefixed by identity
    columns."""
    with open(json_path) as f:
        d = json.load(f)

    identity = _identity_from_json(d, model_name)

    # Bytes statistics (absent for nn_only / entropy_only)
    bpi = d.get("bytes_per_iter")
    if bpi:
        arr = np.array(bpi, dtype=float)
        bytes_mean = float(arr.mean())
        bytes_std = float(arr.std())
    else:
        bytes_mean = float("nan")
        bytes_std = float("nan")

    # Power — nested: power.{idle,active}.groups.<name> (float) and .rails.<name>.avg_power_w
    power = d.get("power", {})
    idle = power.get("idle", {})
    act = power.get("active", {})
    idle_groups = idle.get("groups", {})
    act_groups = act.get("groups", {})
    idle_rails = idle.get("rails", {})
    act_rails = act.get("rails", {})

    idle_v = idle_rails.get("VCCINT", {}).get("avg_power_w", float("nan"))
    act_v = act_rails.get("VCCINT", {}).get("avg_power_w", float("nan"))
    idle_m = idle_groups.get("MPSoC", float("nan"))
    act_m = act_groups.get("MPSoC", float("nan"))
    idle_d = idle_groups.get("DPU_fabric", float("nan"))
    act_d = act_groups.get("DPU_fabric", float("nan"))

    lat_ms = d.get("total_latency_mean_ms", float("nan"))
    energy = act_m * lat_ms if not (np.isnan(act_m) or np.isnan(lat_ms)) else float("nan")

    return {
        **identity,
        "warmup": d.get("warmup"),
        "iters": d.get("iters"),
        "subset_patches": d.get("subset_patches"),
        "evaluated_at": d.get("evaluated_at"),
        "wall_time_s": d.get("wall_time_s"),
        "throughput_fps": d.get("throughput_fps"),
        "total_latency_mean_ms": lat_ms,
        "bytes_per_iter_mean": bytes_mean,
        "bytes_per_iter_std": bytes_std,
        "idle_VCCINT_W": idle_v,
        "active_VCCINT_W": act_v,
        "idle_MPSoC_W": idle_m,
        "active_MPSoC_W": act_m,
        "idle_DPU_fabric_W": idle_d,
        "active_DPU_fabric_W": act_d,
        "energy_mJ_per_patch": energy,
        "_path": str(json_path),
    }


def _check_uniqueness(df: pd.DataFrame, json_paths: dict) -> None:
    """Raise ValueError if any two rows share the same composite key (IDENTITY_COLS)."""
    dupes = df[df.duplicated(subset=IDENTITY_COLS, keep=False)]
    if dupes.empty:
        return
    msg_lines = ["Composite key collision — two JSON files share the same identity:"]
    for _, row in dupes.iterrows():
        msg_lines.append(f"  {row['_path']}  →  ({', '.join(str(row[c]) for c in IDENTITY_COLS)})")
    raise ValueError("\n".join(msg_lines))


def groupby_check(df: pd.DataFrame, groupby: list[str]) -> None:
    """Raise if any IDENTITY_COL is silently dropped by the groupby spec."""
    missing = [c for c in IDENTITY_COLS if c not in groupby and c in df.columns]
    if missing:
        raise ValueError(
            f"groupby drops identity columns {missing}. "
            "Either include them or explicitly document why averaging across them is valid."
        )
