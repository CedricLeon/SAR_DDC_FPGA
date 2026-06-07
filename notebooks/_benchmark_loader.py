"""_benchmark_loader.py — load benchmark JSON results into DataFrames (all platforms).

ONE loader for both the FPGA-only notebook and the cross-platform notebook. It reads two trees:
    results/benchmark_hardware/<model>/<config>_<scenario>[_dpuN][_entN].json   (FPGA, C++ benchmark_hardware)
    results/benchmark_unified/<model>/baseline_<scenario>_<platform>.json        (host GPU/CPU, benchmark_gpu.py)

Both emit the same aligned schema (config/scenario/stages{mean_ms}/throughput_fps/power/...).
FPGA JSONs have no `platform` field → tagged `platform="fpga"`; host JSONs carry `platform` ("gpu"|"cpu").

Public API:
    load_runs(hardware_dir, unified_dir=None) -> DataFrame (one row per run; pass only hardware_dir for FPGA-only)
    load_stage_breakdowns(hardware_dir, unified_dir=None) -> DataFrame (long: one row per run x stage)
    load_quality_metrics(compiled_dir) -> DataFrame (one row per compiled model)
    join_hw_quality(runs_df, quality_df) -> DataFrame (left join on model_name)
Notebooks discard whatever columns they don't need after loading.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Identity columns — the composite key that must be unique in runs_df.
# Never drop these in groupby; the hygiene check enforces their presence.
# `platform` distinguishes fpga / gpu / cpu (and future HW).
# ---------------------------------------------------------------------------
IDENTITY_COLS = ["platform", "model_name", "config", "scenario", "dpu_cores", "entropy_threads"]

# NN (accelerator) stages vs everything else (host/entropy/normalize) — for the nn/host split.
_NN_STAGES = {"g_a", "h_a", "h_s", "g_s"}


def _iter_run_jsons(*dirs: Path | str):
    """Yield (json_path, model_name) over one or more results trees, skipping meta subdirs."""
    for d in dirs:
        if d is None:
            continue
        for json_path in sorted(Path(d).glob("*/*.json")):
            if json_path.parent.name.startswith("_"):
                continue  # skip _roofline and other meta subdirs
            yield json_path, json_path.parent.name


def load_runs(hardware_dir: Path | str, unified_dir: Path | str | None = None) -> pd.DataFrame:
    """Return one row per benchmark JSON across the FPGA + (optional) host trees.

    Backward-compatible: ``load_runs(benchmark_hardware_dir)`` loads FPGA-only.
    Raises ValueError if two rows share the same composite key (IDENTITY_COLS).
    """
    rows = [_parse_run(p, m) for p, m in _iter_run_jsons(hardware_dir, unified_dir)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    _check_uniqueness(df)
    return df.drop(columns=["_path"]).reset_index(drop=True)


def load_stage_breakdowns(
    hardware_dir: Path | str, unified_dir: Path | str | None = None
) -> pd.DataFrame:
    """Return long-format DataFrame: one row per (run, stage) pair, across both trees.

    Columns: IDENTITY_COLS + arch + stage, mean_ms, std_ms, median_ms, p95_ms, min_ms, max_ms, n.
    Stage names are canonical/bare for both FPGA and host (host-only ops are `host_*`).
    """
    rows = []
    for json_path, model_name in _iter_run_jsons(hardware_dir, unified_dir):
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
                "epd_MERLIN": merlin.get("epd"),
                "mse_MERLIN": merlin.get("mse"),
                "psnr_ADAM": adam.get("psnr"),
                "ssim_ADAM": adam.get("ssim"),
                "epd_ADAM": adam.get("epd"),
                "mse_ADAM": adam.get("mse"),
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
    """Extract identity columns from a benchmark JSON dict.

    `platform` defaults to "fpga" (the C++ benchmark_hardware JSONs carry no platform field);
    host JSONs (benchmark_gpu.py) set it to "gpu"/"cpu".
    """
    m = _ARCH_FROM_MODEL.match(model_name)
    arch = m.group(1) if m else d.get("arch", "unknown")
    return {
        "platform": d.get("platform", "fpga"),
        "model_name": model_name,
        "arch": arch,
        "config": d.get("config"),
        "scenario": d.get("scenario"),
        "dpu_cores": int(d.get("dpu_cores", 1)),
        "entropy_threads": int(d.get("entropy_threads", 1)),
    }


def _norm_power(d: dict, platform: str, lat_ms: float) -> dict:
    """Unified cross-platform power columns (identical names for every platform).

    FPGA: scope = MPSoC SoC power (from power.active.groups.MPSoC); also keeps the FPGA-specific
          VCCINT/DPU_fabric columns (NaN for host rows).
    host: reads the normalized block benchmark_gpu.py emits (active_w/idle_w/...).
    energy_mJ_per_patch = active_W (W) x total_latency_mean_ms (ms) = mJ/patch.
    """
    nan = float("nan")
    power = d.get("power", {}) or {}
    # FPGA-specific rails/groups (kept for the FPGA-only notebook; NaN for host)
    fpga = {
        "idle_VCCINT_W": nan,
        "active_VCCINT_W": nan,
        "idle_MPSoC_W": nan,
        "active_MPSoC_W": nan,
        "idle_DPU_fabric_W": nan,
        "active_DPU_fabric_W": nan,
    }

    if platform == "fpga":
        idle_g = power.get("idle", {}).get("groups", {})
        act_g = power.get("active", {}).get("groups", {})
        idle_r = power.get("idle", {}).get("rails", {})
        act_r = power.get("active", {}).get("rails", {})
        fpga.update(
            idle_VCCINT_W=idle_r.get("VCCINT", {}).get("avg_power_w", nan),
            active_VCCINT_W=act_r.get("VCCINT", {}).get("avg_power_w", nan),
            idle_MPSoC_W=idle_g.get("MPSoC", nan),
            active_MPSoC_W=act_g.get("MPSoC", nan),
            idle_DPU_fabric_W=idle_g.get("DPU_fabric", nan),
            active_DPU_fabric_W=act_g.get("DPU_fabric", nan),
        )
        active_w, idle_w, scope = fpga["active_MPSoC_W"], fpga["idle_MPSoC_W"], "SoC_MPSoC"
    else:  # gpu / cpu — benchmark_gpu.py normalized block (None when RAPL unavailable)
        active_w = power.get("active_w", nan)
        idle_w = power.get("idle_w", nan)
        scope = power.get("power_scope", "host")

    active_w = nan if active_w is None else active_w
    idle_w = nan if idle_w is None else idle_w
    dynamic_w = active_w - idle_w if not (np.isnan(active_w) or np.isnan(idle_w)) else nan
    energy = active_w * lat_ms if not (np.isnan(active_w) or np.isnan(lat_ms)) else nan
    dyn_energy = dynamic_w * lat_ms if not (np.isnan(dynamic_w) or np.isnan(lat_ms)) else nan

    return {
        **fpga,
        "power_scope": scope,
        "active_W": active_w,
        "idle_W": idle_w,
        "dynamic_W": dynamic_w,
        "energy_mJ_per_patch": energy,
        "dynamic_energy_mJ_per_patch": dyn_energy,
    }


def _nn_host_split(d: dict) -> dict:
    """Sum stage means into NN (accelerator) vs host(CPU/entropy/normalize) latency, from
    stages."""
    stages = d.get("stages", {}) or {}
    nn = sum(s.get("mean_ms", 0.0) for k, s in stages.items() if k in _NN_STAGES)
    host = sum(s.get("mean_ms", 0.0) for k, s in stages.items() if k not in _NN_STAGES)
    return {"nn_latency_ms": nn or float("nan"), "host_latency_ms": host or float("nan")}


def _parse_run(json_path: Path, model_name: str) -> dict:
    """Parse a single benchmark JSON file into a flat dict of metrics, prefixed by identity
    columns."""
    with open(json_path) as f:
        d = json.load(f)

    identity = _identity_from_json(d, model_name)
    lat_ms = d.get("total_latency_mean_ms", float("nan"))

    # Bytes statistics (absent for nn_only / entropy_only)
    bpi = d.get("bytes_per_iter")
    if bpi:
        arr = np.array(bpi, dtype=float)
        bytes_mean, bytes_std = float(arr.mean()), float(arr.std())
    else:
        bytes_mean = bytes_std = float("nan")

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
        **_nn_host_split(d),
        **_norm_power(d, identity["platform"], lat_ms),
        "_path": str(json_path),
    }


def _check_uniqueness(df: pd.DataFrame) -> None:
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
