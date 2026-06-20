"""Shared plotting/convention kit for the SAR-DDC analysis notebooks.

Single source of truth for: the Okabe-Ito colour palette (replaces plots_colors.json),
naming conventions (SH/ResSH labels, platform/stage order), the per-architecture
production learning rate, W&B/FPGA quality loaders, the manuscript export helper,
and the tile rendering utilities used by reconstruction_visualization.ipynb.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ======================================================================================
# 1. Paths & env
# ======================================================================================
ROOT_DIR = Path(__file__).resolve().parents[1]  # notebooks/_plotkit.py -> project root
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

WANDB_CSV = ROOT_DIR / "notebooks" / "SAR_DDC_FPGA_all_runs_WandB.csv"
COMPILED_MODELS_DIR = ROOT_DIR / "results" / "fpga" / "compiled_models"
BENCH_HW_DIR = ROOT_DIR / "results" / "benchmark_hardware"
BENCH_UNIFIED_DIR = ROOT_DIR / "results" / "benchmark_unified"
PLOTS_DIR = ROOT_DIR / "results" / "plots"
MANUSCRIPT_DIR = ROOT_DIR / "LaTeX" / "SAR_DDC_FPGA_TGRS_2026" / "figures" / "images"

# ANSI shortcuts (kept identical to the per-notebook definitions)
r, g, b, y, e = "\033[31m", "\033[32m", "\033[34m", "\033[33m", "\033[0m"

# ======================================================================================
# 2. Palette (Okabe-Ito, colourblind-safe). Migrated verbatim from plots_colors.json.
# ======================================================================================
PALETTE: dict[str, dict[str, str]] = {
    "platforms": {
        "fpga_dynamic": "#009E73",
        "fpga_idle": "#85d4bc",
        "gpu_dynamic": "#0072B2",
        "gpu_idle": "#7ab6d9",
        "cpu_dynamic": "#D55E00",
        "cpu_idle": "#ebb99a",
    },
    "fpga_power_groups": {
        "PL": "#003f5c",
        "PS": "#2f9e6e",
        "MGT": "#7fccb0",
        "peripherals": "#b2dfdb",
        "TBP_line": "#D55E00",
    },
    "gpu_power_domains": {
        "gpu_board_dynamic": "#0072B2",
        "gpu_board_idle": "#7ab6d9",
        "cpu_rapl_dynamic": "#D55E00",
        "cpu_rapl_idle": "#ebb99a",
    },
    "nn_vs_cpu_domain": {"nn_dynamic": "#0072B2", "cpu_dynamic": "#D55E00"},
    "latency_steps": {
        "nn_g_a": "#003f5c",
        "nn_h_a": "#0072B2",
        "nn_h_s": "#56B4E9",
        "nn_g_s": "#a8d4f0",
        "cpu_eb_compress": "#8B0000",
        "cpu_eb_decompress": "#CC4444",
        "cpu_gc_compress": "#D55E00",
        "cpu_gc_decompress": "#E69F00",
        "cpu_concat_abs": "#aab7b8",
        "cpu_split_y_hat": "#d5dbdb",
    },
    "architectures": {  # ResSHyp=green, SHyp=orange, ResFP=blue, FP=mauve
        "ResSHyp": "#009E73",
        "SHyp": "#E69F00",
        "ResFP": "#0072B2",
        "FP": "#CC79A7",
    },
    "activations": {"relu": "#009E73", "gdn": "#CC79A7"},
    "output_padding": {"with_out_pad": "#0072B2", "no_out_pad": "#D55E00"},
    "metrics": {"psnr": "#0072B2", "ssim": "#009E73", "epd": "#E69F00", "error_bars": "#555555"},
    "compare_gpu_fpga": {"gpu": "#0072B2", "fpga": "#009E73"},
    "bench_configs": {
        "s0": "#003f5c",
        "s1": "#0072B2",
        "nn_only": "#56B4E9",
        "entropy_only": "#E69F00",
        "roofline": "#D55E00",
    },
    "bench_scenarios": {"compress": "#009E73", "full": "#CC79A7"},
    "cpp_stages": {
        "normalize": "#aab7b8",
        "g_a": "#003f5c",
        "h_a": "#0072B2",
        "h_s": "#56B4E9",
        "g_s": "#a8d4f0",
        "eb_compress": "#8B0000",
        "eb_decompress": "#CC4444",
        "gc_compress": "#D55E00",
        "gc_decompress": "#E69F00",
        "denorm": "#d5dbdb",
    },
}

# Convenience aliases used across notebooks
ARCH_COLORS = PALETTE["architectures"]
BACKEND_COLORS = PALETTE["compare_gpu_fpga"]
METRIC_COLORS = PALETTE["metrics"]
ACTIVATION_COLORS = PALETTE["activations"]


# ======================================================================================
# 3. Naming & ordering conventions
# ======================================================================================
ARCH_ORDER = ["ResSHyp", "SHyp", "ResFP", "FP"]  # code names (plot order)
ARCH_LABEL = {"ResSHyp": "ResSH", "SHyp": "SH", "ResFP": "ResFP", "FP": "FP"}  # display


def arch_label(code: str) -> str:
    return ARCH_LABEL.get(code, code)


# Platform display label keyed by (platform, config) — config distinguishes s0/s1 on FPGA.
def platform_label(platform: str, config: str | None = None) -> str:
    p = platform.lower()
    if p == "cpu":
        return "CPU"
    if p == "gpu":
        return "GPU"
    if p == "fpga":
        return "FPGA" if config in (None, "s0") else f"FPGA {config}"
    return platform


# Per-patch C++ stage order (host + DPU), for stacked latency bars
STAGE_ORDER = [
    "normalize",
    "g_a",
    "h_a",
    "eb_compress",
    "eb_decompress",
    "h_s",
    "gc_compress",
    "gc_decompress",
    "g_s",
    "denorm",
]

# lr encodings (lr_sweep_RD): marker = lr, colour reserved for architecture
LR_MARKER = {1e-5: "v", 5e-5: "o", 1e-4: "^", 5e-4: "s"}
LR_LABEL = {1e-5: "1e-5", 5e-5: "5e-5", 1e-4: "1e-4", 5e-4: "5e-4"}

# backend encodings (compare_gpu_fpga): colour = arch, linestyle/marker = backend
BACKEND_MARKER = {"gpu": "o", "fpga": "s"}
BACKEND_LS = {"gpu": "-", "fpga": "--"}

# ======================================================================================
# 4. Run selection & quality loaders
# ======================================================================================
# Per-architecture production learning rate (selected by the lr-sweep). Selection is by
# the lr VALUE (CSV `model.net_optimizer.lr` -> `lr` column), never by tag.
PROD_LR = {"FP": 5e-4, "ResFP": 5e-4, "SHyp": 5e-4, "ResSHyp": 1e-4}
ABLATION_LR = 5e-5  # GDN variants only exist here; controlled ReLU-vs-GDN A/B

METRIC_PREFIX = "test_sub500"  # 500-image MERLIN subset, matches the FPGA eval split
# Flat (backend-agnostic) metric columns in the tidy long DataFrame.
QUALITY_METRICS = [
    "bpp_likelihood",
    "bpp_bitstream",
    "psnr_merlin",
    "mse_merlin",
    "ssim_merlin",
    "ms_ssim_merlin",
    "psnr_adam_noc",
    "mse_adam_noc",
    "ssim_adam_noc",
    "ms_ssim_adam_noc",
    "enl_recon",
    "ratio_mean",
    "ratio_enl",
    "epd_merlin",
    "epd_adam_noc",
]
_CSV_METRIC = {  # flat name -> test_sub500/<col> source
    "bpp_likelihood": "bpp",
    "bpp_bitstream": "bpp_bitstream",
    "psnr_merlin": "psnr_merlin",
    "mse_merlin": "mse_merlin",
    "ssim_merlin": "ssim_merlin",
    "ms_ssim_merlin": "ms_ssim_merlin",
    "psnr_adam_noc": "psnr_adam_noc",
    "mse_adam_noc": "mse_adam_noc",
    "ssim_adam_noc": "ssim_adam_noc",
    "ms_ssim_adam_noc": "ms_ssim_adam_noc",
    "enl_recon": "enl_recon",
    "ratio_mean": "ratio_mean",
    "ratio_enl": "ratio_enl",
    "epd_merlin": "epd_merlin",
    "epd_adam_noc": "epd_adam_noc",
}


def clean_runs(df: pd.DataFrame, seeds: Sequence[int] = range(6)) -> pd.DataFrame:
    """Validity filters shared by every quality figure: dataset, seeds, drop broken/gdn1.
    Does NOT select lr or activation — callers do that explicitly."""
    out = df[df["data_name"].isin(["TSXSSCDataModule"])].copy()
    out = out[out["seed"].isin(list(seeds))]
    for tag in ("debug", "crashed"):  # validity tags, not an lr selector
        out = out[~out["tags"].apply(lambda ts: tag in ts)]
    out = out[~out["model_name"].str.contains("gdn1", na=False)]
    return out


def select_lr(df: pd.DataFrame, lr: str | float | None = "prod") -> pd.DataFrame:
    """Select rows by lr VALUE.

    lr='prod' -> per-arch PROD_LR; 'ablation' -> ABLATION_LR; a float -> that exact lr; None -> no
    lr filter.
    """
    if lr is None:
        return df
    if lr == "prod":
        keep = df.apply(
            lambda row: (
                row["architecture"] in PROD_LR
                and abs(float(row["lr"]) - PROD_LR[row["architecture"]]) < 1e-12
            ),
            axis=1,
        )
        return df[keep]
    target = ABLATION_LR if lr == "ablation" else float(lr)
    return df[df["lr"].round(12) == round(target, 12)]


def load_quality_runs(
    csv_path: Path = WANDB_CSV,
    *,
    lr: str | float | None = "prod",
    archs: list[str] | None = None,
    relu_only: bool = True,
    nop_only: bool = True,
    seeds: Sequence[int] = range(6),
    dedup: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load GPU (FP32) quality runs from the W&B CSV into a tidy long DataFrame (one row per run;
    backend='gpu'). Replaces the per-notebook clean+filter cells.

    Metrics come from the ``test_sub500/`` prefix. lr is selected by VALUE (see ``select_lr``).
    """
    raw = pd.read_csv(csv_path, index_col="id")
    raw["tags"] = raw["tags"].apply(lambda v: json.loads(v) if isinstance(v, str) else [])
    raw["no_output_padding"] = raw["no_output_padding"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    for col_name in ("lmbda", "seed", "lr"):
        raw[col_name] = pd.to_numeric(raw[col_name], errors="coerce")

    filt = clean_runs(raw, seeds=seeds)
    if relu_only:
        filt = filt[filt["model_name"].str.contains("_relu", case=False, na=False)]
    if nop_only:
        filt = filt[filt["no_output_padding"] == True]  # noqa: E712
    filt = select_lr(filt, lr)
    if archs is not None:
        filt = filt[filt["architecture"].isin(archs)]
    if dedup:
        # Full identity: model_name encodes activation, so this keeps the 4 ablation
        # variants distinct while still collapsing lr_search∩retrain duplicates.
        filt = filt.sort_index().drop_duplicates(
            ["architecture", "model_name", "no_output_padding", "lmbda", "seed"], keep="last"
        )

    def col(flat: str) -> np.ndarray:
        src = f"{METRIC_PREFIX}/{_CSV_METRIC[flat]}"
        return (
            np.asarray(pd.to_numeric(filt[src], errors="coerce"), dtype=float)
            if src in filt.columns
            else np.full(len(filt), np.nan)
        )

    out = (
        pd.DataFrame(
            {
                "lambda": filt["lmbda"].values,
                "seed": filt["seed"].astype(int).values,
                "arch": filt["architecture"].values,
                "activation": filt["model_name"].str.split("_").str[-1].values,
                "no_output_padding": filt["no_output_padding"].values,
                "lr": filt["lr"].values,
                "backend": "gpu",
                "wandb_id": filt.index.values,
                "wandb_name": filt["run_name"].values,
                "run_dir": filt["run_dir"].fillna("").values if "run_dir" in filt.columns else "",
                **{m: col(m) for m in QUALITY_METRICS},
            }
        )
        .sort_values(["lambda", "seed"])
        .reset_index(drop=True)
    )
    if verbose:
        print(f"Loaded {g}{len(out)}{e} GPU runs  ({out['arch'].value_counts().to_dict()})")
    return out


def load_fpga_quality(
    gpu_df: pd.DataFrame | None = None,
    *,
    compiled_dir: Path = COMPILED_MODELS_DIR,
    archs: list[str] | None = None,
    seeds: Sequence[int] = range(6),
    verbose: bool = True,
) -> pd.DataFrame:
    """Scan compiled_models/ -> tidy long DataFrame (backend='fpga').

    Folder names don't encode lr, so this picks up whichever INT8 model is currently deployed.
    """
    _seeds = set(seeds)
    lookup = (
        {(int(row.seed), float(row["lambda"])): row.wandb_id for _, row in gpu_df.iterrows()}
        if gpu_df is not None
        else {}
    )
    rows = []
    for model_dir in sorted(Path(compiled_dir).iterdir()):
        man_p, met_p = model_dir / "manifest.json", model_dir / "results" / "metrics.json"
        if not man_p.exists() or not met_p.exists():
            continue
        man = json.loads(man_p.read_text())
        seed, lmbda = man.get("seed"), man.get("lambda")
        if seed not in _seeds:
            continue
        name = man.get("model_name", model_dir.name)
        arch = man.get("architecture") or next((a for a in ARCH_ORDER if a in name), "unknown")
        if archs is not None and arch not in archs:
            continue
        m = json.loads(met_p.read_text())
        merlin, adam, recon = m.get("MERLIN", {}), m.get("ADAM", {}), m.get("recon", {})
        rows.append(
            {
                "lambda": lmbda,
                "seed": seed,
                "arch": arch,
                "activation": "relu",
                "no_output_padding": True,
                "lr": np.nan,
                "backend": "fpga",
                "wandb_id": man.get("wandb_run_id") or lookup.get((seed, lmbda), ""),
                "wandb_name": name,
                "run_dir": "",
                "model_dir": str(model_dir),
                "bpp_likelihood": np.nan,
                "bpp_bitstream": merlin.get("bpp"),
                "psnr_merlin": merlin.get("psnr"),
                "mse_merlin": merlin.get("mse"),
                "ssim_merlin": merlin.get("ssim"),
                "ms_ssim_merlin": np.nan,
                "psnr_adam_noc": adam.get("psnr"),
                "mse_adam_noc": adam.get("mse"),
                "ssim_adam_noc": adam.get("ssim"),
                "ms_ssim_adam_noc": np.nan,
                "enl_recon": recon.get("enl"),
                "ratio_mean": recon.get("ratio_mean"),
                "ratio_enl": recon.get("ratio_enl"),
                "epd_merlin": merlin.get("epd"),
                "epd_adam_noc": adam.get("epd"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["lambda", "seed"]).reset_index(drop=True)
    if verbose:
        print(f"Loaded {g}{len(out)}{e} FPGA results  ({out['arch'].value_counts().to_dict()})")
    return out


def aggregate_rd(
    df: pd.DataFrame,
    by: Sequence[str],
    metrics: Sequence[str] = QUALITY_METRICS,
    stats: Sequence[str] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Group a tidy long DataFrame by ``by`` and aggregate ``metrics`` over seeds.

    Returns a DataFrame with MultiIndex columns (metric, stat). Replaces both
    ``build_statistics`` (ablation) and the ``stats_df`` block (gpu_fpga) via ``by``.
    """
    present = [m for m in metrics if m in df.columns]
    return df.groupby(list(by))[present].agg(list(stats)).sort_index()


# ======================================================================================
# 5. Plot core
# ======================================================================================
def export_manuscript(fig: Figure, manuscript_name: str | None, *, save: bool = True) -> None:
    """Save a figure to the manuscript figures/images dir as PDF (overrides old PNG)."""
    if not (save and manuscript_name):
        return
    dst = MANUSCRIPT_DIR / f"{manuscript_name}.pdf"
    fig.savefig(dst, bbox_inches="tight")
    print(f"{g}→ manuscript:{e} {dst.relative_to(ROOT_DIR)}")


# ---- statistical comparison ----------------------------------------------------------
def compare_at_lambda(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    *,
    lmbda: float = 1000.0,
    lambda_col: str = "lambda",
    metric_prefix: str = "",
    bpp_col: str = "bpp_bitstream",
) -> None:
    """Compare two independent model groups at a fixed λ.

    Reports mean±std for each group and mean±SE for the difference, where SE is the
    Welch standard error of the difference of means for two independent samples:

        SE(Δ) = sqrt(var_A / n_A + var_B / n_B)

    Args:
        df_a / df_b:      Pre-filtered DataFrames (one group each), one row per seed.
        label_a / label_b: Display names (e.g. ``"GPU"``, ``"FPGA"``, ``"ResSH"``).
        lmbda:            λ value to compare at.
        lambda_col:       Column holding λ values. ``"lambda"`` in the tidy long df
                          returned by ``load_quality_runs``; ``"lmbda"`` in raw W&B CSV.
        metric_prefix:    Prefix prepended to default metric column names, including any
                          separator (e.g. ``"test_sub500/"`` for raw CSV; ``""`` for tidy df).
        bpp_col:          BPP column name, without prefix (e.g. ``"bpp_bitstream"``).
    """
    _METRICS = [
        (f"{metric_prefix}psnr_merlin", "PSNR [dB]", ".3f"),
        (f"{metric_prefix}ssim_merlin", "SSIM", ".4f"),
        (f"{metric_prefix}{bpp_col}", "bpp", ".4f"),
    ]
    sub_a = df_a[df_a[lambda_col] == lmbda]
    sub_b = df_b[df_b[lambda_col] == lmbda]
    if sub_a.empty or sub_b.empty:
        print(f"  No data at λ={int(lmbda)}")
        return

    col_w = 16
    _b, _e = "\033[34m", "\033[0m"

    def ms(m, s, f):
        return f"{format(m, f)}±{format(s, f)}"

    print(
        f"\n{_b}{label_a} vs {label_b}  (λ={int(lmbda)},  n_A={len(sub_a)}  n_B={len(sub_b)}):{_e}"
    )
    print(
        f"  {'metric':<10}  {label_a + ' mean±std':>{col_w}}  "
        f"{label_b + ' mean±std':>{col_w}}  {'Δ mean±SE':>{col_w}}"
    )
    print(f"  {'-' * 10}  {'-' * col_w}  {'-' * col_w}  {'-' * col_w}")
    for col, name, fmt in _METRICS:
        if col not in sub_a.columns or col not in sub_b.columns:
            continue
        va = pd.to_numeric(sub_a[col], errors="coerce").dropna()
        vb = pd.to_numeric(sub_b[col], errors="coerce").dropna()
        if va.empty or vb.empty:
            continue
        mean_diff = va.mean() - vb.mean()
        se_diff = np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
        print(
            f"  {name:<10}  {ms(va.mean(), va.std(), fmt):>{col_w}}"
            f"  {ms(vb.mean(), vb.std(), fmt):>{col_w}}"
            f"  {ms(mean_diff, se_diff, fmt):>{col_w}}"
        )


# ---- tile (reconstruction) rendering -------------------------------------------------
def linA_to_logI(linA: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Linear amplitude -> log-intensity.

    eps defaults to src.utils.constants.EPS.
    """
    if eps is None:
        from src.utils.constants import EPS  # lazy: only the recon notebook needs it

        eps = EPS
    return np.log(np.square(linA.astype(np.float64)) + eps)


def show_logI(
    ax: Axes,
    logI: np.ndarray,
    title: str = "",
    subtitle: str = "",
    subtitle_fontsize: float = 6.5,
    border_color: str | None = None,
    clip_factor: int = 3,
) -> None:
    """Render a log-intensity tile (per-image mean±clip_factor·σ) on ``ax``."""
    from src.utils.processing_utils import clip  # lazy import

    ax.imshow(
        clip(logI, mean_std_norm=True, clip_factor=clip_factor),
        cmap="gray",
        aspect="equal",
        interpolation="none",
    )
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    if subtitle:
        ax.text(
            0.5,
            -0.03,
            subtitle,
            transform=ax.transAxes,
            fontsize=subtitle_fontsize,
            ha="center",
            va="top",
            color="gray",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(border_color is not None)
        if border_color is not None:
            sp.set_edgecolor(border_color)
            sp.set_linewidth(2.0)
