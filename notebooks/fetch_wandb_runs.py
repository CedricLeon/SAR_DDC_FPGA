#!/usr/bin/env python3
"""Fetch all W&B runs from SAR_DDC_FPGA and save as a flat CSV.

Usage (from any directory):
    python notebooks/fetch_wandb_runs.py

Output: notebooks/SAR_DDC_FPGA_all_runs_WandB.csv

The CSV has one row per run. Metadata columns (run_name, architecture, lmbda, …)
are followed by one column per W&B summary metric (test/bpp, test/psnr_merlin, …).
Tags are stored as a JSON array string (e.g. '["ablation", "final"]').

This file is a companion to RD-curve_ablation.ipynb and test_subset_analysis.ipynb.
Regenerate it whenever new runs complete (~1 min for the full project).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import wandb
from omegaconf import OmegaConf

ENTITY = "cedric-leonard"
PROJECT = "SAR_DDC_FPGA"
OUTPUT_CSV = Path(__file__).parent / "SAR_DDC_FPGA_all_runs_WandB.csv"


def _derive_architecture(dl_model_name: str, no_residual_blocks: bool) -> str:
    has_hyper = "ScaleHyperprior" in dl_model_name
    if has_hyper:
        return "SHyp" if no_residual_blocks else "ResSHyp"
    return "FP" if no_residual_blocks else "ResFP"


def fetch_runs() -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    total = len(runs)
    print(f"Found {total} runs in {ENTITY}/{PROJECT}. Parsing...")

    rows = []
    for i, run in enumerate(runs):
        config = OmegaConf.create(run.config)

        dl_model_name = (OmegaConf.select(config, "model.net._target_") or "").split(".")[-1]
        activation = OmegaConf.select(config, "model.net.activation")
        no_res = bool(OmegaConf.select(config, "model.net.no_residual_blocks") or False)

        row: dict = {
            "id": run.id,
            "run_name": run.name,
            "data_name": (OmegaConf.select(config, "data._target_") or "").split(".")[-1],
            "model_name": f"{dl_model_name}_{activation}",
            "architecture": _derive_architecture(dl_model_name, no_res),
            "lmbda": OmegaConf.select(config, "lambda"),
            "seed": OmegaConf.select(config, "seed"),
            "lr": OmegaConf.select(config, "model.net_optimizer.lr"),
            "no_output_padding": OmegaConf.select(config, "model.net.no_output_padding"),
            "no_residual_blocks": no_res,
            "tags": json.dumps(run.tags),  # JSON array string, e.g. '["debug"]'
            # Hydra output directory — used to locate Hamburg tile .npy files
            "run_dir": str(OmegaConf.select(config, "paths.output_dir") or ""),
        }

        # Flatten W&B summary — skip internal W&B keys (prefixed with "_")
        for k, v in run.summary._json_dict.items():
            if not k.startswith("_"):
                row[k] = v

        rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} runs parsed...")

    df = pd.DataFrame(rows).set_index("id")
    return df


if __name__ == "__main__":
    df = fetch_runs()
    df.to_csv(OUTPUT_CSV)
    n_metric_cols = sum(1 for c in df.columns if "/" in c)
    print(f"\nSaved {len(df)} runs → {OUTPUT_CSV}")
    print(f"  {len(df.columns) - n_metric_cols} metadata columns + {n_metric_cols} metric columns")
