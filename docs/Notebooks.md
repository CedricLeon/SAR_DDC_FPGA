# Notebooks

*Analysis and figure-generation notebooks for the manuscript. Purpose, data flow, and the shared modules behind them.*

All notebooks live in `notebooks/` and run in the local `DDC_FPGA` conda env (Python 3.11). They consume already-computed results — none run training or board inference.

---

## Data flow

```
W&B (cedric-leonard/SAR_DDC_FPGA)
    └─ fetch_wandb_runs.py ──> notebooks/SAR_DDC_FPGA_all_runs_WandB.csv   (GPU FP32 quality, one row/run)

results/fpga/compiled_models/<model>/results/        (FPGA INT8 quality + reconstructions)
results/benchmark_hardware/<model>/<config>_<scenario>.json   (FPGA latency/power, C++ benchmark_hardware — archived 2026-08-31; DATE'27 board data in results/date27/)
results/benchmark_unified/<model>/baseline_<scenario>_<platform>.json   (host CPU/GPU, benchmark_gpu.py)

        │ loaded via shared modules
        ▼
   _plotkit.py          quality runs + palette/labels + manuscript export + tile rendering
   _benchmark_loader.py  benchmark JSON (latency/power/stages), all platforms

        │ consumed by notebooks
        ▼
   figures ──> results/plots/<topic>/   and   LaTeX/SAR_DDC_FPGA_TGRS_2026/figures/images/ (manuscript PDFs)
```

The W&B CSV and the result trees are the inputs; regenerate the CSV with `python notebooks/fetch_wandb_runs.py` (~1 min) whenever new runs complete. The benchmark JSON trees are produced by the FPGA/cross-platform benchmark runners (see `docs/FPGA_benchmark.md`, `docs/GPU_benchmark.md`).

---

## Shared modules

| Module | Role |
| --- | --- |
| `_plotkit.py` | Single source of truth for the Okabe-Ito **palette**, **naming** (`ARCH_LABEL`: ResSHyp→ResSH, SHyp→SH), per-arch **`PROD_LR`**, the W&B/FPGA **quality loaders** (`load_quality_runs`, `load_fpga_quality`, `aggregate_rd`), `export_manuscript`, and tile rendering (`linA_to_logI`, `show_logI`). |
| `_benchmark_loader.py` | Standalone loader for the benchmark JSON trees (`load_runs`, `load_stage_breakdowns`, `load_quality_metrics`, `join_hw_quality`). Same aligned schema for FPGA + host; FPGA rows tagged `platform="fpga"`. |
| `fetch_wandb_runs.py` | Pulls all W&B runs into the flat CSV. One row per run; metrics from the `test_sub500/` prefix; tags as a JSON-array string. |

**Production-run filter**: quality figures use only `_relu` activation + `no_output_padding=True` runs, enforced by `load_quality_runs(relu_only=True, nop_only=True)`. lr is selected by value via `PROD_LR` (`lr="prod"`). Never mix GDN or output-padded variants into production comparisons.

---

## Notebook map

| Notebook | Purpose | Inputs | Output |
| --- | --- | --- | --- |
| `RD-curve_ablation.ipynb` | RD curves: ResSHyp ablation (activation × output_padding) + all four architectures | W&B CSV | Manuscript (`fig_ablation_RD`, `fig_4arch_RD`) |
| `lr_sweep_RD.ipynb` | RD curves across the four swept learning rates per arch (diagnostic; picks `PROD_LR`) | W&B CSV | `results/plots/lr-sweep/` |
| `compare_gpu_fpga.ipynb` | GPU (FP32) vs FPGA (INT8) RD-quality comparison across archs | W&B CSV + `compiled_models/.../metrics.json` | Manuscript (`fig_crossprecision_RD`, BPP) |
| `reconstruction_visualization.ipynb` | Hamburg tile visual comparison (arch × {GPU, FPGA} × λ), log-intensity | W&B `run_dir` (GPU) + `compiled_models/.../*_recon_linA.npy` (FPGA) | Manuscript tile figures |
| `hardware_benchmark.ipynb` | **Publication HW figures**: latency breakdown + energy/patch (4-arch FPGA, 3-platform) | `benchmark_hardware/` + `benchmark_unified/` | Manuscript (`fig_latency_fpga_4arch`, …) |
| `benchmark_hardware_analysis.ipynb` | Exploratory FPGA analysis (inventory, stage breakdown, S0→S1, roofline, power, tradeoff) | `benchmark_hardware/` | `results/plots/` — **not in release** |
| `benchmark_cross_platform_analysis.ipynb` | Exploratory CPU/GPU/FPGA latency/energy on one schema | `benchmark_unified/` + `benchmark_hardware/` | `results/plots/` — **not in release** |

---

## Release scope

`benchmark_hardware_analysis.ipynb` and `benchmark_cross_platform_analysis.ipynb` are the exploratory working notebooks behind M1–M3. The manuscript release ships only the clean `hardware_benchmark.ipynb` cut. **Before the `main` release, remove both notebooks and every reference to them** (including their rows in this table). They are **not deleted outright** — they stay on `dev` (or a dedicated feature branch) as the basis for future implementation-focused manuscripts.
