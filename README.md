# Synthetic Aperture Radar (SAR) Despeckling and Data Compression (DDC) on Field Programmable Gate Arrays (FPGA)

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Config: Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

Implementation of [Amao-Oliva et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0924271624004866) on FPGA: joint SAR image despeckling and compression using hyper-autoencoders (CompressAI) trained with the MERLIN self-supervised strategy. Neural network subgraphs run on the DPU of a Xilinx ZCU102 via Vitis-AI; entropy coding runs on the ARM CPU in C++.

## TODOs

### Open investigations

- [ ] EPD values > 1 in [RD-curve_ablation.ipynb cell 11](notebooks/RD-curve_ablation.ipynb). EPD should be ≤ 1.
- [ ] Clarify exact Python version per environment (main conda env appears to use 3.11, Vitis-AI Docker 3.8, board may be 3.9). Update `copilot-instructions.md` and code once confirmed.

### Code quality / housekeeping

- [x] Reorganize `scripts/` into `dataset/`, `training/`, `evaluation/`, `fpga/{deploy,benchmark}/`, `vitis_ai/`.

### Planned experiments

- [ ] Hardware-aware or per-layer quantization via `vaiq_pytorch` ([strategy doc](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Hardware-Aware-Quantization-Strategy), [JSON config doc](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Quantization-Strategy-Configuration?tocId=rGCaO9QY6VvNbAJV7l9i7Q)).
- [ ] Quantization-aware training (`fast_finetuning` or full QAT) to recover accuracy lost during PTQ.
- [ ] Investigate skip/merge of the real/imag concatenation step before the hyperprior.
- [ ] Evaluate `ResidualBlockWithStride` / `ResidualBlockUpsample` from CompressAI as alternatives to current manual residual blocks.

## Project Structure

```text
DDC_FPGA/
├── configs/                       # Hydra configuration
│   ├── train.yaml                 # Default training config
│   ├── experiment/                # Experiment preset overrides
│   └── model/, data/, trainer/,...
├── data/                          # Data files (git-ignored; use symlinks to avoid copies)
│   ├── TSX_cos_files/             # Raw CoSAR/SSC downloads
│   ├── processed_hdf5/            # HDF5 patch datasets (output of create_dataset.py)
│   ├── visualization/             # Large tiles + ground-truth references for visualization
│   ├── fpga_eval/                 # Test-patch .npy sets used for FPGA quality evaluation
│   └── method_ground_truths/      # ADAM-NOC and MERLIN reference model checkpoints
├── docs/                          # Documentation (see table below)
├── notebooks/                     # Analysis and visualization notebooks
├── results/                       # Outputs (git-ignored)
│   ├── benchmark/                 # JSON benchmark results per model
│   └── fpga/
│       ├── active_model -> ...    # Symlink to the currently deployed model
│       └── compiled_models/       # Compiled xmodels + inference results, one dir per model
├── scripts/
│   ├── dataset/                   # create_dataset.py, compute_stats.py, convert_h5_to_np.py
│   ├── training/                  # debug_training.py, schedule_training.sh
│   ├── evaluation/                # benchmark_gpu.py, update_wandb_runs.py
│   ├── fpga/
│   │   ├── deploy/                # deploy.py, batch_deploy.py, model_quant.py, DPU_archs/, batch_deploy_configs/
│   │   └── benchmark/             # benchmark_sweep.py, run_benchmarks.py, collect_roofline.py
│   └── vitis_ai/                  # Docker container management scripts
├── src/                           # Training source code
│   ├── train.py                   # Main training entry point (Hydra)
│   ├── data/                      # LightningDataModule
│   ├── models/                    # LightningModules + model components (components/)
│   └── utils/                     # Constants (AMP_MIN/MAX, EPS), processing utils
├── tests/                         # Pytest unit tests
└── CompressAI/                    # CompressAI fork (git submodule)
```

## Documentation

| File | Content |
| ------ | --------- |
| [docs/Method.md](docs/Method.md) | MERLIN theory, model architecture, references |
| [docs/Data.md](docs/Data.md) | Data source, preprocessing pipeline, naming conventions, normalisation constants |
| [docs/FPGA_inference.md](docs/FPGA_inference.md) | C++ FPGA inference pipeline, DPU runners, entropy models, future work |
| [docs/FPGA_benchmark.md](docs/FPGA_benchmark.md) | Benchmark: ZCU102 hardware, methodology, power, results, future work + journal |
| [docs/python_to_cpp_migration_journal.md](docs/python_to_cpp_migration_journal.md) | Python→C++ migration: why, before/after numbers, bug archive |
| [docs/GPU_benchmark.md](docs/GPU_benchmark.md) | GPU/CPU benchmark tooling (legacy, raw — pending unified-runner refactor) |
| [docs/Vitis-AI_journey.md](docs/Vitis-AI_journey.md) | Deployment journal, known issues, changelog |

## Workflows

Quick reference for all recurring workflows. All commands run from the project root (`DDC_FPGA/`) unless noted.
Conda environment: `SAR_DDC` for local work; Vitis-AI Docker container for FPGA quantization (handled transparently by `deploy.py`).

### 1 · Dataset creation

**Script**: `scripts/dataset/compute_stats.py` and `scripts/dataset/create_dataset.py`
**When**: To compute normalization constants and to create HDF5 set of `.cos` files for a split definition.

**Compute dataset statistics**
By default statistics for the intensity and the amplitude in log-scale (natural log with an epsilon of $1e-2$) are computed. Modify the file to compute more.

```bash
cd scripts/dataset
python compute_stats.py > ../../data/analysis/dataset_stats.log
```

```bash
python scripts/dataset/create_dataset.py \
    --input-dir data/TSX_cos_files/ \
    --output-dir data/processed_hdf5/ \
    --split-file data/TSX_cos_files/spatial_splits_5.json \
    --patch-size 256 \
    --seed 42
```

Output directory: `data/processed_hdf5/TSX_<split>_<patch_size>x<patch_size>/` containing `train.h5`, `val.h5`, `test.h5`. See [docs/Data.md](docs/Data.md) for the full HDF5 schema.

### 2 · Training

**Entry point**: `src/train.py` (Hydra)
**Config root**: `configs/`, experiment overrides in `configs/experiment/`.

```bash
# Standard run
python src/train.py experiment=<experiment_name>

# Debug (fast dev run)
python src/train.py experiment=<experiment_name> debug=fdr

# Multirun
python src/train.py -m experiment=<experiment_name> seed=0,1,2,3,4,5 model.net.lmbda=1,2,5,10,20,50,100,200,500,1000
```

Checkpoints land in `logs/train/<task>/<model>/<multi>runs/<date>/<id>/checkpoints/`.

See [docs/Method.md](docs/Method.md) for model architecture, training strategy, and signal equations.

### 3 · FPGA deployment — single run

**Script**: `scripts/fpga/deploy/deploy.py`
**When**: Compile + transfer + infer a single training checkpoint to the ZCU102.
**Requires**: Vitis-AI Docker container (started automatically), passwordless SSH to ZCU102.

```bash
# Full pipeline (compile → transfer → infer → fetch)
python scripts/fpga/deploy/deploy.py --run-dir DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/<date>/<id>

# Skip phases selectively
python scripts/fpga/deploy/deploy.py --run-dir <...> \
    --skip-compile          # skip quantization + xmodel generation
    --skip-transfer         # skip SCP to board
    --skip-infer            # skip board inference
    --skip-fetch            # skip fetching results back

# Optional compile flags
python scripts/fpga/deploy/deploy.py --run-dir <...> \
    --arch ZCU102           # or Leopard (default: ZCU102)
    --inspect               # run DPU inspector before calibration
    --eval-float            # evaluate float model
    --eval-quant            # evaluate quantized model
    --fast-finetune         # enable fast finetuning during calibration
    --image-graph           # generate SVG of compiled xmodel
    --subset 100            # number of test samples for inference (default: 100)
```

Results land in `results/fpga/compiled_models/<model_name>/` and are accessible via the `results/fpga/active_model/` symlink.

See [docs/FPGA_inference.md](docs/FPGA_inference.md) for the on-board inference pipeline architecture.

### 4 · FPGA deployment — batch

**Script**: `scripts/fpga/deploy/batch_deploy.py`
**When**: Compile + deploy a set of W&B runs matching filters defined in `FILTERS_CONFIG`.

```bash
# Preview matched runs (dry run)
python scripts/fpga/deploy/batch_deploy.py --tag <label> --dry-run

# Deploy all matching runs (skips already-compiled models)
python scripts/fpga/deploy/batch_deploy.py --tag <label>

# Deploy specific run IDs directly (bypasses FILTERS_CONFIG)
python scripts/fpga/deploy/batch_deploy.py --tag <label> --run-ids <id1> <id2>

# Force recompile even if model already exists
python scripts/fpga/deploy/batch_deploy.py --tag <label> --force-recompile
```

Supports all phase-skip and compile flags from `deploy.py` (`--skip-transfer`, `--arch`, `--eval-float`, etc.).
Batch log: `results/fpga/batch_deploy/<tag>_<timestamp>.log`.

See [docs/FPGA_inference.md](docs/FPGA_inference.md) for the on-board inference pipeline architecture.

### 5 · Benchmarking

See [docs/FPGA_benchmark.md](docs/FPGA_benchmark.md) for the complete technical reference
(hardware, configs, methodology, results, future work).

**Script**: `scripts/fpga/benchmark/benchmark_sweep.py` (host-side; drives the C++ `benchmark_hardware` binary).
**When**: Measure FPGA latency, throughput, per-stage breakdown, and power across all 4 architectures.
**Requires**: compiled models in `results/fpga/compiled_models/`; SSH alias `ZCU102` configured.

```bash
# Full FPGA sweep — all 4 archs (deploy + benchmark + roofline + fetch), ~30-40 min
python scripts/fpga/benchmark/benchmark_sweep.py

# Subset / options
python scripts/fpga/benchmark/benchmark_sweep.py --models ResSHyp-relu_s0_L1000_pt,FP-relu_s0_L1000_pt \
    --skip-ceiling --skip-roofline --rebuild-cpp --force
```

> GPU/CPU benchmarking (`scripts/evaluation/benchmark_gpu.py`) and unified cross-platform comparison are
> legacy/pending — see [docs/GPU_benchmark.md](docs/GPU_benchmark.md) and the unified-runner TODO
> in [docs/FPGA_benchmark.md](docs/FPGA_benchmark.md).

Runs 5 scenarios: `full`, `compress`, `decompress`, `nn_only`, `entropy_only`.
Results: `results/benchmark/<model_name>/` as JSON. Analysis notebook: `notebooks/benchmark_analysis.ipynb`.
