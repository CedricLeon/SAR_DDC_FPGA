# CLAUDE.md — DDC_FPGA

Joint SAR image despeckling and data compression (DDC) on a Xilinx ZCU102 FPGA. Four neural network subgraphs (`g_a`, `h_a`, `h_s`, `g_s`) run as INT8 on the DPU; entropy coding runs on the ARM CPU. Models are CompressAI hyper-autoencoders (ScaleHyperprior, FactorizedPrior topologies) trained with the MERLIN self-supervised strategy, using PyTorch Lightning + Hydra. Reference: [Amao-Oliva et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0924271624004866).

---

## Environments

Three distinct environments — always use the right one.

| Environment | Activate | Python | Use for |
| --- | --- | --- | --- |
| Local | `conda activate DDC_FPGA` | 3.11 | Training, analysis notebooks, preprocessing, unit tests, GPU benchmarking |
| Vitis-AI Docker | auto (via `deploy.py`) | 3.8 | PTQ quantisation, DPU compilation, xmodel export — **3.8 strict** |
| ZCU102 board | `ssh ZCU102` | 3.9 | On-board inference and benchmarking |

**Python 3.8 compatibility** (Docker): use `from typing import Union, Optional, List, Tuple` — never `T | None` or `list[int]` (PEP 604/585).

---

## Key Paths

| Location | Path |
| --- | --- |
| Host source | `/home/leon_ce/dev/Vitis-AI/DDC_FPGA/` (also mounted at `/mnt/vitisAI/Vitis-AI/DDC_FPGA/`) |
| Board project root | `ZCU102:/home/root/SAR_DDC/` |
| Board inference binary | `ZCU102:/home/root/SAR_DDC/build_cpp/inference_hybrid` |
| Board active model | `ZCU102:/home/root/SAR_DDC/active_model/` (xmodel + entropy_params/) |
| Board test data | `ZCU102:/home/root/SAR_DDC/data/test_sub500_seed42.npy` |
| Active model (host) | `results/fpga/active_model/` (symlink) |

---

## Documentation Map

Read only what's relevant to the task at hand.

| Task | Document |
| --- | --- |
| Model architecture, MERLIN theory, loss, signal equations | `docs/Method.md` |
| Data source, preprocessing, HDF5 schema, normalisation | `docs/Data.md` |
| FPGA inference pipeline (C++), DPU runners, entropy models | `docs/FPGA_inference.md` ← read first for inference work |
| Benchmark: ZCU102 hardware, methodology, power, results, future work + journal | `docs/FPGA_benchmark.md` |
| Why/how we ported Python→C++, before/after numbers, bug archive | `docs/python_to_cpp_migration_journal.md` |
| GPU/CPU benchmark tooling (legacy, raw — pending unified-runner refactor) | `docs/GPU_benchmark.md` |
| Vitis-AI issues, deployment journal | `docs/Vitis-AI_journey.md` (check here before debugging Vitis-AI issues) |

---

## Common Commands

### Training

```bash
conda activate DDC_FPGA
python src/train.py experiment=<name>
python src/train.py experiment=<name> debug=fdr   # fast dev run (1 batch smoke test)
```

### FPGA Deploy (single model)

```bash
python scripts/fpga/deploy/deploy.py --run-dir DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/<date>/<id>
# Skip phases selectively:
python scripts/fpga/deploy/deploy.py --run-dir <...> --skip-compile --skip-transfer  # infer + fetch only
# Rebuild C++ binary on board before inference (Phase 3 always uses the C++ binary):
python scripts/fpga/deploy/deploy.py --run-dir <...> --skip-compile --rebuild-cpp
# Skip 100-patch test-set sweep, run only Hamburg tile eval (fast tile re-evaluation):
python scripts/fpga/deploy/deploy.py --run-dir <...> --skip-compile --skip-test-set
```

### C++ Build on Board (manual)

```bash
# Push sources and rebuild (done automatically by batch_deploy.py and --rebuild-cpp).
# Use rsync (not scp -r): scp -r creates nested src/src/ when the remote dir already exists.
rsync -av inference_cpp/src/ ZCU102:/home/root/SAR_DDC/inference_cpp/src/
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && make -j4"
```

### C++ Inference (on board)

```bash
build_cpp/inference_hybrid --xmodel active_model/*.xmodel \
  --params active_model/entropy_params \
  --data data/test_sub500_seed42.npy --subset 100 \
  [--debug-patch N] [--verbose]
```

---

## DPU / Model Constraints

Code compiled for the DPU must follow these rules:

- No `GDN` activations — use `ReLU`
- No `LowerBoundFunction` — use `torch.clamp` or `torch.max`
- `ConvTranspose2d` must have `output_padding=0`
- Entropy coding: real rANS bitstream (C++ rANS coder in `inference_cpp/src/rans/`) for all evaluation — not the likelihood-estimation path used during training

---

## Data & Normalisation

- Patches stored raw (unnormalised); log-amplitude normalisation applied at runtime inside LightningDataModule
- Constants `AMP_MIN`, `AMP_MAX`, `EPS` — single source of truth: `src/utils/constants.py` — never duplicate
- MERLIN reconstruction: average real + imag predictions (×0.5) when computing intensity for metrics
- Visualisations → log-scale intensity; metrics (PSNR, ENL, …) → linear amplitude — never mix

---

## Current Status: C++ inference + benchmark — COMPLETE

The Python→C++ migration is done; **C++ is the only inference path** (no Python on the board).

- **`inference_hybrid`** — board-validated: 100/100 patches pass the gate
  `|PSNR_cpp_vs_MERLIN − PSNR_py_vs_MERLIN| < 0.1 dB` (mean Δ = +0.083 dB; C++ ≥ Python vs MERLIN).
  Architecture + run reference → `docs/FPGA_inference.md`; bug archive → `docs/python_to_cpp_migration_journal.md` §6.
- **`benchmark_hardware`** — M1–M3 board-verified: S0 per-stage baseline, native INA226/PMBus power
  sampler, S1 channel-parallel (g_a 1.95× / g_s 1.96×, byte-identical to S0), data-parallel ceilings
  (nn_only/entropy_only ~1.97× at N=2). Hardware, methodology, results → `docs/FPGA_benchmark.md`.
- Python legacy removed; `scripts/` reorganised into `dataset/ training/ evaluation/ fpga/{deploy,benchmark}/ vitis_ai/`.

**Scope closed at M3.** Pipelining (M4 P0, M5 P2, P3) and the unified GPU/CPU/FPGA runner are
**future work, not implemented** — documented in `docs/FPGA_benchmark.md` §10.
*Do not implement parallelism without a design discussion first.*

### Benchmark run commands

**Canonical results**: `results/benchmark_hardware/<model_name>/<config>_<scenario>[_dpuN][_entN].json`.
Arch is auto-detected from `active_model/manifest.json` (errors if absent/malformed).

```bash
# Full sweep, all 4 archs (host-side, ~30-40 min): deploy + benchmark + roofline + fetch
python scripts/fpga/benchmark/benchmark_sweep.py
#   --models ResSHyp-relu_s0_L1000_pt,FP-relu_s0_L1000_pt  --skip-ceiling --skip-roofline --rebuild-cpp --force

# Single run (manual, on board)
build_cpp/benchmark_hardware --xmodel active_model/*.xmodel --params active_model/entropy_params \
    --data data/test_sub500_seed42.npy --config s0 --scenario compress --warmup 5 --iters 50 --output out.json

# On board, single model: benchmark sweep + roofline collection
python3 /home/root/SAR_DDC/run_benchmarks.py ResSHyp-relu_s0_L1000_pt
python3 /home/root/SAR_DDC/collect_roofline.py ResSHyp-relu_s0_L1000_pt
```

---

## Conventions

- **Minimal changes**: do not reformat unrelated files; `ruff` handles formatting automatically.
- **Doc consistency**: after code changes, update the relevant doc file if the change affects overall understanding.
- **Markdown tables**: always `| --- | --- |` with spaces (markdownlint MD055/MD056).
- **C++ decisions**: provide clear explanations and context in chat.
- **Destructive actions**: always ask before deleting files, force-pushing, or anything irreversible.
- **Regressions**: run `make test` after Python edits to catch regressions.
- **Errors over silent fallbacks**: if code expects a file, field, or format that should always be present (e.g. `manifest.json`, a specific JSON key, a model name pattern), throw an explicit error when it is missing or malformed — never silently fall back to a default. A missed fallback produces wrong results that may go unnoticed; a hard error forces an immediate fix.
