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
| Manuscript figures | `LaTeX/SAR_DDC_FPGA_TGRS_2026/figures/images/` (PDF export target for notebook figures) |

---

## Documentation Map

Read only what's relevant to the task at hand.

| Task | Document |
| --- | --- |
| Model architecture, MERLIN theory, loss, signal equations | `docs/Method.md` |
| Data source, preprocessing, HDF5 schema, normalisation | `docs/Data.md` |
| FPGA inference pipeline (C++), DPU runners, entropy models | `docs/FPGA_inference.md` ← read first for inference work |
| Benchmark: ZCU102 hardware, methodology, power, results, future work + journal | `docs/FPGA_benchmark.md` |
| GPU/CPU host benchmark + cross-platform (CPU/GPU/FPGA) comparison & unified runner | `docs/GPU_benchmark.md` |
| Onboard streaming pipeline (SLC→`.ddc`): design + measurements — fan-out core-scaling, symmetrization/overlap studies, full-scene throughput/energy/deadline results | `docs/onboard_pipeline.md` ← **systems paper** |
| Analysis notebooks: purpose, data flow, shared modules (`_plotkit`, `_benchmark_loader`) | `docs/Notebooks.md` |

---

## Model Architectures & Checkpoints

Current focus is **inference**. Available checkpoints span **4 architectures × 10 λ
`{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}` × 6 seeds `{0, 1, 2, 3, 4, 5}`**.

Each architecture is one combination of two independent binary choices:

- **Prior** — **F**actorized **P**rior (`FP`) vs **S**cale-**Hyp**erprior (`SHyp`, a.k.a. `SH`); the
  hyperprior adds the `h_a` / `h_s` subgraphs.
- **Residual** — heavy residual blocks in the main encoder `g_a` / decoder `g_s` (the `Res` prefix),
  carrying **~10× the OPs** of the plain variant (`net.no_residual_blocks: false` in `train_config.yaml`).

This results in the FP, SHyp, ResFP, and ResSHyp architectures (where the first is the lightest and the last the most consequent).
Because they differ so much computationally it is VERY important to always specify which architecture is mentioned/described when reporting numbers.

> **Naming trap**: `FP` / `ResFP` and `SHyp` / `ResSHyp` both compile to the *same* xmodel class: the filename does **not** identify the architecture.
> Read `model_name` in `manifest.json` (or `net.no_residual_blocks` in `train_config.yaml`).

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

### Benchmark run commands

**Canonical results**: `results/benchmark_hardware/<model_name>/<config>_<scenario>[_dpuN][_entN].json`.
Arch is auto-detected from `active_model/manifest.json` (errors if absent/malformed).

```bash
# Full sweep, all 4 archs (host-side): deploy + benchmark + roofline + fetch
python scripts/fpga/benchmark/benchmark_sweep.py

# Single run (manual, on board)
build_cpp/benchmark_hardware --xmodel active_model/*.xmodel --params active_model/entropy_params \
    --data data/test_sub500_seed42.npy --config s0 --scenario compress --warmup 5 --iters 50 --output out.json

python3 /home/root/SAR_DDC/run_benchmarks.py ResSHyp-relu_s0_L1000_pt
python3 /home/root/SAR_DDC/collect_roofline.py ResSHyp-relu_s0_L1000_pt
```

### Cross-platform benchmark (CPU / GPU / FPGA — one command)

Full doc → `docs/GPU_benchmark.md`.

```bash
python scripts/benchmark/run_unified_benchmark.py --model-dir results/fpga/active_model/ --power
python scripts/evaluation/benchmark_gpu.py --model-dir results/fpga/active_model/ \
    --scenario compress --no-gpu --iters 100 --warmup 20 --power
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

- Patches stored raw (unnormalised, but symetrized); log-amplitude normalisation applied at runtime inside LightningDataModule
- Constants `AMP_MIN`, `AMP_MAX`, `EPS` — single source of truth: `src/utils/constants.py` — never duplicate
- MERLIN reconstruction: average real + imag predictions (×0.5) when computing intensity for metrics
- Visualisations → log-scale intensity; metrics (PSNR, ENL, …) → linear amplitude — never mix

---

## Current Status: C++ inference + benchmark — COMPLETE

The Python→C++ migration is done; **C++ is the only inference path** (no Python on the board).

- **`inference_hybrid`** — board-validated: 100/100 patches pass the gate
  `|PSNR_cpp_vs_MERLIN − PSNR_py_vs_MERLIN| < 0.1 dB` (mean Δ = +0.083 dB; C++ ≥ Python vs MERLIN).
  Architecture + run reference → `docs/FPGA_inference.md`.
- **`benchmark_hardware`** — M1–M3 board-verified: S0 per-stage baseline, native INA226/PMBus power
  sampler, S1 channel-parallel (g_a 1.95× / g_s 1.96×, byte-identical to S0), data-parallel ceilings
  (nn_only/entropy_only ~1.97× at N=2). Hardware, methodology, results → `docs/FPGA_benchmark.md`.
- Python legacy removed; `scripts/` reorganised into `dataset/ training/ evaluation/ fpga/{deploy,benchmark}/ vitis_ai/`.

**Scope closed at M3.** The unified GPU/CPU/FPGA runner **is implemented**
(`scripts/benchmark/run_unified_benchmark.py` — see the Cross-platform benchmark section below and
`docs/GPU_benchmark.md`).

**Onboard streaming pipeline — implemented + board-verified.** The "SLC tile → despeckle + compress →
`.ddc`" compressor is built and measured (`inference_cpp/src/stream/`, binary `stream_pipeline`; design
+ results → `docs/onboard_pipeline.md`). Composable, byte-identical optimizations
(`--s1`/`--p0`/`--fanout`/`--prefetch`/`--neon`); studies done — symmetrization (E1), overlap (U5), DPU
fan-out core-/lane-scaling, full-scene throughput/energy sweep at the coherent setup (λ=20, overlap 2,
snap grid, 4 archs). **Currently: results analysis + figures** for the DATE'27 paper; remaining =
Jetson (N2) / CCSDS (N5) baselines. Further schedule changes → discuss in `docs/onboard_pipeline.md` first.

---

## Conventions

- **Minimal changes**: do not reformat unrelated files; `ruff` handles formatting automatically.
- **Doc consistency**: after code changes, update the relevant doc file if the change affects overall understanding.
- **Living design docs**: each initiative has one **main doc** — its design + reference (e.g. `docs/onboard_pipeline.md`), sometimes with smaller docs orbiting it — kept clean enough to hand to a stranger. *Resolve, don't accumulate*: when a question/TODO is answered, fold the insight into the section it belongs in and delete the TODO — no "done" markers, changelog blockquotes, or bug-archaeology (that goes to a journal). *One home per fact*: describe each thing once, cross-reference instead of re-describing. *Self-contained prose*: every paragraph readable standalone, no half-formed inline asides.
- **Markdown tables**: always `| --- | --- |` with spaces (markdownlint MD055/MD056).
- **C++ decisions**: provide clear explanations and context in chat.
- **Destructive actions**: always ask before deleting files, force-pushing, or anything irreversible.
- **Errors over silent fallbacks**: if code expects a file, field, or format that should always be present (e.g. `manifest.json`, a specific JSON key, a model name pattern), throw an explicit error when it is missing or malformed — never silently fall back to a default. A missed fallback produces wrong results that may go unnoticed; a hard error forces an immediate fix.
- **Verified arithmetic**: compute every number (unit conversions, patch counts, totals, ratios, percentages) with a `python3`/Bash command before writing it into a file, message, or doc — never inline mental math. Wrong numbers silently end up in docs and need corrective passes.
- **Sourced numbers**: challenge every hardware/spec number before relying on it, and cite its origin inline — a doc (name + page/section), a URL, or a measurement command. **Explicitly flag any number that is an estimate or recollection** not backed by one of those, so it can be verified. When a number comes from a doc/web/measurement, record it in the relevant project doc with the citation (as `docs/TerraSAR-X_objective.md` does with page numbers).
- **Project-root detection**: use `rootutils.setup_root(..., indicator=".project-root")` — never hand-rolled `__file__` parent walks. Exception: scripts that run inside the Vitis-AI Docker (e.g. `model_quant.py`), where rootutils is not installed → manual `.project-root` parent walk.
- **Production-run filter (quality analysis)**: figures use only `_relu` activation + `no_output_padding=True` runs, enforced by `_plotkit.load_quality_runs(relu_only=True, nop_only=True)`. Never mix GDN or output-padded variants into production comparisons.
