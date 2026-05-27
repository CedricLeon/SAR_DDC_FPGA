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
| FPGA pipeline, DPU runners, entropy models | `docs/FPGA_inference.md` (C++ is now the primary inference path; Python pipeline kept as reference) |
| **C++ port design, decisions, open bugs** | **`docs/cpp_inference_design.md`** ← read first for inference work |
| ZCU102 hardware, benchmark methodology, power measurement | `docs/performance_benchmark_implementation.md` |
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
python scripts/fpga/deploy.py --run-dir DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/<date>/<id>
# Skip phases selectively:
python scripts/fpga/deploy.py --run-dir <...> --skip-compile --skip-transfer  # infer + fetch only
# Rebuild C++ binary on board before inference (Phase 3 always uses the C++ binary):
python scripts/fpga/deploy.py --run-dir <...> --skip-compile --rebuild-cpp
# Skip 100-patch test-set sweep, run only Hamburg tile eval (fast tile re-evaluation):
python scripts/fpga/deploy.py --run-dir <...> --skip-compile --skip-test-set
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
- Entropy coding: real rANS bitstream (C++ `ans.so`) for all evaluation — not the likelihood-estimation path used during training

---

## Data & Normalisation

- Patches stored raw (unnormalised); log-amplitude normalisation applied at runtime inside LightningDataModule
- Constants `AMP_MIN`, `AMP_MAX`, `EPS` — single source of truth: `src/utils/constants.py` — never duplicate
- MERLIN reconstruction: average real + imag predictions (×0.5) when computing intensity for metrics
- Visualisations → log-scale intensity; metrics (PSNR, ENL, …) → linear amplitude — never mix

---

## Current Focus: Hardware Benchmarking and C++ Parallelism

`inference_hybrid` is complete and validated. Next phase is `benchmark_hardware` — latency profiling and parallelism design. **Do not implement parallelism without a design discussion first** (see `docs/cpp_inference_design.md` §9).

### `inference_hybrid` Status (2026-05) — COMPLETE

| Item | Status |
| --- | --- |
| Build system, NPY loader, DPU wrappers | ✅ done |
| rANS C++ fork (pybind11-free) + unit tests | ✅ done |
| EntropyBottleneck + GaussianConditional | ✅ done |
| Full pipeline (SHyp + FP paths), Hamburg tile blending | ✅ done |
| Bug 4 — 9 patches had pixel diff vs Python | ✅ closed — C++ is closer to MERLIN; not a bug |
| Bug 5 — FP `_run_fp` y block layout | ✅ fixed — NHWC interleaved, consistent with SHyp |
| `--debug-patch N` broken in C++ | ✅ fixed — all guards now use `Logger::is_verbose()` |
| **On-board validation: 100/100 patches pass** | ✅ **DONE** — mean Δpm = +0.083 dB (C++ ≥ Python vs MERLIN) |
| Bug 6 — Hamburg tile pure noise (float64) | ✅ fixed — `sym_Noisy.npy` is float64; added `to_float32_vec()` with dtype-aware cast |

**Validation gate**: `|PSNR_cpp_vs_MERLIN − PSNR_py_vs_MERLIN| < 0.1 dB` per patch. This replaced the former pixel_tol=0.5 gate — see §0 Q9 Bug 4 closure in `docs/cpp_inference_design.md`.

### Phase 2: `benchmark_hardware` binary

Profile the sequential pipeline first to get a latency breakdown (DPU vs CPU entropy vs overhead). Design is fully documented in `docs/benchmark_hardware_design.md`.

**M1 (S0 + stage_timer)**: ✅ complete and board-verified (2026-05-23).
**M3 (S1 + ceilings)**: ✅ complete and board-verified (2026-05-24). g_a 1.95× speedup; g_s 1.96× (both pairs on distinct DPU cores after creation-order fix). Byte-identical to S0. nn_only N=2: 1.96× throughput; entropy_only N=2: 1.97× CPU scaling.

Binary: `build_cpp/benchmark_hardware`. Build: same `make -j4` in `build_cpp/` as `inference_hybrid`.

**Canonical results storage**: `results/benchmark_hardware/<model_name>/<config>_<scenario>[_dpuN][_entN].json`

Arch is auto-detected from `active_model/manifest.json` (`model_name` field prefix).
Throws an error if manifest is absent or `model_name` format is unexpected.

```bash
# Single run (manual, on board)
build_cpp/benchmark_hardware \
    --xmodel active_model/*.xmodel \
    --params active_model/entropy_params \
    --data   data/test_sub500_seed42.npy \
    --config s0 --scenario compress --warmup 5 --iters 50 \
    --output /path/to/s0_compress.json

# Full sweep for all 4 archs (host-side, ~30-40 min):
python scripts/fpga/benchmark_sweep.py
# Options: --models ResSHyp-relu_s0_L1000_pt,FP-relu_s0_L1000_pt
#          --skip-ceiling  --skip-roofline  --rebuild-cpp  --force

# Manual single-model sweep (on board after deploying):
python3 /home/root/SAR_DDC/run_benchmarks.py ResSHyp-relu_s0_L1000_pt

# Roofline only (on board, ~4 min for SHyp, ~2 min for FP):
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
