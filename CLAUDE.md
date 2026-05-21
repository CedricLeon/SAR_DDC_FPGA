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
| Board inference binary | `ZCU102:/home/root/SAR_DDC/active_model/inference_hybrid` |
| Board Python scripts | `ZCU102:/home/root/SAR_DDC/active_model/` |
| Board test data | `ZCU102:/home/root/SAR_DDC/data/test_sub500_seed42.npy` |
| Active model (host) | `results/fpga/active_model/` (symlink) |

---

## Documentation Map

Read only what's relevant to the task at hand.

| Task | Document |
| --- | --- |
| Model architecture, MERLIN theory, loss, signal equations | `docs/Method.md` |
| Data source, preprocessing, HDF5 schema, normalisation | `docs/Data.md` |
| FPGA pipeline, DPU runners, entropy models | `docs/FPGA_inference.md` (describes Python pipeline; partially outdated — C++ port in progress) |
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
```

### C++ Build and Deploy to Board
```bash
# 1. Push sources
scp inference_cpp/src/*.cpp inference_cpp/src/*.hpp \
    ZCU102:/home/root/SAR_DDC/inference_cpp/src/

# 2. Build on board
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && make -j4"

# 3. Run directly (no deploy step needed — binary stays in build_cpp/)
# ZCU102:/home/root/SAR_DDC/build_cpp/inference_hybrid --xmodel active_model/*.xmodel ...
```

### C++ Inference Validation
```bash
# On board — C++ inference (binary is in build_cpp/, not active_model/)
build_cpp/inference_hybrid --xmodel active_model/*.xmodel \
  --params active_model/entropy_params \
  --data data/test_sub500_seed42.npy --subset 100 \
  --compare-out /tmp/cpp_out [--debug-patch N] [--verbose]

# On board — Python reference
python3 active_model/inference_hybrid.py --xmodel active_model/*.xmodel \
  --data data/test_sub500_seed42.npy --subset 100 --compare-out /tmp/py_out

# Fetch and compare (on host)
scp -r ZCU102:/tmp/cpp_out tmp/cpp_out && scp -r ZCU102:/tmp/py_out tmp/py_out
python scripts/fpga/compare_py_cpp.py --py tmp/py_out --cpp tmp/cpp_out
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

## Current Focus: C++ Inference Port

**Goal**: C++ port of `scripts/fpga/inference_hybrid.py` → `inference_cpp/` → binary `inference_hybrid`.

### Status (2026-05) — Validation COMPLETE

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

**Validation gate**: `|PSNR_cpp_vs_MERLIN − PSNR_py_vs_MERLIN| < 0.1 dB` per patch. This replaced the former pixel_tol=0.5 gate — see §0 Q9 Bug 4 closure in `docs/cpp_inference_design.md`.

### Phase 2: `benchmark_hardware` binary

Profile the sequential pipeline first to get a latency breakdown (DPU vs CPU entropy vs overhead). Then design a parallelism strategy — §9.3 of `docs/cpp_inference_design.md` has two options (intra-patch parallel channels vs inter-tile pipelining). **Do not implement without a design discussion.** Board has a 3-core B4096 DPU; VART assigns cores round-robin at runner construction time.

---

## Conventions

- **Minimal changes**: do not reformat unrelated files; `ruff` handles formatting automatically.
- **Doc consistency**: after code changes, update the relevant doc file if the change affects overall understanding.
- **Markdown tables**: always `| --- | --- |` with spaces (markdownlint MD055/MD056).
- **C++ decisions**: provide clear explanations and context in chat.
- **Destructive actions**: always ask before deleting files, force-pushing, or anything irreversible.
- **Regressions**: run `make test` after Python edits to catch regressions.
