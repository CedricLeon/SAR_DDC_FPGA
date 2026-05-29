# DDC_FPGA AI Instructions

## Project Overview
This project implements joint SAR image despeckling and data compression (DDC) on a Xilinx ZCU102 FPGA, using CompressAI hyper-autoencoders trained with the MERLIN self-supervised strategy via PyTorch Lightning and Hydra.

## 📚 Documentation Map

Read **only** the document(s) relevant to your current task — do not load all docs upfront.

| Task domain | Document |
| --- | --- |
| Model architecture, MERLIN theory, loss, signal equations | [docs/Method.md](../docs/Method.md) |
| Data source, preprocessing pipeline, HDF5 schema, normalisation | [docs/Data.md](../docs/Data.md) |
| FPGA inference pipeline, DPU runners, entropy models, known issues | [docs/FPGA_inference.md](../docs/FPGA_inference.md) |
| Benchmark: ZCU102 hardware, methodology, power, results, future work | [docs/FPGA_benchmark.md](../docs/FPGA_benchmark.md) |
| Deployment journal, resolved/open Vitis-AI issues, changelog | [docs/Vitis-AI_journey.md](../docs/Vitis-AI_journey.md) |

> Before deep-diving an FPGA-related bug, check `docs/Vitis-AI_journey.md` — the issue may already be documented.

This documentation is a live, condensed representation of the code. While it provides a high-level overview and key details, it is not a substitute for reading the code itself. Always refer to the source code for the definitive implementation and logic.
Always update the documentation if you make changes to the code that affect the overall understanding of the project.

## ⚡ Execution Environments

Three environments are used. Each has a distinct Python version and set of tasks. Always activate the right one.

### 1. Local — `DDC_FPGA` conda env (Python 3.11)
- **Activate**: `conda activate DDC_FPGA` — always do this first; it ensures all dependencies are available.
- **Tasks**: model training (`src/train.py`), data preprocessing (`scripts/dataset/`), unit tests, analysis notebooks, general code editing, GPU benchmarking.
- Python 3.11 features are fine here.

### 2. Quantization & Compilation — Vitis-AI Docker (Python 3.8)
- **Managed automatically** by `scripts/fpga/deploy/deploy.py` — no need to start the container manually.
- **Tasks**: PTQ quantization, DPU compilation, `xir` graph manipulation, entropy model export, file packaging for the board.
- **Python 3.8 strict** — any code that runs here must be compatible:
  - ✅ `from typing import Union, Optional, List, Tuple`
  - ❌ `T | None` or `list[int]` (PEP 604/585 not supported)

### 3. Hardware — Xilinx ZCU102 (Python 3.9)
- **Hardware**: ZCU102 evaluation kit — Zynq UltraScale+ MPSoC with FPGA fabric + quad-core ARM Cortex-A53, access via `ssh ZCU102`.
- **Tasks**: on-board inference via the C++ `build_cpp/inference_hybrid` binary, real bitstream benchmark, codec quality evaluation.
- The user manages board access, SSH, and dataset transfers. Only proceed with board operations when explicitly asked.

## ⚙️ Codebase Conventions & Practices

### Training & Lightning

- **Training entry point**: `python src/train.py experiment=<name>`. Use `debug=fdr` (fast dev run, 1 batch) as a quick smoke-test before committing to a full run.
- **Hydra config consistency**: when changing an `__init__` signature or adding parameters to any class instantiated via Hydra (models, modules, callbacks…), always find the corresponding YAML in `configs/` and update it.
- **Manual optimization**: `SARDDCModule` sets `self.automatic_optimization = False` because it needs two independent optimizers — one for the main network parameters and one for the entropy bottleneck auxiliary loss.

### Compression & Entropy

- Two entropy modes exist and must not be confused:
  - **GPU / training mode**: likelihood estimation via the entropy bottleneck — differentiable, no real bitstream, used during training only.
  - **FPGA / evaluation mode**: real rANS bitstream via C++ `ans.so` extension — used for all testing and metric comparison.
- Currently **only the real bitstream mode** is used for quality/rate comparisons. Do not add evaluation code that depends on the likelihood estimation path.

### Vitis-AI / DPU Model Constraints

These apply to any model code that will be quantized and compiled for the DPU:

- The four neural network subgraphs (Encoder, HyperEnc, HyperDec, Decoder) run on the DPU; entropy coding runs on the ARM CPU via a C++ rANS codec.
- No `GDN` activations — use `ReLU`.
- No `LowerBoundFunction` — replace with `torch.clamp` or `torch.max`.
- `ConvTranspose2d` must have `output_padding=0`.

### Data & Normalisation

- Patches are stored **raw** (no normalisation at creation time). Log-amplitude normalisation is applied inside the `LightningDataModule` at training/inference time. Constants (`AMP_MIN`, `AMP_MAX`, `EPS`) are the single source of truth in `src/utils/constants.py`. Never apply normalization twice.
- MERLIN reconstruction: the network sees one component and reconstructs at full power. Average Real and Imag predictions (factor 0.5) when computing intensity for metrics. See `docs/Method.md` for the derivation.
- Visualisations → log-scale intensity. Metrics (PSNR, ENL, …) → linear amplitude. Never mix the two.

### Repository Layout Notes

- `data/` is **git-ignored** — never commit datasets.
- `CompressAI/compressai/` is a local clone used for context only.
- `context/` contains reference implementations and documentation context for the developer — not production code, not imported anywhere.
- Results land in `results/` (git-ignored). FPGA compiled models: `results/fpga/compiled_models/<model>/`; active model symlink: `results/fpga/active_model/`.

## 🤖 AI Assistant Protocol

1.  **Tool usage**: Preface any automated tool call (test run, Docker step, board SSH) with a one-line description of what it does and why.
2.  **Planning**: Use `manage_todo_list` for multi-step tasks. Mark one item in-progress at a time; mark it completed immediately after finishing.
3.  **Editing**: Keep changes minimal and focused. Do not reformat unrelated files. Skip changes that are purely cosmetic — ruff handles formatting automatically.
4.  **Documentation consistency**: After any change, check the relevant documentation files and update them if needed. Skip if the change is minor and does not affect the overall understanding.
5.  **Verification**: This codebase has no meaningful automated test suite — verify changes by running the affected script/notebook and checking its output, not `make test`.
6.  **Reversibility**: Ask before deleting files, force-pushing, dropping datasets, or any action that cannot be undone. Local edits and test runs are fine without asking.
7.  **Markdown tables**: Always use spaces around separators — `| --- | --- |` not `|---|---|` (markdownlint MD055/MD056).
8.  **C++ explanation**: When dealing with C++ code, provide clear explanations and context in the chat.
