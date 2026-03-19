# DDC_FPGA AI Instructions

## Project Overview
This project implements SAR Despeckling and Data Compression (DDC) on FPGA using PyTorch Lightning, Hydra, and CompressAI. It focuses on the MERLIN self-supervised strategy for despeckling and hyper-autoencoders for compression.

## ⚡ Execution Environments & Boundaries
This project operates across two distinct environments. You must distinguish between them:

### 1. Local Development (Training & Preprocessing)
- **Environment**: Conda environment `SAR_DDC`.
- **Tasks**: Model training, data preprocessing, unit tests, general code editing.
- **Activation**: Assume `conda activate SAR_DDC` is available for these tasks.

### 2. FPGA Deployment (Quantization & Compilation)
- **Key Script**: `scripts/fpga/quantize.sh` prepares the model for deployment. It calls `scripts/fpga/model_quant.py` and other scripts to structure the files to be ported to the FPGA.
- **Environment**: Vitis-AI Docker Container (Strict Python 3.8) (start with `scripts/vitis-ai-automation/setup_container.sh`).
- **Tasks**: Quantization, Compilation, `xir` graph manipulation, entropy models export, file organization.
- **Constraint**: Code **MUST** remain Python 3.8 compatible to run here.

### 3. Physical Hardware (Xilinx Board)
- ⛔ **RESTRICTION**: Do not SSH or run inference on the physical board without explicit user instruction.
- The user manages board access and dataset transfers manually.

## 🏗 Architecture & Patterns

### Core Frameworks
- **PyTorch Lightning**: Modules in `src/models/` often use **manual optimization** (`self.automatic_optimization = False`), especially `SARDDCModule` which requires handling multiple optimizers (main model + entropy bottleneck).
- **Hydra**: Configuration management. Defaults in `configs/train.yaml`, overrides in `configs/experiment/`.
- **CompressAI**: Integrated for compression layers and entropy models.
  - **GPU**: Uses likelihood estimation (entropy bottleneck) for training. No bitstream generation.
  - **FPGA**: Uses real bitstream generation via C++ extension (`ans.so`) and rANS.

### Critical Constraints
- **Python 3.8 Compatibility** (Hardware Requirement):
  - ✅ **USE**: `from typing import Union, Optional, List, Tuple`
  - ❌ **AVOID**: `type | None` or `list[int]` (PEP 604/585 are not supported).
- **Vitis-AI Support**:
  - **Ops**: No `GDN` (not adapted yet, use `ReLU`), without `LowerBoundFunction` (use `torch.max`).
  - **Layers**: `ConvTranspose2d` must have `output_padding=0`.
  - **Entropy**: DPU cannot handle entropy coding. Models are split into 4 subgraphs (Encoder, HyperEnc, HyperDec, Decoder) + CPU based entropy coding (C++ `ans.so`).

## 🛠 Developer Workflow

### Training
Run training via the entry point with Hydra overrides:
```bash
# Run a specific experiment (recommended)
python src/train.py experiment=<example>

# Debug mode
python src/train.py experiment=<example> debug=fdr
```

### Data Pipeline
- **Script**: `scripts/dataset/create_dataset.py` processes raw CoSAR/SSC files into patches.
- **Format**: HDF5 with naming convention `<split>_<nb_imgs>_<preservation>_<normalization>.hdf5`.
- **Math**: All models expect data in log-scale, normalized via min-max. The normalization is done in the LightningModules. Be mindful about double-normalization and correctly denormalizing outputs.
- **Reconstruction Scaling**: Due to the MERLIN self-supervised loss strategy, the network reconstructs reflectivity with full power despite partial input. When computing reconstruction intensity from Real/Imag outputs, average them (factor 0.5) to match the original signal scale.
- **Visualization**: All visualizations must be done with log-scale intensity data, opposed to metrics computations that must be done in linear scale, on amplitude data.

### Testing
- **Fast Tests**: `pytest -k "not slow"` (or `make test`)
- **Full Suite**: `pytest` (or `make test-full`)

## 🤖 AI Assistant Protocol

1.  **Tool Usage**: Preface any automated tool call (e.g., tests, or container start) with a one-line preamble describing what you will do and why.
2.  **Planning**: Use the `manage_todo_list` tool to create a plan for multi-step tasks and update it as you progress.
3.  **Editing**: Keep changes minimal and focused. Do not reformat unrelated files. Skip any changes solely linked to code formatting, ruff automatically handles that.
