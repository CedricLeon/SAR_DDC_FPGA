# Synthetic Aperture Radar (SAR) Despeckling and Data Compression (DDC) on Field Programmable Gate Arrays (FPGA)

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Config: Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

Yes, that's a lot of acronyms. But now you know why it's called SAR_DDC_FPGA.
This project implements the solution presented by Amao-Oliva et al. [1] available at [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0924271624004866) on FPGA.

## TODOs

*I'll use this section as a TODO list, including ideas for future projects.*

### Small things to check/investigate

- [ ] EPD values > 1 in [RD-curve_ablation.ipynb python cell 11](notebooks/RD-curve_ablation.ipynb). EPD max value is supposed to be 1, how is that possible? I don't remember having that problem before re-evaluating all the old runs.

### Small things to add/fix

- [ ] Add in README that the syntax is Python 3.8 compatible because it is used by the Vitis-AI container and we can't bump it. (It mostly implies using `Option[]` and `Union[]` from `typing` instead of `|`)
- [ ] Use [rootutils](https://github.com/ashleve/rootutils) better, for example using `find_root()` instead of `setup_root(Path(__file__).resolve().parent.parent` ...

### Long-term Experiments/Upgrades

#### About SAR_DDC

- [ ] Maybe there is a way to avoid the concatenation and average latent representations before hyperprior
- [ ] `compressai` seems to have `ResidualBlockWithStride` and `ResidualBlockUpsample` that are probably used in other architectures. Maybe check if they perform better than our manual ones.

#### About FPGA deployment

- [ ] vaiq_pytorch should allow hardware-aware and partial quantization, see the [doc](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Hardware-Aware-Quantization-Strategy). Alternatively, one can configure the quantization quite a bit with a JSON file, see the [doc](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Quantization-Strategy-Configuration?tocId=rGCaO9QY6VvNbAJV7l9i7Q)
- [ ] Try the `fast_finetuning` option, or if still bad, the QAT

## Method

The pipeline relies on Pytorch Ligthning on [Compressai](https://github.com/InterDigitalInc/CompressAI) [2] to implement Hyper-autoencoders solutions based on Johannes Ballé's work [3-5].
In addition, the despeckling task is inspired from MERLIN's self-supervised training pipeline [6].

### MERLIN Theory

**The big picture (mostly written by ChatGPT)**:
Dalsasso et al. introduce MERLIN, a fully self-supervised strategy for training deep despeckling networks directly on single-look complex (SLC) SAR images. By exploiting Goodman’s speckle model—which shows that the real and imaginary components of an SLC pixel are two independent, Gaussian-distributed realizations with variance proportional to the local reflectivity $r$—they train a U-Net to predict pixel-wise variance maps (i.e., the effective “blurred” reflectivity $r$) from one component (say, the real part) and evaluate the loss on the other component (the imaginary part).

Taking a SLC SAR image with Real part $\tilde{a}$ and Imaginary part $\tilde{b}$ (We use tilde notation to indicate the transformation from the SAR transfer function **H**, see MERLIN Eq.(4) and (5)). Its intensity $I$ is
$$
I = \tilde{a}^2 + \tilde{b}^2
\tag{1}
$$
MERLIN aims at depesckling, i.e., reconstructing the underlying reflectivity image $\tilde r$, with a self-supervised model $f_\theta()$ as $\frac{f_\theta(\tilde{a}^2) + f_\theta(\tilde{b}^2)}{2}$.
To achieve this, the model is trained with the loss function below, where $k$ represent the iteration over pixels:
$$
\mathcal{L}(\tilde{r}, \tilde{b}) = \sum_k \frac{log(\tilde{r}_k)}{2} + \frac{\tilde{b}_k^2}{\tilde{r}_k}
\tag{2a}
$$
In practice, it is useful to work in log-scale to reduce the dynamic range. We use the check notation to represent log-scaled variables: $\check r = log~\tilde r$, $\check a = log|\tilde a|$ and $\check b = log|\tilde b|$.
> Note: The absolute value operator |.| is only used for mathematical correctness as $\tilde a$ and $\tilde b$ can have negative values. In practice it is not needed as these values are squared before being fed to the network. As a last detail, it is also necessary to add a small $\epsilon$ to $\tilde a$ or $\tilde b$, to avoid $log(0)$. In python, this is done with `1e-6` or, preferably, `np.spacing(1)`.

Working in log-scale implies to modify the loss:
$$
\mathcal{L}(\check r, \check b) = \sum_k \frac{\check r_k}{2} + exp(2 \check b_k - \check r_k)
\tag{2b}
$$

**A few mode details**:  ==@TO UPDATE==

- **Further normalization**. In addition of the log-scale, it is beneficial to "normalize the images using a fixed affine transform".

> What this mean is using the minmax formula, but not with the minimum and maximum values of the image (because of the strong outliers, the whole distribution would end up being very narrow and the network would struggle differentiating values). Instead percentile values are used, typically 5 and 95%.

- **Misconception about the data range of the output**. As (most of) the data lies between $[0;1]$, one could expect the reconstruction of the network be in the same interval. However, the network learns to map from noisy realizations of $\tilde a \sim \mathcal{N}(0,r/2)$ to the total reflectivity $r$, not to $\frac{r}{2}$. Same for $\tilde b$.

> This means that to compare 2 images of the same scale one must visualize the Intensity $I = \tilde{a}^2 + \tilde{b}^2$ and a single prediction , e.g., $f_\theta(\tilde{a}^2)$. During inference both network estimations are averaged to decrease the variance of the reconstructed reflectivity, but if a simple proxy is needed, one could use only one of the reconstructions.

- **An $ln(2)$ offset**. @TODO. One thing I still do not understand comes from the loss optimization. If the network tries to optimize Eq. (2b), that I simplify with $\hat x$ the reconstruction and $y$ the target:

$$
\text{Optimizing}~f(\hat x, y) = \frac{\hat x}{2} + exp(2 y - \hat x) \\
\text{Means finding where}~f'(\hat x, y): \frac{1}{2} - exp(2 y - \hat x) = 0 \\
\Leftrightarrow 2y - \hat x = ln(\frac{1}{2}) \\
\Leftrightarrow \hat x = 2y + ln(2)
$$
> Given that we reconstruct the estimated reflectivity and not half of it $2y$ makes kind of sense, however, the ln(2) offset does not have an explanation to me.

### Data

TerraSAR-x StripMap (SM) SSC (Single Look Slant Range Complex) images downloaded from [ESA's platform](https://earth.esa.int/eogateway/catalog/terrasar-x-esa-archive).
> Submitting a form is required to access the data (2 days max delay).

#### Contents of the data/visualization folder

*This information is also detailed in the README specific to `data/` but it is not tracked by Git.*
This folder contains large tiles used for reconstruction visualization.
All the files were created by the notebbok `create_evaluation_references.ipynb`.
Each large tile is stored in its own folder named `<Region>_<Optional: coordinates>`.

Inside these folders, each `.npy` correspond to a different product and follows the naming convention:

`<data scale or format>_<product type>.npy`

For example, inside the `Hamburg_[11000:12024-8500:9524]` folder you will find:

- `sym_Noisy.npy` correspond to the noisy patch symmetrized (linear scale) of the Hamburg product, over these specific coordinates.
- `linA_MERLIN-DDS.npy` corresponds to the product denoised by MERLIN Deep Despeckling over the same region, and stored in linear amplitude scale.

`MERLIN-DDS` is a specific folder where I stored the images I denoised with the original checkpoint from [Hi-Paris GitHub](https://github.com/hi-paris/deepdespeckling/).

#### Pre-processing ==@TO UPDATE==

`script/TSX_dataset_creation.py` creates `hdf5` datasets more convenient for training that re-processing and patchifying the entire TSX SSC images every time.
In particular, each .cos file present in `data/TSX_cos_files/` is open, images are patchified, symmetrized, strong scatterers are preserved, normalized, and the whole set of resulting patches is split in training/validation/test datasets.
The data is pre-processed into train/val/test HDF5 files using the `script/TSX_dataset_creation.py`. This script has many options, use `--help` for details.

#### Datasets naming

I'll try to follow the dataset naming conventions below: `<split_type><nb_images>_<preservation>_<normalization>.hdf5`, where:

- `split_type` is how the patches where split ("randomsplit" or "spatialsplit")
- `nb_images` corresponds to the number of `.cos` files used for the dataset (typically 5)
- `preservation` indicates if strong point-like scatterers were preserved following (@TODO add equation in [[Math.md]]) and the threshold, for example "nopres" or "pres60dB".
- `normalization` consists of `norm<normalization_percentiles><log_base><clipped>`
  - `normalization_percentiles` indicates if min-max normalization was performed on the patches, and which percentiles were used as "min" and "max". For example, "nonorm" or "norm1".
  - `log_base` the logarithmic base was used in the normalization, either "db" (`np.log10()`) or "nat" (natural: `np.log()`)
  - `clipped` is "clip" or "" depending if the data was clipped to [0,1] or not. If `normalization_percentiles=0` clipping is deactivated by default, as after a minmax normalization the data lies already in [0,1]
Examples:
- "randomsplit5_pres60dB_norm5db" was processed with preservation of scatterers with signals above 60dB, the patches were placed in log10 base before being "min-maxed" with p5 and p95 (i.e., value 0 corresponds to p5 and value 1 to p95)
- "randomsplit5_nopres_norm0nat" was processed without scatterer preservation but with normalization to natural logarithm and traditional min-max (0 means 0%)

#### Training

After downloading, unzipping, accessing the `.cos` files (deep in the archive in `IMAGEDATA/`), and copying all CoSAR file format in a common folder, e.g., `data/TSX_cos_files/`, datasets are created to ease training.
> The `data` folder is ignored by Git, but typically contains softlinks to the datasets (to avoid several copies of big files).

==To place somewhere else==:
`dataset_creation.py` creates `hdf5` datasets more convenient for training that re-processing and patchifying the entire SAR SLC images every time.
In particular, each .cos file present in `data/TSX_cos_files/` is open, images are symmetrizeda and patchified, and the whole set of resulting patches is split in training/validation/test datasets.
> Following MERLIN's pipeline: Patches are not normalized between 0 and 1. The log-normalization happens before feeding them to the network. They are also de-normalized afterwards.

#### Testing / Inference

==@TODO==, implement a script that automatically find all TSX archives, unzip them, extract the `.cos` file, pre-process the data,loads the model, despeckle and compress the images, and why not reconstruct them. (Will most likely follow MERLIN's inference workflow)

### Hardware implementation ==@TO UPDATE==

We use Vitis AI [7] for the FPGA deployment. See [Vitis-AI_journey.md](Vitis-AI_journey.md) for details about my struggles.

### Benchmarking

Performance benchmarks measure latency, throughput, and power consumption across GPU (RTX A4000), host CPU (x86), and FPGA (Xilinx ZCU102).  All results are stored as JSON in `results/benchmark/<model_name>/` and analysed in `notebooks/benchmark_analysis.ipynb`.

#### Prerequisites

- A compiled FPGA model in `results/fpga/active_model/` (run `deploy.py` first).
- ZCU102 accessible via `ssh ZCU102` (passwordless SSH configured, see `docs/Vitis-AI_journey.md`).
- For GPU power: `nvidia-smi` available.  For CPU RAPL power: run as root or `sudo chmod o+r /sys/class/powercap/intel-rapl/*/energy_uj`.
- **For meaningful power results**: cold-reboot the ZCU102 before each run.

#### Full benchmark

```bash
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ \
    --warmup 20 --iters 100 \
    --power --idle-baseline 10 \
    --power-hz-gpu 10 --power-hz-fpga 50
```

Runs all 5 scenarios (`full`, `compress`, `decompress`, `nn_only`, `entropy_only`) on GPU + CPU (host) + FPGA (ZCU102 via SSH), with a 10 s idle power baseline before each scenario. *Estimated runtime: ~15 min.*

```bash
# GPU + CPU only (no board required)
python scripts/fpga/run_full_benchmark.py --model-dir results/fpga/active_model/ --no-fpga

# FPGA only (model already on board, skip transfer)
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ --no-gpu --no-cpu --skip-transfer
```

#### Protocol for reproducible results

1. **Cold-reboot the ZCU102** — ensures no residual DPU/VART state from previous runs.
2. **Wait ~60 s** after PetaLinux boot before starting the benchmark.
3. **Close GPU workloads** on the host (`nvidia-smi` should show 0 MiB compute usage).
4. Run the command above — idle baselines are captured automatically per scenario.
5. For publication: **repeat 3×** (reboot between runs) and report mean ± std.

#### Interpreting results

Key fields in each JSON file:

| Field | Meaning |
| --- | --- |
| `latency_total_mean_ms` | Per-tile end-to-end wall time |
| `latency_dpu_total_mean_ms` / `latency_gpu_total_mean_ms` | Hardware accelerator time only |
| `latency_cpu_total_mean_ms` | CPU-side entropy coding + pre/postprocessing |
| `power.groups_avg_w.DPU_fabric` | VCCINT+VCCBRAM (DPU switching power) on FPGA |
| `power.groups_avg_w.PS_compute` | ARM A53 APU power (entropy coding) on FPGA |
| `power.groups_avg_w.MPSoC` | PL + PS total SoC power on FPGA |
| `power.idle_board_total_avg_w` | Idle baseline for dynamic power subtraction |

**Dynamic power** = `power.groups_avg_w.X` − `power.idle_groups_avg_w.X` (post-hoc from JSON).

> **Key finding**: CPU-side Gaussian Conditional entropy coding (C++ rANS) accounts
> for ~60 % of total latency on FPGA.  Use the `entropy_only` scenario to isolate it
> and `nn_only` to isolate pure DPU inference.

#### Power measurement scope

| Platform | Instrument | Scope | Unmonitored |
| --- | --- | --- | --- |
| **GPU** | `nvidia-smi` | Full GPU board power | Host CPU, memory, motherboard |
| **CPU** | Intel RAPL | CPU package + DRAM | Motherboard, fans, storage |
| **FPGA** | 18× TI INA226 + 3× Maxim PMBus | PL, PS, MGT + DDR4/UTIL rails | 6 secondary bias rails (< 500 mW, workload-invariant) |

See `docs/performance_benchmark_implementation.md` §4 for the complete technical reference, including INA226 register configuration, I2C bus topology, rail-to-sensor mapping, and paper-ready measurement descriptions (§14).

#### References

- [1] Joel Amao-Oliva, Nils Foix-Colonier, Francescopaolo Sica. (2024). Joint compression and despeckling by SAR representation learning. ISPRS Journal of Photogrammetry and Remote Sensing.
- [2] J. Bégaint, F. Racapé, S. Feltman, and A. Pushparaja, “CompressAI: a PyTorch library and evaluation platform for end-to-end compression research,” Nov. 05, 2020, arXiv: arXiv:2011.03029. doi: 10.48550/arXiv.2011.03029.
- [3] D. Minnen, J. Ballé, and G. D. Toderici, “Joint Autoregressive and Hierarchical Priors for Learned Image Compression,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2018. Accessed: Jun. 14, 2023.
- [4] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” presented at the International Conference on Learning Representations, Feb. 2018. Accessed: Apr. 10, 2024.
- [5] J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end Optimized Image Compression,” Mar. 03, 2017, arXiv: arXiv:1611.01704. doi: 10.48550/arXiv.1611.01704.
- [6] Dalsasso, E., Denis, L., & Tupin, F. (2022). As if by magic: Self-supervised training of deep despeckling networks with MERLIN. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–13. <https://doi.org/10.1109/TGRS.2021.3128621>
- [7] AMD Vitis™ AI Software. (2019). AMD. <https://www.amd.com/en/products/software/vitis-ai.html>

## Workflows

***

Quick reference for all recurring workflows. All commands run from the project root (`DDC_FPGA/`) unless noted.
Conda environment: `SAR_DDC` for local work; Vitis-AI Docker container (Python 3.8) for FPGA quantization (handled transparently by `deploy.py`).

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

Output follows the naming convention `<split>_<nb_imgs>_<preservation>_<normalization>.hdf5` (see README §Data).

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

### 3 · FPGA deployment — single run

**Script**: `scripts/fpga/deploy.py`
**When**: Compile + transfer + infer a single training checkpoint to the ZCU102.
**Requires**: Vitis-AI Docker container (started automatically), passwordless SSH to ZCU102.

```bash
# Full pipeline (compile → transfer → infer → fetch)
python scripts/fpga/deploy.py --run-dir DDC_FPGA/logs/train/sar_ddc/hyperprior/runs/<date>/<id>

# Skip phases selectively
python scripts/fpga/deploy.py --run-dir <...> \
    --skip-compile          # skip quantization + xmodel generation
    --skip-transfer         # skip SCP to board
    --skip-infer            # skip board inference
    --skip-fetch            # skip fetching results back

# Optional compile flags
python scripts/fpga/deploy.py --run-dir <...> \
    --arch ZCU102           # or Leopard (default: ZCU102)
    --inspect               # run DPU inspector before calibration
    --eval-float            # evaluate float model
    --eval-quant            # evaluate quantized model
    --fast-finetune         # enable fast finetuning during calibration
    --image-graph           # generate SVG of compiled xmodel
    --subset 100            # number of test samples for inference (default: 100)
```

Results land in `results/fpga/compiled_models/<model_name>/` and are accessible via the `results/fpga/active_model/` symlink.

### 4 · FPGA deployment — batch

**Script**: `scripts/fpga/batch_deploy.py`
**When**: Compile + deploy a set of W&B runs matching filters defined in `FILTERS_CONFIG`.

```bash
# Preview matched runs (dry run)
python scripts/fpga/batch_deploy.py --tag <label> --dry-run

# Deploy all matching runs (skips already-compiled models)
python scripts/fpga/batch_deploy.py --tag <label>

# Deploy specific run IDs directly (bypasses FILTERS_CONFIG)
python scripts/fpga/batch_deploy.py --tag <label> --run-ids <id1> <id2>

# Force recompile even if model already exists
python scripts/fpga/batch_deploy.py --tag <label> --force-recompile
```

Supports all phase-skip and compile flags from `deploy.py` (`--skip-transfer`, `--arch`, `--eval-float`, etc.).
Batch log: `results/fpga/batch_deploy/<tag>_<timestamp>.log`.

### 5 · Benchmarking

**Script**: `scripts/fpga/run_full_benchmark.py`
**When**: Measure latency, throughput, and power across GPU + CPU + FPGA.
**Requires**: compiled model in `results/fpga/active_model/` (run `deploy.py` first).

```bash
# Full benchmark — all platforms, with power measurement
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ \
    --warmup 20 --iters 100 \
    --power --idle-baseline 10 \
    --power-hz-gpu 10 --power-hz-fpga 50

# GPU + CPU only (no board required)
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ --no-fpga

# FPGA only (model already on board)
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ \
    --no-gpu --no-cpu --skip-transfer
```

Runs 5 scenarios: `full`, `compress`, `decompress`, `nn_only`, `entropy_only`.
Results: `results/benchmark/<model_name>/` as JSON. Analysis notebook: `notebooks/benchmark_analysis.ipynb`.
