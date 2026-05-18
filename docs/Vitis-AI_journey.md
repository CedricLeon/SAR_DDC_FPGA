# Vitis AI journey

I'll use this file as a journal, just to keep track of what I tried and when.
Once I understand the toolchain and its processes better, I'll make a step-by-step instructions for deployment, like so:

---

## Benchmark Analysis — Figure Polish & Multi-Model Expansion (2026-05-01)

Tracking the cleanup and expansion of `notebooks/benchmark_analysis.ipynb`, creation of a shared
color system, and design of a new cross-model notebook.

Tasks are ordered for linear execution: color system first (everything else depends on it),
then power plot fixes, then new notebook.

---

### Step 1 — Color system: create `configs/plots_colors.json` ✅

`configs/plots_colors.json` created. Okabe-Ito base palette, colorblind-safe.
Explicit semantic keys covering all color use-cases across all three notebooks:

| Key group | Used in |
| --- | --- |
| `platforms` | any figure with FPGA/GPU/CPU bars (idle + dynamic variants) |
| `fpga_power_groups` | §8a FPGA stacked power composition |
| `gpu_power_domains` | §8b GPU board + CPU RAPL stacked bars |
| `nn_vs_cpu_domain` | §5b NN-vs-CPU latency split |
| `latency_steps` | §5 per-step stacked horizontal bars |
| `architectures` | `hardware_model_comparison.ipynb` |
| `activations` | `RD-curve_ablation.ipynb` |
| `output_padding` | `RD-curve_ablation.ipynb` |
| `metrics` | `compare_gpu_fpga.ipynb` |
| `compare_gpu_fpga` | `compare_gpu_fpga.ipynb` |

---

### Step 2 — Migrate `benchmark_analysis.ipynb` to shared colors + power plot fixes

All sub-tasks below are part of a single editing pass on `benchmark_analysis.ipynb`:

- [x] **2a · Load colors in config cell (cell 3)** *(agent)*
  Add `C = json.load(open(ROOT_DIR / "configs" / "plots_colors.json"))` to the config cell.
  Replace all hard-coded hex literals with `C[group][key]` lookups.

- [x] **2b · §8a — Fix idle line colors, labels, drop MGT bar** *(agent)*
  - Match idle line colors to their group bar: PL line → `C["fpga_power_groups"]["PL"]`,
    PL+PS line → `C["fpga_power_groups"]["PS"]`, TBP line → `C["fpga_power_groups"]["TBP_line"]`.
  - Labels: replace `"← idle TBP  ≈ 11.0 W"` with short black text `"idle TBP"` / `"idle PL"` /
    `"idle PL+PS"`, smaller font, placed so they don't overlap bars.
  - Remove MGT as a plotted bar; fold its value into a caption footnote:
    `"TBP includes ~0.11 W MGT (constant, transceivers unused)."`
  - Add FMC to TBP: `benchmark_fpga.py` currently excludes VADJ_FMC from TBP; update
    `_extract_power` to include FMC in TBP sum and update `_FPGA_FMC_RAILS` from dead code
    to active use. Add footnote documenting measured FMC value.

- [x] **2c · §8b — Redesign GPU power bar as "GPU system" stacked + fix CPU bar** *(agent)*
  - Merge GPU board (NN domain, blue) + CPU RAPL (entropy/OS domain, orange) into one
    stacked "GPU system" bar per scenario — directly comparable to FPGA TBP in §8a.
  - Keep a separate standalone bar for the CPU-only benchmark scenario (CPU platform).
  - Annotation format: value label inside each sub-bar segment (idle / dynamic),
    total on top. Replace `"X (+Y)"` format.

- [x] **2d · Drop §8d (DPU_fabric vs PS_compute cell)** *(agent)*
  Delete cell 22 entirely.

- [ ] **2e · Long plot titles → captions** *(me)*
  Move verbose in-plot titles/legends to markdown caption cells.

---

### Step 3 — Migrate `RD-curve_ablation.ipynb` to shared colors

- [x] **3 · Color migration** *(agent)*
  Add `json.load` config cell. Replace `colors = {"with_out_pad": "tab:blue", "no_out_pad": "tab:red"}`
  and linestyle dict with `C["output_padding"]` and `C["activations"]` lookups.

---

### Step 4 — Migrate `compare_gpu_fpga.ipynb` to shared colors

- [x] **4 · Color migration** *(agent)*
  Add `json.load` config cell. Replace implicit matplotlib defaults for GPU/FPGA curves
  with `C["compare_gpu_fpga"]` and `C["metrics"]` lookups.

---

### Step 5 — Create `notebooks/hardware_model_comparison.ipynb`

- [x] **5a · Scaffold notebook** *(agent)*
  New notebook with config cell, data loader that scans all
  `results/benchmark/<model_name>/` directories, parses model name into
  (architecture, lambda, seed, activation) columns, builds unified `df_all` DataFrame.

- [x] **5b · Latency vs lambda plot** *(agent)*
  Line chart: x = lambda, y = latency (ms), one line per (scenario, platform).
  Shows entropy-coding lambda-dependence (`full` increases with lambda, `nn_only` flat).

- [x] **5c · Architecture comparison bars** *(agent)*
  Side-by-side latency/energy bars: ResSHyp vs SHyp per scenario and platform.

- [x] **5d · Hardware cost scatter** *(agent)*
  x = PSNR (with BPP in parenthesis), y = latency or energy.
  One point per (architecture, lambda, platform). Design to be refined during implementation.

---

### Notebook scope (stable)

| Notebook | Scope |
| --- | --- |
| `benchmark_analysis.ipynb` | Single-model deep-dive: latency, power, energy, throughput |
| `hardware_model_comparison.ipynb` | Cross-model: latency/energy vs lambda, architecture |
| `compare_gpu_fpga.ipynb` | Quality delta: GPU vs FPGA reconstruction quality |
| `RD-curve_ablation.ipynb` | Training ablation: activation functions, output padding |

---

## W&B config vs local Hydra config — known discrepancy (2026-05-01)

When new config keys are added to the model (e.g. `no_residual_blocks`), old runs already
stored on W&B are missing that key in their W&B config (it reads as `None` via `OmegaConf.select`).

`scripts/fix_wandb_run_names.py --fix-config` backfills the correct value (`False`) into the
**W&B config** for those runs. However, the corresponding **local `.hydra/config.yaml`** files
on disk are not updated. This creates a permanent cosmetic discrepancy:

| Source | Key present? | Value |
| --- | --- | --- |
| W&B config (after fix) | ✓ explicit | `False` |
| Local `.hydra/config.yaml` | ✗ absent | (inherits Python default: `False`) |

**This is safe**: `update_wandb_runs.py` and `deploy.py` load model configs from the local
`.hydra/config.yaml`, not from W&B. Hydra falls back to the Python default (`False`) for absent
keys, which is identical to the backfilled value. No model will be instantiated differently.

The discrepancy is simply accepted as a cost of retroactively adding config fields. If absolute
consistency is required in the future, local configs would need to be patched manually.

## Power Measurement Deep-Dive and Benchmark Infrastructure (2026-04-30)

A focused review and improvement of the power measurement methodology across
`benchmark_fpga.py`, `benchmark_gpu.py`, and `run_full_benchmark.py`.

### What changed

**Bug fix — busy-wait in polling loops**:
Both `INA226PowerSampler._poll_loop` and `PMBusRailSampler._poll_loop` used a
`time.perf_counter()` busy-wait between polls.  On the ARM A53, this pinned a
full CPU core continuously, competing with VART inference threads and inflating
PS-side power readings.  Fixed to `time.sleep(remaining)` in both samplers.

**Power group hierarchy rewritten** to match the Xilinx EDA365 reference formula:

- Old (wrong): `PL_total` included VADJ_FMC and MGT rails; `PS_total` included
  MGTRAVCC/MGTRAVTT.
- New (correct): `PL` = VCCINT+VCCBRAM+VCCAUX+VCC1V2+VCC3V3; `PS` = 8 ARM/DDR
  rails; `MGT` = 4 transceiver rails; `MPSoC` = PL + PS (MGT excluded — the
  transceiver subsystem is not used by the DPU); `TBP` = MPSoC + MGT + peripherals.

**Default idle baseline changed from 0 → 10 s** in all three scripts
(`--idle-baseline` CLI default).

**INA226 hardware configuration confirmed on board** via Python I2C RDWR:
All 18 sensors have `CFG = 0x4327` (4× hardware averaging, Vct = Ict = 1100 µs,
hardware update period = 8.8 ms).  The effective sysfs poll rate is ~5 Hz (not the
requested 50–100 Hz) due to I2C bus round-trip overhead; 200 samples over 41 s is
still sufficient for a stable mean.

**Unmonitored rails clarified**: 6 secondary bias/PHY rails have no telemetry at
all (PL_DDR4_VTT, PS_DDR4_VPP_2V5, VCCADC, MGT bias companions, USB/DP LDOs).
Their combined estimated contribution is < 500 mW and invariant to workload.

### Why it matters

The corrected power groupings eliminate MGT (~0.1 W) from the reported SoC compute
figure (`MPSoC`).  The busy-wait fix removes a confounding source of PS idle power
inflation.  Together, these changes make the dynamic power numbers (`ΔPL`, `ΔPS`,
`ΔMPSoC`) more accurate and directly comparable to datasheet estimates.

See `docs/performance_benchmark_implementation.md` §4 for the complete technical
details, including INA226 register decoding, I2C topology, and paper-ready
measurement descriptions.

---

## Automating deployment and evaluation with one orchestrator script (2026-03-02)

`deploy.py` is a Python orchestrator that manages the full compile → transfer → inference → fetch pipeline from the project root (`~/dev/Vitis-AI/DDC_FPGA/`) using the `SAR_DDC` conda environment. Requires [passwordless SSH access to ZCU102](#ssh-key-setup-passwordless-access-to-zcu102).

### Quick start

Set the target run via `--run-dir` (or by editing the `RUN_DIR` constant at the top of `deploy.py`), then:

```bash
# Full pipeline (default: ZCU102 arch, 100 inference samples)
python scripts/fpga/deploy.py --run-dir DDC_FPGA/logs/train/sar_ddc/hyperprior/<date>/<id>

# RUN_DIR constant acts as default when --run-dir is omitted
python scripts/fpga/deploy.py

# Skip phases selectively (e.g. rerun inference without recompiling)
python scripts/fpga/deploy.py --run-dir <...> --skip-compile [--skip-transfer] [--skip-infer] [--skip-fetch]

# Optional compile sub-flags
python scripts/fpga/deploy.py --run-dir <...> [--inspect] [--eval-float] [--eval-quant] \
                               [--fast-finetune] [--image-graph] \
                               [--arch ZCU102|Leopard]

# FPGA inference subset size
python scripts/fpga/deploy.py --run-dir <...> [--subset 100]
```

`RUN_DIR` is a module-level constant at the top of `deploy.py` and serves as the default for `--run-dir`. `--arch` is a short name (`ZCU102` or `Leopard`); `deploy.py` has a lookup dict to resolve the full JSON path.

### Batch deployment

To evaluate many W&B runs in sequence, use `batch_deploy.py`:

```bash
# Preview matched runs without deploying
python scripts/fpga/batch_deploy.py --tag relu_seed0 --dry-run

# Deploy all matching runs (skips already-compiled models by default)
python scripts/fpga/batch_deploy.py --tag relu_seed0

# Deploy specific run IDs directly (bypasses FILTERS_CONFIG, fast W&B fetch)
python scripts/fpga/batch_deploy.py --tag relu_seed0 --run-ids abc123 def456

# Force recompile even if model already exists
python scripts/fpga/batch_deploy.py --tag relu_seed0 --force-recompile
```

Edit `FILTERS_CONFIG` at the top of `batch_deploy.py` (same `(key, op, value)` tuple format as `update_wandb_runs.py`). The default filters only select DPU-compatible runs (`relu` activation + `no_output_padding=True`). Batch coordination log: `results/fpga/batch_deploy/<tag>_<timestamp>.log`.

### Phases

| Phase | Where | What |
| --- | --- | --- |
| 0. Container | Host | Ensure `vai_container` running + GPU-healthy |
| 1.1. Inspect *(opt)* | Container | `model_quant.py --quant_mode float --inspect` |
| 1.2. Eval float *(opt)* | Container | `model_quant.py --quant_mode float` |
| 1.3. Calibrate | Container | `model_quant.py --quant_mode calib` |
| 1.4. Eval quant *(opt)* | Container | `model_quant.py --quant_mode test` |
| 1.5. Deploy xmodel | Container | `model_quant.py --quant_mode test --deploy --subset_len 1 --batch_size 1` |
| 1.6. Compile | Container | `vai_c_xir -x ... -a <arch_json>` |
| 1.7. SVG graph *(opt)* | Container | `xdputil xmodel ...` |
| 1.8. Export entropy | Host | `_export_entropy_params()` — instantiates model, calls `.update()`, saves `.npz` |
| 1.9. Organize output | Host | `_organize_compiled_output()` — manifest, move to `compiled_models/`, update `active_model` symlink |
| 2. Transfer | Host→FPGA | Clean `active_model/` on FPGA, `scp -r` compiled model |
| 3. Infer | FPGA | `python3 inference_hybrid.py` (output streamed to terminal) |
| 4. Fetch | FPGA→Host | `scp -r` results back to `results/fpga/active_model/` |

Each phase after Phase 0 can be skipped with `--skip-compile`, `--skip-transfer`, `--skip-infer`, `--skip-fetch`.

### Design notes

#### scripts/fpga/ structure

```text
scripts/fpga/
    deploy.py                    # Main orchestrator — all host-side logic lives here
    batch_deploy.py              # Batch wrapper: fetch W&B runs, loop deploy.py per run
    model_quant.py               # Container-only quantization script (called via docker exec)
    inference_hybrid.py          # FPGA inference script (copied into compiled model dir at phase 1)
    inference_utils.py           # FPGA inference utilities (copied into compiled model dir at phase 1)
    entropy_models_inference.py  # FPGA entropy model (copied into compiled model dir at phase 1)
scripts/vitis-ai-automation/
    setup_container.sh           # Kept for interactive/manual debug use
    start_container_bg.sh        # Starts vai_container in detached mode for deploy.py
```

`model_quant.py` remains a subprocess called via `docker exec` — the CLI boundary is intentional because `pytorch_nndct` is only available inside the container and cannot be imported on the host. Phases 1.8 and 1.9 run on the host and are inlined as private functions in `deploy.py`.

#### Container setup and lifecycle

`deploy.py` needs the Vitis-AI Docker container (`vai_container`) running in detached mode. Xilinx's `docker_run.sh` hardcodes `-it` (interactive TTY), which blocks until the user exits — unsuitable for automation. `start_container_bg.sh` replicates the essential mounts (same `/workspace` bind, `/opt/xilinx`, `--gpus all`, `--network host`) but uses `--detach` and `sleep infinity` as PID 1 so it returns immediately. `setup_container.sh` is kept for interactive/debug use.

Lifecycle in `deploy.py`:

1. Check if `vai_container` is running: `docker ps --filter name=vai_container`
2. If running, check GPU health: `docker exec vai_container nvidia-smi`
3. If GPU dead (NVML error): kill with `docker rm -f` and restart.
4. If not running: call `start_container_bg.sh`, then run one-time `pip install` via `docker exec`.

All container commands are executed via:

```bash
docker exec vai_container bash -c "
  source /opt/vitis_ai/conda/etc/profile.d/conda.sh &&
  conda activate vitis-ai-pytorch &&
  export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6:\$LD_PRELOAD &&
  cd /workspace &&
  <command>"
```

Path translation: host paths are converted to container paths via `host_path.relative_to(VITIS_AI_ROOT)` where `VITIS_AI_ROOT = PROJECT_ROOT.parent` (the `/workspace` mount point inside the container maps to `~/dev/Vitis-AI/` on the host).

#### Host-side steps (phases 1.8 and 1.9)

Both steps run in the SAR_DDC environment on the host, not in the container.

- `_export_entropy_params()`: instantiates `ResidualScaleHyperpriorPatched`, calls `.update()` to populate CDF/quantile tables, and saves `entropy_params.npz` alongside the compiled model.
- `_organize_compiled_output()`: loads the Hydra config, derives the model name (e.g. `ResSHyp-relu_s2_L200_pt`), saves `train_config.yaml` and `manifest.json`, moves the compiled directory to `results/fpga/compiled_models/<name>/`, and updates the `active_model` symlink. `deploy.py` then reads `manifest.json` via the symlink to get the model name for phases 2–4.

#### FPGA transfer and result fetching

`~/SAR_DDC/active_model/` on the FPGA is a plain directory — one model at a time to avoid OOM. Old files are wiped before each deployment to prevent stale artifacts. `scp -r` follows the `active_model` symlink on the host, so the actual compiled model directory is transferred.

`inference_hybrid.py` writes results to `~/SAR_DDC/active_model/results/` on the FPGA. Fetched back to `results/fpga/active_model/results/` on the host — because `active_model` is a symlink pointing to `compiled_models/<model_name>/`, results land inside the correct model folder automatically.

### Logging: the `Tee` class

The `Tee` class in `deploy.py` is named after the Unix `tee(1)` command, which reads from stdin and writes simultaneously to stdout and one or more files — like a T-shaped pipe fitting in plumbing. The class intercepts `sys.stdout` so that every `print()` call and every line streamed from a subprocess lands in both the terminal and the log file at once.

Two log files are produced per deployment run, both stored inside the compiled model folder:

- **`compile.log`** — Phase 0 + Phase 1 (container checks, all quantization/compilation steps). Written to a temporary file at `$VITIS_AI_ROOT/.deploy_compile_tmp.log` while the compile runs, then moved to the model folder after Phase 1.9 (once the final model name is known).
- **`deploy.log`** — Phases 2–4 (transfer, inference, fetch). Opened directly in the model folder.

Terminal verbosity is controlled by two constants at the top of `deploy.py`:

| Constant | Default | Controls |
| --- | --- | --- |
| `PHASE1_VERBOSE` | `False` | Docker / Vitis-AI compile output (very noisy) |
| `PHASE3_VERBOSE` | `True` | FPGA inference output |

Setting either to `False` silences that phase on screen while still writing every line to the file.

#### tqdm progress bar handling

`tqdm` writes to `stderr` by default, but since `run_in_container()` uses `stderr=STDOUT` (merging both streams into the pipe), progress bars flow through `Tee.write()`. Because `Tee` has patched `sys.stdout` with a file-like object, `tqdm` no longer detects a real TTY and emits each update as a separate `\n`-terminated line instead of rewriting the same line with `\r`.

The module-level regex `_TQDM_RE = re.compile(r"^\s*(\d+)%\|")` detects these lines. `Tee.write()` then applies split treatment:

- **Terminal**: writes `\r` + the stripped line so the bar overwrites itself in place; emits `\n` only at 100%.
- **Log file**: drops all intermediate updates (0 %, 25 %, …) and records only the final 100 % completion line, keeping logs human-readable without kilobytes of `\r`-polluted progress spam.

---

## Full deployment and evaluation of the models (January 2026)

### Requirements

- If needed, perform [onetime setups](#fpga-board-setup-one-time).
- Have a trained checkpoint (and its configuration)

### 1. Initialize Vitis-AI docker container

If the current `vai_container` is dead (see NVMH error where CUDA is not available), `exit` it and restart it with:

```bash
[HOST](DDC_FPGA) leon_ce@bart:~/dev/Vitis-AI/DDC_FPGA$ ./scripts/vitis-ai-automation/setup_container.sh
```

which is equivalent to what we did [in October 2025](#adapting-vitis-ai-docker-container-to-my-requirements-october-2025).

> Note: using this script to prepare the docker container implies that we won't see the name of our conda environment `(vitis-ai-pytorch)` in our terminal.
> However, it is already activated. This can be checked with `which python`.

### 2. Let Vitis-AI do its job

Deploying a model using Vitis-AI requires to inspect it, quantize it, deploy it, compile it and evaluate it at various stages of the process.
I have created a script that does all of that for use:

```bash
[HOST](vitis-ai-pytorch) vitis-ai-user@bart:/workspace$ ./DDC_FPGA/scripts/fpga/quantize.sh
```

> Output: a folder, e.g., `ResSHyp-relu_s2_L1000_pt` located in `results/fpga/compiled_models/`. The symlink `results/fpga/active_model` is updated to point to this folder

### 3. Transfer to Target

```bash
[HOST](vitis-ai-pytorch) vitis-ai-user@bart:/workspace$ scp -r DDC_FPGA/results/fpga/active_model/ root@10.0.0.2:/home/root/SAR_DDC/
```

### 4. Run Inference on FPGA

If needed open a connection to the Target:

```bash
[HOST] ssh root@10.0.0.2
```

Go into the deployed model directory and run inference

```bash
[TARGET] root@xilinx-zcu102-20222:~$ cd ~/SAR_DDC/active_model/
[TARGET] root@xilinx-zcu102-20222:~/SAR_DDC/active_model/# python3 inference_hybrid.py --xmodel ./*.xmodel --data ../data/test_1000.npy --subset 100
```

### 5. Retrieve Results to Host

```bash
[TARGET] root@xilinx-zcu102-20222:~/SAR_DDC/# scp -r results/ leon_ce@10.0.0.1:~/dev/Vitis-AI/DDC_FPGA/results/fpga/active_model/
```

## FPGA Board Setup (one-time)

### SSH key setup (passwordless access to ZCU102)

PetaLinux uses the **Dropbear** SSH daemon, which resolves `authorized_keys` relative to the `HOME`
environment variable — set to `/home/root/` on this board. The standard `ssh-copy-id` fails
silently because it writes to `/root/.ssh/` which Dropbear does not check.

```bash
# 1. Generate a key pair on host (no passphrase)
ssh-keygen -t ed25519 -f ~/.ssh/bart_to_zcu102 -C "bart-to-zcu102" -N ""

# 2. Manually push the public key to the correct path (one-time, with password)
ssh root@10.0.0.2 "mkdir -p /home/root/.ssh && chmod 700 /home/root/.ssh"
cat ~/.ssh/bart_to_zcu102.pub | ssh root@10.0.0.2 "cat >> /home/root/.ssh/authorized_keys && chmod 600 /home/root/.ssh/authorized_keys"
```

Then add to `~/.ssh/config` on the host:

```ssh-config
Host ZCU102
    HostName 10.0.0.2
    User root
    IdentityFile ~/.ssh/bart_to_zcu102
```

After this, `ssh ZCU102` and `scp ... ZCU102:/path/` work without a password.

### Compile the C++ rANS entropy encoder for the FPGA

To be able to use the rANS entropy coder to generate real bitstreams on the FPGA we need to compile the CompressAI custom C++ implementation into a shared library (`ans.so`). As the PetaLinux image of the ZCU102 has `g++` we do the compilation directly on the target to avoid cross-compilation hassles.

1. **Package the C++ environment (Host)**:

   ```bash
   cd scripts/fpga/deploy_cpp_entropy/
   ./setup_fpga_cpp.sh
   ```

   > Output: `fpga_cpp_pkg/`
2. **Transfer and Compile (Target)**:

   ```bash
   [HOST] scp fpga_cpp_pkg root@10.0.0.2:/home/root/SAR_DDC/
   [TARGET] cd fpga_cpp_pkg && make
   ```

   > Output: `ans.cpython-39-aarch64-linux-gnu.so`
3. **Install**:
   In order to be able to import the newly compiled library:

   ```bash
   cp ans.cpython-39-aarch64-linux-gnu.so ..
   ```

   Then I added `export PYTHONPATH=$PYTHONPATH:/home/root/SAR_DDC/` to the `~/.bashrc` and refreshed it with `source ~/.bashrc`.

#### Implementation verification

To verify that this implementation works I created `debug_entropy_dpu_equivalence.py` that runs on Host to compare reconstructions of the same checkpoint ran in "Training" mode, with likelihoods, and in "Inference" mode, i.e., the call of the C++ Entropy coder.

1. **Compile and make available the C++ ANS package:

   ```bash
   [Host] cd scripts/compare_FPGA_to_GPU/
   [Host] python setup.py build_ext --inplace
   ```

2. **Run the debug script**:

   ```bash
   [Host] cd ../..
   [Host] python scripts/debug_entropy_dpu_equivalence.py
   ```

### Dataset export to the FPGA

```bash
[HOST](DDC_FPGA) leon_ce@bart:~/dev/Vitis-AI/DDC_FPGA/$ python scripts/dataset/convert_h5_to_np.py --dataset_path data/processed_hdf5/TSX_spatial_splits_5_256x256/test.h5 --subset 500 --seed 42
# Afterwards transfer the dataset to the Target (It also took 6:41 mins)
[HOST](DDC_FPGA) leon_ce@bart:~/dev/Vitis-AI/DDC_FPGA/$ scp DDC_FPGA/data/processed_hdf5/TSX_spatial_splits_5_256x256/test_sub500_seed42.npy root@10.0.0.2:/home/root/SAR_DDC/data/test_sub500_seed42.npy
```

## Creating a DPU-friendly inference pipeline / Updating the model to be Vitis-AI-friendly

### Real Compression/Decompression in C++ (2026-02-17)

Instead of simulating BPP with likelihoods we implemented actual compression/decompression of latents with the rANS encoder. This file comes from `CompressAI/compressai/cpp_exts/rans/rans_interface.cpp` and is used inside CompressAI Python code using PyBind11. So Copilot created a script that allows to export everything necessary onto the FPGA and compile the file there (as the PetaLinux image has `g++`) whiich produces a shared library `ans.so` that we can access during inference.

### "Mocking" the behavior of the Entropy Models in `numpy` for the DPU (2026-02-09)

I got Copilot to create a couple of files that allowed me to export the series of parameters necessary for the entropy encoders/decoders to be used on the FPGA  `export_entropy_params.py` (mainly scale tables and other), as well as a `entropy_models_dpu.py`.
However, I realized later that was dummy because what we want to do on the FPGA is to perform real compression/decompression and generate Byte-strings not likelihoods.

### Dealing with the multiple DPU subgraphs (2026-01-16)

Currently the models passes Vitis AI inspection, quantization, and compilation, but I struggle to execute it on the FPGA.
That's because the full model is divided in **many** subgraphs, there are 19 DPU subgraphs and probably a lot more CPU ones.
Then I have several options:

1. Commit to the "Hybrid" execution: painfully rewrite my inference script to manually orchestrate the flow between CPU and DPU. This is very tedious and will lead to terrible performance.
2. Simplify the model: avoid all unsupported ops to get 1 Subgraphs (or significantly less so that the Hybrid execution is possible).
3. Register custom ops, most likely in C++.

#### Simplifying the model

There are several ops that are not supported, mostly coming from the `th` (PyTorch C++ backend) library or custom layers. Here is the breakdown:

- `aten::pow`: **Power function ($x^y$)**. Used in `GDN` (computing $x^2$) and `NonNegativeParametrizer`. DPU does not support arbitrary power/exponents.
- `aten::max`: **Maximum value**. Used in `LowerBound` (custom clamping). DPU theoretically supports ReLU (max(0,x)), but `max(x, constant)` often falls back to CPU.
- `nndct_sqrt`: **Square Root**. Used in `GDN` (inverse=True) and `Parametrizer` initialization. DPU does not support hardware square root.
- `aten::_convolution`: **Standard Convolution**. Usually supported, but seeing it as a warning usually means the graph partitioner failed to merge it with other DPU-compatible ops (likely because it's sandwiched between unsupported ops).
- `aten::rsqrt`: **Reciprocal Square Root ($1/\sqrt{x}$)**. Used in `GDN` (inverse=False). Not supported on DPU.
- `aten::abs`: **Absolute Value ($|x|$)**. Used in `z = h_a(|y|)` to prepare the hyperprior. Not supported on DPU.
- `aten::ones_like`: **Tensor creation**. Used in internal helper functions for shape handling.
- `aten::clone`: **Memory copy**. Used in `EntropyModel.quantize`.
- `aten::round`: **Rounding**. Used in `EntropyModel.quantize` to simulate integer discrete quantization. DPU works on integer arithmetic but doesn't expose a "round float to int" layer for the graph logic itself.
- `aten::erfc`: **Complementary Error Function**. Used in `GaussianConditional` to estimate the Cumulative Distribution Function (CDF) for bit-rate estimation. This is a complex statistical function (Probability Math) completely outside the scope of DPU acceleration.

Conclusion: Almost all unsupported ops come from **GDN** (Normalization) and **Entropy Modeling** (Probability/Quantization). The Convolutional layers themselves are fine.

### Overwriting CompressAI custom `torch.autograd.Function` (December 2025)

*Vitis-AI only supports a handful of operations. Obviously, custom backward operation are not supported, but they are also not needed during inference.*

#### Redefining LowerBoundFunction

*See [this issue](https://github.com/InterDigitalInc/CompressAI/issues/345) to better understand the role of the function in the first place.*
The original error from Vitis-AI model Inspector is `[VAIQ_ERROR][QUANTIZER_TORCH_UNSUPPORTED_OPS]: Unsupported Ops: {'LowerBoundFunction'}.`. To avoid that, we implement a `LowerBoundFunctionPatched` that simply uses `torch.max()`.

#### Re-writing all CompressAI componentsto use LowerBoundFunctionPatched

Now we create `Patched` versions of `GDN`, `EntropyBottleneck`, and `GaussianConditional` to use `LowerBoundFunctionPatched`.
During the process I copied some compressai code for all these components. However, I got some problems with "unexpected keys" when loading the checkpiont. That's because they renamed some parameters between 1.2.6 and 1.2.8. Therefore, I bumped compressai to 1.2.8.

### Adapting Vitis-AI Docker container to my requirements (October 2025)

```bash
# If old container still running, but must restart because NVMH
exit
docker rm vai_container_2
# ---
cd /mnt/vitisAI/Vitis-AI/
./docker_run.sh xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde
conda activate vitis-ai-pytorch
pip install h5py omegaconf compressai torchmetrics  # hydra-core
# We cannot install lightning, because it will bump torch 1.13.1+cu117 to torch-2.4.1 which is not compatible with vaic and vart that were built with PyTorch 1.13 (I suppose) and results in an OSError
# Tell the container to use conda's `libstdc++` which is more recent and satisfies GLIBCXX_3.4.29
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6:$LD_PRELOAD
```

### Convolution settings (tackled before LPS ~ 23/06/2025)

*Vitis-AI has a problem with `output_padding`, I detailed below how I solved that.*
Because my patches will always be square I wrote the formulas for only 1 dimension, i.e., height=width. This also holds for the stride, kernel_size, padding, and output_padding.

**Conv2D**, see [Pytorch doc](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
$$
H_{out} = \lfloor\frac{H_{in} + 2 * padding−dilation * (kernel\_size−1) - 1}{stride} + 1 \rfloor
$$
So: `nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2)` gives $H_{out} = \lfloor\frac{256 + 2 * 2 − 1 * (5 − 1) - 1}{2} + 1 \rfloor = \lfloor \frac{255}{2} + 1\rfloor = 128$

Alternative: `nn.Conv2d(N, N, kernel_size=4, stride=2, padding=1)` gives $H_{out} = \lfloor\frac{256 + 2 * 1 − 1 * (4 − 1) - 1}{2} + 1 \rfloor = \lfloor \frac{254}{2} + 1\rfloor = 128$

**ConvTranspose2D**, see the [Pytorch doc](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
$$
H_{out} = (H_{in} − 1) * stride − 2 * padding + dilation * (kernel\_size−1) + output\_padding + 1
$$

So: `nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1)` gives $H_{out} = (128 − 1) * 2 − 2 * 2 + 1 * (5 − 1) + 1 + 1 = 254 - 4 + 4 + 1 +1 = 256$

Alternative: `nn.ConvTranspose2d(N, N, kernel_size=4, stride=2, padding=1, output_padding=0)` gives $H_{out} = (128 − 1) * 2 − 2 * 1 + 1 * (4 − 1) + 0 + 1 = 254 - 2 + 3 + 0 + 1 = 256$

**Smaller kernels**: the first and last layer hyperprior have $kernel\_size = 3$ `nn.Conv2d(M, M, kernel_size=3, stride=2, padding=1)` gives $H_{out} = \lfloor\frac{16 + 2 * 1 − 1 * (3 − 1) - 1}{2} + 1 \rfloor = \lfloor \frac{15}{2} + 1\rfloor = 8$. And `nn.ConvTranspose2d(M, M, kernel_size=3, stride=2, padding=1, output_padding=0)` gives $H_{out} = (8 − 1) * 2 − 2 * 1 + 1 * (3 − 1) + 0 + 1 = 14 - 2 + 2 + 0 + 1 = 15$
> I need to use the output_padding for this one.

## Running the model on the ZC102

So I realized (a bit late) that, it's not enough to compile the model. In their tutorial, Vitis AI uses some additional scripts to run inference of the compiled model on specific task/images. Because these scripts are not suitable for my application I need to write my own.
I found barely any documentation about the features these scripts should implement or functions from `xir` or `vart` they should call, so I will proceed brute-force: have an LLm (perplexity) hallucinate some procedure/script and iteratively debug that thing, just so I start from somewhere.

==@TODO for the moment I'm going for python as inference language, but as soon as my understanding of the board gets better, I can switch to C++==

### 23.11.2025

#### Setup and dependencies

Below is the intended project structure on the Target:

```text
/root/home/SAR_DDC/
├── model/
│   ├── <your_model>.xmodel           # Compiled model
│   └── <your_model>.prototxt         # Optional config file
├── data/
│   └── val.h5                        # Your test dataset
├── scripts/
│   ├── inference.py                  # Main inference script
│   └── metrics.py                    # Your metrics module
└── libs/
    └── (any custom modules needed)
```

==@TODO Didn't check that `.prototxt` file yet, see [this tuto](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html#compile-the-model) about it, I don't know how to adapt it for my application.==

I couldn't find a smart way to install `h5py` on the Target, even using AMD package manager `dnf`. So I created a new script `convert_h5_to_np.py`:

```bash
[HOST](DDC_FPGA) leon_ce@bart:~/dev/Vitis-AI/DDC_FPGA/$ python scripts/dataset/convert_h5_to_np.py --dataset_path data/processed_hdf5/test_with_GT/TSX_preprocessed_spatial_splits_5_256x256/test.h5 --subset 500
# Afterwards transfer the dataset to the Target (It also took 6:41 mins)
[HOST](DDC_FPGA) leon_ce@bart:~/dev/Vitis-AI/DDC_FPGA/$ scp data/processed_hdf5/test_with_GT/TSX_preprocessed_spatial_splits_5_256x256/test_500.npy root@10.0.0.2:/home/root/SAR_DDC/data/test_500.npy
```

> Be careful, the Zynq US+ does not have infinite RAM, so choose the subset smartly. Or implement a better loading function in `inference.py`.

#### Inference script

Because I don't want to create a VSCode server on the FPGA directly and I don't know how to open a file from a "recursive" SSH session inside of the Remote Explorer extension, I will develop the script "locally" (on BART) and `scp` it every time. The script will be in `scripts/fpga/inference.py`.
*Move the script to the Target*:

```bash
[HOST](vitis-ai-pytorch) vitis-ai-user@bart:/workspace$ scp scripts/fpga/inference.py root@10.0.0.2:/home/root/SAR_DDC/scripts/inference.py
```

*Perform inference*:

```bash
[HOST](DDC_FPGA) leon_ce@bart:~$ ssh root@10.0.0.2
[TARGET]root@xilinx-zcu102-20222:~# cd SAR_DDC
[TARGET]root@xilinx-zcu102-20222:~/SAR_DDC# python3 scripts/inference.py --xmodel model/ResAE_pt.xmodel --data data/test_500.npy --subset 100
```

## What I did for a first deployment (20/06/2025)

*This list was accompanied by several changes. Find them at commit f1fc9b616a9f988493121b39967d8efc20b5e032 (branch `fpga_delpoyment`). I had to stash them when I cleaned the repo.*

1. Created a similar `model_quantization.py` in DDC_FPGA, realized I need it in the Vitis-AI folder (because the docker image root is in Vitis-AI/), so I copied it there.
2. Wanted to start docker, the GPU image seems to have disappeared, so I started the `cpu:latest`
3. I ran into partition space problems so I started mounting 1 of the 2 1TB disks available and copying vitisAI and my repo there. I've used ChatGPT to do so and for the moment I'll use a symlink between the partition and my `~/dev/Vitis-AI/`.
4. So I figured out that NVIDIA-drivers were outdated and it was causing a daemon error when trying to run the docker GPU image `./docker_run.sh xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde`. After a bit of a fight with `/etc/fstab/` I manage to reboot the computer
4-bis. I updated `/etc/fstab/` to automount the `/dev/sda1` 1TB partition by adding the line `UUID=ed3655cd-3524-421b-a268-3f89d05ffbee /mnt/vitisAI ext4 defaults 0 2` where **UUID** was obtained with `sudo blkid /dev/sda1`.
5. I installed the missing packages directly in the docker image: `h5py`, `compressai` (with pip)
6. Then I encountered the classic `compressai` problem:

   ```bash
   ImportError: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by <conda_path>/envs/rs_dc/lib/python3.11/site-packages/compressai/_CXX.cpython-311-x86_64-linux-gnu.so)
   ```

   The solution is to install gcc and g++ and make sure they have the same version number. I traditionally do that via conda, but xilinx docker image was set to use anaconda as a default channel, which DLR does not allow. so I had to remove the default and local (`file:///scratch/conda-channel`) channels from the config with `conda config --remove channels <name>` and set a strict channel priority with `conda config --set channel_priority strict` then after 30 damn min of "Solving environment" I ... gave up.
7. Because I was in a deadend with `conda` I tried installing gcc and g++ with `apt-get`. However, the xilinx docker image uses Ubuntu 20.04 and the latest available version of `libstdcxx-ng` was `GLIBCXX_3.4.28`... so another deadend. Ultimately, I solved this problem by downgrading compressai to 1.2.3 with `pip` because it uses `GLIBCXX_3.4.28`.
8. After a little bit more of playing around and redefining my loss in the `model_quant.py` to **not** use `torchmetrics` (because it imports matplotlib which also needs `GLIBCXX_3.4.29`, and my downgrading trick did not seem to work). I finally got the script to run in "float" mode 😍
9. So I could evaluate my model in float mode, but as expected the "Inspection" fails because Vitis-AI spots function not supported by the DPU, e.g., `LowerBoundFunction`. This is a custom autograd function used by compressai for their `LowerBound` operator used for example in some EntropyModels, e.g., `GaussianConditional`. I fixed it (temporarily, I have a feeling it will come back to me next training) by registering a new custom_op like in the [doc](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Register-Custom-Operation) or this [blog](https://adaptivesupport.amd.com/s/article/Custom-OP-complete-example-design-for-Pytorch?language=en_US) and doing MonkeyPatching.
10. I lost a day trying to fix a mismatching Tensor shape happening after the first layers of the decoder `g_s`. There was an off-by-one error in the first elementwise operation (a multiplication in GDN `out = x * norm`) happening after a ConvTranspose2d. It appears AMD `pytorch_nndct` does not really like "un-friendly" padding settings. My ConvTranspose2d all used `output_padding=1` and I'm pretty sure this was the problem. I "fixed" it my using safe convolutions, i.e.:

   ```python
   # I transformed:
   nn.ConvTranspose2d(c_in, c_out, kernel_size=5, stride=2, padding=2, output_padding=1)
   # Into:
   nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, output_padding=0)
   ```

   This worked.
11. Moving onto the next step: the model compilation using `vai_c_xir`, it crashes with a simple message `[UNILOG][FATAL][XCOM_UTIL_INVALID_VALUE][A invalid value is given for specific function.] Division by 0 or minus in div_ceil, 3 / 0` with no indication whatsoever where it could happen. I figured out it had to be in the EntropyBottleneck and tried to add shape, NaN, Inf, and even 0 values check everywhere, but found nothing.
Solution: either replace by an easier Entropy model, even one without parameters. Or copy compressai code over and laboriously work your way in what could be the problem.

@TODO: I could not have a look into the explicit definition of `pytorch_nndct.nn.Module.ConvTranspose2d` but it probably can be find around [this path](/opt/vitis-ai/src/vai_quantizer/vai_q_pytorch/pytorch_binding/pytorch_nndct/nn/modules/conv_transpose.py) in the docker image.

---

## ResidualFactorizedPriorPatched Integration (2026-05-06)

`ResidualFactorizedPriorPatched` (`res_factorized_prior_dpu.py`) is a simpler alternative to `ResidualScaleHyperpriorPatched`: it removes the hyperprior path (`h_a`, `h_s`, `gaussian_conditional`) and uses `EntropyBottleneck` directly on the main latent `y`.

### Architecture diff

| Aspect | `ResidualScaleHyperpriorPatched` | `ResidualFactorizedPriorPatched` |
| --- | --- | --- |
| NN subgraphs | `g_a`, `h_a`, `h_s`, `g_s` (4) | `g_a`, `g_s` (2) |
| Entropy models | `EntropyBottleneck` + `GaussianConditional` | `EntropyBottleneck` only |
| `forward()` output | `x_hat`, `likelihoods: {y, z}` | `x_hat`, `likelihoods: {y}` |
| `compress()` output | `strings: [y_strings, z_strings], shape: z.shape[-2:]` | `strings: [y_strings], shape: y.shape[-2:]` |
| DPU split graph | 4 subgraphs + EB + GC on CPU | 2 subgraphs + EB on CPU |

### Codebase impact

| File / component | Status | Issue |
| --- | --- | --- |
| `src/` training pipeline | ✅ OK | `MerlinRDLoss` and `SARDDCModule` are architecture-agnostic; `compress`/`decompress`/`update`/`aux_loss` all implemented |
| `scripts/update_wandb_runs.py` | ✅ OK | Uses `hydra.utils.instantiate` — never touches `h_a`/`h_s`/`gc` directly |
| `notebooks/*.ipynb` | ✅ OK | Load from JSON/W&B, metric keys are the same |
| `deploy.py::_export_entropy_params` | ❌ Crash | Hardcoded `ResidualScaleHyperpriorPatched` instantiation + unconditional `gc.*` access |
| `deploy.py::DPU_WRAPPER_NAME` | ⚠️ Wrong name | Hardcoded string, needs to be model-derived |
| `model_quant.py::load_model` | ❌ Crash | `raise ValueError` on unknown model name |
| `model_quant.py` dummy inputs | ❌ Crash | 4-tensor dummy `(x, abs_y, z_hat, y_hat)` hardcoded for 4-subgraph wrapper |
| `dpu_wrapper.py` | ❌ Crash | `ResidualScaleHyperpriorDPUWrapper.__init__` accesses `h_a`, `h_s`, `gc` |
| `inference_hybrid.py` | ❌ Crash | Unconditional `gc_scale_table` load + `runners["h_a"]`/`runners["h_s"]` calls |
| `inference_utils.py::identify_subgraphs` | ❌ Crash | `required_keys = ["g_a","h_a","h_s","g_s"]` hardcoded — raises `ValueError` for 2-subgraph model |
| `benchmark_gpu.py::run_scenario_*` | ❌ Crash | Every scenario function calls `net.h_a(...)` and `net.gaussian_conditional.*` |
| `benchmark_fpga.py::run_scenario_*` | ❌ Crash | Same pattern + `runners["h_a"]`/`runners["h_s"]` |

> **Bonus cleanup:** `ResidualScaleHyperpriorPatched.forward()` returns a `"y_hat"` key that nothing in the codebase reads. Safe to remove for API consistency.

---

### Fix plan

🤖 = agent, 👤 = human. Steps are ordered by dependency: train first, then deploy chain, then benchmarks.

#### Step 0 — Configs + naming ✅

- [x] **`configs/model/res_factorized_prior.yaml`** *(👤)* — new model config targeting `ResidualFactorizedPriorPatched`
- [x] **`configs/experiment/ADAM-ResFP.yaml`** *(👤)* — experiment override (relu, no_output_padding, etc.)
- [x] **`from __future__ import annotations` in `res_factorized_prior_dpu.py`** *(👤)* — Python 3.8 compat for Vitis-AI container
- [x] **`deploy.py::_make_compiled_model_name`** *(👤)* — added `elif "ResidualFactorizedPrior" in target: model = "ResFP"/"FP"`
- [x] **`utils.py::make_wandb_run_name`** *(👤)* — added `elif "ResidualFactorizedPrior" in model: model = "ResFP"/"FP"`

```bash
# Verify
python src/train.py experiment=ADAM-ResFP debug=fdr
```

**Commit:** `feat: add ResidualFactorizedPriorPatched Hydra configs, naming, and Python 3.8 compat`

---

#### Step 1 — Train a checkpoint *(👤, gating step for Steps 3–5)*

```bash
python src/train.py experiment=ADAM-ResFP model.criterion.lmbda=1000 seed=42
```

---

#### Step 2 — DPU Wrapper ✅

- [x] **`FactorizedPriorDPUWrapper` added to `dpu_wrapper.py`** *(🤖)* — 2-subgraph wrapper (`g_a` + `g_s` only). Calibration mode: `g_a → EB passthrough → g_s`. Deployment mode: `forward(x_in, y_hat_in) → (y_out, x_out)`.

```bash
# Verify
python -c "
from src.models.components.res_factorized_prior_dpu import ResidualFactorizedPriorPatched
from src.models.components.dpu_wrapper import FactorizedPriorDPUWrapper, ResidualScaleHyperpriorDPUWrapper
from src.models.components.res_scale_hyperprior_dpu import ResidualScaleHyperpriorPatched
import torch; x = torch.randn(1, 2, 256, 256)
m = ResidualFactorizedPriorPatched(nb_channels_main=32, activation='relu', no_output_padding=True, export_dpu=True)
w = FactorizedPriorDPUWrapper(m); w.eval()
out = w(x); print('calib OK:', out['x_hat'].shape)
y_out, x_out = w(x[:,:1], torch.randn(1,32,16,16)); print('deploy OK:', x_out.shape)
m2 = ResidualScaleHyperpriorPatched(nb_channels_main=32, activation='relu', no_output_padding=True, export_dpu=True)
out2 = ResidualScaleHyperpriorDPUWrapper(m2).eval()(x); print('ResSHyp regression OK:', out2['x_hat'].shape)
"
```

**Commit:** `feat(fpga): add FactorizedPriorDPUWrapper for 2-subgraph DPU split graph`

---

#### Step 3 — Deploy + Quantization entry points *(🤖, needs Step 2)*

- [x] **`deploy.py::_make_compiled_model_name`** *(👤, done in Step 0)*
- [x] **`deploy.py::_export_entropy_params`** — dispatches on `_target_`; saves only EB keys for ResFP, adds GC keys only for ResSHyp (uses `isinstance(model, ResidualScaleHyperpriorPatched)` — no `has_gc` flag)
- [x] **`deploy.py::DPU_WRAPPER_NAME`** — removed constant; replaced with `_get_dpu_wrapper_name(cfg)` helper; `phase_compile` loads config **once** and propagates `cfg` to all helpers
- [x] **`model_quant.py::load_model`** — added `elif "ResidualFactorizedPriorPatched"` branch wrapping with `FactorizedPriorDPUWrapper`
- [x] **`model_quant.py::evaluate`** — split into `evaluate_hyperprior` (4-input) and `evaluate_factorized` (2-input); `main()` dispatches via `eval_fn`
- [x] **`model_quant.py` dummy inputs** — shapes derived from model properties (`full_model.nb_channels_main`, `full_model.main_downsampling_factor`, `full_model.hyper_downsampling_factor`) instead of hardcoded values
- [x] **`res_factorized_prior_dpu.py` / `res_scale_hyperprior_dpu.py`** — added `self.nb_channels_main: int = N` attribute to both `__init__` methods

Bonus — also fixed in this step (needed for safe entropy_params.npz loading):

- [x] **`inference_hybrid.py`** — gate `GaussianConditional` on `"gc_scale_table" in data`; rename `process_single_tile` → `process_single_tile_SHyp`, add `process_single_tile_FP`; extract `_prepare_tile_input` / `_collect_tile_output` shared helpers; derive `model_name` from manifest and dispatch with `"SHyp" in model_name`
- [x] **`inference_utils.py::identify_subgraphs`** — accepts `model_name: str = ""`; `required_keys` is `["g_a","g_s"]` (FP) or `["g_a","h_a","h_s","g_s"]` (SHyp)

**Commit:** `feat(fpga): add FactorizedPriorPatched dispatch in deploy.py, model_quant.py, inference_hybrid.py`

---

#### Step 4 — FPGA Inference pipeline *(✅ covered in Step 3 bonus)*

#### Step 5 — GPU / FPGA Benchmarks *(🤖)*

- [x] **`benchmark_gpu.py`** — all `run_scenario_*` + `_precompress` + `get_model_info`: `hasattr(net, 'h_a')` topology guard; FP path `g_a → EB → g_s`, no h_a/h_s/GC
- [x] **`benchmark_fpga.py`** — `gc: Optional[GaussianConditional]`; all five scenario functions + `_precompress` have FP branches; `run_benchmark()` reads `model_name` from `manifest.json`, passes to `identify_subgraphs`, loads `gc` only when `"gc_scale_table" in data`
- [x] **Refactor** — introduced `tmark(label, timer, cuda_timer, device)` (replaces 3-line sync+mark×2 blocks) and `_count_bytes(strings)` (replaces nested sum comprehension); all 5 scenario functions and `_precompress` simplified significantly

```bash
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ \
    --warmup 20 --iters 100 \
    --power --idle-baseline 10 \
    --power-hz-gpu 10 --power-hz-fpga 50
```

**Commit:** `feat(benchmark): add FactorizedPrior scenario branching in gpu and fpga benchmarks`

---

#### Step 6 — Cleanup *(🤖, optional)*

- [x] **Remove `"y_hat"` key** from `ResidualScaleHyperpriorPatched.forward()` and `ResidualScaleHyperpriorDPUWrapper.forward_mode1_calibration()` — nothing reads it

```bash
pytest -k "not slow"
```

**Commit:** `refactor: remove unused y_hat key from ResSHyp forward output dict`

---

## Doc Re-organisation (2026-05-15)

Restructured project documentation into clearly scoped files. Notes placed here either
because they are unresolved research questions or because they record what was removed/
corrected, so nothing is truly lost.

### Unresolved: ln(2) offset in MERLIN loss optimisation

*Moved from `docs/Method.md` — not yet understood, not operational, kept for future
investigation.*

Optimising the MERLIN log-scale loss (Eq. 2b) with respect to the reconstruction $\hat x$:

$$
\mathcal{L}(\check r, \check b) = \sum_k \frac{\check r_k}{2} + \exp(2\check b_k - \check r_k)
$$

$$
\frac{\partial \mathcal{L}}{\partial \hat x} = \frac{1}{2} - \exp(2y - \hat x) = 0
\quad\Leftrightarrow\quad \hat x = 2y + \ln(2)
$$

The $2y$ factor is intuitive (the network reconstructs full reflectivity, not half of it).
The $\ln(2) \approx 0.693$ offset has no obvious explanation. It is small relative to the
signal dynamic range and has never caused a visible artefact; it may reflect the asymmetric
curvature of the loss around its minimum.

### Corrections and removals

- **Stale script reference fixed** in `docs/Method.md`: `script/TSX_dataset_creation.py`
  → `scripts/dataset/create_dataset.py` (correct path and correct script name).
- **"Data" section removed** from `docs/Method.md` → content now in new `docs/Data.md`.
- **Duplicate paragraph** (`==To place somewhere else==` block about `dataset_creation.py`)
  removed from `docs/Method.md` — covered in `docs/Data.md` preprocessing section.
- **`==@TO UPDATE==` markers** removed from `docs/Method.md` — content was already correct,
  markers were stale.
- **"mostly written by ChatGPT"** annotation removed from the MERLIN Theory section header.
- **README TODOs** trimmed: "Add Python 3.8 note to README" removed (done in the rewrite);
  "Use rootutils better" removed (minor internal preference, not actionable).
- **Full Method + Data sections removed** from `README.md` — content now lives in
  `docs/Method.md` and `docs/Data.md` respectively; README is now a navigation + workflow
  reference only.
