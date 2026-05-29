# GPU / CPU Benchmark (RAW DUMP — to refactor)

> **Status: raw extraction, not cleaned.** These sections were lifted verbatim from the
> former `performance_benchmark_implementation.md` when the FPGA benchmark docs were
> consolidated into `FPGA_benchmark.md`. The GPU/CPU benchmark (`scripts/benchmark_gpu.py`)
> and the legacy cross-platform orchestrator/notebook are kept here for reference. Refactor
> this doc when addressing the **unified GPU/CPU/FPGA benchmark runner** TODO (see
> `FPGA_benchmark.md`). Note: `run_full_benchmark.py` and `benchmark_fpga.py` referenced below
> are **deleted** — kept here only as historical reference.

---

## GPU/CPU idle-baseline note (from perf-doc §4.9)

**GPU/CPU idle baseline**: `benchmark_gpu.py` captures the same 10 s idle window
before each scenario.  GPU idle is measured *after* `net.to(device)` to match the
P0 CUDA-context-loaded state during inference (not P8 deep-sleep).  Measured RTX
A4000 idle: **33.4 W**; load (full scenario): **58.2 W** → dynamic ≈ **25 W**.
CPU idle baseline uses Intel RAPL, but **RAPL `energy_uj` files require root on
Linux ≥ 5.10**; if unavailable, CPU idle is not recorded.  Fix:
`sudo chmod o+r /sys/class/powercap/intel-rapl/*/energy_uj`.

---

### 8.3 Comparison Fairness (GPU vs FPGA)

1. **Data format**: FPGA uses INT8, GPU uses FP32 — the FPGA's lower precision
   introduces quantization error. Quality comparison (PSNR, SSIM) is essential.
2. **Batch size**: GPU batching amortises overhead; FPGA benchmark uses batch=1.
   For fair throughput comparison, normalise to per-image values.
3. **Power comparison**: GPU power from `nvidia-smi` is board-level GPU power.
   FPGA power from INA226 is PL+PS. Neither captures host/memory system power.
   Use energy-per-inference as the fairest comparison metric.

---

## 9. GPU / CPU Benchmark: `scripts/benchmark_gpu.py`

### 9.1 Purpose & Relationship to FPGA Benchmark

`benchmark_gpu.py` measures per-component latency, throughput, and power for the
**same model** running on a CUDA GPU and/or host CPU.  It produces JSON output with
the **same schema** as `benchmark_fpga.py` so results can be loaded into a single
comparison table or plot.

Key differences from the FPGA benchmark:

- **Precision**: FP32 on GPU/CPU vs. INT8 on FPGA — quality (PSNR/SSIM) **must**
  be compared alongside speed.
- **Batch size**: Always 1 (matching the FPGA baseline).
- **Entropy coding**: Same CompressAI Python implementation runs on **both** GPU
  and CPU host.  On the FPGA, this runs on the ARM A53 via C++ `ans.so`.

### 9.2 Dual-Device Mode

By default, a single invocation measures on **GPU first, then CPU sequentially**.
Skip either with:

- `--no-gpu` — skip GPU measurement (useful on CPU-only machines)
- `--no-cpu` — skip CPU measurement (faster iteration on GPU numbers)

Each device produces its own JSON file:

```bash
results/benchmark/<run_name>/benchmark_gpu_<scenario>.json
results/benchmark/<run_name>/benchmark_cpu_<scenario>.json
results/benchmark/<run_name>/benchmark_fpga_<scenario>.json
```

### 9.3 Scenarios

| Scenario | Steps Timed | Purpose |
| --- | --- | --- |
| `full` | All (compress + decompress) | End-to-end latency for one tile |
| `compress` | g_a → h_a → EB → h_s → GC | Encode-only latency |
| `decompress` | EB → h_s → GC → g_s | Decode-only latency (from cached bitstream) |
| `nn_only` | g_a, h_a, h_s, g_s | Isolate NN latency, no entropy coding |
| `entropy_only` | EB + GC (compress + decompress) | Isolate CPU entropy coding |

The `nn_only` scenario is the same on all platforms (FPGA, GPU, CPU).

### 9.4 Step Labels & Prefixing Convention

| Prefix | Meaning | When used |
| --- | --- | --- |
| `gpu_` | NN subgraph running on GPU | GPU measurement mode |
| `nn_` | NN subgraph running on CPU | CPU measurement mode |
| `cpu_` | CPU-side operation (entropy coding, concat, split) | Both modes |

This allows automatic aggregation (e.g., sum all `gpu_*` steps for total GPU NN time).

### 9.5 Timing Methodology

| Device | Method | Precision |
| --- | --- | --- |
| **GPU** (wall-clock) | `time.perf_counter()` with `torch.cuda.synchronize()` | ~µs |
| **GPU** (CUDA events) | `torch.cuda.Event(enable_timing=True)` | ~µs, no CPU-side jitter |
| **CPU** | `time.perf_counter()` | ~µs |

The GPU measurement records **both** wall-clock and CUDA event timings for every
step.  CUDA events are stored in `latency_breakdown_cuda_events` — prefer these
for NN sub-graph comparisons as they exclude Python/CPU overhead.

### 9.6 Power Measurement

| Source | Metric | How |
| --- | --- | --- |
| **GPU** | Board-level GPU draw | `nvidia-smi --query-gpu=power.draw` polled at `--power-hz` (default 10 Hz) in a background thread |
| **CPU** | Package + DRAM power | Intel RAPL via `/sys/class/powercap/intel-rapl/` — energy counter delta between start/stop |

**Limitations**:

- `nvidia-smi` power is the **full GPU board** (incl. idle), not incremental.
- RAPL reports **package** (all cores + uncore) and **DRAM**, but not
  motherboard, PSU, fans, etc.
- Neither captures host system total power.  For publication, consider an
  external wall-plug meter.

### 9.7 Usage

The script accepts the model either via `--model-dir` (recommended — reads the
checkpoint path from `manifest.json`) or `--ckpt` (direct checkpoint path).
The two are mutually exclusive.

```bash
# ---- Recommended: use --model-dir (reads manifest.json → checkpoint) ----

# Full pipeline, GPU + CPU, with power measurement
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario full \
    --warmup 20 --iters 100 \
    --power

# GPU only, compress scenario, idle baseline
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario compress \
    --no-cpu --power --idle-baseline 10

# CPU only, entropy isolation
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario entropy_only \
    --no-gpu --iters 200

# ---- Alternative: direct checkpoint path ----

python scripts/benchmark_gpu.py \
    --ckpt logs/train/runs/<run>/checkpoints/last.ckpt \
    --scenario full \
    --warmup 20 --iters 100

# Custom output directory (works with either source)
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario full \
    --output-dir results/my_benchmark
```

When `--model-dir` is used, the output directory defaults to
`results/benchmark/<model_name>/` (e.g. `ResSHyp-relu_s1_L1000_pt`).
When `--ckpt` is used, it defaults to `results/benchmark/<run_timestamp>/`.

### 9.8 JSON Output Schema

```json
{
  "platform": "GPU_NVIDIA_RTX_A4000",
  "device": "cuda",
  "scenario": "full",
  "timestamp": "2025-...",
  "n_warmup": 20,
  "n_iters": 100,

  "latency_breakdown": {
    "preprocess":       {"mean_s": 0.000001, "std_s": ..., "median_s": ..., "min_s": ..., "max_s": ..., "p95_s": ..., "n": 100},
    "gpu_g_a":          {"mean_s": 0.0012,   ...},
    "cpu_concat_abs":   {"mean_s": 0.00003,  ...},
    "gpu_h_a":          {"mean_s": 0.0005,   ...},
    "cpu_eb_compress":  {"mean_s": 0.035,    ...},
    "cpu_eb_decompress":{"mean_s": 0.002,    ...},
    "gpu_h_s":          {"mean_s": 0.0005,   ...},
    "cpu_gc_compress":  {"mean_s": 0.12,     ...},
    "cpu_gc_decompress":{"mean_s": 0.14,     ...},
    "cpu_split_y_hat":  {"mean_s": 0.000002, ...},
    "gpu_g_s":          {"mean_s": 0.0014,   ...},
    "postprocess":      {"mean_s": 0.000001, ...}
  },
  "latency_breakdown_cuda_events": {
    "preprocess":       {"mean_s": ..., ...},
    "gpu_g_a":          {"mean_s": 0.00098, ...},
    "..."
  },

  "latency_total_mean_s": 0.301,
  "latency_total_mean_ms": 301.0,
  "latency_gpu_total_mean_ms": 3.6,
  "latency_cpu_total_mean_ms": 297.4,
  "latency_wall_total_s": 30.1,

  "throughput_fps": 3.32,
  "avg_compressed_bytes": 1234,

  "power": {
    "gpu":            {"avg_power_w": 85.0, "energy_j": 2550.0, "n_samples": 300, "duration_s": 30.1},
    "cpu_rapl": {
      "package-0":    {"avg_power_w": 42.0, "energy_j": 1264.2, ...},
      "dram":         {"avg_power_w": 5.2,  "energy_j": 156.5, ...}
    },
    "gpu_avg_w":           85.0,
    "cpu_rapl_total_avg_w": 47.2,
    "idle_gpu":            {"avg_power_w": 15.0, ...},
    "idle_cpu_rapl": {
      "package-0":    {"avg_power_w": 12.0, ...},
      "dram":         {"avg_power_w": 3.0,  ...}
    }
  },

  "hw_info": {
    "name": "NVIDIA RTX A4000",
    "cuda_version": "12.4",
    "cudnn_version": "90100",
    "torch_version": "2.5.1+cu124",
    "memory_total_mb": 16376,
    "power_limit_w": 140.0,
    "max_sm_clock_mhz": 1560.0,
    "max_mem_clock_mhz": 7001.0
  },

  "model_info": {
    "total_params": 15000000,
    "trainable_params": 15000000,
    "weights_size_mb": 57.22,
    "N": 128,
    "M": 256
  }
}
```

### 9.9 Interpreting Results & Cross-Platform Comparison

#### 9.9.1 What to compare

| Metric | Fair comparison? | Notes |
| --- | --- | --- |
| **NN latency** (gpu/dpu/nn steps) | ✅ Comparable | Different devices executing the same subgraphs |
| **Entropy latency** (cpu_ steps) | ⚠️ Be careful | GPU benchmark runs entropy on x86; FPGA on ARM A53. The x86 is vastly faster |
| **Total latency** | ✅ Comparable | Apples-to-apples if batch=1 |
| **Throughput** (fps) | ✅ Comparable | Derived from total latency |
| **Power** | ⚠️ Different scopes | GPU = board GPU power; FPGA = SoC rails. Not directly comparable |
| **Energy per inference** | ✅ Best metric | $E = P_\text{avg} \times t_\text{total}$ for each platform |
| **Quality** (PSNR/SSIM) | ✅ Essential | FP32 vs INT8 quality gap must be reported alongside speed |

#### 9.9.2 Pitfalls

1. **Entropy coding dominates on all platforms.**  On both GPU and FPGA, entropy
   coding (CompressAI's rANS) runs on the CPU.  The x86 host is ~10–30× faster
   than the ARM A53, so total latency differences are dominated by this component
   rather than NN inference speed.
2. **GPU warmup.**  The first CUDA kernel launch incurs JIT compilation overhead.
   Always use `--warmup ≥ 10` to ensure steady-state.
3. **GPU power states.**  An idle GPU draws ~15W.  Under load, RTX A4000 can reach
   ~140W.  If `--iters` is too low, the GPU may not reach steady-state power,
   inflating apparent energy efficiency.
4. **CUDA events vs. wall-clock.**  `latency_breakdown_cuda_events` excludes
   CPU-side overhead (Python dispatch, memory copies).  For GPU NN steps, CUDA
   events are more accurate; for end-to-end latency, use the wall-clock breakdown.
5. **CPU benchmark is single-threaded by default.**  PyTorch uses `torch.get_num_threads()`
   threads for CPU ops.  This may differ across machines.  Report `torch_threads`
   from the JSON output alongside results.

#### 9.9.3 Recommended comparison table format

For a publication, present results as:

| | FPGA (ZCU102) | GPU (RTX A4000) | CPU (host) |
| --- | --- | --- | --- |
| **NN latency** (ms) | 167 | ? | ? |
| **Entropy latency** (ms) | 258 | ? | ? |
| **Total latency** (ms) | 425 | ? | ? |
| **Throughput** (patches/s) | 2.35 | ? | ? |
| **Power** (W) | 11.62 (SoC) | ? (GPU board) | ? (RAPL pkg) |
| **Energy/patch** (J) | 4.94 | ? | ? |
| **PSNR** (dB) | from eval | from eval | from eval |
| **Precision** | INT8 | FP32 | FP32 |

---


---

## 11. Full Benchmark Orchestrator: `scripts/fpga/run_full_benchmark.py`

The orchestrator script runs the complete benchmark suite for one compiled model across
all three platforms in four phases:

| Phase | Description | Location |
| --- | --- | --- |
| **1 — GPU + CPU** | `benchmark_gpu.py` × 5 scenarios | Host (this machine) |
| **2 — FPGA setup** | Copy `benchmark_fpga.py` + `scp` model → ZCU102 | Host → ZCU102 |
| **3 — FPGA run** | `benchmark_fpga.py` × 5 scenarios via SSH | ZCU102 |
| **4 — Fetch** | `scp` JSON results back to host | ZCU102 → Host |

### 11.1 Usage

```bash
# Full run with power measurement (~14 min with idle baseline)
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ \
    --power --idle-baseline 10

# GPU + CPU only (no board access required)
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ --no-fpga

# FPGA only (model already on board)
python scripts/fpga/run_full_benchmark.py \
    --model-dir results/fpga/active_model/ --no-gpu --no-cpu --skip-transfer
```

### 11.2 Output Layout

All results are stored in `results/benchmark/<model_name>/`:

```text
results/benchmark/ResSHyp-relu_s1_L1000_pt/
├── benchmark_gpu_full.json
├── benchmark_gpu_compress.json
├── benchmark_gpu_decompress.json
├── benchmark_gpu_nn_only.json
├── benchmark_gpu_entropy_only.json
├── benchmark_cpu_full.json
├── benchmark_cpu_compress.json
├── benchmark_cpu_decompress.json
├── benchmark_cpu_nn_only.json
├── benchmark_cpu_entropy_only.json
├── benchmark_fpga_full.json
├── benchmark_fpga_compress.json
├── benchmark_fpga_decompress.json
├── benchmark_fpga_nn_only.json
├── benchmark_fpga_entropy_only.json
└── benchmark_fpga_full_parallel.json
```

### 11.3 Estimated Runtime

| Component | No power | `--power --idle-baseline 10` |
| --- | --- | --- |
| GPU + CPU (5 scenarios) | ~5 min | ~8 min |
| FPGA (6 scenarios) | ~5 min | ~6 min |
| Transfer (scp) | ~1 min | ~1 min |
| **Total** | **~10 min** | **~14 min** |

Estimates assume `--warmup 20 --iters 100` (defaults).

---

## 12. Analysis Notebook: `notebooks/benchmark_analysis.ipynb`

### 12.1 Purpose

The analysis notebook loads all JSON result files produced by the benchmark suite
(`benchmark_gpu.py`, `benchmark_fpga.py`, via `run_full_benchmark.py`) and produces
cross-platform comparison visualisations.  It is the **single source of truth** for
interpreting benchmark data and generating publication figures.

### 12.2 Data Loading & Schema Unification

All `benchmark_*.json` files in `results/benchmark/<model_name>/` are loaded and
classified by filename pattern:

| Pattern | Platform |
| --- | --- |
| `benchmark_gpu_<scenario>.json` | GPU (CUDA) |
| `benchmark_cpu_<scenario>.json` | CPU (x86) |
| `benchmark_fpga_<scenario>.json` | FPGA (ZCU102) |

**Scenario canonicalisation**: All platforms now use `nn_only` for the NN-only
(no entropy) scenario. No renaming is required at load time.

**Step label canonicalisation**: Per-step breakdown labels are renamed from
platform-specific prefixes (`gpu_`, `dpu_`) to a canonical `nn_` prefix, enabling
direct visual comparison of the same logical step across platforms.

### 12.3 Derived Metrics & Formulas

The following metrics are derived from the raw JSON fields during loading:

#### NN vs CPU Latency Split

$$t_\text{NN} = \texttt{latency\_\{gpu,dpu,nn\}\_total\_mean\_ms}$$
$$t_\text{CPU} = \texttt{latency\_cpu\_total\_mean\_ms}$$
$$f_\text{NN} = \frac{t_\text{NN}}{t_\text{total}}$$

These are read directly from JSON.  The per-step breakdown provides a finer view:

$$t_\text{NN}^{(\text{steps})} = \sum_{s \in \texttt{nn\_*}} s.\texttt{mean\_s} \times 1000$$
$$t_\text{CPU}^{(\text{steps})} = \sum_{s \in \texttt{cpu\_*}} s.\texttt{mean\_s} \times 1000$$

#### Power Aggregation

Power is aggregated differently per platform due to different measurement instruments:

| Platform | `power_total_w` | `power_nn_w` | `power_cpu_w` |
| --- | --- | --- | --- |
| **GPU** | `nvidia-smi` + RAPL total | `nvidia-smi` avg | RAPL total |
| **CPU** | RAPL total | — | RAPL total |
| **FPGA** | `board_total_avg_w` (INA226) | `groups.DPU_fabric` | `groups.PS_compute` |

Idle baseline (when captured via `--idle-baseline`) is stored as `power_idle_total_w`.

#### Energy per Inference

$$E_\text{tile}\;[\text{mJ}] = P_\text{total}\;[\text{W}] \times t_\text{total}\;[\text{ms}]$$

Dynamic energy removes idle/static power:

$$E_\text{dyn}\;[\text{mJ}] = (P_\text{load} - P_\text{idle})\;[\text{W}] \times t_\text{total}\;[\text{ms}]$$

#### Bits per Pixel (BPP)

$$\text{BPP} = \frac{\texttt{avg\_compressed\_bytes} \times 8}{256 \times 256}$$

#### Throughput

$$\text{Throughput}\;[\text{patches/s}] = \frac{N_\text{iters}}{t_\text{wall}\;[\text{s}]}$$

### 12.4 Notebook Sections

| § | Title | Visualisation | Key insight |
| --- | --- | --- | --- |
| 1 | Setup | — | Set `BENCHMARK_DIR` to target model |
| 2 | Load & merge | Print summary | Verify completeness, spot anomalies |
| 3 | Overview table | Styled DataFrame | Quick scan of all metrics |
| 4 | End-to-end latency | Grouped bars (log) | Compare total latency per scenario |
| 5 | Per-step breakdown | Stacked horizontal bars | Where time is spent within each scenario |
| 6 | NN vs entropy split | Grouped bars + pie | Bottleneck identification (NN vs entropy) |
| 7 | Throughput | Grouped bars (log) | System sizing (patches/s) |
| 8 | Power | Per-scenario bars + idle overlay | Absolute power draw with caveats |
| 9 | Energy per inference | Grouped bars + dynamic printout | Fairest cross-platform metric |
| 10 | Summary table | Publication DataFrames + CSV | Final numbers for the thesis |

### 12.5 Key Measurement Caveats (Summary)

These are discussed in detail within the notebook's markdown cells:

1. **Power scope mismatch**: nvidia-smi (GPU board), RAPL (CPU package + DRAM),
   INA226 (18 SoC rails) measure different system subsets.  Numbers are indicative
   but not perfectly comparable.
2. **RAPL requires root** on kernels ≥ 5.10 (`energy_uj` files are `-r--------`).
   If unavailable, CPU power is reported as 0 W.  The `RAPLPowerSampler` now
   probe-reads during discovery and skips unreadable domains.
3. **Batch=1 everywhere**: GPU is underutilised.  FPGA DPU B4096 only supports batch=1.
4. **Entropy coding dominates total latency** on all platforms.  The `nn_only` scenario
   isolates the NN accelerator ceiling; `entropy_only` isolates the coding bottleneck.
5. **FPGA DPU times include Python/VART overhead** (10–20% above raw hardware time).

### 12.6 Operational Comparison

The notebook concludes with a **satellite downlink** scenario table:

- **Satellite side** (compress): FPGA `compress` scenario metrics
- **Ground side** (decompress): GPU and CPU `decompress` scenario metrics

This is the most deployment-relevant comparison for the SAR DDC use case.

---
