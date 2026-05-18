# FPGA Inference — Architecture, Pipeline & Roadmap

> Reference document for the on-board hybrid inference workflow on the Xilinx ZCU102.
> Covers current state, known bottlenecks, and planned work on full-image streaming inference.

---

## 1. Overview

The compiled model runs in a **hybrid DPU+CPU** configuration:

- **DPU** (DPUCZDX8G B4096, 300 MHz): runs the 4 neural-network subgraphs
  (`g_a`, `h_a`, `h_s`, `g_s`) as INT8 operations.
- **ARM A53 CPU**: handles all entropy coding (rANS via C++ `ans.so`), normalisation,
  tensor splitting/concatenation, and metrics computation.

The code that implements this lives entirely in `scripts/fpga/` and is **only
executable on the ZCU102 board** (requires `vart`, `xir`, and the `ans.so` extension).

---

## 2. Relevant Files

| File | Role |
| ------ | ------ |
| `scripts/fpga/inference_hybrid.py` | Main orchestrator: loads model, runs the inference loop over patches/tiles |
| `scripts/fpga/inference_utils.py` | DPU runners (`DPUSubgraphRunner`, `DPUJob`), subgraph identification, metrics, tiling utilities |
| `scripts/fpga/entropy_models_inference.py` | Pure-NumPy + C++ rANS wrappers for `EntropyBottleneck` and `GaussianConditional` |
| `scripts/fpga/benchmark_fpga.py` | Latency/power benchmarking (not quality evaluation) |
| `scripts/fpga/deploy.py` | Host-side orchestrator: quantize → compile → scp → run inference → fetch results |
| `scripts/fpga/deploy_cpp_entropy_coder/` | Build scripts for the `ans.so` C++ extension (cross-compile for aarch64) |

---

## 3. Data Formats

### 3.1 Test Patch Set (`*.npy`)

Loaded by `load_npy_test_set()`:

```text
Shape: [N, 256, 256, 4]
  ch 0:  real amplitude  (unnormalized, raw complex)
  ch 1:  imag amplitude  (unnormalized, raw complex)
  ch 2:  ADAM-NOC ground truth (linear amplitude)
  ch 3:  MERLIN ground truth   (linear amplitude)
```

The noisy input (`ch 0–1`) is in raw complex amplitude. **Normalisation is done inside the pipeline** (in `_prepare_tile_input`), not in the dataset.

### 3.2 Large Tile (`sym_Noisy.npy`)

```text
Shape: [H, W, 2]  — raw complex amplitude, H and W arbitrary (e.g. 1024×1024)
```

Located in `data/visualization/<Region>/sym_Noisy.npy`. Reference ground truths are `linA_MERLIN.npy`, `linA_ADAM_NOC.npy` etc. in the same folder.

---

## 4. Normalisation Convention

All models expect **log-scale, min-max normalised input**. The transform applied per tile is:

```python
# In _prepare_tile_input (inference_hybrid.py)
noisy_sq    = noisy**2                          # square: amplitude -> intensity
noisy_logI  = log(noisy_sq + EPS)               # log intensity, EPS=1e-2
noisy_norm  = (noisy_logI - 2*AMP_MIN) / (2*AMP_MAX - 2*AMP_MIN)
```

where:

- `AMP_MIN = 4.605170…`  (p5 of `log(amplitude + EPS)` over the full dataset)
- `AMP_MAX = 10.742239…` (p95 of `log(amplitude + EPS)` over the full dataset)
- The factor of 2 comes from squaring: `log(a²) = 2·log(a)`.

Canonical values and derivation: `src/utils/constants.py` and [docs/Data.md](Data.md).

The real and imag channels are processed **separately** through `g_a` and `g_s`.

### Denormalisation (output)

```python
recon_logI = recon_norm * (AMP_MAX - AMP_MIN) + AMP_MIN
recon_linI = exp(recon_logI)
# MERLIN factor: average both reconstructions to recover reflectivity
recon_linI = 0.5 * (recon_real**2 + recon_imag**2)
recon_linA = sqrt(recon_linI)           # -> linear amplitude for metrics
```

---

## 5. Current Pipeline (Single Tile, Sequential)

### 5.1 ScaleHyperprior topology (`process_single_tile_SHyp`)

```text
[CPU]   Normalise real/imag                       in: [H,W,2]       -> [1,1,H,W] x2
[DPU]   g_a(real), g_a(imag)                      in: [1,1,256,256] -> [1,16,16,128]
[CPU]   concat + abs                               -> y[1,16,16,256], y_abs[1,16,16,256]
[DPU]   h_a(y_abs)                                -> z[1,2,2,256]
[CPU]   EB.compress(z)                             -> z_strings (bytes)
[CPU]   EB.decompress(z_strings)                  -> z_hat[1,2,2,256]
[DPU]   h_s(z_hat)                                -> scales[1,16,16,256]
[CPU]   GC.compress(y, scales, means=0)           -> y_strings (bytes)
[CPU]   GC.decompress(y_strings, scales, means=0) -> y_hat[1,16,16,256]
[CPU]   split y_hat                               -> y_hat_real/imag[1,16,16,128]
[DPU]   g_s(y_hat_real), g_s(y_hat_imag)         -> recon[1,256,256,1] x2
[CPU]   stack real/imag                           -> recon[256,256,2]
```

**Total DPU calls per tile: 6** (g_a×2, h_a×1, h_s×1, g_s×2).

### 5.2 FactorizedPrior topology (`process_single_tile_FP`)

```text
[CPU]   Normalise real/imag
[DPU]   g_a(real), g_a(imag)                     -> y_real/imag[1,16,16,128]
[CPU]   concat                                    -> y[1,16,16,256]
[CPU]   EB.compress(y), EB.decompress(y_strings) -> y_hat[1,16,16,256]
[CPU]   split                                     -> y_hat_real/imag[1,16,16,128]
[DPU]   g_s(y_hat_real), g_s(y_hat_imag)         -> recon x2
[CPU]   stack                                     -> recon[256,256,2]
```

### 5.3 Large Image Inference (`patch_infer_fpga`)

Large tiles (e.g. 1024×1024) are split into overlapping 256×256 windows with a
configurable stride (`patch_size - overlap`). Each patch goes through the single-tile
pipeline above and results are blended using a sigmoid/linear/cosine feathering ramp
in overlap zones to avoid visible seams.

Current call is **fully sequential**: patch 0 is processed end-to-end, then patch 1, etc.

---

## 6. DPU Runner Abstraction

### `DPUSubgraphRunner`

Wraps a `vart.Runner` for a single compiled subgraph. Exposes:

- `run(input_float) -> output_float` — blocking synchronous call (quantize → DPU → dequantize)
- `submit(input_float) -> DPUJob` — non-blocking async dispatch (returns immediately)
- `collect(job) -> output_float` — wait for a previously submitted job

INT8 ↔ float conversion uses the fixed-point scale read from the xmodel tensor attributes
(`fix_point`):

```text
input_int8 = float * 2^(+fix_point_in)
output_float = int8 * 2^(-fix_point_out)
```

### `DPUJob`

Handle keeping both I/O buffers alive during async DMA. Must not be GC'd before `collect()`.

### Multi-core parallelism

`benchmark_fpga.py` already exploits the 3-core B4096 by dispatching `g_a(real)` and
`g_a(imag)` in parallel via `ThreadPoolExecutor(2)`, achieving ~1.9× speedup on `g_a`.
The key constraint is that VART assigns runners to cores **at creation time** (round-robin),
so the creation order of runner instances determines which core they land on.

---

## 7. Entropy Models

Both `EntropyBottleneck` and `GaussianConditional` are pure-Python/NumPy wrappers around
the C++ rANS coder (`ans.so`). They are loaded from a pre-exported `.npz` file
(`entropy_params.npz`) that contains the CDF tables baked at training time.

The key exported arrays are:

| Array | Shape | Description |
| ------- | ------- | ------------- |
| `eb_quantized_cdf` | `[C, cdf_size]` | Per-channel CDF tables for EB |
| `eb_cdf_length` | `[C]` | Valid CDF entries per channel |
| `eb_offset` | `[C]` | Symbol offset (so symbols can be negative) |
| `eb_medians` | `[C]` | Per-channel medians used for quantisation |
| `gc_scale_table` | `[n_scales]` | Discrete scale table for GC |
| `gc_quantized_cdf` | `[n_scales, cdf_size]` | Per-scale CDF tables for GC |
| `gc_cdf_length` | `[n_scales]` | Valid CDF entries per scale |
| `gc_offset` | `[n_scales]` | Symbol offset for GC |

**The entropy coding is the primary latency bottleneck** for each tile (significantly
slower than the DPU subgraph calls), particularly for `GaussianConditional` on `y`
(`[1, 16, 16, 256]` = 65,536 symbols per channel pass).

---

## 8. Known Issues & Messiness

- `inference_hybrid.py` uses `tuple[...]` and `dict[...]` syntax (PEP 585) which is
  **not Python 3.8 compatible** (e.g. `load_npy_test_set` return type).
- The `log()` function references a global `log_file` variable — fragile.
- Constants (`IMAGE_SIZE`, `C_MAIN`, `C_HYPER`, `S_MAIN`, `S_HYPER`) are duplicated
  between `inference_hybrid.py` and `benchmark_fpga.py`.
- `patch_infer_fpga` is in `inference_utils.py` but is a fairly large, self-contained
  function — coupling is high.
- The decompress/compress cycle in `process_single_tile_*` blocks the CPU entirely;
  there is no way for the DPU to overlap with entropy coding in the current design.

---

## 9. Planned Work — Full Image Streaming Inference

### 9.1 Motivation

Current inference operates on **pre-extracted 256×256 patches** (the test-set `.npy` file).
The new workflow should accept a **full SAR image** (arbitrary size, e.g. 1024×1024 or
larger), apply the DDC pipeline end-to-end, and report reconstruction quality + throughput.

The goal is to answer: *what would end-to-end latency and quality look like in a realistic
operational scenario where the FPGA receives a full image, compresses it, and transmits
the bitstream to a host?*

### 9.2 Target Execution Model

The natural pipeline for one image has these **stages per patch**:

```text
Stage A  [CPU]  Extract + normalise patch
Stage B  [DPU]  g_a (encoder)                ← DPU-bound, ~35 ms
Stage C  [CPU]  concat/abs, then h_a input prep
Stage D  [DPU]  h_a                           ← fast, ~1 ms
Stage E  [CPU]  EB compress + decompress      ← CPU-bound bottleneck
Stage F  [DPU]  h_s                           ← fast, ~1 ms
Stage G  [CPU]  GC compress + decompress      ← CPU-bound bottleneck
Stage H  [DPU]  g_s (decoder)                ← DPU-bound, ~35 ms
Stage I  [CPU]  denormalise, write to canvas
```

With sequential execution the total per-patch cost is roughly:
`~35 + ~35 + ~30 (entropy) ≈ 100 ms/patch`.

With a **producer-consumer pipeline** across patches, we could overlap:

- Patch N entropy coding (Stage E/G) with Patch N+1 DPU encoding (Stage B)
- Patch N DPU decoding (Stage H) with Patch N+1 entropy coding (Stage E/G)

### 9.3 Parallelisation Opportunities

| Opportunity | Mechanism | Expected gain |
| ------------- | ----------- | --------------- |
| g_a real ‖ g_a imag (per patch) | `submit`/`collect` on 2 runners (already done in benchmark) | ~1.9× on g_a |
| g_s real ‖ g_s imag (per patch) | same | ~1.9× on g_s |
| Patch N DPU ‖ Patch N-1 entropy | Separate threads with a queue | up to 1× entropy effectively hidden |
| C++ entropy instead of Python loops | Port Python loops in `EntropyBottleneck.compress` to C++ | 5–10× on entropy |

### 9.4 Migration to C++

The Python overhead in entropy coding comes from:

1. `.tolist()` conversions of large numpy arrays before passing to `ans.encode_with_indexes`
2. Python-level loops over batch items (N=1 always, so minor)
3. The GIL preventing true concurrency in the Python wrapper

A C++ wrapper around the existing `ans.so` (or a direct rANS reimplementation) would
eliminate (1) and (3). The interface would be:

```cpp
std::vector<uint8_t> eb_compress(float* symbols, int H, int W, int C,
                                  int32_t* cdf, int32_t* cdf_len, int32_t* offset,
                                  float* medians);
```

### 9.5 New Script: `inference_fullimage.py`

Proposed script (to be created) that wraps the full workflow:

```text
load_full_image(path)           # [H, W, 2] sym_Noisy.npy
pad_to_multiple(image, 256)     # ensure divisibility
extract_patch_grid(image, 256, overlap)
for each patch:                 # producer-consumer pipeline
    preprocess
    DPU encode
    entropy compress
    entropy decompress
    DPU decode
    postprocess
reconstruct_from_patches(blended)
save output + metrics
```

---

## 10. Deployment Workflow (Recap)

The host-side orchestrator `deploy.py` manages the full pipeline:

```text
[Host/Container]  quantize.sh → model_quant.py (calib + deploy xmodel)
[Host/Container]  vai_c_xir   → compiled *.xmodel
[Host]            export_entropy_params() → entropy_params.npz
[Host]            scp compiled_model/ → ZCU102:/home/root/SAR_DDC/active_model/
[ZCU102]          python3 inference_hybrid.py --xmodel *.xmodel --data test.npy
[Host]            scp results/ ← ZCU102
```

Results land in `results/fpga/active_model/results/`.

For step-by-step deployment notes, environment-specific settings, and known issues encountered along the way, see [docs/Vitis-AI_journey.md](Vitis-AI_journey.md).
