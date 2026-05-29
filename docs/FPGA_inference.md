# FPGA Inference — C++ Pipeline Reference

> How on-board inference works **now**: the C++ `inference_hybrid` binary on the Xilinx ZCU102.
> For *why* it was ported from Python and the before/after numbers, see
> `python_to_cpp_migration_journal.md`. For benchmarking, see `FPGA_benchmark.md`.

---

## 1. Overview

The compiled model runs in a **hybrid DPU+CPU** configuration:

- **DPU** (DPUCZDX8G **B4096 @ 300 MHz**, 3 cores): the four NN subgraphs (`g_a`, `h_a`, `h_s`,
  `g_s`) as INT8.
- **ARM A53 CPU** (×4): entropy coding (C++ rANS), log-amplitude normalization, tensor
  interleave/deinterleave, denormalization, metrics.

Inference is a single native C++ binary, `build_cpp/inference_hybrid`, built on the board. There is
no Python on the board. It processes the 256×256 test patches and the Hamburg tile, computes task
metrics (bpp/PSNR/SSIM/ENL/EPD), and writes `metrics.json` + reconstructions.

---

## 2. Relevant files

| Path | Role |
| --- | --- |
| `inference_cpp/src/` | All C++ inference sources (see §6) |
| `build_cpp/inference_hybrid` | The inference binary on the board |
| `inference_cpp/src/rans/` | pybind11-free rANS fork (entropy codec) |
| `scripts/fpga/deploy/deploy.py` | Host orchestrator: quantize → compile → export params → transfer → run C++ → fetch |
| `scripts/fpga/deploy/batch_deploy.py` | Batch wrapper over `deploy.py` (rebuilds the C++ binary once per batch) |
| `scripts/fpga/deploy/model_quant.py` | PTQ (calibration + deploy xmodel) inside the Vitis-AI Docker |

The deployable model lives in `active_model/` on the board: the `.xmodel` + an `entropy_params/`
directory of individual `.npy` CDF tables.

---

## 3. Data formats

### 3.1 Test patch set (`*.npy`)

```text
Shape: [N, 256, 256, 4]
  ch 0:  real amplitude  (unnormalized, raw complex)
  ch 1:  imag amplitude  (unnormalized, raw complex)
  ch 2:  ADAM-NOC ground truth (linear amplitude)
  ch 3:  MERLIN ground truth   (linear amplitude)
```

Normalisation is done **inside the pipeline**, not in the dataset. Board path:
`data/test_sub500_seed42.npy` (float32).

### 3.2 Large tile (`sym_Noisy.npy`)

`[H, W, 2]` raw complex amplitude (e.g. 1024×1024), under `data/visualization/<Region>/`.
Reference ground truths (`linA_MERLIN.npy`, `linA_ADAM_NOC.npy`) sit alongside.
**Note:** `sym_Noisy.npy` is stored **float64** — the NPY loader casts dtype-aware
(`to_float32_vec()`), never a raw reinterpret (see migration journal §6, Bug 6).

---

## 4. Normalisation convention

All models expect **log-scale, min-max normalised** input:

```text
noisy_sq    = noisy**2                          # amplitude -> intensity
noisy_logI  = log(noisy_sq + EPS)               # EPS = 1e-2
noisy_norm  = (noisy_logI - 2*AMP_MIN) / (2*AMP_MAX - 2*AMP_MIN)
```

`AMP_MIN = 4.605170…`, `AMP_MAX = 10.742239…` (p5/p95 of `log(amp+EPS)`); the factor 2 comes from
`log(a²) = 2·log(a)`. Single source of truth: `src/utils/constants.py` / `inference_cpp/src/constants.hpp`.

Denormalisation (output): `recon_logI = recon_norm·(AMP_MAX−AMP_MIN) + AMP_MIN`; `recon_linI =
exp(recon_logI)`; MERLIN reflectivity = `0.5·(recon_real² + recon_imag²)`; `recon_linA = sqrt(linI)`.
Denorm runs in **double precision** internally to match the Python float64 reference (journal §6).

---

## 5. Pipeline (single tile, sequential)

### 5.1 ScaleHyperprior (SHyp / ResSHyp)

```text
[CPU] normalize real/imag
[DPU] g_a(real), g_a(imag)                      -> y       (NHWC interleaved, C = 2·C_MAIN)
[CPU] concat + |y|                              -> y_abs
[DPU] h_a(y_abs)                                -> z
[CPU] EntropyBottleneck compress/decompress(z)  -> z_hat
[DPU] h_s(z_hat)                                -> scales
[CPU] GaussianConditional compress/decompress(y, scales) -> y_hat
[CPU] deinterleave y_hat                        -> y_hat_real/imag
[DPU] g_s(real), g_s(imag)                      -> recon
[CPU] denormalize                               -> linear amplitude
```

DPU calls per patch: **6** (g_a×2, h_a, h_s, g_s×2).

### 5.2 FactorizedPrior (FP / ResFP)

No `h_a`/`h_s`/GC: `normalize → g_a×2 → EB compress/decompress(y) → g_s×2 → denormalize`.
DPU calls per patch: **4**.

> **Channel layout (journal §6, Bug 2/5):** `y` is built **NHWC-interleaved** (`C = 2·C_MAIN`), not
> as `[real_block│imag_block]` — otherwise each symbol pairs with the wrong CDF/scale.

### 5.3 Large-image inference

Large tiles are split into overlapping 256×256 windows (stride = patch − overlap); each window runs
the single-tile pipeline; outputs are blended with a sigmoid feathering ramp in the overlap zones to
avoid seams. (Currently sequential per window.)

---

## 6. Module layout (`inference_cpp/`)

```text
src/
  main.cpp                       CLI: --xmodel/--params/--data/--output/--subset/--verbose
  inference_runner.{cpp,hpp}     InferenceRunner/Pipeline (SHyp+FP paths, tile blending, metrics.json)
  dpu_runners.{cpp,hpp}          DPUSubgraphRunner + XModelLoader (VART/XIR, HAVE_DPU-gated)
  entropy_models.{cpp,hpp}       EntropyBottleneck, GaussianConditional
  metrics.{cpp,hpp}              PSNR / SSIM (OpenCV) / ENL / EPD / BPP
  npy_io.{cpp,hpp}               NPY v1/v2 loader+writer (no deps; dtype-aware float32/float64)
  patch_transforms.hpp           normalize / interleave / denorm pure fns (shared with benchmark)
  constants.hpp, logger.hpp, scoped_timer.hpp
  rans/                          pybind11-free rANS fork (rans_interface_cxx + rans64.h)
  benchmark/                     benchmark_hardware sources (see FPGA_benchmark.md)
tests/test_rans_roundtrip.cpp    host unit test (rANS encode↔decode)
```

**Board libraries:** OpenCV 4.5.2 (incl. `opencv_quality` for SSIM), Eigen 3, nlohmann/json 3.10.2.
spdlog is absent → custom `Logger`. **CMake options:** `HAVE_DPU` (default ON; needs VART/XIR +
nlohmann/json), `WITHOUT_OPENCV` (stub SSIM for host syntax checks).

### DPU runners & VART

`DPUSubgraphRunner` wraps a `vart::Runner`: quantize float→INT8, `execute_async` + `wait`,
dequantize INT8→float via the tensor `fix_point` (`input_int8 = float·2^{+fp_in}`,
`output_float = int8·2^{−fp_out}`). `XModelLoader` finds the named subgraphs.

> **VART core assignment (footgun):** `create_runner()` has no core index — VART assigns DPU cores
> **round-robin at creation time**. Two runners on the same core run serially. For any parallel
> scheme, runner **creation order** is the only lever; verify placement by timing. (Relevant to the
> benchmark's S1/ceilings — see `FPGA_benchmark.md`.)

### Entropy models (rANS)

`EntropyBottleneck` / `GaussianConditional` mirror the CompressAI semantics on top of a
**pybind11-free rANS fork** (`rans/`) — bit-compatible with the original Python `ans` encoder
(roundtrip unit test). CDF tables load from `entropy_params/*.npy`
(`eb_quantized_cdf`, `eb_cdf_length`, `eb_offset`, `eb_medians`, `gc_scale_table`,
`gc_quantized_cdf`, `gc_cdf_length`, `gc_offset`). Rounding uses an explicit FPU-mode-independent
`round_half_to_even()` to match numpy at half-integer boundaries (journal §6).

---

## 7. Build & run (board)

```bash
# build (native on the ZCU102; deploy.py --rebuild-cpp / batch_deploy.py automate this)
rsync -av inference_cpp/src/ ZCU102:/home/root/SAR_DDC/inference_cpp/src/
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && make -j4"

# run (from /home/root/SAR_DDC on board)
build_cpp/inference_hybrid \
    --xmodel active_model/*.xmodel \
    --params active_model/entropy_params \
    --data   data/test_sub500_seed42.npy \
    --subset 100 [--debug-patch N] [--verbose]
```

Outputs in `<model>/results/`: `metrics.json` (averaged task metrics), `<tile>_metrics.json` +
`<tile>_recon_linA.npy` per tile, visualization NPYs, `inference.log`, `inference_meta.json`.

**DPU model constraints** (codified in `CLAUDE.md`): no `GDN` (use ReLU), no `LowerBoundFunction`
(use clamp), `ConvTranspose2d` must have `output_padding=0`.

---

## 8. Performance characteristic

In C++ the pipeline is **DPU-dominated** for the residual models (ResSHyp ≈ 78% DPU, 12% entropy,
10% normalize/denorm) and more balanced for the small models (FP ≈ 37% DPU; normalize/denorm and
entropy dominate). This is the *reverse* of the Python era, where entropy coding dominated — the C++
rANS collapsed the entropy cost. Full numbers: `python_to_cpp_migration_journal.md` §3 / `FPGA_benchmark.md`.

---

## 9. Future work (inference)

- **NEON-vectorize `normalize` + `denorm`** (`patch_transforms.hpp`): ~18.7 ms/patch of scalar
  `std::log`/`exp` (131K calls), now ~10% of the full pipeline (much larger share for FP). Highest-value
  CPU optimization.
- **Datatype/precision audit (inference + benchmark).** `denorm_to_lina()` runs in **double** to match
  the Python float64 reference (§4/§8) — but it is unclear we need that precision. Study the dtypes used
  throughout the C++ pipeline (input data, intermediate buffers, denorm math) and check whether float32
  (or NEON-friendly types) gives the same metrics within tolerance while saving compute. Small,
  self-contained study; pairs naturally with the NEON work above.
- **Full-image streaming inference**: accept a full SAR image (arbitrary size), pad to ×256, extract
  an overlapping patch grid, run a producer-consumer pipeline (overlap patch N+1 DPU with patch N
  entropy), blend, report quality + throughput — the realistic "receive image → compress → transmit
  bitstream" scenario. Would reuse the benchmark's pipelining work (`FPGA_benchmark.md`, M4+).

---

## 10. Deployment (recap)

`deploy.py` (host): quantize/compile in the Vitis-AI Docker (`model_quant.py` + `vai_c_xir`) →
`export_entropy_params` writes `entropy_params/*.npy` → `rsync --exclude=results active_model/` →
build/run the C++ binary on the board → fetch `results/`. `--rebuild-cpp` (or `batch_deploy.py`)
pushes updated C++ sources and recompiles. Step-by-step notes + historical issues:
`Vitis-AI_journey.md`.
