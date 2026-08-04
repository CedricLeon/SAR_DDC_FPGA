# GPU / CPU Benchmark & Cross-platform Comparison

> Host-side (GPU + CPU) benchmarking of the SAR-DDC pipeline, and how it joins the FPGA results for
> a fair cross-platform comparison. FPGA-only benchmarking is in `FPGA_benchmark.md`; inference in
> `FPGA_inference.md`; the Python→C++ history in `python_to_cpp_migration_journal.md`.

---

## 1. Platform semantics (the standard)

Every platform splits into **NN compute** + **CPU-side entropy/normalize**, mirroring the FPGA. This
removes the long-standing "is the CPU number the whole pipeline or just entropy?" confusion.

| `platform` | NN stages (g_a/h_a/h_s/g_s) | entropy + normalize/denorm | precision | `accelerator` |
| --- | --- | --- | --- | --- |
| `fpga` | DPU B4096 | ARM A53 | INT8 | `DPU` |
| `gpu`  | CUDA GPU | host x86 CPU | FP32 | `GPU` |
| `cpu`  | host x86 CPU | host x86 CPU | FP32 | `none` |

So **`gpu` is a hybrid** (GPU NN + CPU entropy) — the closest analogue to the FPGA's DPU+ARM split —
while **`cpu` is pure host x86**. The loader derives `nn_latency_ms` vs `host_latency_ms` from the
canonical stage names, so the split is always explicit.

---

## 2. `scripts/evaluation/benchmark_gpu.py`

Measures per-stage latency, throughput, bytes/BPP and (optionally) power for the model on **CUDA GPU
and/or host CPU**. One invocation runs both devices unless `--no-gpu`/`--no-cpu`.

- **Input**: cycles the **same 20-patch real subset the FPGA uses** (`--data`, `--subset 20`) — so
  bytes/BPP are genuinely comparable across platforms (not a synthetic single patch).
- **Timing**: `time.perf_counter()` with `torch.cuda.synchronize()` before each mark; GPU runs also
  record CUDA-event timings (`latency_breakdown_cuda_events`, µs-precise) as an extra.
- **Model source**: `--model-dir <compiled model dir>` (reads `manifest.json` → checkpoint; required,
  also gives `arch`) or `--ckpt`. Arch resolution errors rather than guesses.
- **Output**: `results/benchmark_unified/<model>/baseline_<scenario>_<platform>.json` — the
  **FPGA-aligned schema** (see §4).

```bash
# GPU + CPU, compress + power, default 20-patch subset
python scripts/evaluation/benchmark_gpu.py --model-dir results/fpga/active_model/ \
    --scenario compress --power --iters 100 --warmup 20
python scripts/evaluation/benchmark_gpu.py --model-dir results/fpga/active_model/ \
    --scenario full --no-cpu          # GPU-only, full pipeline
```

Scenarios: `compress`, `full` (the cross-platform set), plus host-only extras `decompress`,
`nn_only`, `entropy_only` (component isolation — **not** the FPGA's data-parallel ceilings of the
same name; don't co-plot them).

---

## 3. `scripts/benchmark/run_unified_benchmark.py` — one command, all platforms

Drives the host GPU/CPU benchmark locally **and** the FPGA sweep over SSH, into one results tree, via
a **pluggable backend registry** (`BACKENDS`). Adding new hardware (e.g. a Jetson) = add a backend +
a `--no-<hw>` toggle, no rewrite.

```bash
conda activate DDC_FPGA
python scripts/benchmark/run_unified_benchmark.py --model-dir results/fpga/active_model/ --power
python scripts/benchmark/run_unified_benchmark.py --model-dir results/fpga/active_model/ --no-fpga      # host only
python scripts/benchmark/run_unified_benchmark.py --model-dir results/fpga/active_model/ --no-gpu --no-cpu  # FPGA only
```

- **host backend** → `benchmark_gpu.py` per scenario → `results/benchmark_unified/`.
- **fpga backend** → `scripts/fpga/benchmark/benchmark_sweep.py` over SSH (deploy + s0/s1 ×
  compress/full + fetch) → `results/benchmark_hardware/`. Requires the ZCU102 reachable; `--no-fpga`
  reuses existing FPGA results.
- Flags: `--scenarios compress,full`, `--warmup/--iters/--fpga-iters/--subset/--power/--idle-baseline`,
  `--rebuild-cpp`. A backend whose hardware is unreachable warns and is skipped (others continue).

---

## 4. Aligned output schema (host)

Mirrors the C++ `benchmark_hardware` schema so one loader reads both trees:

```jsonc
{
  "platform": "gpu",            // "gpu" | "cpu"
  "accelerator": "GPU",         // "GPU" | "none"
  "config": "baseline",         // host has no s0/s1 axis
  "scenario": "compress",       // compress | full | (host extras)
  "arch": "SHyp",
  "warmup": 20, "iters": 100, "subset_patches": 20,
  "wall_time_s": ..., "throughput_fps": ..., "total_latency_mean_ms": ...,
  "evaluated_at": "...",
  "stages": { "normalize": {"mean_ms":..,"std_ms":..,...}, "g_a": {...}, "eb_compress": {...},
              "host_concat_abs": {...}, ... },   // bare canonical names; host-only ops -> host_*
  "bytes_per_iter": [...],
  "nn_latency_mean_ms": ..., "host_latency_mean_ms": ...,
  "power": {
    "power_scope": "GPU_board+RAPL",            // or "RAPL_pkg+dram" (cpu)
    "active_w": ..., "idle_w": ..., "dynamic_w": ...,        // null if RAPL unavailable
    "energy_mj_per_patch": ..., "dynamic_energy_mj_per_patch": ...,
    "native": { "gpu_active": {...}, "rapl_active": {...}, "gpu_idle": {...}, "rapl_idle": {...} }
  },
  // extras: device_name, latency_breakdown (seconds), latency_breakdown_cuda_events, hw_info, model_info
}
```

Stage-name alignment: NN `gpu_/nn_` prefixes stripped to bare (`g_a`, …); `cpu_eb_compress`→`eb_compress`;
host `preprocess`/`postprocess`→`normalize`/`denorm` (same CPU work as the FPGA stages);
`cpu_concat_abs`/`cpu_split_y_hat`→`host_concat_abs`/`host_split_y_hat` (no FPGA stage equivalent).

---

## 5. Power measurement

- **GPU**: `nvidia-smi --query-gpu=power.draw` polled in a background thread — full board draw.
- **CPU**: Intel RAPL energy counters at `/sys/class/powercap/intel-rapl/*/energy_uj` (package + DRAM).
  **Needs read permission** — if absent, host power is emitted as `null` (not 0). Enable with:
  `sudo chmod o+r /sys/class/powercap/intel-rapl/*/energy_uj`.
- An idle baseline (`--idle-baseline N` s, model loaded, no inference) is captured before each run so
  `dynamic_w = active_w − idle_w` isolates the workload.

**Scopes differ** (GPU board vs CPU package+DRAM vs FPGA MPSoC) → compare **energy/inference**, not
raw watts. For the `gpu` platform, `active_w` = GPU board **+** CPU package (entropy runs on CPU), so
it is the true total-system power.

---

## 6. Cross-platform analysis — `notebooks/benchmark_cross_platform_analysis.ipynb`

Loads both trees via `notebooks/_benchmark_loader.py` (`platform` in the identity key; unified power
columns; `nn_latency_ms`/`host_latency_ms`). Quality is precision-correct: **FP32** from W&B (linked by
`wandb_run_id` from each model's manifest) for GPU/CPU, **INT8** from `metrics.json` for FPGA.

Every plot is **argument-driven** (`series=[...]`, `scenario=`, `qmetric=`, `size_by_bpp=`, `save=`):

- grouped bars: latency, throughput, energy/inference, energy-delay product (EDP) — ×-vs-CPU annotated.
- per-stage stacked breakdown (shared Y) — shows the NN↔entropy bottleneck shift across platforms.
- quality-vs-cost scatter — `qmetric=` selects any canonical quality key (`quality_keys()`); circle ∝ bpp.

**Derived metrics**: `energy_mj_per_patch = active_w × latency_ms`; `dynamic_energy` similarly;
`edp_mJ_ms = energy × latency` (lower = fast AND low-energy); `BPP = avg_compressed_bytes × 8 / 256²`.

---

## 7. Fairness caveats

- **FP32 vs INT8**: GPU/CPU run FP32, FPGA INT8 → quality differs (PSNR/SSIM not equal); always pair
  speed/energy plots with the quality-vs-cost view.
- **CPU = 6-thread x86** (not single core) — note when comparing to the FPGA ARM.
- **Power scopes differ** (see §5) — energy/inference is the fair axis.
- **20-patch subset, real data** on all platforms (host no longer uses a synthetic single patch).

---

## 8. Future work

- **Seed-averaged quality** (mean±std over the 6 compiled seeds) on the quality-vs-cost plots.
- **Throughput-per-watt** and **full-scenario** cross-platform figures.
- **Operational downlink table**: satellite side = FPGA `compress`; ground side = GPU/CPU
  `decompress`/`full` — the most deployment-relevant framing for SAR DDC.
- Multi-λ cross-platform sweeps (host quality already exists in W&B across λ; host latency/power is
  λ-invariant in topology, so one representative λ usually suffices).
