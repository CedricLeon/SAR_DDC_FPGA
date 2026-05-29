# FPGA Benchmark — Hardware, Methodology & Development Journal

> The benchmark reference for the SAR DDC pipeline on the Xilinx ZCU102: hardware, the
> `benchmark_hardware` C++ binary, measurement methodology, results, future work, and a decisions
> log. **Inference** itself is documented in `FPGA_inference.md`; the **Python→C++ migration**
> before/after is in `python_to_cpp_migration_journal.md`; the **GPU/CPU** benchmark and the legacy
> cross-platform tooling live (raw) in `GPU_benchmark.md`.
>
> This doc consolidates the former `performance_benchmark_implementation.md` (Python-era hardware /
> methodology, kept) and `benchmark_hardware_design.md` (C++ dev journal). Python-specific *how-to*
> (`benchmark_fpga.py`, `run_full_benchmark.py`) is dropped; Python-era *data* is retained where it
> is hardware-truth (language-independent).

---

## 1. Hardware platform — Xilinx ZCU102

| Component | Spec |
| --- | --- |
| SoC | Zynq UltraScale+ XCZU9EG-2FFVB1156 |
| CPU | 4× ARM Cortex-A53 (ARMv8-A) |
| DPU IP | DPUCZDX8G v4.1 — ISA1 **B4096**, **3 cores**, **300 MHz**, fingerprint `0x101000056010407` |
| Vitis-AI | 3.0 · batch size 1 · OS PetaLinux (aarch64) · board Python 3.8 |

**Peak compute (one B4096 core):** 4096 ops/clk × 300 MHz = **1.2288 TOPS** (INT8; Xilinx counts
1 MAC = 2 ops). Resource use for a 3-core B4096 (additive estimate) on the XCZU9EG: ~57% LUTs,
~54% FF, ~84% BRAM, ~85% DSP.

---

## 2. Model architecture & subgraph workload

Hyper-autoencoder (Scale Hyperprior, Ballé 2018) with residual blocks, ReLU (GDN replaced for DPU),
**N=128** main / **M=256** hyper channels, ~15M params. Real and imag channels processed separately.

| Subgraph | Role | INT8 weights (CONST) | Workload (OPs) |
| --- | --- | --- | --- |
| g_a | encoder | 3,731,456 B | 39.75 G |
| h_a | hyper-encoder | 4,927,488 B | 0.275 G |
| h_s | hyper-decoder | 3,158,016 B | 0.176 G |
| g_s | decoder | 3,289,088 B | 38.13 G |
| **total** | | **15.1 MB** | **78.34 G** |

**Why h_a/h_s have huge weights but tiny OPs:** OPs ∝ `k²·C_in·C_out·H_out·W_out`. g_a/g_s operate on
large feature maps (256² → 16²) → high OPs; h_a/h_s on 2×2–16×16 grids → sub-ms on the DPU despite
wide 256-channel kernels. **Per-patch DPU calls: 6** (g_a×2, h_a, h_s, g_s×2) for SHyp; 4 for FP.
λ and seed do not change topology (N/M fixed) — workload/bytes/peak are λ-invariant.

---

## 3. `xdputil` tooling & roofline data

| Command | Use |
| --- | --- |
| `xdputil query` | DPU cores, arch, clock, fingerprint |
| `xdputil xmodel <m> -l` | static analysis → per-subgraph `workload` (OPs), `reg info` (REG_0 CONST/REG_1 WORKSPACE/REG_2 INPUT/REG_3 OUTPUT bytes), shapes, fixpos |
| `xdputil benchmark <m> -i <idx>` | synthetic single-subgraph peak FPS (random data, zero host overhead) |

**Per-subgraph peaks (single-thread, ~89% DPU utilisation on g_a/g_s):**

| Subgraph | Workload | Peak FPS | Throughput | Util |
| --- | --- | --- | --- | --- |
| g_a | 39.75 G | 27.8 | 1.105 TOPS | 89.9% |
| g_s | 38.13 G | 28.5 | 1.087 TOPS | 88.5% |
| h_a | 0.275 G | 1,225 | 0.337 GOPS | 0.03% (tiny maps, not an efficiency issue) |
| h_s | 0.176 G | 1,375 | 0.242 GOPS | 0.02% |

**Memory per subgraph** (bytes): g_a CONST 3.56 MB / WORKSPACE 6.0 MB; g_s 3.14 MB / 7.87 MB;
h_a 4.70 MB / 20 KB; h_s 3.01 MB / 20 KB. Total weights ~14.4 MB — negligible vs 4 GB DDR4.

**Collection:** `scripts/fpga/benchmark/collect_roofline.py` (board) runs `xdputil benchmark` + parses
`xdputil xmodel -l` per subgraph → `results/benchmark_hardware/_roofline/<model>_xmodel_info.json`
(`peak_fps`, `workload_ops`, `const/workspace/input/output_bytes`, `fixpos_*`, totals). Consumed by
`benchmark_hardware_analysis.ipynb` for the Williams roofline (arithmetic intensity × throughput vs
the 1.2288 TOPS compute ceiling and DDR-bandwidth memory ceiling).

---

## 4. The `benchmark_hardware` binary

**Question it answers (for publication):** what is the best schedule for this heterogeneous DPU+CPU
pipeline on the ZCU102 (3× B4096 + 4× A53)? **Headline metric = throughput (patches/s)**;
deployment-realistic workload = compress-only.

**Locked decisions:** throughput objective; compress-only for pipelined configs; data-parallel
ceilings (`nn_only`/`entropy_only`) + pipelined `compress`, with `full`/`decompress` kept sequential
as references; 4 archs benchmarked separately (SHyp/ResSHyp/FP/ResFP — `Res` = residual blocks on,
heavier DPU); native C++ power sampler; modular, single-source-of-truth build.

| Config | Pattern | Resources | Purpose |
| --- | --- | --- | --- |
| **S0** | sequential | 1 core, 1 thread | C++ per-stage baseline |
| **S1** | channel fan-out | 2 g_a + 2 g_s runners | `g_a(real)‖g_a(imag)`, `g_s(real)‖g_s(imag)` (fixed parallelism) |
| **nn_only** | fan-out | `--dpu-cores N` replicas | data-parallel DPU ceiling |
| **entropy_only** | fan-out | `--entropy-threads N` workers | CPU/entropy ceiling |
| **P0** (future) | patch pipeline | 1 DPU lane + 1 entropy worker | patch N entropy ‖ patch N+1 DPU |
| **P2** (future) | patch pipeline | 1 DPU lane + K workers | attack entropy ceiling with spare A53s |

**`--dpu-cores` means different things — do not conflate:** S1 ignores it (structurally 2 runners/pair);
`nn_only` = N independent patch runners (the roofline); P0/P2 = concurrent pipeline DPU lanes.
**No core pinning:** VART assigns cores round-robin at *runner-creation time* — `--dpu-cores N`
*influences* placement (N replicas → up to 3 distinct cores), not guarantees it; overlap is verified
empirically (concurrent wall-time ≈ serial/N). **S1 creation-order is critical** (see §11 / journal §6).
`--entropy-threads` = genuine OS threads on the A53s; each needs its own `BenchPipeline` (thread safety).

```bash
build_cpp/benchmark_hardware --xmodel active_model/*.xmodel --params active_model/entropy_params \
    --data data/test_sub500_seed42.npy --config s0 --scenario compress \
    --warmup 5 --iters 50 [--power --idle-baseline-s 10] --output out.json
# host-side sweep over all 4 archs: scripts/fpga/benchmark/benchmark_sweep.py
```

---

## 5. Module architecture (`inference_cpp/src/benchmark/`)

The **stage decomposition is the single source of truth**; executors (seq/chan/pipe) only differ in
*how* they schedule the same stages. Leaf blocks reused unchanged from inference.

| File | Role |
| --- | --- |
| `stage_timer.hpp` | per-stage labelled timer (mark/commit/summary) |
| `bench_pipeline.{cpp,hpp}` | `BenchPipeline` + `PatchState`; each pipeline step a callable stage. Shared deinterleave/interleave/pack helpers (`channel_split`/`interleave_and_abs`/`deinterleave_yhat`/`pack_recon`) |
| `bench_configs.{cpp,hpp}` | seq / chan / pipe executors + ceiling fan-outs |
| `power_sampler.{cpp,hpp}` | INA226 (sysfs) + PMBus (I²C) sampler, background thread, `--power`-gated |
| `main_benchmark.cpp` | CLI → config → dispatch → JSON; soft `[warn]` on inapplicable flags |

Reused: `dpu_runners`, `entropy_models` + `rans/`, `npy_io`, `metrics`, `patch_transforms.hpp`
(extracted, board-verified behavior-preserving). `thread_pool.hpp` planned for M4/M5.

---

## 6. Measurement methodology

- **Latency configs (seq/chan):** warmup → N iters → per-stage mean/std via `StageTimer`.
- **Throughput configs (pipe):** warmup fills the pipeline → steady-state over N patches → drain;
  headline = patches/wall_time, per-stage kept distinct in JSON.
- **Power:** sampler window + separate idle-baseline window; pace with `sleep`, never busy-wait
  (a busy-wait pins an A53 and corrupts timing — documented lesson).
- **JSON schema:** `config`, `dpu_cores`, `entropy_threads`, `throughput_fps` headline, per-stage
  breakdown, optional `power.{idle,active}.{rails,groups}`.

**Power sampler details (verified M2):** 18 TI INA226 monitors via sysfs hwmon (4× averaging,
1100 µs conversion → ~8.8 ms hw period; effective poll ~5–100 Hz depending on bus); 3 Maxim PMBus
rails (DDR4 VDDQ, UTIL_3V3/5V0) via `/dev/i2c-4` `I2C_RDWR`. Idle ≈ VCCINT 6.0 W; active VCCINT 7.67 W
(+1.66 W DPU). Groups: **PL** (DPU fabric: VCCINT+VCCBRAM+VCCAUX+VCC1V2+VCC3V3), **PS** (ARM+DDR I/O),
**MGT** (transceivers, unused), **MPSoC = PL+PS**, **DPU_fabric** (VCCINT+VCCBRAM), **PS_compute**
(A53 APU). Six secondary bias rails (<500 mW, workload-invariant) are unmonitored → no effect on
dynamic-power deltas.

**Paper-ready FPGA power text:** *FPGA power is measured via 18 TI INA226 current/power monitors on
the ZCU102 (Linux sysfs hwmon) plus 3 Maxim PMBus rails over I²C; six secondary bias rails (<500 mW,
DPU-workload-invariant) are unmonitored. Power is aggregated into PL (DPU fabric), PS (ARM + DDR I/O),
and MGT (unused), with MPSoC = PL + PS as SoC compute power. A 10 s idle baseline (VART runners +
entropy model loaded, no inference) is subtracted to isolate dynamic inference power.* (GPU/CPU
paragraph + the short ≤2-sentence version are in `GPU_benchmark.md`.)

---

## 7. Results & status

Milestones (all board-verified):

- **M1 — S0 + stage_timer** (2026-05-23): per-stage compress/full breakdown; `patch_transforms.hpp`
  extraction verified behavior-preserving.
- **M2 — power_sampler** (2026-05-23): INA226 + PMBus, group aggregates, idle baseline.
- **M3 — S1 + ceilings** (2026-05-24): see below.

**M3 (ResSHyp L1000):** g_a S0→S1 **1.95×**, g_s **1.96×** (after the creation-order fix); total
compress 97.2→60.6 ms (1.59×), full 187.8→116.5 ms (1.61×); S1 output **byte-identical** to S0 on all
20 patches; nn_only N=2 **1.96×**, entropy_only N=2 **1.97×**. FP S1: g_a 1.71×, g_s 1.79× (smaller —
shorter DPU work, fixed thread overhead proportionally larger).

**Bottleneck in C++:** DPU-dominated for residual models (ResSHyp ~78% DPU / 12% entropy / 10%
normalize+denorm); FP ~37% DPU (normalize/denorm + entropy dominate → NEON is the lever, not
scheduling). The full Python-vs-C++ per-stage before/after lives in `python_to_cpp_migration_journal.md` §3.

**xdputil synthetic vs real-world:** real per-call adds ~10–19% on g_a/g_s (quant/dequant + dispatch)
and far more on h_a/h_s (fixed overhead dominates their sub-ms compute).

---

## 8. Derived metrics for publication

- **Latency** = `latency_total_mean_ms`; **throughput** = `throughput_fps`; **DPU fraction** =
  `latency_dpu_total_mean_ms / latency_total_mean_ms`.
- **Model size:** ~15M FP32 params (~60 MB) → 14.4 MB INT8 (~4.2×).
- **Compute efficiency:** GOPS/W = throughput_TOPS / power_W (DPU_fabric or board).
- **Energy/patch:** `P × t` (DPU_fabric × DPU time, or board × total).
- **Bitrate:** `avg_compressed_bytes × 8 / (256×256)`.

---

## 9. Caveats

- **Timing:** wall-clock includes ~µs Python/dispatch overhead (use `vaitrace` for DPU-level truth);
  sub-ms subgraphs (h_a/h_s) are overhead-dominated; thread-pool lifecycle must live *outside* the
  timed loop (an earlier per-call pool teardown caused spurious ±10 ms stage shifts).
- **Power:** INA226 undersamples sub-ms transients (fine for steady state); idle baseline captures
  static + OS overhead, so `P_dynamic = P_load − P_idle` isolates inference.
- **Cross-platform fairness (GPU vs FPGA):** INT8 vs FP32 (quality must be compared too); batch=1 vs
  GPU batching (normalise per-image); energy-per-inference is the fairest metric. Detail in `GPU_benchmark.md`.

---

## 10. Future work / TODOs

**Pipelining (not implemented; scope closed at M3):**

- **M4 — P0 coarse pipe:** patch-level DPU‖entropy overlap. Modest ceiling — DPU dominates (~1.16×
  compress / ~1.28× full for ResSHyp; FP bounded by its ~15% CPU fraction). SHyp `h_s` needs
  `z_hat` from `eb_decompress` → no clean single-patch DPU‖CPU split; overlap unit is patch-level.
  Each pipeline thread needs its **own `BenchPipeline`** (runners/entropy hold mutable state).
- **M5 — P2 multi-entropy + sweeps:** K entropy consumers; run `--dpu-cores`/`--entropy-threads` sweeps.
- **P3 (deferred):** fine-grained per-subgraph pipeline + `DPUCoreAllocator` (track subgraph→core by
  creation order, validate concurrent pairs on distinct cores). Gated on P0/P2 data.

**Other:**

- **[E] Unified GPU/CPU/FPGA runner** *(design needed)* — `benchmark_hardware` is FPGA-only;
  `run_full_benchmark.py` (the old GPU+FPGA orchestrator) is removed. Need a common runner + unified
  JSON schema driving `benchmark_gpu.py` and `benchmark_hardware`. **Schema reconciliation is the crux:**
  - *Scenario/config mismatch.* Legacy (`benchmark_gpu.py`, kept) uses flat scenarios
    `full/compress/decompress/nn_only/entropy_only`; the C++ benchmark uses `config` (s0/s1/nn_only/
    entropy_only) × `scenario` (compress/full). `decompress` is standalone in legacy but folded into
    `full` in C++. **`nn_only`/`entropy_only` mean different things**: legacy = *component isolation*
    (NN-only / entropy-only latency); C++ = *data-parallel ceilings* (`--dpu-cores`/`--entropy-threads`
    sweeps). They must not be plotted as the same axis.
  - *No s0/s1 on GPU/CPU.* The S0/S1 (DPU-core) distinction has no GPU/CPU analogue — the unified
    schema needs a `platform` field and a per-platform notion of "parallelism config".
  - *Per-stage keys differ.* Legacy `latency_breakdown{step:{mean_s,…}}` vs C++ `stages{stage:{mean_ms,…}}`;
    units differ (s vs ms). The Python↔C++ stage-name map is in the migration journal §5 / the (removed)
    `compare_benchmark_s0.py` logic.
  - *Workload mismatch.* GPU/CPU run FP32 batch-amortised; C++ FPGA cycles a 20-patch subset, INT8 — normalise per-image and compare energy/inference (fairness notes in `GPU_benchmark.md`).
  - **Two design options:** (a) keep `benchmark_gpu.py`'s legacy schema and write an *adapter* that
    re-maps C++ `benchmark_hardware` output into it, or (b) re-emit both platforms in a new common
    schema (`platform`, `config`, `scenario`, `throughput_fps`, canonical stage names). (b) is cleaner
    long-term; (a) reuses the legacy plotting notebooks as-is.
  - **Assets to reuse:** the frozen `results/benchmark/<model>/` JSONs (legacy cross-platform data, a
    reference set) and the legacy plotting notebooks `benchmark_analysis.ipynb` (cross-platform) +
    `hardware_model_comparison.ipynb` (cross-model) — both now banner-marked LEGACY. Quality (RD) GPU-vs-FPGA
    already works independently via `compare_gpu_fpga.ipynb` (W&B + `metrics.json`); the unified runner
    is **latency/throughput/power only**, not quality. See `GPU_benchmark.md` for the legacy tooling dump.
- **NEON-vectorize normalize/denorm** (~18.7 ms/patch scalar `log`/`exp`) — see `FPGA_inference.md` §9.
- **vaitrace** for DPU-level hardware timing (validate wall-clock); **tile-level** benchmarking on
  512²/1024²; **thermal** characterization (DPU junction temp under sustained load); **external USB
  power meter** for true board power vs rail-sum.
- **Analysis-notebook upgrades** (`benchmark_hardware_analysis.ipynb` §8 cell): seed-averaged
  cost-vs-quality, whole-model roofline, RD-curve-by-compute, DPU-only energy, APM DDR-traffic
  verification; + 5 methodology items to verify.

**Done cleanups:** [A] S1 single-core build warning; **[B]** deinterleave/interleave dedup (helpers
extracted); **[D]** argument safeguards (`warn_ignored`). **[C]** subcommand CLI — deferred to pre-M5.

---

## 11. Decisions log / journal (newest first)

- **2026-05-24 — M3 verified.** `init_s1` creation-order fix made model-aware: SHyp has 4 primary
  runners (g_a→0, h_a→1, g_s→2, h_s→0) so the aux pair must be created g_s_1→1 then g_a_1→2 to land
  on distinct cores; FP (2 primaries) keeps g_a_1 first. Wrong order silently collides two runners on
  one core → zero speedup (the recurring footgun). Byte-identity S0=S1 confirmed.
- **2026-05-23 — stage split.** `eb`/`gc` split into `_compress`/`_decompress` stages (compress
  scenario was timing an unused gc_decompress, inflating ~9 ms); foundation for P0/P2 thread split.
- **2026-05-23 — M1/M2 verified** (see §7). Known M1 artifact: `eb/gc compress()` heap-allocates
  inside the timed stage (variable-length output) — minor, unfixed.
- **2026-05-22 — phase planned.** VART has no core-pinning (UG1414); `--dpu-cores` reframed as
  influence + empirical verification. Scope S0/S1/ceilings/P0/P2; P3 deferred. Doc strategy: living
  journal → graduate into the benchmark reference (this doc realizes that graduation).

---

## 12. References

Vitis-AI 3.0 User Guide (VART/xdputil); DPUCZDX8G PG338 (resource utilisation); TI INA226 datasheet;
Maxim PMBus telemetry. Model/MERLIN theory: `docs/Method.md`. Migration before/after:
`python_to_cpp_migration_journal.md`. Inference: `FPGA_inference.md`.
