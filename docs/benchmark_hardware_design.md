# benchmark_hardware — Design & Development Journal

**Status**: 🚧 in development (Phase 2). This is a *living* doc — decisions log + journal +
progress tracker. When the binary is complete and validated, the stable parts (architecture,
run commands, methodology) graduate into `performance_benchmark_implementation.md`; this file is
then retained as a historical development record, not deleted.

---

## 1. Purpose

`inference_hybrid` (C++ port) is complete and validated. The reason for the port was control
over parallelism: in Python the GIL blocked overlapping the dominant CPU entropy stage with DPU
work. This phase builds a `benchmark_hardware` C++ binary to answer, for an upcoming publication:

> **What is the best schedule for this heterogeneous DPU+CPU SAR compression pipeline on the
> ZCU102 (3× DPU B4096 cores + 4× ARM A53)?**

- **Headline metric**: throughput (patches/s).
- **Deployment-realistic workload**: compress-only (satellite side).
- **Working hypothesis to test first**: the Python profile is entropy-dominated (GC ≈ 58%), but
  ~half of GC's Python cost is numpy index-building that collapses in native C++ — so the C++
  bottleneck may shift back toward the DPU. *Deliverable #0 is the C++ per-stage breakdown that
  settles this.* Every scheduling decision depends on it.

---

## 2. Locked decisions

| Topic | Decision |
| --- | --- |
| Objective | Throughput (latency loses meaning once patches pipeline) |
| Pipeline scope | Compress-only for pipelined configs |
| Scenario coverage | Data-parallel ceilings (`nn_only`, `entropy_only`) + pipelined `compress`; `full`/`decompress` kept **sequential** as Python-comparison references |
| Initial scope | S0, S1, ceilings, P0, P2. **P3 deferred** (fine-grained per-subgraph pipeline + DPU core allocator) |
| Models | 4 architectures (SHyp, ResSHyp, FP, ResFP), one representative each. Seed is timing-irrelevant; lambda sensitivity is a *separate later experiment* — the script does not special-case it. **Naming note**: `Res` prefix = `no_residual_blocks=False` (3 extra ResidualBlocks per stage in g_a and g_s, each block = conv+act+conv with skip connection). SHyp/FP = `no_residual_blocks=True` (lighter, faster DPU stages). The benchmarked ResSHyp/FP-relu numbers are **not** interchangeable with SHyp/FP; both must be benchmarked separately before publishing. |
| Sweeps | `--dpu-cores {1,2,3}`, `--entropy-threads {1,2,3,4}` as free knobs for scaling curves |
| Power | Native C++ sampler, self-contained module, runtime-gated by `--power` |
| Build style | Slow, controlled, milestone-by-milestone; modular, single-source-of-truth, no redundant functions/classes |

---

## 3. Configs

| Config | Pattern | Resources | Purpose |
| --- | --- | --- | --- |
| **S0** seq | none | 1 DPU core, 1 thread | C++ per-stage baseline (deliverable #0) |
| **S1** chan | fan-out | 2 g_a runners + 2 g_s runners | `g_a(real)‖g_a(imag)`, `g_s(real)‖g_s(imag)` — channel-parallel, fixed parallelism |
| **nn_only** | fan-out | `--dpu-cores` replicas | N independent patch-processing runners; data-parallel DPU ceiling |
| **entropy_only** | fan-out | `--entropy-threads` workers | N independent patch-entropy workers; CPU ceiling |
| **P0** pipe (coarse) | patch-level pipeline, 1 queue | 1 DPU lane + 1 entropy worker | patch N's entropy (CPU) overlaps patch N+1's DPU stages |
| **P2** pipe (multi-entropy) | patch-level pipeline, 1 queue, K consumers | 1 DPU lane + `--entropy-threads` workers | attack entropy ceiling with spare A53 cores |

**`--dpu-cores` means different things in different configs — do not conflate:**

- **S1**: Parallelism is **structural**, not a user knob. S1 always creates exactly **2 runners for g_a** (one for real, one for imag) and 2 for g_s. `--dpu-cores` has no effect in S1. The two runners run concurrently from two `std::thread`s, each calling `runner.run()`.
- **nn_only**: `--dpu-cores N` = N independent runners, each processing a **different patch** concurrently (data-parallel). Speedup is measured against single-runner throughput. This is the DPU roofline.
- **P0 / P2**: `--dpu-cores` = number of concurrent pipeline DPU lanes (each lane handles the DPU stages of one patch at a time, while entropy workers handle another patch's CPU stages). In practice, with a single DPU lane, P0 overlaps DPU of patch N+1 with CPU entropy of patch N.

**VART core assignment (no core-pinning):** `create_runner(subgraph, mode)` takes no core/device
index; assignment is automatic **round-robin at runner creation time**. The only lever is
creation order and replica count. `--dpu-cores N` for nn_only therefore *influences* placement
(N replicas → tend toward N distinct cores, up to 3); it does not *guarantee* it. **We verify
overlap empirically**: concurrent wall-time materially below serial sum (≈N× for N runners on N
cores) is the evidence. Missing speedup ⇒ collision on one core.

**S1 creation-order constraint (critical for M3):** Both g_a runners must be created
**consecutively and first**, before h_a/h_s runners, so VART round-robin places them on cores 0
and 1 (not both on core 0). Same applies to g_s if S1 is extended to the full scenario. Creating
runners in the wrong order is the known `inference_hybrid` footgun and will produce a result that
looks like S1 running but delivers no speedup.

**`--entropy-threads`:** N genuine OS threads on the 4 A53 cores — no VART constraint; placement
is handled by the OS scheduler. Each thread needs its own `BenchPipeline` instance (see §4).

**P0/P2 thread safety — one `BenchPipeline` per thread (critical for M4/M5):** both `DPUSubgraphRunner`
and `EntropyBottleneck`/`GaussianConditional` hold mutable state (DMA buffers, internal scratch).
Calling the same runner from two threads concurrently is a data race. The fix: each pipeline thread
(DPU lane or entropy worker) owns a **separate `BenchPipeline` instance**, constructed independently.
One XModelLoader per BenchPipeline means separate runner objects per thread. This was noted in M1
as the known obstacle; M4/M5 implementations must respect it.

**P0 SHyp intra-patch dependency (critical for M4):** the SHyp compress pipeline is
`g_a → h_a → eb_compress → eb_decompress → h_s → gc_compress`. The DPU cannot run `h_s` until
the CPU finishes `eb_decompress` (h_s needs `z_hat`). A coarse "DPU thread ‖ CPU thread" split
does not apply cleanly to a single SHyp patch — the two threads must synchronise mid-patch.
The clean overlap unit is patch-level: the DPU processes patch N+1's `g_a`/`h_a` while the CPU
handles patch N's `eb_compress→eb_decompress→h_s→gc_compress`. For FP the dependency is clean
(no h_s/h_a), but the DPU dominates timing (g_a = 73 ms vs CPU entropy = 12 ms), so overlap
benefit is bounded by the CPU fraction (~15%).

---

## 4. Module architecture

Design principle: the **stage decomposition is the single source of truth** for the compress
pipeline's steps; the executors (seq/chan/pipe) only differ in *how* they schedule those stages.
Leaf building blocks from the validated inference code are reused unchanged.

**New files under `inference_cpp/src/benchmark/`:**

| File | Role |
| --- | --- |
| `stage_timer.hpp` | Per-stage labelled timer (mark/commit/summary), mirrors Python `StepTimer`. Header-only, zero coupling. |
| `bench_pipeline.{hpp,cpp}` | `BenchPipeline` — holds loaded models (XModelLoader, EB, GC); exposes each pipeline step as a callable stage over a `PatchState` struct. The schedulable units. |
| `bench_configs.{hpp,cpp}` | seq / chan / pipe(P0,P2) executors + ceiling fan-outs; drive `BenchPipeline` stages + `StageTimer`. |
| `power_sampler.{hpp,cpp}` | INA226 (sysfs) + PMBus (I2C_RDWR) sampler; background `std::thread`; `--power`-gated. |
| `thread_pool.hpp` | Minimal fixed-size pool (mutex + condition_variable task queue) for fan-out and pipeline workers. **Planned for M3/M4 — not yet created.** |
| `main_benchmark.cpp` | CLI parse → config → dispatch → JSON out. |

**Reused unchanged (leaf building blocks):**
- `dpu_runners.{hpp,cpp}` — `DPUSubgraphRunner::run(const float*, float*)`, `XModelLoader`.
- `entropy_models.{hpp,cpp}` (`EntropyBottleneck`, `GaussianConditional`) + `rans/`.
- `npy_io.hpp`, `logger.hpp`, `scoped_timer.hpp`, `constants.hpp`, `metrics.{hpp,cpp}`.

**Shared transforms — `patch_transforms.hpp` (extracted, board-verified M1):** the per-patch
pure functions `normalize_patch`, channel interleave/deinterleave, `denorm_to_lina`, `raw_to_lina`
were extracted from the anonymous namespace in `inference_runner.cpp` into a header-only
`inference_cpp/src/patch_transforms.hpp`, included by both `inference_runner.cpp` and
`bench_pipeline`. Board re-verified in M1: bpp/PSNR/SSIM bit-identical pre- and post-extraction
(behavior-preserving move confirmed).

**Build**: add a `benchmark_hardware` executable target in `inference_cpp/CMakeLists.txt` linking
`inference_lib` (+ `nlohmann/json`, OpenCV like the existing targets). Power module compiles
everywhere (plain file/ioctl ops) but is runtime-gated; host build (`HAVE_DPU=OFF`) still compiles
the non-DPU modules.

---

## 5. Measurement methodology

- **Latency configs (seq/chan)**: warmup → N iters → per-stage mean/std via `StageTimer`.
- **Throughput configs (pipe)**: warmup *fills* the pipeline → measure steady-state over N
  patches → drain. Headline = `patches / wall_time`; per-stage latency still recorded but kept
  distinct from throughput in the JSON.
- **Power**: sampler around the measured window + a separate idle-baseline window (as Python).
  Reuse rail maps + group formulas (PL/PS/MGT/MPSoC/DPU_fabric/PS_compute). Pace with `sleep`,
  never busy-wait (documented A53-pinning lesson).
- **JSON schema**: extend `benchmark_fpga.py`'s schema (add `config`, `dpu_cores`,
  `entropy_threads`; `throughput_fps` headline) so files drop into `benchmark_analysis.ipynb`.

---

## 6. Milestone checklist

- [x] **M1 — S0 + `stage_timer`**: stage decomposition (`BenchPipeline`), shared
  `patch_transforms.hpp` extraction (+ **board re-verify** inference unchanged — required before
  trusting results), sequential executor, CMake target, minimal CLI → C++ per-stage compress/full
  breakdown. *Settles the real bottleneck.* Board-verified 2026-05-23.
- [x] **M2 — `power_sampler`**: INA226 sysfs (18 rails, 10 ms) + PMBus I2C_RDWR (3 rails, 40 ms)
  in background threads. Board-verified 2026-05-23: idle VCCINT = 6.010 W (✓ known ~6 W),
  UTIL_3V3 = 2.11 W (✓), DDR4_DIMM_VDDQ = 0.48 W (✓). Active VCCINT = 7.67 W (+1.66 W DPU).
  JSON schema: `power.{idle,active}.{rails,groups}` with all 7 group aggregates. `--idle-baseline-s N` flag.
- [x] **M3 — S1 + ceilings**: board-verified 2026-05-24 (ResSHyp + FP-relu). g_a speedup 1.95×;
  g_s speedup 1.96× (after creation-order fix); byte-identical output vs S0 (all 20 patches);
  inference_hybrid PSNR unchanged; FP S1 branch verified (g_a 1.71×, g_s 1.79×, bytes pass).
  See M3 verification journal below. Remaining pre-M4 task: item [B] refactor.
- [ ] **M4 — P0**: coarse 2-stage DPU‖entropy pipe.
- [ ] **M5 — P2 + sweeps**: K entropy consumers; run `--dpu-cores`/`--entropy-threads` sweeps.
- [ ] **(later) P3**: fine-grained per-subgraph pipeline + DPU core allocator — gated on P0/P2 data.

---

## 7. Verification

- **Concurrency correctness**: pipelined `compress` output bytes == sequential for the same
  patches (threading must not change results). Reuse the `tests/` rANS roundtrip pattern.
- **DPU overlap (timing is the proof)**: no reliable API to query a runner's core, so concurrent
  wall-time materially below serial sum (≈1.9× for a pair) is the evidence. Missing speedup ⇒
  round-robin collided two replicas on one core (the `inference_hybrid` creation-order footgun).
- **Python S0/S1 cross-check**: diff C++ `seq`/`chan` vs `benchmark_fpga.py` JSON in a
  `benchmark_analysis.ipynb` cell. Expect DPU stages to match; quantify the C++ entropy speedup.
- **Host build** (`HAVE_DPU=OFF`): non-DPU modules compile; `--power` no-ops off-board.

---

## 8. Decisions log / journal

*(Append-only. Newest at top. Record surprises, dead-ends, and why choices were made.)*

- **2026-05-24** — M3 board-verified. ResSHyp L1000, compress + full scenarios, 20 iters, 20 patches.

  **init_s1 creation-order fix:** the original code always created `g_a_1` then `g_s_1`. For SHyp
  (4 primary runners → next VART slot = core 1), this placed `g_s_1` on core 2 (same as `g_s`) —
  eliminating the g_s speedup in the full scenario. Fix: make `init_s1` model-aware: SHyp creates
  `g_s_1` first (→ core 1 ≠ g_s core 2 ✓) then `g_a_1` (→ core 2 ≠ g_a core 0 ✓); FP keeps the
  original order (2 primaries → next slot = core 2; `g_a_1` first ✓, `g_s_1` second ✓).
  Runner creation order observed in log after fix (SHyp): primary g_a→0, h_a→1, g_s→2, h_s→0;
  init_s1: g_s_1→1, g_a_1→2. Both concurrent pairs now on distinct cores.

  **M3 results (after fix):**

  | Config | Scenario | g_a (ms) | g_s (ms) | Total (ms) | Throughput | vs S0 |
  | --- | --- | --- | --- | --- | --- | --- |
  | S0 | compress | 73.13 | — | 97.2 | 10.3 fps | baseline |
  | S1 | compress | 37.53 | — | 60.6 | 16.4 fps | 1.59× total; **g_a 1.95×** |
  | S0 | full | 73.18 | 71.76 | 187.8 | 5.3 fps | baseline |
  | S1 | full | 37.50 | 36.53 | 116.5 | 8.6 fps | 1.61× total; **g_a 1.95×; g_s 1.96×** |

  **Correctness checks (all passed):**
  - S0 regression: 97.2 ms compress, 187.8 ms full — matches M1 baseline ✓
  - S1 byte identity: `bytes_per_iter` from S0 and S1 compress are element-wise identical across
    all 20 patches (8848, 9944, … 9748 — all 20 values match exactly) ✓
  - inference_hybrid regression: PSNR 32.09 dB (10 patches, ResSHyp) — unchanged after
    `subgraphs_` addition to `XModelLoader` ✓
  - nn_only N=2: 20.4 fps (1.96×), entropy_only N=2: 163 fps (1.97×) — unchanged from M3 impl ✓
  - FP model (FP-relu L1000, 2-runner model): S1 FP branch verified. g_a 10.4→6.1 ms (1.71×);
    g_s 8.65→4.84 ms (1.79×); byte identity PASS. Lower speedup than ResSHyp (1.71-1.79× vs
    1.95-1.96×) because FP has no residual blocks — shorter DPU inference means fixed thread
    overhead is proportionally larger. See §9 architecture naming note.

  **FP-relu L1000 S0/S1 timing (for reference):**

  | Stage | S0 compress (ms) | S0 full (ms) | S1 compress (ms) | S1 full (ms) |
  | --- | --- | --- | --- | --- |
  | normalize | 8.96 | 9.00 | 8.96 | 8.93 |
  | g_a | 10.39 | 10.39 | 6.05 (1.71×) | 6.00 (1.73×) |
  | eb_compress | 6.66 | 6.66 | 6.68 | 6.65 |
  | eb_decompress | — | 6.96 | — | 6.97 |
  | g_s | — | 8.65 | — | 4.84 (1.79×) |
  | denorm | — | 9.75 | — | 9.74 |
  | **Total** | **26.0** | **51.4** | **21.7 (1.20×)** | **43.1 (1.19×)** |

  **Key FP insight:** DPU is only 37% of the full pipeline (vs 78% for ResSHyp). S1 gives only
  1.19× end-to-end because normalize+denorm (36%) and entropy (27%) don't parallelize. NEON
  vectorization of normalize/denorm would have much larger relative impact for FP than scheduling
  changes.

- **2026-05-23** — Stage split fix: `stage_eb` and `stage_gc` each split into separate `_compress`
  and `_decompress` stages. Motivation: (1) `stage_gc` was running both GC compress+decompress in
  the `compress` scenario — y_hat was computed but unused, inflating gc timing by ~9 ms (full-path
  gc_decompress); (2) FP `stage_eb` had the same problem in compress scenario; (3) split enables
  individual timing of compress/decompress and is the foundation for future P0/P2 where compress
  and decompress may run on different threads.

  New stages: `stage_eb_compress`, `stage_eb_decompress`, `stage_gc_compress`, `stage_gc_decompress`.
  SHyp `eb_compress`/`eb_decompress` always paired (z_hat needed by h_s) but split for timing.
  Bitstreams stored in `PatchState` (`z_bits`, `y_bits`).

  Stage sequences after fix:
  - SHyp compress: `normalize→g_a→h_a→eb_compress→eb_decompress→h_s→gc_compress→_end`
  - SHyp full: `…→gc_compress→gc_decompress→g_s→denorm→_end`
  - FP compress: `normalize→g_a→eb_compress→_end`
  - FP full: `…→eb_compress→eb_decompress→g_s→denorm→_end`

  Board-verified: smoke run (compress 10 iters + full 5 iters) shows correct stage names and
  values — gc_compress ≈ 11.7 ms, gc_decompress ≈ 9.8 ms (sum ≈ 21.5 ms, matches old `gc` entry).

- **2026-05-23** — Python/C++ S0 comparison complete (ResSHyp, both scenarios, 50 iters,
  --no-parallel, --power, --idle-baseline 10). JSONs in
  `/tmp/py-cpp_bench_results_comparison/` (not committed to `results/benchmark/` — that dir is
  notebook-managed; rerun all benchmarks when results are final).
  Comparison script: `scripts/fpga/compare_benchmark_s0.py` (now scenario-aware; power section added).

  **Compress scenario** (200 ms Python → 96.5 ms C++):

  | Stage | Python (ms) | C++ (ms) | Speedup | Notes |
  | --- | --- | --- | --- | --- |
  | normalize | 0.00* | 8.97 | — | *Python cached outside loop |
  | g_a | 75.71 | 73.18 | 1.03× | DPU match; Python wrapper overhead |
  | h_a | 2.15 | 1.40 | 1.54× | Same DPU; Python wrapper overhead |
  | eb_compress | 4.90 | 0.12 | **42×** | numpy index-building eliminated |
  | eb_decompress | 5.31 | 0.11 | **47×** | numpy index-building eliminated |
  | h_s | 2.03 | 0.84 | 2.43× | Same DPU |
  | gc_compress | 112.22 | 11.89 | **9.4×** | compress-only correctly separated |
  | **TOTAL** | 202.33 | 96.50 | **2.10×** | |

  **Full scenario** (409 ms Python → 188 ms C++):

  | Stage | Python (ms) | C++ (ms) | Speedup | Notes |
  | --- | --- | --- | --- | --- |
  | gc_decompress | 133.46 | 9.89 | **13.5×** | biggest single speedup |
  | gc_compress | 111.79 | 11.86 | 9.4× | |
  | g_a | 75.56 | 73.18 | 1.03× | DPU match |
  | g_s | 74.22 | 71.76 | 1.03× | DPU match |
  | denorm | 0.005* | 9.74 | — | *Python caches log-amp; C++ recomputes |
  | **TOTAL** | 409.28 | 187.80 | **2.18×** | |

  **Power findings** (both scenarios):
  - Active MPSoC power: Python ≈ 11.4–11.8 W, C++ ≈ 11.5–11.7 W — **nearly identical**.
  - Energy savings come purely from lower latency, not lower power draw.
  - Compress: 2301 mJ/patch (Python) → 1106 mJ/patch (C++) — **2.08× less energy**.
  - Full: 4833 mJ/patch → 2193 mJ/patch — **2.20× less energy**.
  - VCCINT idle ≈ 6.03 W (consistent across runs); active ≈ 8.8–9.1 W (+2.8–3.1 W DPU).
  - DPU_fabric Δ (active − idle) ≈ 2.8–3.2 W for both (DPU hardware draw same regardless of language).

  **Key insights**:
  - C++ scalar `std::log()` normalize (131K calls × 68 ns = 9 ms) is a new bottleneck vs Python.
  - `denorm_to_lina` same issue: 9.7 ms in C++ vs ~0.005 ms in Python (log-amp cached).
  - These two pure-CPU loops (18.7 ms total, 10% of full pipeline) are candidates for NEON vectorization.
  - CPU entropy total: Python 256 ms → C++ 41 ms (**6.3× speedup**); DPU total essentially unchanged (1.04×).
  - `gc_decompress` is the biggest single speedup (13.5×) — Python GC decompress has expensive index lookup.

- **2026-05-23** — M2 board verified. `power_sampler.{hpp,cpp}` implemented and tested.
  INA226: 18 rails discovered via `/sys/class/hwmon` scan for `ina226_uXX` names; sysfs poll at
  10 ms (≈100 Hz, confirmed 497 samples in 5 s). PMBus: `/dev/i2c-4`, `I2C_RDWR` ioctl with
  2-msg write+read; VOUT_MODE probed once at init (returns 0x14 on all 3 rails → exp = -12);
  poll at 40 ms (≈25 Hz, confirmed 125 samples in 5 s). Values match Python reference:
  VCCINT idle = 6.010 W, UTIL_3V3 = 2.11 W, UTIL_5V0 ≈ 0. Active VCCINT = 7.67 W.
  `LocalI2cRdwr` struct defined locally to avoid `<linux/i2c-dev.h>` / `<linux/i2c.h>` header
  conflict. Groups include MPSoC computed as PL+PS (not listed in kPowerGroups table to avoid
  circular refs). `--idle-baseline-s N` added to CLI (default 10); JSON adds `power.idle` and
  `power.active` windows only when `--power` is passed and sensors are found.

- **2026-05-23** — M1 board verified. S0 results (ResSHyp L1000, compress+full scenarios, 50 iters, 20 patches cycled).
  Board re-verify: `patch_transforms.hpp` extraction confirmed behavior-preserving (bpp/PSNR/SSIM
  bit-identical pre- and post-push). One compile error caught: most-vexing-parse on
  `BenchPipeline pipeline(fs::path(...), fs::path(...))` — fixed with brace init. CMakeLists.txt
  must also be pushed (not just `src/`) when adding new targets; cmake re-configure needed.

  **S0 deliverable #0 results** (ResSHyp L1000, SHyp path, 50 iters, 20 patches cycled):

  | Stage | Mean (ms) | % of full total |
  | --- | --- | --- |
  | normalize | 8.92 | 4.8% |
  | g_a | 73.18 | **39.0%** |
  | h_a | 1.38 | 0.7% |
  | eb_compress + eb_decompress | 0.23 | 0.1% |
  | h_s | 0.84 | 0.4% |
  | gc_compress | 11.86 | 6.3% |
  | gc_decompress | 9.89 | 5.3% |
  | g_s | 71.76 | **38.2%** |
  | denorm | 9.74 | 5.2% |
  | **total (full)** | **187.80** | |
  | compress only | 96.50 | |

  **Key findings vs working hypothesis:**
  - **Working hypothesis CONFIRMED**: C++ bottleneck shifted back to DPU.
    DPU total (g_a + h_a + h_s + g_s): 147.2 ms (**78.4%**); CPU entropy: 22.1 ms (11.8%); normalize+denorm: 18.7 ms (10.0%).
  - GC numpy index-building eliminated: gc_compress 112 ms → 11.9 ms; gc_decompress 133 ms → 9.9 ms.
  - EB negligible (0.1%) — z is only 2×2×256 = 1024 floats.
  - normalize + denorm (18.7 ms, 10%) are new bottlenecks invisible in Python (numpy vectorizes internally).
    ARM A53 float `log()` ≈ 68 ns/call; 256×256×2 = 131K calls → 8.9 ms expected. Candidates for NEON.
  - Total C++ full: 188 ms vs Python 409 ms — **2.18× speedup**; C++ compress: 97 ms vs Python 202 ms — **2.10×**.
  - **Implication for P0**: DPU dominates (78.4%). Max throughput gain from entropy overlap:
    pipeline bottleneck = DPU (147 ms); theoretical ceiling ≈ 1/(147 ms) = 6.8 fps vs sequential 5.3 fps → **~1.28× for full**.
    For compress: DPU = 75.4 ms (g_a+h_a+h_s), CPU entropy = 12.1 ms → ceiling ≈ 1.16×.
    See §3 for the SHyp intra-patch dependency constraint that shapes the P0 pipeline split.

- **2026-05-23** — M1 implementation complete (pending board re-verify). Files written:
  `src/patch_transforms.hpp` (extracted `normalize_patch`, `denorm_to_lina`, `raw_to_lina`
  from anonymous namespace in `inference_runner.cpp`); `src/benchmark/bench_pipeline.{hpp,cpp}`
  (`BenchPipeline` + `PatchState`, 7 stages as callable methods); `src/benchmark/bench_configs.{hpp,cpp}`
  (`run_s0` sequential executor with `StageTimer`); `src/benchmark/main_benchmark.cpp` (CLI + JSON);
  `CMakeLists.txt` updated with `benchmark_hardware` target. Board re-verify step: push sources,
  build, run `inference_hybrid` on 100 patches, confirm PSNR unchanged (|Δ| < 0.1 dB gate).
  Known measurement artifact: `eb_.compress()` / `gc_.compress()` do a heap alloc inside the timed
  stage (variable-length entropy output — unavoidable without pre-sizing a max-size buffer). Noted
  but not fixed in M1. For P0/P2: `BenchPipeline` has one `XModelLoader` (one set of runners);
  concurrent stage calls across threads would race on the same runner — multiple `BenchPipeline`
  instances (one per thread) will be required.

- **2026-05-22** — Phase planned and approved. Confirmed via VART 3.0 C++ API doc (UG1414) that
  there is no core-pinning mechanism; `--dpu-cores` reframed as influence-plus-verification.
  Initial scope set to S0/S1/ceilings/P0/P2; P3 deferred. Doc strategy: living journal →
  graduate to perf doc → archive.

---

## 9. Pending Improvements & Open Questions

*(Items to tackle in order of relevance to the active milestone. Add new items here.)*

**[A] Single-core build warning for S1** *(done — comment + doc)*
`init_s1()` now carries a `WARNING: assumes 3 DPU cores (ZCU102 B4096×3)` comment with a
pointer to verify via `xdputil query` (which reads DPU hardware registers and reports the
physical core count, architecture, and clock frequency — see
`docs/performance_benchmark_implementation.md` §3). VART provides no programmatic API to
query core count, so the warning is documentation-level only. No further action needed unless
the binary is ported to different hardware.

**[B] Refactor deinterleave / interleave duplicates** *(planned for pre-M4 cleanup)*
`stage_ga` and `stage_ga_s1` share identical deinterleave + interleave + `|y|` loops;
`stage_gs` and `stage_gs_s1` share identical deinterleave + pack loops. Currently noted
with `// Same as stage_X` comments.

Planned extraction into `bench_pipeline.cpp` anonymous namespace:
- `channel_split(norm_hwc, real_ch, imag_ch, H, W)` — deinterleave input
- `interleave_and_abs(y_real, y_imag, y, y_abs, yh, yw, yc)` — interleave outputs + `|y|`
- `deinterleave_yhat(y_hat, yh_real, yh_imag, yh, yw, yc)` — split y_hat for g_s input
- `pack_recon(recon_real, recon_imag, recon_norm_logI, H, W)` — pack g_s outputs

Each stage function becomes ~10 lines; ~120 lines of duplication removed. Do in a single
focused commit before M4 (no new features, pure refactor).

**[C] CLI design: single main vs subcommands** *(medium priority — usability)*
Three options considered:
- **Separate binaries** (`benchmark_hardware_s0`, …): compile-time prevention of invalid
  arg combinations, but duplicates all shared CLI boilerplate (xmodel, params, data,
  power, iters, warmup) and causes version drift.
- **Subcommands** (`benchmark_hardware s0 [opts]`, `benchmark_hardware nn_only [opts]`):
  one binary, each subcommand has its own option set — invalid combinations prevented by
  construction. Modest refactor: `main_s0()`, `main_s1()`, etc. dispatched from a thin
  `main()` after parsing `argv[1]`; shared args parsed once. **Recommended** when the full
  config set is settled (before or at M5/publication).
- **Flat CLI + guards** (current): simplest; extend with validation (see [D]).
**Decision**: add guards [D] now; defer subcommand refactor to pre-M5 cleanup.

**[D] Argument safeguards for nonsensical combinations** *(medium priority — correctness)*
Several config/flag combinations silently ignore or misapply arguments:
- `--config s0 --dpu-cores N>1` → silently ignored
- `--config s1 --dpu-cores N>1` → silently ignored (S1 always uses exactly 2 runners/pair)
- `--config entropy_only --dpu-cores N>1` → ignored; use `--entropy-threads` instead
- `--config nn_only --entropy-threads N>1` → ignored; use `--dpu-cores` instead
Add a post-parse validation block in `main_benchmark.cpp` that emits `[warn]` for each
inapplicable non-default flag (soft warning, not a hard error, to allow scripted sweeps
that pass a fixed flag set).
