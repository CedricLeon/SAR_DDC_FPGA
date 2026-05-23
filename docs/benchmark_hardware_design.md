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
| Models | 4 architectures (SHyp, ResSHyp, FP, ResFP), one representative each. Seed is timing-irrelevant; lambda sensitivity is a *separate later experiment* — the script does not special-case it |
| Sweeps | `--dpu-cores {1,2,3}`, `--entropy-threads {1,2,3,4}` as free knobs for scaling curves |
| Power | Native C++ sampler, self-contained module, runtime-gated by `--power` |
| Build style | Slow, controlled, milestone-by-milestone; modular, single-source-of-truth, no redundant functions/classes |

---

## 3. Configs

| Config | Pattern | Resources | Purpose |
| --- | --- | --- | --- |
| **S0** seq | none | 1 DPU core, 1 thread | C++ per-stage baseline (deliverable #0) |
| **S1** chan | fan-out | 2+ DPU cores | `g_a(real)‖g_a(imag)`, `g_s(real)‖g_s(imag)` |
| **nn_only** | fan-out | `--dpu-cores` | DPU-bound throughput ceiling |
| **entropy_only** | fan-out | `--entropy-threads` | CPU entropy throughput ceiling |
| **P0** pipe (coarse) | pipeline, 1 queue | DPU thread + 1 entropy thread | overlap DPU stage with entropy stage |
| **P2** pipe (multi-entropy) | pipeline, 1 queue, K consumers | DPU + `--entropy-threads` | attack entropy ceiling with spare A53 cores |

**Knob reality (important)**: VART 3.0 C++ has **no core-pinning** — `create_runner(subgraph,
mode)` takes no core/device index; assignment is automatic round-robin at runner creation (same
constraint as Python). `--dpu-cores N` therefore *influences* placement via replica count +
creation order; it does not *control* it. We **verify overlap empirically** (concurrent
wall-time < serial sum) and report measured effective parallelism, never assume a core count.
`--entropy-threads` are genuine OS threads on the 4 A53 cores — no placement issue.

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
| `thread_pool.hpp` | Minimal fixed-size pool (mutex + condition_variable task queue) for fan-out and pipeline workers. |
| `main_benchmark.cpp` | CLI parse → config → dispatch → JSON out. |

**Reused unchanged (leaf building blocks):**
- `dpu_runners.{hpp,cpp}` — `DPUSubgraphRunner::run(const float*, float*)`, `XModelLoader`.
- `entropy_models.{hpp,cpp}` (`EntropyBottleneck`, `GaussianConditional`) + `rans/`.
- `npy_io.hpp`, `logger.hpp`, `scoped_timer.hpp`, `constants.hpp`, `metrics.{hpp,cpp}`.

**Shared transforms — `patch_transforms.hpp` (planned extraction):** the per-patch pure
functions `normalize_patch`, channel interleave/deinterleave, `denorm_to_lina`, `raw_to_lina`
currently live in an anonymous namespace in `inference_runner.cpp` (file-local, not reusable).
To avoid duplicating them in the benchmark, extract them into a header-only
`inference_cpp/src/patch_transforms.hpp` and have **both** `inference_runner.cpp` and
`bench_pipeline` include it. ⚠️ This touches validated code, so it is a behavior-preserving move
(identical math, no logic change) and **must be board-re-verified** (inference output unchanged)
before we build on it. Until then it's an open step — see milestone 1.

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
  breakdown. *Settles the real bottleneck.* ⚠️ Board push + re-verify pending.
- [ ] **M2 — `power_sampler`**: validate vs known idle (~6 W VCCINT, perf doc §4.6).
- [ ] **M3 — S1 + ceilings**: confirm ~1.9× g_a channel-parallel speedup; **gate**: prove N
  concurrent runners use N cores; spike `xir::Attrs` core-hint.
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

- **2026-05-23** — M1 board verified. S0 first results (ResSHyp model, compress+full scenarios, 50 iters).
  Board re-verify: `patch_transforms.hpp` extraction confirmed behavior-preserving (bpp/PSNR/SSIM
  bit-identical pre- and post-push). One compile error caught: most-vexing-parse on
  `BenchPipeline pipeline(fs::path(...), fs::path(...))` — fixed with brace init. CMakeLists.txt
  must also be pushed (not just `src/`) when adding new targets; cmake re-configure needed.

  **S0 deliverable #0 results** (ResSHyp L1000, SHyp path, 50 iters, 20 patches cycled):

  | Stage | Mean (ms) | % of total |
  | --- | --- | --- |
  | normalize | 8.92 | 14.4% |
  | g_a (×2 runs) | 10.40 | 16.8% |
  | h_a | 1.33 | 2.2% |
  | eb (z only) | 0.23 | 0.4% |
  | h_s | 0.83 | 1.3% |
  | gc (y) | 21.52 | **34.8%** |
  | g_s (×2 runs) | 8.84 | 14.3% |
  | denorm | 9.75 | 15.8% |
  | **total (full)** | **61.83** | |
  | compress only | 43.26 | |

  **Key findings vs working hypothesis:**
  - Total C++ pipeline: 62 ms vs Python ~425 ms — **7× speedup** from language change alone.
  - GC is still #1 at 34.8% (Python: 58%) — the numpy overhead collapsed as expected.
  - normalize + denorm together: 18.7 ms (30.2%) — pure A53 loop arithmetic, *not accelerated*.
    ARM A53 float `log()` is ~68 ns/call (no NEON). With 256×256×2 = 131K calls, 8.9 ms is expected.
    This is **a new bottleneck not visible in Python** (numpy vectorizes it internally).
  - DPU total (g_a + h_a + h_s + g_s): 21.4 ms (34.6%); CPU total: 40.4 ms (65.4%).
  - EB is negligible (0.4%) — z is only 2×2×256 = 1024 floats.
  - **Implication for P0 coarse pipe**: max(DPU, CPU_entropy) = max(21.4, 21.75) ≈ 21.75 ms —
    theoretical ≈1.85× speedup if we overlap DPU (of next patch) with entropy (of current).
    But: h_s depends on z_hat (EB output), so the DPU thread must stall waiting for EB mid-patch.
    The clean pipeline split point is *before h_a* (hand off z after h_a; entropy thread runs EB,
    sends z_hat back; DPU thread continues with h_s). This is more granular than "coarse 2-stage".
    normalize + denorm (30%) remain on CPU and don't benefit from DPU/entropy overlap.

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
