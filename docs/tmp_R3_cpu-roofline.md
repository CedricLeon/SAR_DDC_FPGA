# R3 — CPU-roofline feasibility for `normalize` (memo)

> Scratch memo for `docs/DATE27_paper_plan.md` §4.2 task **R3**. Exploratory, timeboxed. Everything
> below is either a direct board measurement (this session, 2026-09-01, idle ZCU102) or a citation —
> nothing here is an unflagged estimate.

## Verdict

**Confirms the plan's default: no CPU roofline in the paper.** `normalize` (NEON) reaches only
**≈3.3 % of the single-core NEON compute peak**, and the shortfall is not memory bandwidth (ruled
out below) — it's an unrolled, strictly-serial dependency chain that Cortex-A53's in-order pipeline
can't hide. A "3 %" number needs the whole mechanism explained to not look like a regression, which
is disproportionate to a one-sentence salvage. **Recommendation: keep W4 exactly as planned — the
measured 2.42–2.49× isolated speedup already tells the NEON story cleanly; drop the roofline framing
entirely, including the salvage sentence.**

Side finding, independent of the paper decision: **the plan's `19.2 GFLOP/s` A53 NEON estimate (§4.0)
was UNVERIFIED and is wrong by exactly 2×** — see below. It isn't quoted standalone anywhere in
`main.tex` today (checked), so this is a correction to the planning doc only, not an erratum.

---

## 1. A53 NEON compute peak — spec vs. measured

**Board identity** (`ssh ZCU102`, 2026-09-01, idle — `uptime` load average 0.00): 4 cores, `CPU part
0xd03` = Cortex-A53 (`/proc/cpuinfo`, `lscpu`), `scaling_cur_freq` = 1,199,999 kHz ≈ **1.2 GHz**
(governor `userspace`, fixed), `Features: fp asimd ...` (NEON present, no SVE — expected for A53).

### 1.1 Why the plan's formula was unverified — and what's actually documented

The plan's number, `4 cores × 1.2 GHz × 4 FLOP/cyc = 19.2 GFLOP/s`, implicitly assumes **4 FLOP/cycle
per core** — i.e. one 128-bit NEON op/cycle (4×fp32 lanes) at **1 FLOP/lane**. That's only correct if
the op is a plain multiply *or* add, not a fused multiply-add (FMA = 2 FLOP/lane: one multiply, one
add, in one instruction). Whether the peak-defining op is fused matters a lot, and the plan flagged
this exact ambiguity as unverified.

ARM's own **public** documentation doesn't settle it: the Cortex-A53 Technical Reference Manual
(DDI0500J, r0p4 — non-confidential, fetched in full, 623 pages) states only "in-order pipeline with
symmetric dual-issue of most instructions" (§1.3) and a control-register bit to disable "dual issue
of floating-point, Advanced SIMD and Cryptography instructions" (§4.3.36) — no per-instruction
throughput table. The document that *does* have that table — ARM's confidential "Cortex-A53 SW
Optimisation Specification" (`ARM-EPM-022955`) — is not publicly published by ARM for the A53 (unlike
the A57/A72, which do have public `UAN00xx` optimization guides); the only copy found in this search
is CONFIDENTIAL and circulating via an unrelated 2020 corporate leak. **Not used or cited here** —
inappropriate to rely on for a publication regardless of easy availability.

Instead: **LLVM's AArch64 Cortex-A53 scheduling model**
(`llvm/lib/Target/AArch64/AArch64SchedA53.td`, LLVM project, Apache-2.0 WITH LLVM-exception, accessed
2026-09-01 from `raw.githubusercontent.com/llvm/llvm-project/main/...`) is the best available public,
citable, redistributable source — it's derived from that same ARM spec (its own header comment cites
*"Cortex-A53 Software Optimisation Specification - Instruction Timings v1.0 Spreadsheet"*) but
released as open-source compiler infrastructure, used in production by every AArch64 GCC/LLVM build.
It models exactly two FP/ASIMD execution resources per core:

```tablegen
def A53UnitFPALU : ProcResource<1> { let BufferSize = 0; } // FP ALU (add/cmp/cvt/copy)
def A53UnitFPMDS : ProcResource<1> { let BufferSize = 0; } // FP Mult/Div/Sqrt/MAC
...
def A53WriteFMAC : SchedWriteRes<[A53UnitFPMDS]> { let Latency = 10; }
def : InstRW<[A53WriteFMAC], (instregex "^FN?M(ADD|SUB).*")>;   // scalar FMADD/FMSUB
def : InstRW<[A53WriteFMAC], (instregex "^FML(A|S).*")>;         // vector FMLA/FMLS
```

**Exactly one `FPMDS` unit** handles every multiply/FMA-class op (scalar or 128-bit vector, same
resource) → throughput ≤ 1 FMA-class instruction/cycle/core, full 10-cycle latency but 1-cycle
throughput (fully pipelined, no `ResourceCycles` override). That gives a **derived peak of 1
FMLA/cycle × 4 lanes × 2 FLOP/FMA × 1.2 GHz = 9.6 GFLOP/s/core**, i.e. **8 FLOP/cyc/core, not 4** —
the plan's assumption undercounts by exactly the FMA fusion credit.

### 1.2 Measured (this session — the actual verification)

Wrote a NEON FMA microbenchmark (16 independent `float32x4_t` accumulators via `vfmaq_f32` — more
than the 10-cycle latency needs, so the single `FPMDS` port stays saturated; full source in the
Appendix) and ran it pinned to each core (`taskset -c N`):

| Test | Result |
| --- | --- |
| Core 0, solo | 9.586 GFLOP/s |
| Core 1, solo | 9.581 GFLOP/s |
| **All 4 cores concurrently** | 9.586 / 9.587 / 9.588 / 9.588 GFLOP/s — **aggregate 38.349 GFLOP/s** |

Disassembly confirms the microbenchmark's loop is genuinely vectorized as designed (16×
`fmla vN.4s, v2.4s, v1.4s` per iteration, no scalar fallback).

- Single-core measured/derived: 9.584 (avg) / 9.6 = **99.83 % efficiency** — the LLVM model's "1
  FMLA/cycle" prediction is correct to within measurement noise.
- 4-core concurrent aggregate: 38.349 / 38.4 theoretical = **99.87 %**, and only 0.013 % below naive
  `9.586 × 4` — cores don't contend for FP throughput (each has its own private FPMDS port; only L2/DDR
  is shared), confirming linear scaling is real, not assumed.
- **Corrected ceiling: 4-core chip peak = 38.4 GFLOP/s, not the plan's 19.2 GFLOP/s — off by exactly
  2.00×.**

A scalar-FMA variant (same shape, 1 lane) was also run as an intended lane-width sanity check but
measured 3.834 GFLOP/s — 1.6× *higher* than the naive "1 lane, same port" prediction of 2.4 GFLOP/s,
even though disassembly confirms it's genuinely scalar (`fmadd sN, sN, s2, s1`, no auto-vectorization).
**Flagging honestly, not explaining away**: this doesn't match the simple single-FPMDS-port model, and
I did not chase why (the abstracted LLVM model is explicitly a *scheduling* heuristic, not a
cycle-exact simulator, so some divergence from real silicon on scalar-width ops is plausible — e.g. a
narrower path not represented in the model). **It doesn't affect the verdict**: the actual kernel is
100 % NEON, and the NEON number is the one that matches theory almost exactly (99.83–99.87 %), so it's
the one used throughout.

---

## 2. STREAM-triad DDR bandwidth from the A53s (board experiment)

NEON-vectorized triad kernel `c[i] = a[i] + scalar*b[i]` (128-bit `vld1q_f32`/`vst1q_f32`, matching
how `normalize` itself moves data), arrays sized 32 MB each (`a`,`b`,`c` — far beyond any plausible
on-chip cache), best-of-10 timed passes per the standard STREAM methodology. This measures **CPU
load/store bandwidth through the cache hierarchy** — a different path from the DPU/AXI-HP-port DMA
bandwidth the existing R2 literature numbers describe, so **not directly comparable** to
Lu 2022 / Manev 2019 in `onboard_pipeline.md` §2; both are reported below for context only.

| Config | Result |
| --- | --- |
| 1 thread (core 0 solo) | **2.436 GB/s** |
| 4 threads concurrent (1/core, private arrays) | 1.688 / 1.777 / 1.673 / 1.726 GB/s per core → **aggregate 6.863 GB/s** |

Unlike compute, DDR bandwidth **does** contend: per-core rate drops from 2.436 → ~1.7 GB/s (−30 %)
under 4-way concurrent load, consistent with a shared DDR4 channel. Context vs. existing ceilings:
6.863 GB/s is **40.2 %** of the theoretical chip DDR ceiling (17.06 GB/s, `onboard_pipeline.md` §2)
and **50.1 %** of Lu 2022's measured DMA-path ceiling (13.7 GB/s, same DDR4-2133 config as our board)
— i.e. plain CPU load/store traffic achieves roughly half of what dedicated DMA achieves, which is
the expected qualitative direction (DMA engines sustain longer bursts / more outstanding requests
than scalar load/store issue) and not itself surprising enough to add to `onboard_pipeline.md`.

---

## 3. `normalize` kernel: FLOP/byte accounting

Kernel: `ddc::normalize_patch_neon()` (`inference_cpp/src/patch_transforms.hpp:54-78`), which inlines
`log_ps()` — a Cephes/Pommier degree-8 polynomial + IEEE-754 bit-trick log approximation
(`inference_cpp/src/neon_mathfun.h:48-86`), **not** a `logf`/`log1pf` call. `float32x4_t` throughout
(4 lanes), one 4-element group per loop iteration.

Counted twice, independently, and cross-checked: (a) by hand from the C++ source, (b) instruction-by-
instruction from `objdump -d` of the **real production binary** (`stream_pipeline` as currently built
on the board, function `ddc::BenchPipeline::stage_normalize`, which inlines the same kernel) — both
give the same total, since FMA-contraction doesn't change the FLOP count under a fixed convention
(the compiler auto-fused one extra `mul`+`sub` pair into `fmls` beyond what the source spells out
explicitly, but that changes instruction count, not FLOP count).

**Convention**: add/sub/mul = 1 FLOP, fused multiply-add/sub (`fmla`/`fmls`, or `vmlaq_f32` in source)
= 2 FLOP; compare/clamp/exponent bit-tricks excluded (standard HPC FLOP convention — these run on the
`FPALU`/integer ports, not `FPMDS`, and aren't "arithmetic").

| Quantity | Value | Basis |
| --- | --- | --- |
| FMA-class ops / 4-lane group | 12 (1 pre-log + 8 Horner + 2 combine + 1 auto-fused) | disasm-verified |
| add / sub / mul (non-fused) / group | 5 / 1 / 4 | disasm-verified |
| **FLOPs/element** | **8.5** | (12×2 + 5+1+4) / 4 lanes |
| **bytes/element** | **8** (4 B fp32 in + 4 B fp32 out) | `ldr`/`str q0` = 16 B / 4 elements |
| **Arithmetic intensity** | **1.0625 FLOP/byte** | — |
| elements/patch (256×256×2, re+im) | 131,072 | matches `FPGA_inference.md`'s "131K calls" figure |
| **FLOPs/patch** | **1,114,112** | |
| bytes/patch (algorithmic, in+out) | 1,048,576 B = exactly 1 MiB | |

AI ≈ 1.06 FLOP/byte is very low — for scale, the DPU roofline ridge in this same repo is **72
OP/byte** (`onboard_pipeline.md` §2). At face value this reads as memory-bound; §4 checks that
directly rather than assuming it.

---

## 4. Roofline placement

### 4.1 Measured kernel time (isolated, real production kernel, not a proxy)

Wrote a harness that `#include`s the actual `patch_transforms.hpp` from the board's source tree and
times `ddc::normalize_patch`/`ddc::normalize_patch_neon` directly, single-threaded, pinned to core 0,
iters=50 (mirroring `benchmark_hardware`'s s0 methodology for direct comparability):

| Path | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| scalar | 8.860 ms | 8.854 ms | 8.843 ms | 8.929 ms |
| NEON | 3.564 ms | 3.562 ms | 3.548 ms | 3.613 ms |

Scalar cross-checks against the independent `results/date27/s0/FP/s0_compress_entoff.json` stage
timing (mean 8.943 ms, n=50) to within 1 % — different random patch data, same order, good agreement.
**Isolated speedup this session: 2.486×** (doc previously quoted "~2.42×" from a different measurement
context — same ballpark, consistent). Output cross-checked scalar-vs-NEON: max abs diff = 1.19e-7,
same order as the doc's existing "kernel error vs libm = 7e-8" figure.

### 4.2 Is it memory-bound? (checked, not assumed)

At the measured single-core STREAM bandwidth (2.436 GB/s, §2), moving `normalize`'s 1 MiB/patch would
take **0.430 ms** if DDR-bandwidth-bound. Measured time is **3.564 ms — 8.3× longer**. There's no
world in which the kernel is bandwidth-bound at that gap; **the compute-peak comparison is the right
ceiling**, not a memory one. (This doesn't prove the 512 KB+512 KB working set fits L2 — sysfs cache
topology wasn't populated on this board/kernel to check directly — but it doesn't need to: the 8.3×
margin is conclusive regardless of where the bytes actually live.)

### 4.3 The salvage sentence

```
achieved throughput = 1,114,112 FLOPs / 3.564 ms = 0.313 GFLOP/s
                     = 3.26 % of the measured (and theoretical) single-core NEON peak (9.6 GFLOP/s)
ideal time at peak   = 0.116 ms/patch  (measured is 30.7x slower than that ideal)
```

**"`normalize` reaches ≈3.3 % of the single-core CPU compute bound."** True, verified, and the exact
deliverable the plan asked for — but see the verdict at the top for why this sentence is a liability,
not an asset, without a paragraph of context.

### 4.4 Mechanism (why 3 %, not 30 % or 80 %) — directly observed, not inferred

Disassembled the actual compiled loop (`objdump -d -C` on `stream_pipeline`'s
`stage_normalize`). It is **not unrolled**: one `ldr q0` → the full chain (12 fused + 10 more
dependent FP instructions, dominated by the 8-deep serial Horner evaluation, each `y_i` depending on
`y_{i-1}`) → one `str q0`, then branch back — repeated 32,768 times per patch with **zero
cross-iteration overlap**. Different iterations process independent data and *could* overlap on an
out-of-order core, but Cortex-A53 is strictly in-order: it cannot issue iteration *i+1* around a
stalled dependent instruction in iteration *i*. The compute-peak microbenchmark in §1 hits 99.8 %
of theoretical only *because* it was deliberately built with 16 independent accumulator chains to hide
exactly this latency — `normalize` as compiled has one chain, not sixteen. This is a latency-bound
kernel wearing a throughput-bound roofline's clothes; the 3 % figure is a real, well-understood
property of the code shape, not a flaw in the vectorization (the NEON path is still a legitimate
2.49× win over scalar — it just isn't the kind of win a FLOPs/peak ratio flatters).

*(Not pursued further — out of scope for this task — but noted for anyone optimizing this kernel
later: manually interleaving 2–4 independent `log_ps` calls per loop iteration, the same trick used in
the §1 microbenchmark, would plausibly recover a large fraction of the gap between 3 % and peak.)*

---

## Disposition

- [x] A53 NEON peak GFLOP/s — spec (LLVM `AArch64SchedA53.td`, ARM-derived, public/citable) +
  microbenchmark (measured, this session): **9.6 GFLOP/s/core, 38.4 GFLOP/s/chip** (theoretical,
  99.8%+ achieved in both single- and 4-core-concurrent microbenchmarks). Plan's `19.2 GFLOP/s` was
  UNVERIFIED and is corrected — **off by exactly 2×** (missing the FMA fusion credit).
- [x] STREAM-triad DDR bandwidth from the A53s (board experiment): **2.436 GB/s** single-core,
  **6.863 GB/s** 4-core aggregate (40.2 % of the 17.06 GB/s theoretical chip DDR ceiling). Different
  access path from the existing DPU/DMA literature numbers (R2) — not merged with them.
  Not folded into `onboard_pipeline.md` §2 — flag if useful there later.
- [x] FLOP/byte count for `normalize`: **8.5 FLOP/element, 8 B/element, AI ≈ 1.06 FLOP/byte**
  (source- and disassembly-verified).
- [x] Verdict: **not paper-worthy.** `normalize` achieves ≈3.3 % of single-core NEON peak — real,
  memory-bound ruled out, mechanism identified (unrolled serial dependency chain, in-order core) —
  but a 3 % figure needs its own explanation to read as a result rather than a regression. W4 should
  keep the existing 2.42–2.49× isolated-speedup framing and drop the roofline angle entirely.

**No files outside this memo were touched** (figures/`main.tex` untouched, per task scope). Benchmark
source lives in `$CLAUDE_JOB_DIR/tmp` on the host running this session (not committed — throwaway per
the plan's default disposition); full source reproduced in the Appendix below so the memo is
self-contained even after that directory is cleaned up. Board scratch dir `/tmp/r3_bench/` will be
removed at the end of this session.

---

## Appendix: reproducibility

Board: `ssh ZCU102`, GCC 11.2.0, build: `g++ -O3 -mcpu=cortex-a53 -std=c++17 -pthread <file>.cpp -o <bin>`.

**`cpu_roofline_bench.cpp`** (FMA peak + STREAM triad):

```cpp
// R3 CPU-roofline microbenchmark (DATE27 §4.2) — throwaway, not part of the shipped codebase.
// Modes:
//   fma_neon   <iters>                 - NEON 4x-lane FMA peak, single core
//   fma_scalar <iters>                 - scalar FMA peak, single core (sanity/lane-width check)
//   stream     <elements> <reps> <nthreads> - STREAM-triad-style DDR bandwidth, 1 or N cores
//
// Build: g++ -O3 -mcpu=cortex-a53 -std=c++17 -pthread cpu_roofline_bench.cpp -o cpu_roofline_bench
#include <arm_neon.h>
#include <pthread.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>
#include <vector>
#include <string>

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// NEON FMA peak: 16 independent float32x4_t accumulators (A53 FMLA latency
// ~10 cyc, throughput 1/cyc per the FPMDS port -> need >=10 independent
// chains in flight to saturate the pipe). Manually unrolled, no reassociation
// possible without -ffast-math (not used), so the compiler cannot fold the
// loop into a closed form.
double bench_fma_neon(long iters) {
    float32x4_t a = vdupq_n_f32(1.0000001f);
    float32x4_t b = vdupq_n_f32(0.9999999f);
    float32x4_t acc0 = vdupq_n_f32(1.0f), acc1 = vdupq_n_f32(1.1f), acc2 = vdupq_n_f32(1.2f), acc3 = vdupq_n_f32(1.3f);
    float32x4_t acc4 = vdupq_n_f32(1.4f), acc5 = vdupq_n_f32(1.5f), acc6 = vdupq_n_f32(1.6f), acc7 = vdupq_n_f32(1.7f);
    float32x4_t acc8 = vdupq_n_f32(1.8f), acc9 = vdupq_n_f32(1.9f), acc10 = vdupq_n_f32(2.0f), acc11 = vdupq_n_f32(2.1f);
    float32x4_t acc12 = vdupq_n_f32(2.2f), acc13 = vdupq_n_f32(2.3f), acc14 = vdupq_n_f32(2.4f), acc15 = vdupq_n_f32(2.5f);

    double t0 = now_sec();
    for (long it = 0; it < iters; ++it) {
        acc0 = vfmaq_f32(acc0, a, b);   acc1 = vfmaq_f32(acc1, a, b);
        acc2 = vfmaq_f32(acc2, a, b);   acc3 = vfmaq_f32(acc3, a, b);
        acc4 = vfmaq_f32(acc4, a, b);   acc5 = vfmaq_f32(acc5, a, b);
        acc6 = vfmaq_f32(acc6, a, b);   acc7 = vfmaq_f32(acc7, a, b);
        acc8 = vfmaq_f32(acc8, a, b);   acc9 = vfmaq_f32(acc9, a, b);
        acc10 = vfmaq_f32(acc10, a, b); acc11 = vfmaq_f32(acc11, a, b);
        acc12 = vfmaq_f32(acc12, a, b); acc13 = vfmaq_f32(acc13, a, b);
        acc14 = vfmaq_f32(acc14, a, b); acc15 = vfmaq_f32(acc15, a, b);
    }
    double t1 = now_sec();

    float32x4_t sum = vaddq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)),
                                 vaddq_f32(vaddq_f32(acc4, acc5), vaddq_f32(acc6, acc7)));
    sum = vaddq_f32(sum, vaddq_f32(vaddq_f32(vaddq_f32(acc8, acc9), vaddq_f32(acc10, acc11)),
                                    vaddq_f32(vaddq_f32(acc12, acc13), vaddq_f32(acc14, acc15))));
    float tmp[4]; vst1q_f32(tmp, sum);
    fprintf(stderr, "  [sink=%.6f]\n", tmp[0] + tmp[1] + tmp[2] + tmp[3]);

    double flops = (double)iters * 16.0 /*accumulators*/ * 4.0 /*lanes*/ * 2.0 /*FMA=2 FLOPs*/;
    double secs = t1 - t0;
    printf("fma_neon: %.4f s, %.6e FLOPs, %.3f GFLOP/s\n", secs, flops, flops / secs / 1e9);
    return flops / secs;
}

// Scalar FMA peak: same shape, 1 lane instead of 4 -> should be ~1/4 of NEON if
// vector width is really 4x32-bit and both use the same FPMDS pipeline.
double bench_fma_scalar(long iters) {
    float a = 1.0000001f, b = 0.9999999f;
    float acc0=1.0f,acc1=1.1f,acc2=1.2f,acc3=1.3f,acc4=1.4f,acc5=1.5f,acc6=1.6f,acc7=1.7f;
    float acc8=1.8f,acc9=1.9f,acc10=2.0f,acc11=2.1f,acc12=2.2f,acc13=2.3f,acc14=2.4f,acc15=2.5f;

    double t0 = now_sec();
    for (long it = 0; it < iters; ++it) {
        acc0 = __builtin_fmaf(acc0, a, b);   acc1 = __builtin_fmaf(acc1, a, b);
        acc2 = __builtin_fmaf(acc2, a, b);   acc3 = __builtin_fmaf(acc3, a, b);
        acc4 = __builtin_fmaf(acc4, a, b);   acc5 = __builtin_fmaf(acc5, a, b);
        acc6 = __builtin_fmaf(acc6, a, b);   acc7 = __builtin_fmaf(acc7, a, b);
        acc8 = __builtin_fmaf(acc8, a, b);   acc9 = __builtin_fmaf(acc9, a, b);
        acc10 = __builtin_fmaf(acc10, a, b); acc11 = __builtin_fmaf(acc11, a, b);
        acc12 = __builtin_fmaf(acc12, a, b); acc13 = __builtin_fmaf(acc13, a, b);
        acc14 = __builtin_fmaf(acc14, a, b); acc15 = __builtin_fmaf(acc15, a, b);
    }
    double t1 = now_sec();

    float sink = acc0+acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8+acc9+acc10+acc11+acc12+acc13+acc14+acc15;
    fprintf(stderr, "  [sink=%.6f]\n", sink);

    double flops = (double)iters * 16.0 * 1.0 /*lane*/ * 2.0 /*FMA=2 FLOPs*/;
    double secs = t1 - t0;
    printf("fma_scalar: %.4f s, %.6e FLOPs, %.3f GFLOP/s\n", secs, flops, flops / secs / 1e9);
    return flops / secs;
}

// STREAM-triad-style DDR bandwidth: c[i] = a[i] + scalar*b[i], NEON load/store
// (128-bit, matches how the actual kernel moves data), 2 reads + 1 write per
// element. n chosen far larger than any plausible on-chip cache.
struct StreamArgs {
    size_t n;
    int reps;
    int core;
    double best_gbs;
};

void stream_triad_worker(StreamArgs* args) {
    pin_to_core(args->core);
    size_t n = args->n;
    float* a = (float*)aligned_alloc(64, n * sizeof(float));
    float* b = (float*)aligned_alloc(64, n * sizeof(float));
    float* c = (float*)aligned_alloc(64, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.0f; }
    float32x4_t vscalar = vdupq_n_f32(3.0f);

    double best_secs = 1e18;
    for (int r = 0; r < args->reps; ++r) {
        double t0 = now_sec();
        for (size_t i = 0; i + 4 <= n; i += 4) {
            float32x4_t va = vld1q_f32(&a[i]);
            float32x4_t vb = vld1q_f32(&b[i]);
            float32x4_t vc = vfmaq_f32(va, vscalar, vb);
            vst1q_f32(&c[i], vc);
        }
        double t1 = now_sec();
        double secs = t1 - t0;
        if (secs < best_secs) best_secs = secs;
        a[0] += c[n - 1] * 1e-30f; // prevent hoisting across reps
    }
    double bytes = 3.0 * (double)n * sizeof(float); // STREAM convention: 2R + 1W
    args->best_gbs = bytes / best_secs / 1e9;
    free(a); free(b); free(c);
}

void bench_stream(size_t n, int reps, int nthreads) {
    std::vector<StreamArgs> args(nthreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        args[t] = {n, reps, t, 0.0};
    }
    double t0 = now_sec();
    for (int t = 0; t < nthreads; ++t) threads.emplace_back(stream_triad_worker, &args[t]);
    for (auto& th : threads) th.join();
    double t1 = now_sec();

    double agg = 0.0;
    for (int t = 0; t < nthreads; ++t) {
        printf("stream[core %d]: %.3f GB/s\n", args[t].core, args[t].best_gbs);
        agg += args[t].best_gbs;
    }
    printf("stream_aggregate(%d threads): %.3f GB/s (wall %.3f s)\n", nthreads, agg, t1 - t0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s fma_neon <iters> | fma_scalar <iters> | stream <elements> <reps> <nthreads>\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "fma_neon") {
        long iters = argc > 2 ? atol(argv[2]) : 200000000L;
        bench_fma_neon(iters);
    } else if (mode == "fma_scalar") {
        long iters = argc > 2 ? atol(argv[2]) : 200000000L;
        bench_fma_scalar(iters);
    } else if (mode == "stream") {
        size_t n = argc > 2 ? (size_t)atol(argv[2]) : (size_t)8388608;
        int reps = argc > 3 ? atoi(argv[3]) : 10;
        int nthreads = argc > 4 ? atoi(argv[4]) : 1;
        bench_stream(n, reps, nthreads);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 1;
    }
    return 0;
}
```

**`normalize_isolated.cpp`** (isolated timing of the real production kernel):

```cpp
// R3: isolated timing of the REAL production normalize kernels (scalar + NEON),
// mirroring benchmark_hardware's s0 per-stage methodology (iters=50, report
// mean/min/max/median ms) for direct comparability with
// results/date27/s0/FP/s0_compress_entoff.json's normalize stage (mean 8.943 ms).
//
// Build (on board, from the real source tree so it's the exact same kernel):
//   g++ -O3 -mcpu=cortex-a53 -std=c++17 -I<repo>/inference_cpp/src normalize_isolated.cpp -o normalize_isolated
#include "patch_transforms.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <time.h>

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    const int H = 256, W = 256;
    const int n = H * W * 2;
    std::vector<float> in(n), out(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 500.0f); // plausible raw SLC-amplitude range
    for (auto& v : in) v = dist(rng);

    const int iters = 50;
    std::vector<double> t_scalar(iters), t_neon(iters);

    // Warmup + scalar timing
    for (int i = 0; i < 5; ++i) ddc::normalize_patch(in.data(), out.data(), H, W);
    for (int i = 0; i < iters; ++i) {
        double t0 = now_sec();
        ddc::normalize_patch(in.data(), out.data(), H, W);
        double t1 = now_sec();
        t_scalar[i] = (t1 - t0) * 1000.0;
    }

    std::vector<float> out_neon(n);
    for (int i = 0; i < 5; ++i) ddc::normalize_patch_neon(in.data(), out_neon.data(), H, W);
    for (int i = 0; i < iters; ++i) {
        double t0 = now_sec();
        ddc::normalize_patch_neon(in.data(), out_neon.data(), H, W);
        double t1 = now_sec();
        t_neon[i] = (t1 - t0) * 1000.0;
    }

    // byte-identity sanity check vs scalar (should match within the documented ~7e-8 kernel error)
    double max_abs_diff = 0.0;
    for (int i = 0; i < n; ++i) max_abs_diff = std::max(max_abs_diff, (double)std::abs(out[i] - out_neon[i]));

    auto stats = [](std::vector<double> v, const char* label) {
        std::sort(v.begin(), v.end());
        double sum = 0; for (double x : v) sum += x;
        double mean = sum / v.size();
        double median = v[v.size()/2];
        printf("%s: mean=%.5f ms, median=%.5f ms, min=%.5f ms, max=%.5f ms (n=%zu)\n",
               label, mean, median, v.front(), v.back(), v.size());
        return mean;
    };
    double mean_scalar = stats(t_scalar, "scalar");
    double mean_neon = stats(t_neon, "neon  ");
    printf("speedup (scalar/neon) = %.4fx\n", mean_scalar / mean_neon);
    printf("max_abs_diff scalar-vs-neon output = %.3e\n", max_abs_diff);
    return 0;
}
```

**Commands run** (all against the idle board, 2026-09-01):

```bash
# FMA peak, single core
taskset -c 0 ./cpu_roofline_bench fma_neon 300000000
taskset -c 1 ./cpu_roofline_bench fma_neon 300000000
taskset -c 0 ./cpu_roofline_bench fma_scalar 300000000

# FMA peak, 4 cores concurrent (separate processes, taskset-pinned, via a wrapper script)
taskset -c {0,1,2,3} ./cpu_roofline_bench fma_neon 300000000   # backgrounded, waited, one per core

# STREAM triad
./cpu_roofline_bench stream 8388608 10 1   # single core
./cpu_roofline_bench stream 8388608 10 4   # 4 cores concurrent

# Isolated real-kernel timing
g++ -O3 -mcpu=cortex-a53 -std=c++17 -I/home/root/SAR_DDC/inference_cpp/src normalize_isolated.cpp -o normalize_isolated
taskset -c 0 ./normalize_isolated

# Disassembly cross-check
objdump -d -C --disassemble='ddc::BenchPipeline::stage_normalize(ddc::PatchState&)' \
    /home/root/SAR_DDC/build_cpp/stream_pipeline
```
