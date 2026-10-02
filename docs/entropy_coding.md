# CPU rANS entropy coding — internals, profiling & optimization

Deep dive on the CPU-side rANS entropy stage of the onboard streaming pipeline (satellite of
`onboard_pipeline.md`, which covers where this stage sits in the pipeline and its share of per-patch
wall time). Board = ZCU102 (4× Cortex-A53 @ 1.2 GHz, NEON). Source lives in
`inference_cpp/src/rans/rans_interface_cxx.{cpp,hpp}` + `rans64.h` and `inference_cpp/src/entropy_models.{cpp,hpp}`.

## Coders & tables

Two entropy coders run depending on architecture:

- **EB** (entropy-bottleneck) — factorized-prior archs (FP/ResFP) code the main latent with it; the
  hyperprior archs use it only over the tiny `z` side-latent.
- **GC** (Gaussian-conditional) — hyperprior archs (SHyp/ResSHyp) code the main latent with it, using
  scales recovered by re-running `h_s` at decode time.

The rANS probability scale is `precision = 16` (`rans_interface_cxx.cpp`) — frequencies sum to
$2^{16}$ = 65 536. That 65 k is a **normalization total, not a table length**. The CDF tables themselves
are per-channel: EB is **256 × 28 = 7 168 int32 (28 KB)**, GC is **64 × 3 133 = 200 512 int32 (802 KB)**
(`entropy_params/{eb,gc}_quantized_cdf.npy`). GC being ~28× the EB table is the structural reason
`gc_compress` costs ~9.7 ms/patch against `eb_enc`'s ~4.7 ms.

**CDF tables are static** — `quantized_cdf_` is loaded once in `load_params` and only read per patch.
There is no per-patch CDF build; a hypothesised per-patch build cost does not exist.

## Where the per-patch time goes (profiled)

Measured with the opt-in sub-stage profiler (`DDC_RANS_PROFILE`, below). Each coder's per-patch time
splits into **symbuild** (`round(x − median/mean)` + index construction; for GC the per-element
`scale_index()` search), **lookup** (the forward CDF pass that builds `_syms`, `encode_with_indexes`),
and **flush** (the reverse rANS renorm loop `Rans64EncPut` + output copy).

**FP — EB over the main latent, 65 536 symbols/patch:**

| stage | µs/patch | ns/symbol | share |
| --- | --- | --- | --- |
| symbuild | 440 | 6.7 | 6 % |
| **lookup** | **5 198** | **79.3** | **75 %** |
| flush (rANS) | 1 268 | 19.4 | 18 % |
| **TOTAL** | **6 906** | 105 | 100 % |

**SHyp/ResSHyp — GC over main latent + EB over z** (the CPU entropy stage is identical for the two —
they differ only in DPU residual blocks):

| coder · stage | µs/patch | share of coder |
| --- | --- | --- |
| **GC symbuild** (`scale_index` search) | **5 665** | 52 % |
| GC lookup | 4 040 | 37 % |
| GC flush (rANS) | 1 258 | 11 % |
| GC TOTAL | 10 963 | — |
| EB over z (tiny) | 111 | negligible |

**The rANS renorm core (flush) is the *smallest* slice.** The cost lives in the lookups and, for GC, in
the per-element scale search.

## Diagnosis — memory/lookup-bound, not arithmetic

- **FP lookup = 79 ns/symbol (~95 A53 cycles) for ~10 instructions of work → memory-latency-bound.** Two
  structural causes: `_syms` is an **unreserved** `std::vector<RansSymbol>` (6 B/elem) that grows by
  reallocation to ~½ MB, then is **written in lookup and re-read in flush** — a two-pass design touching
  ½ MB twice; and `cdfs` is a **vector-of-vectors** (256 separate heap rows), pointer-chased per symbol
  as the channel index cycles (256 rows × 28 int32 = 28 KB, thrashes L1).
- **FP flush = 19 ns/symbol → divide-bound.** The loop calls the divide-based `Rans64EncPut` (`x / freq`,
  `x % freq`), not the precomputed-reciprocal `Rans64EncPutSymbol` present but unused in `rans64.h`.
- **SHyp symbuild = 86 ns/element → `scale_index()` is a per-element binary search** over a 64-entry
  `gc_scale_table` (`std::upper_bound`, `entropy_models.cpp`) — data-dependent branches, misprediction-heavy.

## The `--entropy` optimization (implemented; byte-identical)

`--entropy` selects the optimized coder; the pre-optimization CDF-lookup + divide path is kept alongside
it for A/B (both live in `entropy_models.{cpp,hpp}` / `rans_interface_cxx.{cpp,hpp}`, selected once per
`compress()` call). At **model-load** (once, not per patch) it (1) **flattens** the 256-row CDF book into
one contiguous block and **`reserve()`s** `_syms` to its exact length — the forward pass stops
pointer-chasing and stops re-growing; and (2) precomputes a **reciprocal** per table entry so the
per-symbol division in flush becomes a multiply (`Rans64EncPutSymbol`). The running list stores a small
index into that table. **Emitted bytes are exactly the same** (`.ddc` sha256 identical, sequential and
`--fanout`; `test_rans` roundtrip passes).

Per-stage effect (single-thread, `benchmark_hardware` s0/compress, 65 patches):

| arch · coder | lookup | flush | TOTAL |
| --- | --- | --- | --- |
| FP · EB | −49.3 % | **+50.9 %** | **−27.8 %** |
| SHyp · GC | −35.7 % | +14.7 % | **−11.7 %** |

**The flush-regression lesson.** Flush got *slower*, not faster: the reciprocal encoder removed the
divide, but the reciprocal table (24 B/entry, ~165 KB) is now **gathered per symbol** in the backward
pass, and on the memory-latency-bound A53 that gather costs more than the divide it replaced. **The net
win comes entirely from the forward (lookup) pass** — flatten + reserve + moving symbol resolution out
of it. A clean axis-2-only variant (flatten + reserve, no reciprocal) confirms the mechanism: flush
returns to baseline (1 270 ≈ 1 268 µs) and lookup still drops via flatten (−33 %). But the full bundle
still wins end-to-end (210.4 vs 205.7 patch/s, +2.3 %), because moving symbol resolution *out* of lookup
saves more there (−49 %) than the flush gather costs (+51 %). The hypothesis that the ~165 KB table would
lose under 16-thread contention was **refuted** — the bundle stays ahead at every thread count. Keep the
bundle; axis-2-only is a viable simpler fallback (closer to upstream CompressAI) at a −2.3 % throughput cost.

**End-to-end, the −28 % single-thread gain dilutes to +8 %** (FP, warm, t16 operating point:
194.7 → 210.3 patch/s). Entropy is only ~⅓ of FP per-patch CPU work (normalize + `g_a` glue are the
rest), and under lane contention the stage is bandwidth-bound — the gain is real and reproducible but not
1:1 with the stage delta.

## Unimplemented optimization axes (ranked by measured leverage)

Byte-identical = same `.ddc` bitstream, no encoder/decoder format rework.

1. **Fuse lookup + flush into one reverse pass** *(byte-identical)* — deletes the ~½ MB `_syms`
   intermediate entirely; attacks the 75 % lookup and part of the 18 % flush. Highest value. rANS already
   encodes in reverse; the only wrinkle is bypass-symbol handling, emittable inline in the reverse walk.
2. **SHyp only — branchless / direct-quantized `scale_index`** *(byte-identical)*, replacing the
   per-element binary search — attacks its 52 % symbuild slice.
3. **Interleaved rANS streams** *(NOT byte-identical — changes `.ddc`, needs encode + decode rework)* —
   only helps the 11–18 % flush slice. Deprioritize until 1–2 are done.

## Reproduce

The profiler is opt-in and lives in `inference_cpp/src/rans/rans_profile.{hpp,cpp}` with timers in
`rans_interface_cxx.cpp` (lookup, flush) and `entropy_models.cpp` (symbuild). It is **off by default,
byte-identical to production, and safe to leave in** — with the option OFF every macro compiles to a
no-op and `test_rans` still passes. Board builds must `make clean` first (the ZCU102 clock is unset →
`make` clock-skew mis-triggers incremental builds — this was learned the hard way here).

```bash
# 1. Sync + build with the flag on (from repo root, host)
rsync -a inference_cpp/src/ ZCU102:/home/root/SAR_DDC/inference_cpp/src/
rsync -a inference_cpp/CMakeLists.txt ZCU102:/home/root/SAR_DDC/inference_cpp/CMakeLists.txt
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && cmake -DDDC_RANS_PROFILE=ON . && make benchmark_hardware -j4"

# 2. Run single-threaded, capture the per-(coder,stage) table from stderr
ssh ZCU102 "cd /home/root/SAR_DDC && ./build_cpp/benchmark_hardware \
  --xmodel active_model/*.xmodel --params active_model/entropy_params \
  --data data/test_sub500_seed42.npy --config s0 --scenario compress \
  --warmup 5 --iters 60 --output /tmp/bench.json 2>/tmp/prof.txt; \
  grep -A20 'rANS encode profile' /tmp/prof.txt"

# 3. Revert the board to a clean production build
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && make clean && cmake -DDDC_RANS_PROFILE=OFF . && make -j4"
```

Per-symbol ns = `µs_per_patch × 1000 / n`, with `n = H_LATENT · W_LATENT · channels = 16 · 16 · 256 =
65 536` for the FP/GC main latent (`channels = C_MAIN · 2 = 256`, interleaved real+imag; EB-over-z uses
`16·16` → far fewer).

For how the entropy stage's cost translates into what binds the CPU-bound archs at the operating point
(uncontended FP spends 4.72 ms/patch in `eb_enc`, SHyp 9.65 ms in `gc`), see `onboard_pipeline.md` §6.
