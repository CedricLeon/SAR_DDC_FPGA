# Entropy-coding optimization opportunities (N6 exploration)

Scratch note profiling the CPU-side rANS entropy stage of the onboard streaming pipeline, to find
where the per-patch time actually goes before committing to any optimization. Board = ZCU102
(4× Cortex-A53 @ 1.2 GHz, NEON). Companion to N6 in `onboard_pipeline.md`.

## TL;DR

- **The rANS core (renorm loop) is the *smallest* slice** (11–18 %). The cost lives in the **lookups**.
- **FP** (headline CPU-bound arch, EB over the main latent): **75 % CDF-lookup**, 18 % rANS, 6 % symbol-build.
- **SHyp/ResSHyp** (GC over main latent + EB over z): **52 % `scale_index` binary search**, 37 % lookup, 11 % rANS.
- Highest-leverage wins are **byte-identical** (no `.ddc` format change): fuse the two passes, `reserve` +
  flatten the CDF table, precomputed-reciprocal encoder. **Interleaved rANS streams** hit the smallest
  slice *and* break the format — deprioritize.
- The hypothesised "per-patch CDF build" cost **does not exist**: CDF tables are static, loaded once.

## Results

Instrumented the real compress path with an opt-in sub-stage profiler (`DDC_RANS_PROFILE`, off by
default, byte-identical to production). Each coder's per-patch time splits into:

- **symbuild** — `round(x − median)` / `round(y − mean)` + index construction (for GC, the per-element
  `scale_index()` search).
- **lookup** — the forward CDF-lookup pass that builds `_syms` (`encode_with_indexes`).
- **flush** — the reverse rANS renorm loop (`Rans64EncPut`) + output byte copy.

**FP — `FP-relu_s0_L20`, EB over the main latent, 65 536 symbols/patch:**

| stage | µs/patch | ns/symbol | share |
| --- | --- | --- | --- |
| symbuild | 440 | 6.7 | 6 % |
| **lookup** | **5198** | **79.3** | **75 %** |
| flush (rANS) | 1268 | 19.4 | 18 % |
| **TOTAL** | **6906** | 105 | 100 % |

(≈6.9 ms/patch, consistent with the ~6.4 ms entropy figure in `onboard_pipeline.md` §CPU occupancy.)

**SHyp/ResSHyp — `ResSHyp-relu_s0_L20`, GC over main latent + EB over z.** The CPU entropy stage is
identical for SHyp and ResSHyp (they differ only in DPU residual blocks), so either model measures it.

| coder · stage | µs/patch | share of coder |
| --- | --- | --- |
| **GC symbuild** (`scale_index` search) | **5665** | 52 % |
| GC lookup | 4040 | 37 % |
| GC flush (rANS) | 1258 | 11 % |
| GC TOTAL | 10963 | — |
| EB TOTAL (over z, tiny) | 111 | negligible |

## Diagnosis — memory/lookup-bound, not arithmetic

- **FP lookup = 79 ns/symbol (~95 A53 cycles) for ~10 instructions of work → memory-latency-bound.**
  Two structural causes in `rans_interface_cxx.cpp` / `rans_interface_cxx.hpp`:
  - `_syms` is an **unreserved** `std::vector<RansSymbol>` (6 B/elem) that grows by reallocation to
    ~½ MB, then is **written in lookup and re-read in flush** — a two-pass design touching ½ MB twice.
  - `cdfs` is a **vector-of-vectors** (`std::vector<std::vector<int32_t>>`, 256 separate heap rows),
    pointer-chased per symbol as the channel index cycles (256 rows × 28 int32 = 28 KB, thrashes L1).
- **FP flush = 19 ns/symbol → divide-bound.** The loop calls the divide-based `Rans64EncPut`
  (`x / freq`, `x % freq`), **not** the precomputed-reciprocal `Rans64EncPutSymbol` already present but
  unused in `rans64.h`.
- **SHyp symbuild = 86 ns/element → `scale_index()` is a per-element binary search over a 64-entry
  `gc_scale_table`** (`entropy_models.cpp`, `std::upper_bound`) — data-dependent branches, misprediction-heavy.
- **Static CDFs.** `quantized_cdf_` is loaded once in `load_params` and only read per patch — no per-patch
  CDF build.

## Optimization axes (ranked by measured leverage)

Byte-identical = same `.ddc` bitstream, no encoder/decoder format rework.

1. **Fuse lookup + flush into one reverse pass** *(byte-identical)* — deletes the ~½ MB `_syms`
   intermediate entirely; attacks the 75 % and part of the 18 %. Highest value. (rANS already encodes in
   reverse; the only wrinkle is bypass-symbol handling, which can be emitted inline in the reverse walk.)
2. **`reserve(n)` on `_syms` + flatten `cdfs`** to one contiguous buffer with per-row offsets
   *(byte-identical)* — removes reallocations and the pointer chase. Cheap; do this even if not fusing.
3. **Precomputed-reciprocal encoder** — build a `Rans64EncSymbol` table once from the static CDFs and use
   `Rans64EncPutSymbol` *(byte-identical)* — removes the per-symbol 64-bit divide in flush.
4. **SHyp only:** replace the `scale_index` binary search with a branchless / direct-quantized index
   *(byte-identical)* — attacks its 52 %.
5. **Interleaved rANS streams** *(NOT byte-identical — changes `.ddc`, needs encode+decode rework)* —
   only helps the 11–18 % flush slice. Deprioritize until 1–4 are done.

## Reproduce

The profiler is opt-in and lives in `inference_cpp/src/rans/rans_profile.{hpp,cpp}` with timers in
`rans_interface_cxx.cpp` (lookup, flush) and `entropy_models.cpp` (symbuild). Enable with the CMake option
`DDC_RANS_PROFILE`; it prints a per-`(coder, stage)` table to **stderr** at process exit.

**1. Sync + build on the board with the flag on:**

```bash
# from repo root (host)
rsync -a inference_cpp/src/ ZCU102:/home/root/SAR_DDC/inference_cpp/src/
rsync -a inference_cpp/CMakeLists.txt ZCU102:/home/root/SAR_DDC/inference_cpp/CMakeLists.txt
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && cmake -DDDC_RANS_PROFILE=ON . && make benchmark_hardware -j4"
```

**2. Run (single-threaded per-patch), capturing the profile from stderr:**

```bash
# FP (EB over main latent) — the headline number
ssh ZCU102 "cd /home/root/SAR_DDC && ./build_cpp/benchmark_hardware \
  --xmodel models/FP_L20/*.xmodel --params models/FP_L20/entropy_params \
  --data data/test_sub500_seed42.npy --config s0 --scenario compress \
  --warmup 5 --iters 60 --output /tmp/bench_fp.json 2>/tmp/prof_fp.txt; \
  grep -A20 'rANS encode profile' /tmp/prof_fp.txt"

# SHyp/ResSHyp (GC + EB) — swap the model dir (e.g. models/ResSHyp_L20 or the active_model symlink)
```

Per-symbol ns = `µs_per_patch × 1000 / n`, with `n = H_LATENT · W_LATENT · channels = 16 · 16 · 256 =
65 536` for the FP/GC main latent (`channels = C_MAIN · 2 = 256`, interleaved real+imag; EB-over-z uses
`16·16` → far fewer). `us/patch` in the table is averaged over all warmup+iter patches.

**Revert the board to a clean production build:**

```bash
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && cmake -DDDC_RANS_PROFILE=OFF . && make -j4"
```

The source instrumentation is safe to leave in: with the option OFF every macro compiles to a no-op, the
bitstream is unchanged, and the rANS roundtrip test (`test_rans`) still passes.

---

# Stage 1 results — reserve + flatten + reciprocal encoder (byte-identical)

Implemented axes 2 (reserve + flatten CDF) and 3 (precomputed-reciprocal encoder) as a single
byte-identical restructure of the encode path. Axes 1 (fuse) and 4 (`scale_index`) are deferred to
Stage 2; axis 5 (interleaved streams) is untouched (breaks the `.ddc` format).

## What changed, in lay terms

Encoding one patch's ~65 k latent symbols runs in two passes. **Before:** the forward pass looked up each
symbol's probability interval in a "book" of 256 per-channel tables stored as 256 separate little arrays
scattered across memory (a *vector-of-vectors*), appending each result to a running list it re-grew on the
fly; the backward pass then walked that list and did the arithmetic-coding step, which included an integer
**division** per symbol. Two wasteful things: (1) the scattered book meant every lookup chased a pointer to
a cold cache line, and the list was reallocated many times as it grew; (2) integer division is slow on the
board's A53.

**After:** at *model-load* (once, not per patch) we (1) flatten the book into one contiguous block and
pre-size the running list to its exact length, so the forward pass stops pointer-chasing and stops
re-growing; and (2) precompute a "reciprocal" for every table entry so the per-symbol division in the
backward pass becomes a multiply (`Rans64EncPutSymbol` instead of `Rans64EncPut`). The running list now
stores a small index into that precomputed table. **The emitted compressed bytes are exactly the same.**

## Per-stage entropy time (single-thread, `benchmark_hardware` s0/compress, 65 patches)

| arch · coder | stage | base µs/patch | Stage-1 µs/patch | delta |
| --- | --- | --- | --- | --- |
| FP · EB | symbuild | 439.6 | 436.1 | −0.8 % |
| FP · EB | lookup | 5198.2 | 2634.9 | **−49.3 %** |
| FP · EB | flush | 1268.3 | 1913.7 | **+50.9 %** |
| FP · EB | **TOTAL** | **6906.1** | **4984.7** | **−27.8 %** |
| SHyp · GC | symbuild | 5665.4 | 5636.3 | −0.5 % |
| SHyp · GC | lookup | 4039.6 | 2595.7 | **−35.7 %** |
| SHyp · GC | flush | 1257.6 | 1442.6 | +14.7 % |
| SHyp · GC | **TOTAL** | **10962.6** | **9674.6** | **−11.7 %** |

(SHyp EB-over-z is ~114 µs/patch, negligible. Run-to-run variance ≈ 2 %.)

## End-to-end throughput (FP, warm, operating point t16 fanout+prefetch+neon, overlap 2)

Same-session interleaved A/B (baseline vs Stage-1 binary, 3 rounds each) on the full scene (7 540 patches):

| binary | patch/s (median of 3) | scene time |
| --- | --- | --- |
| baseline | 194.7 | 38.7 s |
| Stage-1 | 210.3 | 35.8 s |
| **delta** | **+8.0 %** | **−7.5 %** |

Baseline reproduces the committed DATE'27 number (192.83 patch/s), validating the setup.

## Byte-identity — confirmed

- `test_rans` roundtrip PASS.
- FP `.ddc` sha256 identical, sequential: `66366f70…` (Stage-1 == committed baseline).
- FP `.ddc` sha256 identical, **production `--fanout` config**: `2413e881…` (baseline binary == Stage-1 binary).
- SHyp `.ddc` sha256 identical: `943224ef…`.

## Surprises / honest caveats

- **`flush` got *slower*, not faster** (FP +51 %, SHyp +15 %). The reciprocal encoder removed the divide
  but the reciprocal table (24 B/entry, ~165 KB) is now gathered per symbol in the backward pass, and on
  the memory-latency-bound A53 that gather costs more than the divide it replaced. **The net win comes
  entirely from the forward (`lookup`) pass** — flatten + reserve + moving the symbol resolution out of it
  (−49 % / −36 %), which outweighs the flush regression. So axis 3 (reciprocal) in isolation looks
  net-negative here; the value is in axis 2 (flatten/reserve) plus the index-store restructure.
- **The −28 % single-thread entropy gain dilutes to +8 % end-to-end**, as expected: entropy is only ~⅓ of
  the FP per-patch CPU work (normalize + `g_a` glue make up the rest), and under 16-thread contention the
  stage is bandwidth-bound, so the gain is partial. It is real and reproducible, but not 1:1 with the
  stage delta.
- **Board incremental builds are unreliable** (ZCU102 clock is unset → `make` clock-skew mis-triggers).
  Any A/B must use `make clean` full rebuilds, verified here by rebuilding both binaries from scratch.

## Reproduce (Stage-1 additions)

Clean rebuild is mandatory (see caveat above):

```bash
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && make clean && \
  make stream_pipeline benchmark_hardware test_rans -j4"
```

Byte-identity under the production config (baseline binary saved as `/tmp/stream_base`):

```bash
FL='--p0 --fanout --threads 16 --prefetch --neon --windowed --overlap 2 --max-rows 512'
# run both binaries with $FL on data/full_scene_i16.npy, then sha256sum the two .ddc
```

Warm throughput A/B: interleave `stream_base` and the Stage-1 `stream_pipeline` (3 rounds each) on
`data/full_scene_i16.npy` with the operating-point flags above (no `--max-rows`), taking the median
`patch/s` from each binary's stdout summary line.

---

# Stage 1b — axis-2-only variant vs the axis-2+3 bundle (which to keep)

Stage 1's flush regression raised the question: is the reciprocal encoder (axis 3) pulling its weight, or
is the win entirely axis 2 (flatten + reserve)? To settle it, a clean **axis-2-only** variant was built —
flatten the CDF + `reserve()` `_syms`, but store the full `{start,freq}` symbol (no index) and keep the
original divide-based `Rans64EncPut` in flush (no reciprocal table anywhere). Then a 3-way byte-identical
A/B: (i) committed baseline, (ii) Stage-1 bundle (axis 2+3), (iii) axis-2-only.

**Byte-identity (iii):** `test_rans` PASS; `.ddc` sha256 identical, sequential `66366f70…` and production
`--fanout` `2413e881…`.

**Single-thread entropy (µs/patch, `benchmark_hardware` s0/compress):**

| variant | FP·EB lookup | FP·EB flush | FP·EB TOTAL | SHyp·GC TOTAL |
| --- | --- | --- | --- | --- |
| (i) baseline | 5198 | 1268 | 6906 | 10963 |
| (ii) axis 2+3 | 2635 | 1914 | **4985 (−27.8 %)** | 9675 (−11.7 %) |
| (iii) axis-2-only | 3499 | 1270 | 5205 (−24.6 %) | 9643 (−12.0 %) |

Axis-2-only confirms the mechanism cleanly: **flush returns to baseline** (1270 ≈ 1268 µs — the reciprocal
gather is gone) and lookup still drops via flatten+reserve (−33 %). But (ii) is faster overall: moving the
symbol resolution *out* of lookup (storing a compact index, gathering the reciprocal symbol in flush)
saves more in lookup (−49 %) than the flush gather costs (+51 %).

**End-to-end throughput (FP, warm, t16 operating point; median of 3 interleaved rounds):**

| variant | patch/s | vs baseline |
| --- | --- | --- |
| (i) baseline | 194.8 | — |
| (ii) axis 2+3 | **210.4** | **+8.0 %** |
| (iii) axis-2-only | 205.7 | +5.6 % |

Runs are cleanly separated (ii 209.5–210.9, iii 205.4–206.4), so **(ii) is robustly +2.3 % over (iii)**.

**Decision — keep (ii), the axis-2+3 bundle.** It wins throughput (+8.0 % vs +5.6 %) *and* single-thread,
at every thread count tested. The Stage-1 hypothesis that the ~165 KB reciprocal table would lose under
16-thread contention was **refuted** — the gather-in-flush arrangement stays ahead. The correct reading of
the Stage-1 flush regression: axis 3 makes the *flush stage* slower in isolation, but the index-store it
enables makes the *lookup stage* enough cheaper that the net is positive. Axis-2-only remains a viable
**simpler** fallback (no reciprocal table, closer to upstream CompressAI) at a −2.3 % throughput cost;
kept as a documented alternative, not the default.
