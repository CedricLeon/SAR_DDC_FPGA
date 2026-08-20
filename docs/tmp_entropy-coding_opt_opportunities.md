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
