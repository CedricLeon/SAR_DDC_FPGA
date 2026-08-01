# Onboard Streaming Pipeline — Plan & Feasibility

> End-to-end **"receive focused SLC tile → despeckle + compress → write downlink bitstream"**
> streaming demo on the Xilinx ZCU102 — the core contribution of the systems/CS paper, alongside
> the performance analysis. This file is the **design discussion CLAUDE.md requires before
> implementing any parallelism**. It records only *new* decisions + open questions; for existing
> pieces it points to the canonical docs.
>
> Builds on: `FPGA_inference.md` §5 (per-stage pipeline), §9 (streaming inference — earmarked);
> `FPGA_benchmark.md` §10 (P0/P2 patch-pipelining — earmarked, not implemented); `Data.md`
> (`.cos` source, normalisation).

Status: **implemented + measured.** On-board streaming compressor (Steps 1–4), full-scene ablation
sweep (throughput / latency / energy), and power + memory + roofline analysis are done and
board-verified. Remaining: overlap + reconstructed-tile quality (Step 6), on-ground SHyp decode
(Step 7, nice-to-have), and results-figure polish. Last updated 2026-08-01.

---

## 1. Goal & scope

- **Deliverable:** a *configurable, reproducible experimental harness* (not a shipped binary) to
  explore pipeline/thread schedules for the onboard scenario, and measure **steady-state throughput
  (patch/s)** + **full-tile latency** (read → compress → write).
- **Models:** ResSHyp (DPU-bound) + FP (CPU-bound) primary; SHyp/ResFP come ~free (arch
  auto-detected from `manifest.json`).
- **Overlap:** start **non-overlapping** (independent per-patch bitstreams); add overlap later — the
  harness will expose `--stream-overlap {0,4,8,16}` px to sweep it (reconstructed-image quality;
  inflates bitrate).
- **Language:** C++ on-board (only inference path). Python only for the offline symmetrization study
  (§6) and the `.ddc` verifier (§7).

---

## 2. Hardware facts (measured on the board unless noted)

| Memory | Capacity (this board) | Persistent? | Role in the pipeline | Source |
| --- | --- | --- | --- | --- |
| **SD card** (rootfs) | 29.7 GB card; ext4 `/` = 27 GB, **17 GB free** | ✅ | **Steps 1 & 7** — read tile, write `.ddc` | board `lsblk`/`df` |
| **PS DDR4** (RAM) | 4 GB spec; **3.84 GiB** seen, **~3.0 GiB free** | ❌ | Working memory: in-flight patches, queues, weights | `/proc/meminfo` + UG1182 |
| QSPI flash | boot 30 MB + env 256 KB + kernel 36 MB | ✅ | Boot only — not a data store | `/proc/mtd` + UG1182 |
| PL DDR4 | 512 MB (4 Gb, 16-bit) | ❌ | PL-side; unused by our ARM+DPU path | UG1182 |
| OCM (on-chip SRAM) | 256 KB | ❌ | PS boot/scratch; not in our path | DS891 |
| A53 cache | L1 32 KB I + 32 KB D /core; L2 1 MB shared | ❌ | CPU cache (automatic) | DS891 |
| DPU on-chip (BRAM/URAM) | ~84 % of ZU9EG BRAM (3× B4096) | ❌ | DPU weight/activation buffers (VART) | `FPGA_benchmark.md` + PG338 |

- **"Persistent memory" = the SD card** (17 GB free). No eMMC on the ZCU102 (only `mmcblk0`).
- ⚠️ `/tmp`, `/dev/shm`, `/var/volatile` are **tmpfs (RAM-backed, 2 GB each)** → not persistent and
  consume DDR. Step 7 writes to the SD rootfs.
- **4× A53 cores** (`nproc=4`) → parallel-worker budget, shared with DPU dispatch + OS.
- Queues are a non-issue: **~50 MB at depth 16** vs ~3 GB free.
- **SD sequential read ≈ 23.5–23.8 MB/s** (measured cold: 1.93 GB ÷ 81 s) — the **Step-1 read ceiling**.
  `eth0` is GbE (1000 Mb/s ≈ 5× the SD), but reading the tile from the host **breaks the onboard premise**;
  to emulate a *faster persistent store* use a warm read or a `/dev/shm` tmpfs copy (RAM speed, GB/s), not the host.
- **Page cache** (4 KB pages, confirmed): a *warm* read = the file resident in kernel cache — reclaimable, still
  counts as free. Our 1.93 GB tile ≈ **472k pages**; whole-tile *warm* read is fine, whole-tile *f32 in-process
  load* (3.87 GB) OOMs. Cold vs warm = whether those pages are present (cold → real SD read at ~24 MB/s).

- **PS DDR4 peak bandwidth = 17.06 GB/s** — 4 GB DDR4-2133 SODIMM (Kingston KVR21SE15S8/4), 64-bit
  (2133 MT/s × 8 B) [UG1182 + SODIMM part]. Sustained DDR traffic (SD read + DPU DMA + memcpy) sits
  ≈20× below this → DDR is **not** a bottleneck (vaitrace-measured; see §onboard TODOs).
- **DPU = 3× DPUCZDX8G B4096 @ 300 MHz → 1229 GOP/s per core** (4096 ops/cycle × 0.30 GHz; the guide
  lists 1400 @ 350 MHz) [PG338, *DPUCZDX8G Peak Performance*; clock from `xdputil query`]. Roofline
  ridge vs DDR = 1229 ÷ 17.06 = 72 OP/byte.

Sources: [UG1182 ZCU102 Eval Board UG](https://docs.amd.com/v/u/en-US/ug1182-zcu102-eval-bd) ·
[DS891 Zynq UltraScale+ Data Sheet](https://www.mouser.com/datasheet/2/903/ds891_zynq_ultrascale_plus_overview-1662253.pdf) ·
[PG338 DPUCZDX8G Peak Performance](https://docs.amd.com/r/en-US/pg338-dpu/DPUCZDX8G-Peak-Performance).

---

## 3. Data & acquisition geometry

- Tile: `data/TSX_cos_files/Hamburg_…_strip_004.cos` = **14 686 (range) × 32 901 (azimuth)** →
  **57 × 128 = 7 296** non-overlap 256² patches; raw complex-int16 = **1.93 GB**.
- Footprints: full f32 `[H,W,2]` tile = **3.87 GB** (> DDR → must stream); ~1000-patch region ≈ 538 MB.
- **Streaming axis = azimuth.** Range bins (14 686 cols) arrive ~together per radar pulse (fast-time);
  azimuth lines (32 901 rows) accumulate as the platform flies (slow-time). → natural streaming unit
  = **row-block** = 256 azimuth lines × full range = **one patch-row (57 patches)**; **128 row-blocks**
  per scene.
- Working sets: `test_1000.npy` (1000 random, **already whole-image-symmetrized** patches) for
  throughput plumbing **only** — not usable for E1 (needs raw). Switch to the raw `.cos` scene once
  we touch I/O + patchify + normalize + symmetrization.
- Reuse: `load_cosar`, `symmetrize`, `extract_patches` (`src/utils/sar_utils.py`, Python); C++
  `npy_io` already reads `[H,W,2]` tiles.

---

## 4. Pipeline (who does what)

| Step | Unit | Notes |
| --- | --- | --- |
| 1 read tile | SD → DDR | whole (region) or **row-block stream** (full scene) |
| ~~1.5 symmetrize~~ | — | **dropped** (E1 §6: ≤0.54 dB cost); optional one-time whole-image pre-pass if ever wanted |
| 2 patchify | CPU | strided per-row `memcpy` of `[256,256,2]` out of the DDR tile |
| 3 normalize | CPU | log + min/max (~9 ms/patch; `--neon` = 2.42× faster, byte-transparent) |
| 4 g_a×2 | DPU | already S1-parallel (1.95×) |
| 5 h_a/EB/h_s | DPU+CPU | SHyp only (FP skips) |
| 6 entropy | CPU | rANS → bits |
| 7 write bits | DDR → SD | per-tile `.ddc` product |

**Concepts.** *Queue* = bounded producer→consumer FIFO between threaded stages (depth = max buffered
patches; overlap = stage B on patch N while A makes N+1). One slot is the minimum for overlap; a small
depth (a few) absorbs per-patch DPU/rANS jitter so the bottleneck never stalls. The queue *is* the
efficient wait — a blocking pop, not a busy-wait (a spin would pin an A53 and corrupt timing).
*Row-block* = §3. *memcpy* = raw byte block copy; a patch = 256 per-row copies out of the tile.

**Schedules (named presets, not free-form knobs):**

*(Implemented as composable flags on `stream_pipeline`, not fixed presets — every one is byte-identical
to `stream_seq`, enforced by the correctness gate §8.)*

- `stream_seq` — 1 thread; correctness + latency baseline (reads tile, writes `.ddc`).
- `--s1` — `g_a(re) ‖ g_a(im)` across the 2 DPU cores (channel-parallel; 1.6–1.95×).
- `--p0 --threads K` — worker pool: each worker runs normalize → DPU (serialized by a mutex) →
  entropy → write, records placed by index. `K` is the "fine-grained" knob (subsumes the once-planned
  `stream_fine` per-stage-workers idea; ResSHyp plateaus ~3, FP keeps scaling to 4).
- `--prefetch` — producer thread double-buffers row-block N+1 while workers compress block N.
- `--neon` — NEON-vectorized normalize/denorm.

---

## 5. Decisions & open questions

| # | Question | Status |
| --- | --- | --- |
| U1 | Symmetrization granularity (whole / none / patch / block) | ✔ **resolved (E1 §6, definitive): skip it.** Granularity irrelevant; cost ≤0.54 dB (λ1000) → ~0.03 dB (λ2), across ResSHyp/FP over the full 7 296-patch scene. Optional whole-image pre-pass reclaims it |
| U2 | On-disk **file layout** + data type | ✔ (3.5) **row-major int16 `.npy`**, seek-streamed by `TileWindowReader` (header parsed once → byte offset of any patch/row computed from its coordinate; no per-patch file-open). Patchify (row-strided extract) = 0.4%, negligible. Patch-major `[n,256,256,2]` would zero patchify but fragments SD reads + complicates overlap → not worth it for in-order streaming. Data type: **int16** (4 B/px, likely the onboard SLC format). |
| U3 | Harness shape | ✔ **locked** — extend `benchmark_hardware` with named presets + gates (§8) |
| U4 | `.ddc` container format | ✔ **locked v1 + verified** (§7): `src/utils/ddc_format.py` codec + `ddc_selftest.py` pass on ResSHyp/FP (byte-exact round-trip, random access, lossless decode) |
| U5 | Overlap | ⏳ later — harness will expose `--stream-overlap {0,4,8,16}` px to sweep overlap → reconstructed-image quality |

---

## 6. Experiment E1 — symmetrization granularity (local GPU, no retraining)

**Why:** `symmetrize()` is a whole-image FFT → would break streaming. But it is an **integer spectral
roll = spatial phase ramp → |amplitude| is exactly preserved**; it only redistributes energy between
real/imag. **Hypothesis:** despeckling is amplitude-dominated and the model processes real/imag
separately, so local symmetrization should cost little quality → it can live *inside* the
per-patch/per-block pipeline. E1 measures it.

- **Variants** (preprocess the raw `.cos`, then run the DDC model): (1) whole-image *(reference)*,
  (2) none, (3) per-256²-patch, (4) per-1024²-block.
- **Metrics:** MSE, PSNR, SSIM, MS-SSIM (+ENL, EPD), **linear amplitude**, vs **(a)** variant (1)
  [`vs_whole`] and **(b)** MERLIN GT [`vs_merlin`, `--merlin-gt`].
- **Implemented:** `scripts/evaluation/symmetrization_study.py` (conda `DDC_FPGA`, GPU) — one model
  per run, streaming metrics (memory-safe to the full scene), fixed JSON schema + `--aggregate` for
  the cross-model table. Model resolved from the compiled FPGA models
  (`--arch ResSHyp --lambda 1000 [--seed]` → `manifest.json` → checkpoint); MERLIN GT from
  `data/method_ground_truths/MERLIN/checkpoints/last.ckpt`. Default region 4096² (256 patches),
  `--region full` for the whole scene (true whole-image reference).

  ```bash
  python scripts/evaluation/symmetrization_study.py --arch ResSHyp --lambda 1000 [--merlin-gt]
  python scripts/evaluation/symmetrization_study.py --aggregate results/symmetrization_study/
  ```

- **Outcome → placement:** patch-OK → per-patch stage (fold in with normalize); block-OK →
  per-row-block stage; only-whole → one-time whole-image pre-sweep before the patch pipeline.
- **Result** — ResSHyp + FP × λ∈{2,20,1000}, seed 0, **whole scene (7 296 patches)**, vs MERLIN GT
  PSNR:

  | model         | whole | none  | patch | block | Δ(skip) |
  |---------------|-------|-------|-------|-------|---------|
  | ResSHyp λ1000 | 31.65 | 31.10 | 31.10 | 31.11 | 0.54    |
  | ResSHyp λ20   | 28.61 | 28.36 | 28.41 | 28.42 | 0.25    |
  | ResSHyp λ2    | 22.94 | 22.91 | 22.93 | 22.93 | 0.03    |
  | FP λ1000      | 30.67 | 30.24 | 30.26 | 30.27 | 0.43    |
  | FP λ20        | 28.87 | 28.64 | 28.65 | 28.66 | 0.22    |
  | FP λ2         | 24.70 | 24.67 | 24.67 | 24.67 | 0.03    |

- SSIM:

  | model         | whole  | none   | patch  | block  | Δ(skip) |
  |---------------|--------|--------|--------|--------|---------|
  | ResSHyp λ1000 | 0.9666 | 0.9627 | 0.9628 | 0.9628 | 0.0039  |
  | ResSHyp λ20   | 0.9210 | 0.9190 | 0.9189 | 0.9190 | 0.0020  |
  | ResSHyp λ2    | 0.6540 | 0.6539 | 0.6539 | 0.6537 | 0.0001  |
  | FP λ1000      | 0.9592 | 0.9555 | 0.9556 | 0.9556 | 0.0037  |
  | FP λ20        | 0.9348 | 0.9323 | 0.9327 | 0.9327 | 0.0025  |
  | FP λ2         | 0.8464 | 0.8456 | 0.8454 | 0.8457 | 0.0008  |

  **Conclusion → skip symmetrization** (definitive: 2 archs × 3 rates × full scene). Granularity is
  irrelevant everywhere (none ≈ patch ≈ block within ≤0.05 dB). Dropping whole-image symmetrization
  costs **at most 0.54 dB** (ResSHyp λ1000) and **shrinks with compression — only ~0.03 dB at λ2**,
  i.e. negligible at the operating points a compression paper cares about. The pipeline drops the
  symmetrization stage; an optional one-time whole-image pre-pass reclaims the ≤0.54 dB if ever
  wanted. (An earlier 4096² urban-region run agreed; absolute PSNR is higher on the full scene as it
  includes homogeneous terrain — the Δ is what matters.) Runs: `results/symmetrization_study/*_full.json`.

> **linA convention (single source of truth).** `x_hat` is 2-channel `[B,2,H,W]` (real, imag);
> `linA = sqrt(0.5·(exp(x0·Δ+AMP_MIN)² + exp(x1·Δ+AMP_MIN)²))`, Δ = AMP_MAX−AMP_MIN — as in
> `create_dataset.py::_predict_linA` (the code that generated the MERLIN/ADAM GT), now extracted to
> `src/utils/reconstruction.py` and shared by `evaluate.py` / `create_dataset.py` / the study — **fixed**.

---

## 7. `.ddc` downlink product (v1 — locked)

Reference codec: **`src/utils/ddc_format.py`** (authoritative spec-in-code); self-test:
`scripts/evaluation/ddc_selftest.py`. Little-endian, positional. A `.ddc` = **header** (how to
decode) + **body** of per-patch rANS bitstreams + optional **trailer** offset table. Nothing
model-specific (CDF tables, weights) is embedded — the ground station has the decoder + CDFs; the
header only *references* them.

- **Header** (46 fixed bytes + two length-prefixed UTF-8 strings): `magic "DDC1"` · `flags`
  (bit0 = trailer present) · `arch_id` (0 FP/1 ResFP/2 SHyp/3 ResSHyp) · `N`/`M` · `patch`/`stride`
  · `scene_H`/`scene_W` · `grid_r`/`grid_a` · `AMP_MIN/MAX/EPS` (f32) · `params_sha` (8 B guard) ·
  `tile_id` (TSX product name) · `model_id`.
- **Body** × (grid_r·grid_a), row-major: `len_z`(u32) · `z_bits` · `len_y`(u32) · `y_bits`
  (FP: `len_z` = 0).
- **Trailer** (if flags bit0): `grid_r·grid_a × u64` = byte offset of each patch record. Location is
  **derived, not stored**: `table_start = filesize − n·8` (n from the header) → O(1) random access /
  partial + prioritised downlink.

**Locked decisions:** latent shapes are *derived* from patch+arch (a dummy forward), not stored;
`params_sha` is a decodability *guard* (params shipped out-of-band, ground has the decoder+CDF);
little-endian; scene bound by `tile_id`. **Verified** (self-test, ResSHyp + FP): header/body/trailer
round-trip byte-exact, random access matches, decode(file) ≈ decode(direct) within float32 ε. bpp
sanity 1.95 (ResSHyp λ1000) / 0.26 (FP λ20); container overhead ≈ header + 8·n bytes (negligible).
The C++ `stream_seq` writer (step 3) must emit these exact bytes; the Python codec is the oracle.

> **Reviewed + hardened (2026-07-31).** Independent review confirmed the happy-path round-trip (host
> `test_ddc_io` + `ddc_cross_check.py` pass). **`params_sha` algorithm now pinned:** canonical =
> `ddc_format.py::params_guard` = **standard FNV-1a-64** over the sorted `entropy_params/*.npy` bytes
> (little-endian). Two latent bugs closed — the self-test hashed with SHA-256[:8] (never matched the
> board), and the C++ `fnv1a_params` offset basis was a typo (the canonical basis with its last digit
> dropped → non-standard). C++ constant corrected (written in hex) and cross-checked byte-for-byte
> against `params_guard` on real `entropy_params`. Malformed-`.ddc` reads (C++ + Python) now
> hard-error via bounds/length guards instead of over-reading or silently truncating.

---

## 8. Staying sane (anti-chaos safeguards for U3)

- **Named presets** (S0/S1-style), not arbitrary knob combos; knobs (`--queue-depth`,
  `--entropy-threads`) validated at startup, **hard-error on nonsensical combos** (errors-over-fallbacks).
- **Correctness gate:** every schedule must emit **byte-identical** bitstreams to `stream_seq` (same
  gate that verified S0=S1), plus the one-time `.ddc` round-trip PSNR check. This is the anti-"BS
  results" safeguard — wrong schedules fail loudly.
- Streaming executor in its own module (`inference_cpp/src/benchmark/stream_pipeline.{cpp,hpp}`)
  reusing leaf stages; if the CLI ever tangles, splitting to a separate binary is cheap (shared functions).

---

## 9. Feasibility checklist

- [x] DDR / cores / SD capacity (§2).
- [x] Scene dims → patch count / footprints (§3).
- [x] SD read throughput — **23.5 MB/s** cold (dd, cache-dropped, 3.5).
- [x] SD write throughput — ~0.1 ms/patch for the `.ddc` (3.5).
- [x] E1 symmetrization study (§6) — done; symmetrization dropped.
- [x] `.ddc` format + Python codec/verifier (§7) — done; self-test passes on ResSHyp/FP.
- [x] per-patch stage timing (3.5) — DPU 75 / entropy 12 / normalize 9 ms; DPU-bound.
- [x] correctness gate — decode == `inference_hybrid` (MSE=0); windowed == whole byte-identical.

---

## 10. Staged plan

1. ✅ **E1** (local GPU) — done; symmetrization dropped (§6).
2. ✅ **`.ddc` format + Python codec/verifier** — done; v1 locked + verified (§7).
3. ✅ **`stream_pipeline`** on-board — tile → DPU compress → `.ddc` → decode. **Correctness-gated:**
   decoded recon is **bit-identical** to `inference_hybrid` (MSE=0.0, SSIM=1.0, identical bpp) on
   ResSHyp. New: `inference_cpp/src/{ddc_io.hpp, tile_source.hpp, stream/}` + host tests
   `test_ddc_io`/`test_tile_source`.
   ✅ **3.5 baseline** (1024 patches, ResSHyp, **int16 tile streamed row-block by row-block**):
   **10 patch/s**, 100 ms/patch — **DPU 75% / entropy 12% / normalize 9% / I/O ~1%**; windowed ==
   whole-tile **byte-identical** (+1.5% time, 16× less DDR: 33 MB row-block vs 536 MB whole);
   bpp 2.08 → **15.4× vs raw int16**; cold SD read 23.5 MB/s. Full scene ≈ 12 min / 124 MB (projected).
   **DPU-bound → pipelining ceiling ~1.33×** (that's step 4's headroom).
4. **Step 4 (parallelism), ResSHyp 1024 patches.** ✅ **S1** (g_a‖g_a, 2 cores) → 16.1 patch/s
   (1.61×); ✅ **p0** (S1 + CPU‖DPU worker-pool overlap; DPU serialized by a mutex; records placed by
   index) → **21.1 patch/s (2.11×)** at 3–4 threads, **byte-identical** to sequential (no race);
   DPU-serialized ceiling ~25/s (84% reached). The `--threads` knob *is* the "fine" tuning — ResSHyp
   plateaus at ~3 (DPU-bound). **FP (CPU-bound) scales the opposite way:** seq 37 → s1 43 (1.18×) →
   **p0 116 patch/s (3.16×)**, still climbing t2→t4 (+52% vs ResSHyp's +9%). **Parallelism is
   arch-dependent:** ResSHyp wants S1 (DPU channel-parallel), FP wants p0 threads (CPU-parallel);
   both byte-identical. Full scene @ best p0: ResSHyp ~5.8 min, FP ~1.0 min.

   ✅ **Full-scene headline (Option 1 — 7,296 patches, `--p0 --s1 --windowed --threads 4`, measured on board):**

   | arch (λ=1000) | throughput | full-tile latency | `.ddc` | ratio vs raw int16 | bpp | peak DDR |
   | --- | --- | --- | --- | --- | --- | --- |
   | FP (CPU-bound) | 53 patch/s (131 compute-only) | 2.28 min = 81 s cold SD read + 56 s compute | 88.9 MB | **21.8×** | 1.485 | 272 MB |
   | ResSHyp (DPU-bound) | 22 patch/s | 5.61 min (7 s warm read; **~6.85 min** cold) | 74.0 MB | **26.1×** | 1.237 | 307 MB |

   Windowed read holds only a **30 MB row-block + compressed records → peak DDR ~0.3 GB** (board stays
   ~2.8 GB free). The whole f32 tile is 3.87 GB > 3 GB DDR → **windowed streaming is required, not an
   optimization** (whole-load is OOM-killed). Measured compute matches the step-4 projection (FP 0.93 min,
   ResSHyp 5.5 min); the ~81 s cold SD read (23.8 MB/s) is the end-to-end add-on. Full-scene bpp (1.24–1.48)
   beats the 1024 crop (2.08) — more low-texture area → higher ratio. raw int16 SLC = 1.93 GB (`.cos` payload).
5. ✅ **Largely folded into Step 4.** Full 7 296-patch scene streamed from SD (headline above); the
   `--threads` worker sweep *is* the fine-grained sweep, so a separate `stream_fine` was not needed.
   Extra `--queue-depth`/`--entropy-threads` knobs left unbuilt (prefetch depth fixed at 2 — not a
   bottleneck; DPU-serialization, not queueing, is the limit).
6. Overlap + reconstructed-tile quality.
7. **On-ground decode (nice-to-have):** reproduce the on-board INT8 `h_s` on host so SHyp `.ddc`
   files decode in pure Python — until then, SHyp verification is board-side (the FP path already
   decodes in Python; see the INT8-scales note in §7). Handy for a ground station, not required.

> **Step-3 review — done (2026-07-31).** Independent agent audited `ddc_io.hpp`, `tile_source.hpp`,
> `stream/` (+ the `ddc_format.py` oracle). Happy path validated (byte-identical gate + `.ddc`
> round-trip hold); it independently cleared the FP `len_z=0` path and the full-scene 64-bit
> `tile_source` offsets (all `size_t`, no truncation). **Host-verified fixes applied:**
> malformed-`.ddc` bounds/length guards (C++ `read_ddc` + Python reads) + reserve cap; `params_sha`
> canonicalized to standard FNV-1a-64 (§7, incl. the C++ typo fix, cross-checked on real params);
> negative test added to `test_ddc_io`.
>
> **Board bridge — done + verified on board (2026-07-31):** (a) C++ FNV constant fix, (b) decode-side
> guards (arch-class + `params_sha`) in `stream_decode_ddc`, (c) de-duplicated the shared
> writer/header-build between `stream_compress_tile`/`_p0` (6 helpers; per-patch compute loops left
> inline). Gate — ResSHyp λ1000, 128-patch region of `stream_tile_1k_i16`: new `seq` ≡ new `p0+s1`
> (byte-identical); new `seq` vs the pre-rebuild reference differs in **exactly** the 8 `params_sha`
> bytes and nothing else (dedup changed nothing; the FNV fix is the sole output change); board
> `params_sha` = host `params_guard` = `24ee38ee6ec605f2`; decode passes the new guards. Deferred/
> flagged (not fixed): `--max-rows` `scene_H` metadata, u16 length wrap, host-decode whole-scene RAM,
> `tile_source` non-LE-host nit; FP/int16 host-test coverage.

> **Done (2026-08-01) — ablation sweep + table.** `stream_sweep.py` (deploy each model → cumulative
> configs cold + warm-final via the harness) + `stream_table.py` → `results/benchmark_stream/`
> `ablation_table.md`. FP + ResSHyp × λ{1000,20}, full scene. **λ-independent** (L20 == L1000 within
> ~1% everywhere → rANS time ∝ #latents, not bpp; one λ suffices for throughput). Cumulative cold:
> **FP 26→86 patch/s (3.3×), ResSHyp 9.2→22.8 (2.5×)**. Gains are arch-specific (ResSHyp ← s1 halves
> `g_a`; FP ← p0 then prefetch); **NEON ~flat on top of p0** (normalize already overlapped). **Read
> ceiling:** SD ~81 s / ~24 MB/s constant; prefetch defeats the *serial* read (FP p0 137→85 s), then
> FP is read-bound at the SD wall; warm (RAM read) → FP 85→53 s (**1.6× headroom**), ResSHyp warm ≈
> cold (read fully hidden by its long compute → **no** storage headroom). Energy column → power TODO.

> **Done (2026-07-31) — NEON `normalize`/`denorm` (`--neon`).** Vectorised log/exp (Cephes/Pommier,
> `neon_mathfun.h`) behind a runtime flag; scalar path kept for A/B + rollback. Kernel self-check
> (`--neon-check`) max rel err vs libm = 7e-8. **normalize 2.42× faster** (1150→475 ms/128 patches,
> ResSHyp). **Byte-transparent encode:** scalar vs `--neon` `.ddc` byte-identical (7e-8 log error ≪
> the INT8 g_a step 1/64 → no quantisation flips). `denorm` (double→float32, decode-side) changes the
> recon by only ~146 dB PSNR (SSIM 1.00000; identical vs MERLIN 22.16 dB) — float noise. Validated on
> the Hamburg region; metrics + log-intensity panels via `scripts/evaluation/compare_recon.py`. FP
> (CPU-bound, normalize a bigger fraction) should gain more — quantify in the Phase 5 sweep.

> **Done (2026-08-01) — Gantt figures.** `stream_gantt.py --model <M> --detail {merged,full}`: seq /
> s1 / p0(+s1) + a coarse row-block **streaming** panel (read‖compute‖write), from measured per-stage
> means. p0 shows *normalize-early → WAIT (mutex) → DPU0/DPU1 serialized*; the streaming panel shows
> ResSHyp **compute-bound** (read hidden under compute) vs FP **read-bound** (reader packed, compute
> waits). ResSHyp + FP, both detail levels.

> **Done (2026-07-31) — double-buffer (`--prefetch`, windowed seq + p0).** A producer thread reads
> row-block N+1 while the compressor works block N (bounded `RowBlockQueue`, depth 2). Verified on
> board: **byte-transparent** — seq / p0 with and without `--prefetch` are all byte-identical — and
> cold-read `total` drops as the SD read hides behind compute (ResSHyp λ1000, 128-patch region: p0+s1
> 9.9→8.8 s, seq 16.4→15.4 s; the ~1 s hidden ≈ the region's ~1.4 s cold read minus block 0, which
> has nothing to overlap). Scales to the headline: on the full scene the ~81 s cold read hides fully
> behind ResSHyp compute (compute-bound) and lifts **FP toward its ~34 MB/s ceiling** (else read-bound
> at the SD's ~24 MB/s). Full FP + full-scene cold/warm quantification → Phase 5 sweep (needs the
> cold/warm harness below).

> **Done (2026-07-31) — cold/warm harness** `scripts/fpga/benchmark/stream_benchmark.py` (host-side,
> drives `stream_pipeline` over SSH). **Cold is the default:** `sync; echo 3 > /proc/sys/vm/drop_caches`
> before *each* timed run (board is root) — never by hand. `--keep-cache` runs WARM (skips the drop) to
> simulate a much faster persistent store (read from RAM), the compute-ceiling number. iters→median;
> reports patch/s, **SLC MB/s** (the objective's data/s), full-tile latency, bpp; writes
> `results/benchmark_stream/<model>/<label>.json`. Validated (ResSHyp p0+s1+pf, 128-patch region):
> cold 8.80 s vs warm 6.30 s — the ~2.5 s gap is the cold SD read, reproducible across iters.
> ⚠️ small-region totals include the one-time ~1 s model load (`t0` precedes model construction);
> negligible at full scene — use `--max-rows -1` for headline numbers.

### Other TODOs, user-added

- @TODO (figures): first pass **built** — `stream_gantt.py` (seq/s1/p0/streaming timelines),
  `stream_roofline.py` (DPU-kernel roofline, plan A), `stream_sysplot.py` (system views: (i)
  throughput-vs-ladder + (ii) OP/byte roofline, plan B). Remaining for the manuscript batch: legibility
  polish, a dedicated read-ceiling figure, and deciding whether to surface CPU load explicitly (see the
  roofline caveat below).
- @TODO (sweep): ✅ **done** — power implemented first, then one consolidated sweep run
  (`stream_sweep.py`, FP + ResSHyp × λ{1000,20}); the table lands throughput + latency + energy
  together (results in the ablation-sweep note above).
- @TODO (power): ✅ **implemented** — `--power` in `stream_pipeline` (+ harness) reuses `PowerSampler`
  (INA226 sysfs + PMBus I2C) across the compress phase; reports MPSoC (PS+PL) avg-W, total J, and
  **J/patch** + per-group means into `StreamResult`/JSON. First test (ResSHyp, 128 patches): seq
  11.6 W / 1.26 J/patch vs p0+s1 15.9 W / **0.73 J/patch** — p0 draws more power but ~1.7× less
  energy/patch. → energy column in the consolidated sweep.
- @TODO (memory): plan = **(a)** peak footprint via `VmHWM` (`/proc/self/status`) — confirms the
  windowed ~0.3 GB; **(b)** analytical data-movement budget (SD→DDR 1.9 GB, patchify memcpy ~3.8 GB,
  DPU DMA, DDR→SD ~90 MB) → rates vs the 17.06 GB/s (DDR4-2133 ×64b) DDR ceiling; **(c)** `vaitrace --txt_summary` for
  the DPU's per-subgraph DDR traffic — `LdFM` (feature-map load MB), `LdWB` (weight/bias load MB),
  `StFM` (feature-map store MB), `AvgBw` (avg DDR bandwidth). **DPU `AvgBw` + a CPU-side estimate =
  worst-case simultaneous DDR demand** (p0 overlaps them) → the DDR **margin** vs 17.06 GB/s (DDR4-2133 ×64b).
  Measured so far (vaitrace): DPU g_a/g_s `AvgBw` ~651–660 MB/s (dominant), h_a/h_s bursts ~6 GB/s;
  CPU side is an **estimate** (~100–150 MB/s from memcpy+normalize volumes ÷ time). @TODO (if the
  paper needs a defensible CPU DDR number): measure it with `perf stat -e l2d_cache_refill,l2d_cache_wb`
  (×64 B ÷ runtime) — but `perf` is **not installed on the board** (would need adding). AXI
  Performance Monitor (whole-system DDR counters) = overkill, skip. Refs: vaitrace UG1414 —
  <https://docs.amd.com/r/en-US/ug1414-vitis-ai/vaitrace-Usage>,
  <https://docs.amd.com/r/en-US/ug1414-vitis-ai/Text-Summary>.
- @TODO (roofline): ✅ **both built** (vaitrace-measured, s0 L1000). **(A) DPU-kernel** (Williams,
  `stream_roofline.py`): x = `WL/(LdFM+LdWB+StFM)` OP/byte, y = `WL/HW_RT` GOP/s; roofs = B4096 @
  300 MHz = 1229 GOP/s + DDR 17.06 GB/s (ridge 72 OP/byte). *Findings:* g_a/g_s ride **468–1822
  OP/byte** — far right of the ridge, hard **compute-bound** at **70–97%** of the roof (the gap =
  DPU overhead; ResSHyp's residual g_a/g_s hit 94–97%). h_a/h_s sit at **~57 OP/byte** (just left of
  the ridge) at only **~27%** → tiny + weight-load-bound. **FP g_a/g_s are separate points** (residual
  connections make ResSHyp's ~9× larger). **(B) system** (`stream_sysplot.py`): (i) throughput-vs-ladder
  with the SD-read ceiling + per-model warm lines; (ii) OP/byte roofline (x = compress DPU OP per SLC
  byte, y = achieved DPU OP/s) with SD-read (diagonal) + DPU (2-core) roofs — **FP (low intensity)
  rides the SD-read roof (read-bound); ResSHyp (high intensity) sits near the compute roof
  (compute-bound); the warm `*` breaks the SD roof.**
  > ⚠️ **CPU-load caveat (all three figures are DPU + SD-read only).** None decomposes the CPU
  > (entropy + normalize) load: (A) is pure-DPU by construction; (ii)'s only compute roof is the DPU;
  > (i) bakes the CPU into the *measured* throughputs + the warm ceiling but never draws it. The CPU
  > shows up **only implicitly** — and for **FP the warm/compute ceiling *is* the CPU entropy limit**
  > (FP is CPU-bound). To surface CPU load explicitly, two options for the manuscript batch: **(1)** a
  > per-model **compute roof at the warm throughput** on (ii) = the `min(DPU, CPU)` compute bound (folds
  > the CPU in as the effective roof), or **(2)** a separate **stacked time-per-patch** figure
  > (read / DPU / normalize / entropy) — the honest, direct view of where CPU time goes and how NEON
  > shrinks the normalize slice. Recommend (2): the roofline is a single-resource (DPU) tool, so the
  > stacked-time figure is the right complement for the CPU story.

## TerraSAR-X objective (full derivation → `docs/TerraSAR-X_objective.md`)

The mission-throughput objective — *how much SLC a StripMap acquisition produces, how fast we must
process it, and whether our compression makes it downlinkable* — is derived from cited TSX specs in
**`docs/TerraSAR-X_objective.md`** (novice-friendly, page-referenced; reference PDFs in `docs/references/`).
Headline numbers the rest of this doc refers to:

| StripMap SM, 100% duty | working point (real Hamburg scene) | **worst case (headline)** |
| --- | --- | --- |
| incidence / PRF / N_r | 26° / 3600 Hz / 14 686 | **45° / 3800 Hz / 23 570** |
| SLC acq rate (int16 4 B/px) | 211 MB/s | **358 MB/s** |
| 180 s take | 38 GB | **64.5 GB** |
| raw rate → SSMM (BAQ 8:4) | 423 Mbps | 717 Mbps (cf. [Pitz] StripMap-mean 580 Mbps) |

Against this, one ZCU102 `p0` (FP 34.4 MB/s / ResSHyp 5.8 MB/s of SLC) is **~10× short of full-duty
real-time** but **meets the process-before-next-contact deadline** (FP, ~3× headroom) and keeps the
compressed product (~24×) well inside the 270 Mb/s-net downlink. Platform constants: downlink 270 Mb/s
net / 300 gross; SSMM 384 Gbit BOL / 256 EOL; ground swath 30 km. Full tables + the three-deadline
breakdown are in the objective doc §4–5; mission context (power, contacts) in its §9.
