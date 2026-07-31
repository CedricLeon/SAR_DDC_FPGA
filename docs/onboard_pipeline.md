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

Status: **planning + feasibility** (nothing implemented). Last updated 2026-07-28.

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

Sources: [UG1182 ZCU102 Eval Board UG](https://docs.amd.com/v/u/en-US/ug1182-zcu102-eval-bd) ·
[DS891 Zynq UltraScale+ Data Sheet](https://www.mouser.com/datasheet/2/903/ds891_zynq_ultrascale_plus_overview-1662253.pdf).

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
| 3 normalize | CPU | log + min/max (~18.7 ms/patch; NEON target) |
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

- `stream_seq` — 1 thread; correctness + latency baseline (reads tile, writes `.ddc`).
- `stream_p0` — 2-lane: DPU lane ‖ CPU lane (normalize + entropy + I/O). **First target.**
- `stream_fine` — per-stage workers (patchify / normalize / entropy / writer) — the narrative
  "fine-grained" pipeline; measured against `stream_p0` to show where extra threads actually help.

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

> **TODO (user review):** `src/utils/ddc_format.py` + `scripts/evaluation/ddc_selftest.py` were
> committed unreviewed — independently sanity-check the byte layout and the self-test assertions.

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
5. **`stream_fine`** + `--queue-depth`/`--entropy-threads` sweeps; full 7 296-patch scene streamed from SD.
6. Overlap + reconstructed-tile quality.
7. **On-ground decode (nice-to-have):** reproduce the on-board INT8 `h_s` on host so SHyp `.ddc`
   files decode in pure Python — until then, SHyp verification is board-side (the FP path already
   decodes in Python; see the INT8-scales note in §7). Handy for a ground station, not required.

> **TODO (review agent):** commission an independent code-review agent over the Step-3 C++
> (`ddc_io.hpp`, `tile_source.hpp`, `stream/`) — flag convoluted/redundant implementations and
> untested cases (FP path, full-scene 64-bit overflow, error handling, endianness/edge values,
> the FNV params hash) before Step 3 graduates.

> **TODO (paper):** full 7,296-patch scene — **seq vs s1 vs p0(best-threads)** — for all archs ×
> λ{1000,20,2} → throughput / full-tile latency / energy table. Batch it (ResSHyp seq ≈ 12 min/run).

> **TODO (CPU opt):** NEON-vectorize `normalize`/`denorm` (log/exp) — the CPU bottleneck that
> dominates FP (~33% of its per-patch time; would lift FP p0 throughput). See FPGA_inference.md §9.

> **TODO (figure):** Gantt-style timeline diagrams (stages × threads) for seq / s1 / p0 (and B) —
> to communicate the schedules in the paper.

> **TODO (overlap — double-buffer):** prefetch row-block N+1 while compressing block N (currently
> read then process, serialized). Hides the SD read behind compute: lifts **FP 14→24 MB/s** (then
> read-bound at the SD ceiling) and **fully hides ResSHyp's read** (compute-bound, 5.8 MB/s). Single
> biggest lever for the realistic "data/s" number. Pretending faster persistent storage then lifts FP
> toward its **34 MB/s compute ceiling** (the SD's 24 MB/s is our board's artifact, not fundamental).

> **TODO (harness — cold/warm read):** the realistic-scenario runner drops the page cache at the start
> of each timed run **inside the script** (`echo 3 > /proc/sys/vm/drop_caches`, needs root) — never by
> hand (we'd forget). **Cold is the default** (honest + reproducible, matches the real
> acquire→focus→store→read flow: the focuser is likely a separate board, so its SLC is genuinely on
> persistent storage, not in our RAM). Add a `--keep-cache` / warm flag that **skips the drop** to
> *simulate a much faster persistent store* (the read then comes from RAM ≈ removing the SD bottleneck),
> giving the FP compute-ceiling number alongside the cold floor.

> **Open (U2, storage format):** the realistic "SLC on the SD" is complex **int16** (4 B/px, the
> `.cos` payload) — our f32 `.npy` is a 2× convenience. Resolve as part of 3.5 (int16 vs f32 read).

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
