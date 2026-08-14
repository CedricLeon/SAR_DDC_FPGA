# Onboard Streaming Pipeline — Plan & Feasibility

> End-to-end **"receive focused SLC tile → despeckle + compress → write downlink bitstream"**
> streaming demo on the Xilinx ZCU102 — the core contribution of the systems/CS paper, alongside
> the performance analysis. This file **was the design discussion CLAUDE.md requires before
> implementing parallelism, and now records the implemented design + measurements**. It keeps only
> the *decisions* and their rationale; for existing pieces it points to the canonical docs.
>
> Builds on: `FPGA_inference.md` §5 (per-stage pipeline), §9 (streaming inference — earmarked);
> `FPGA_benchmark.md` §10 (P0/P2 patch-pipelining — earmarked there; **P0 now implemented +
> measured here**); `Data.md` (`.cos` source, normalisation).

Status: **implemented + measured.** The on-board streaming compressor, the parallel/I/O optimizations
(§4), the full-scene throughput/latency/energy sweep + memory/roofline analysis (§8), and the overlap
study (§10), and the DPU fan-out core-scaling (§11-N1) are done and board-verified. Remaining (§11): the
DATE'27 experiments (N6 four-arch ladder, N2 Jetson, N5 CCSDS) and figure polish. Last updated 2026-08-12.

**Questions opened/answered**:

- [x] **U1**: Symmetrization is a whole-tile pre-processing step, so it breaks the streaming flow. Can
  we skip it, or drop its granularity to a patch or large block? *We can skip it* — see §5 (E1).
- [x] **U2**: Which file format and layout for the stored tile? *Row-major int16 `.npy`, seek-streamed
  by `TileWindowReader` (no per-patch open)*; patchify (row-strided extract) costs ~0.4% of latency. A
  patch-major format `[n,256,256,2]` would zero that but fragments SD reads and complicates overlap.
  *Data type: **int16**, 4 B per complex sample (I+Q).*
- [x] **U3**: How do we stay organized given all the configs/setups? *See §7 — one base-schedule preset
  + composable flags, all held to a byte-identical gate.*
- [x] **U4**: How do we store the compressed patches + tile? *A `.ddc` container format — see §6.*
- [x] **U5**: Non-overlapping patch compression leaves seam/edge artifacts when the tile is recombined.
  How do we prevent that? *Answered — overlap 2 closes the seam at ~1 % cost (§10); reconstruct there.*

---

## 1. Goal & scope

- **Deliverable:** a *configurable, reproducible experimental harness* (not a shipped binary) to
  explore pipeline/thread schedules for the onboard scenario, and measure **steady-state throughput
  (patch/s)** + **full-tile latency** (read → compress → write).
- **Models:** ResSHyp (DPU-bound) + FP (CPU-bound) primary; SHyp/ResFP come ~free (arch
  auto-detected from `manifest.json`).
- **Overlap:** studied (§10) — reconstruct at overlap 2 (closes the seam at ~1 % cost).
- **Language:** C++ on-board (only inference path). Python only for the offline symmetrization study
  (§5) and the `.ddc` verifier (§6).

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

- **4× A53 cores** (`nproc=4`) → parallel-worker budget, shared with DPU dispatch + OS.
- Queues between stages are a non-issue: **~50 MB at depth 16** vs ~3 GB free.
- **SD sequential read ≈ 23.5–23.8 MB/s** (measured cold: 1.93 GB ÷ 81 s) — the **Step-1 read ceiling**.
  `eth0` is GbE (1000 Mb/s ≈ 5× the SD), but reading the tile from the host **breaks the onboard premise**. To emulate a *faster persistent store* we use a warm read.
- **Page cache** (4 KB pages, confirmed): Our 1.93 GB tile ≈ **472k pages**; whole-tile *warm* read is fine, but whole-tile *f32 in-process load* (3.87 GB) OOMs.

- **PS DDR4 peak bandwidth = 17.06 GB/s** — 4 GB DDR4-2133 SODIMM (Kingston KVR21SE15S8/4), 64-bit
  (2133 MT/s × 8 B) [UG1182 + SODIMM part]. Sustained DDR traffic (SD read + DPU DMA + memcpy) sits
  ≈20× below this → DDR is **not** a bottleneck (vaitrace-measured; §8).
- **DPU = 3× DPUCZDX8G B4096 @ 300 MHz → 1229 GOP/s per core** (4096 ops/cycle × 0.30 GHz; the guide
  lists 1400 @ 350 MHz) [PG338, *DPUCZDX8G Peak Performance*; clock from `xdputil query`]. Roofline
  ridge vs DDR = 1229 ÷ 17.06 = 72 OP/byte.

**Concepts.** *Queue* = bounded producer→consumer FIFO between threaded stages (depth = max buffered patches; overlap = stage B on patch N while A makes N+1). One slot is the minimum for overlap; a small depth (a few) absorbs per-patch DPU/rANS jitter so the bottleneck never stalls.

Sources: [UG1182 ZCU102 Eval Board UG](https://docs.amd.com/v/u/en-US/ug1182-zcu102-eval-bd) ·
[DS891 Zynq UltraScale+ Data Sheet](https://www.mouser.com/datasheet/2/903/ds891_zynq_ultrascale_plus_overview-1662253.pdf) ·
[PG338 DPUCZDX8G Peak Performance](https://docs.amd.com/r/en-US/pg338-dpu/DPUCZDX8G-Peak-Performance).

---

## 3. Data & acquisition geometry

- Tile: `data/TSX_cos_files/Hamburg_…_strip_004.cos` = **14 686 (range) × 32 901 (azimuth)** →
  **58 × 129 = 7 482** 256² patches on the snap grid (the last patch in each axis is flush to the edge,
  overlapping its neighbour by the remainder; a plain floor grid gives 57 × 128 = 7 296 and drops the far
  edge sliver); raw complex-int16 = **1.93 GB**.
- Footprints: full f32 `[H,W,2]` tile = **3.87 GB** (> DDR → must stream); a ~1000-patch region ≈ 524 MB.
- **Streaming axis = azimuth.** Range bins (14 686 cols) arrive ~together per radar pulse (fast-time);
  azimuth lines (32 901 rows) accumulate as the platform flies (slow-time). → natural streaming unit
  = **row-block** = 256 azimuth lines × full range = **one patch-row (58 patches)**; **129 row-blocks**
  per scene (the last row-block and the last patch of each row are snapped partials).
- Reuse: `load_cosar`, `symmetrize`, `extract_patches` (`src/utils/sar_utils.py`, Python); C++
  `npy_io` already reads `[H,W,2]` tiles.

---

## 4. Pipeline & optimizations

| Step | Unit | Notes |
| --- | --- | --- |
| 1 read tile | SD → DDR | whole (region) or **row-block stream** (full scene) |
| ~~1.5 symmetrize~~ | — | **dropped** (see §5/E1: ≤0.54 dB cost); optional one-time whole-image pre-pass if ever wanted |
| 2 patchify | CPU | strided per-row `memcpy` of `[256,256,2]` out of the DDR row-block |
| 3 normalize | CPU | log + min/max (~9 ms/patch; `--neon` = 2.42× faster, byte-identical) |
| 4 g_a×2 | DPU | already S1-parallel (1.95×) |
| 5 h_a/EB/h_s | DPU+CPU | SHyp only (FP skips) |
| 6 entropy | CPU | rANS → bits |
| 7 write bits | DDR → SD | generate `.ddc` product per tile (§6) |


**Optimizations.** Everything below layers on the sequential `seq` baseline (the step table above) and
is **gated byte-identical to it** (correctness gate §7): any schedule that changes a single output byte
fails loudly, and the decoded recon is bit-identical to `inference_hybrid` (MSE=0). So the knobs are
purely about *speed and energy*, never quality. Measured effects → §8.

- `seq` — single thread; correctness + latency baseline (reads the tile, writes the `.ddc`).
- **`--s1` — DPU channel-parallel.** Runs `g_a(re)` and `g_a(im)` on the two DPU cores at once. The
  main lever for the DPU-bound ResSHyp; ~free for FP.
- **`--p0 --threads K` — worker pool.** K workers each run normalize → DPU (serialized by a mutex) →
  entropy → write, with records placed by patch index (deterministic output). The main lever for the
  CPU-bound FP (scales to 4 workers); ResSHyp plateaus at ~3 (DPU-serialized).
- **`--fanout` — DPU data-parallel lanes (N1).** `--p0` modifier: K *independent* pipelines, one DPU
  core per worker, the mutex dropped — so `g_a` runs on all 3 cores across patches (vs `--p0`'s single
  serialized DPU). The DPU-bound lever beyond `--s1` (ResSHyp **2.85×** at 3 lanes; oversubscribing to 4
  regresses — §11-N1). Excludes `--s1` (both fight for the same 3 cores).
- **`--prefetch` — double-buffer I/O.** A producer thread reads row-block N+1 while the workers
  compress block N (bounded `RowBlockQueue`, depth 2). Byte-transparent; hides the SD read behind
  compute — fully for ResSHyp, partially for FP (toward its read ceiling).
- **`--neon` — vectorized normalize/denorm.** NEON log/exp (Cephes/Pommier, `neon_mathfun.h`) behind a
  runtime flag, scalar path kept for A/B. Kernel error vs libm = 7e-8 → **byte-transparent encode** (≪
  the INT8 `g_a` step, so no quantisation flips). 2.42× faster normalize in isolation.
- **`--power` — energy instrumentation.** `PowerSampler` (INA226 sysfs + PMBus) wraps the compress
  phase → MPSoC (PS+PL) avg-W, total J, and **J/patch**.
- **cold/warm harness** (`stream_benchmark.py`, host-side over SSH). Drops the page cache before each
  run (`sync && echo 3 > /proc/sys/vm/drop_caches`) so **cold** is the honest SD-read number;
  `--keep-cache` runs **warm** (tile served from RAM) = the compute ceiling if storage were fast. The
  measurement methodology behind every number in §8.

---

## 5. Experiment E1 — symmetrization granularity (local GPU, no retraining)

**Why:** `symmetrize()` is a whole-image FFT → would break streaming. Its purpose is to **decorrelate
the real/imag parts** so MERLIN's Noise2Noise training can treat them as i.i.d. **Hypothesis:** at
inference the residual Re/Im correlation is small, so skipping symmetrization — or coarsening it to a
patch/block — is only slightly suboptimal, and it can live *inside* the per-patch/per-block pipeline
(or be dropped entirely).

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

- **Result** — ResSHyp + FP × λ∈{2,20,1000}, seed 0, **whole scene (7 482 patches, snap grid §3)**, vs
  MERLIN GT PSNR:

  | model         | whole | none  | patch | block | Δ(skip) |
  |---------------|-------|-------|-------|-------|---------|
  | ResSHyp λ1000 | 31.68 | 31.14 | 31.14 | 31.14 | 0.54    |
  | ResSHyp λ20   | 28.65 | 28.40 | 28.44 | 28.45 | 0.25    |
  | ResSHyp λ2    | 22.98 | 22.95 | 22.97 | 22.97 | 0.03    |
  | FP λ1000      | 30.71 | 30.28 | 30.30 | 30.31 | 0.43    |
  | FP λ20        | 28.90 | 28.68 | 28.69 | 28.69 | 0.22    |
  | FP λ2         | 24.74 | 24.71 | 24.71 | 24.71 | 0.03    |

  *Δ(skip) = whole − none, from unrounded values (may differ ±0.01 from the rounded columns).*

- SSIM (**`AMP_LIN_99` fixed-`data_range` basis**, `src/utils/metrics.py`; lower than any
  pre-2026-08-05 SSIM figure, which predated the fixed-`data_range` fix `1bd79e8` — a metric-basis
  change, **not** a grid effect. Δ(skip), the load-bearing quantity, is basis-robust):

  | model         | whole  | none   | patch  | block  | Δ(skip) |
  |---------------|--------|--------|--------|--------|---------|
  | ResSHyp λ1000 | 0.9153 | 0.9062 | 0.9064 | 0.9065 | 0.0090  |
  | ResSHyp λ20   | 0.8360 | 0.8315 | 0.8319 | 0.8320 | 0.0045  |
  | ResSHyp λ2    | 0.6603 | 0.6601 | 0.6602 | 0.6602 | 0.0002  |
  | FP λ1000      | 0.8929 | 0.8854 | 0.8857 | 0.8858 | 0.0075  |
  | FP λ20        | 0.8379 | 0.8331 | 0.8335 | 0.8337 | 0.0047  |
  | FP λ2         | 0.7079 | 0.7071 | 0.7073 | 0.7074 | 0.0008  |

  **Conclusion → skip symmetrization**. Granularity is irrelevant everywhere (none ≈ patch ≈ block
  within ≤0.06 dB). Dropping whole-image symmetrization costs **at most 0.54 dB** (ResSHyp λ1000) and
  shrinks with compression — only ~0.03 dB at λ2. **The pipeline drops the symmetrization stage**; an
  optional one-time whole-image pre-pass reclaims the ≤0.54 dB if ever wanted.
  Runs: `results/symmetrization_study/*_full.json`.

---

## 6. `.ddc` downlink product (v1 — locked)

Reference codec: **`src/utils/ddc_format.py`** (authoritative spec-in-code); self-test:
`scripts/evaluation/ddc_selftest.py`. Little-endian, positional. A `.ddc` = **header** (how to
decode) + **body** of per-patch rANS bitstreams + optional **trailer** offset table. Nothing
model-specific (CDF tables, weights) is embedded — the ground station has the decoder + CDFs; the
header only *references* them.

- **Header** (46 fixed bytes + two length-prefixed UTF-8 strings): `magic "DDC1"` · `flags`
  (bit0 = trailer present) · `arch_id` (0 FP/1 ResFP/2 SHyp/3 ResSHyp) · `N`/`M` (model channels) · `patch`/`stride`
  · `scene_H`/`scene_W` · `grid_r`/`grid_a` · `AMP_MIN/MAX/EPS` (f32) · `params_sha` (8 B guard) ·
  `tile_id` (TSX product name) · `model_id`.
- **Body** × (grid_r·grid_a), row-major: `len_z`(u32) · `z_bits` · `len_y`(u32) · `y_bits`
  (FP: `len_z` = 0).
- **Trailer** (if flags bit0): `grid_r·grid_a × u64` = byte offset of each patch record. Location is
  **derived, not stored**: `table_start = filesize − n·8` (n from the header) → O(1) random access /
  partial + prioritised downlink.

**Locked decisions:** latent shapes are *derived* from patch+arch (a dummy forward), not stored;
`params_sha` is a decodability *guard* — **FNV-1a-64 over the sorted `entropy_params/*.npy` bytes
(little-endian)**, canonical impl `ddc_format.py::params_guard` (params shipped out-of-band; the ground
station has the decoder + CDFs); little-endian throughout; scene bound by `tile_id`. Malformed `.ddc`
reads **hard-error** (bounds/length guards, C++ + Python) rather than over-reading or silently
truncating. **Verified** (self-test, ResSHyp + FP): header/body/trailer round-trip byte-exact, random
access matches, decode(file) ≈ decode(direct) within float32 ε. bpp sanity 1.95 (ResSHyp λ1000) / 0.26
(FP λ20); container overhead ≈ header + 8·n bytes (negligible). The C++ streaming writer (step 7) must
emit these exact bytes; the Python codec is the oracle.

---

## 7. Staying sane (anti-chaos safeguards for U3)

- **One base preset + composable flags, not arbitrary knob combos.** The harness picks a base executor
  with `--schedule {seq,p0}` and layers composable modifiers on top (`--s1`, `--prefetch`, `--neon`,
  `--threads K` → C++ `stream_pipeline` flags); combos are validated at startup and **hard-error on
  nonsensical inputs** (errors-over-fallbacks).
- **Correctness gate:** every schedule must emit **byte-identical** bitstreams to `seq` (the same gate
  that verified S0=S1), plus the one-time `.ddc` round-trip PSNR check. Because the modifiers are
  composable flags, this gate — not the preset count — is what keeps results honest: wrong schedules
  fail loudly.
- Streaming executor in its own module (`inference_cpp/src/stream/` — `main_stream.cpp` CLI +
  `stream_pipeline.{cpp,hpp}`, binary `stream_pipeline`), reusing leaf stages; if the CLI ever tangles,
  splitting to a separate binary is cheap (shared functions).

---

## 8. Results

**The sweep.** `stream_sweep.py` deploys each model and runs the cumulative optimization ladder on the
full **7,482-patch** Hamburg scene (snap-covered grid, §3), **cold**, plus one **warm** run of the best
config, through the harness (§4); `stream_table.py` builds the table. Ran FP + ResSHyp × λ{1000, 20} and
found it **λ-independent** (L20 = L1000 within ~1% — rANS time scales with the *number of latents*, not
bpp), so one λ characterizes throughput. The table is **λ=20** (the ladder re-run on the snap grid; it
stands for λ=1000 too by λ-independence); the **compression** line is λ-specific and quoted at λ=1000.
Results in `results/benchmark_stream/<model>/*.json` + `ablation_table.md`.

| optimization | FP patch/s (MB/s) | FP latency | FP J/patch (W) | ResSHyp patch/s (MB/s) | ResSHyp latency | ResSHyp J/patch (W) |
| --- | --- | --- | --- | --- | --- | --- |
| seq | 26.7 (6.9) | 4.67 min | 0.359 (9.7) | 9.3 (2.4) | 13.38 min | 1.252 (11.7) |
| + s1 | 30.2 (7.8) | 4.13 min | 0.323 (9.8) | 14.0 (3.6) | 8.93 min | 0.944 (13.2) |
| + p0 | 52.8 (13.6) | 2.36 min | 0.200 (10.7) | 17.9 (4.6) | 6.98 min | 0.815 (14.7) |
| + prefetch | 86.5 (22.3) | 1.44 min | 0.137 (12.1) | 22.4 (5.8) | 5.56 min | 0.721 (16.3) |
| + neon | 86.6 (22.4) | 1.44 min | 0.137 (12.1) | 22.9 (5.9) | 5.43 min | 0.711 (16.4) |
| **warm (+neon)** | **132.5 (34.2)** | **0.94 min** | **0.104 (13.9)** | **23.1 (6.0)** | **5.41 min** | **0.711 (16.4)** |

- **Why "warm" is the last row.** Warm serves the tile from the RAM page cache (`--keep-cache`) instead
  of the SD card — it emulates a *faster persistent store* and isolates the **compute ceiling** from the
  SD-read bottleneck. It's not a further optimization, it's the "if storage weren't the limit" number:
  FP jumps 87→132 patch/s (read-bound → compute-bound), while ResSHyp is unchanged (already
  compute-bound, its read fully hidden).
- **Compression** (byte-identical across every config; quoted at λ=1000, snap grid): FP bpp 1.482 →
  **21.3× vs raw int16** (90.8 MB `.ddc`); ResSHyp bpp 1.232 → **25.6×** (75.5 MB). Full-scene ratios beat
  the 1024-crop (2.08 bpp) — more low-texture area.
- **Peak DDR ~0.3 GB** (windowed: a 30 MB row-block + compressed records) vs ~3 GB free. The whole f32
  tile is 3.87 GB > DDR → **windowed streaming is required, not optional** (whole-load is OOM-killed).
- Energy = MPSoC (PS+PL) INA226; **J/patch is the comparable metric**, avg W indicative (rises partly
  from thermal drift over the continuous sweep — configs measured in order).

**Findings.**

- **Parallelism is arch-dependent.** ResSHyp (DPU-bound) is unlocked by `--s1` (halves `g_a`); FP
  (CPU-bound) by `--p0` threads, then `--prefetch`. Cumulative cold: FP **27→87 patch/s (3.2×)**,
  ResSHyp **9.3→23 (2.5×)**. `--neon` is ~flat end-to-end (normalize already overlapped/hidden) — its
  win is the standalone 2.42× (§4), not throughput here.
- **Parallelism costs power but saves energy** — the speedup outpaces the extra draw. FP seq→neon
  **0.359→0.137 J/patch (2.6× better)** at +25% W; ResSHyp **1.252→0.711 (1.76×)** at +40% W.
  Cross-arch, ResSHyp costs **5.2× the energy/patch** of FP (≈ its ~9× OPs).
- **Read ceiling.** The cold SD read is ~81 s / ~24 MB/s, constant. `--prefetch` hides it behind
  compute: fully for ResSHyp (warm ≈ cold → no storage headroom), partially for FP (warm 1.5× faster →
  FP is read-bound at the SD wall).
- **DDR is not a bottleneck.** vaitrace: the dominant DPU traffic (`g_a`/`g_s`) is ~651–660 MB/s, ~20×
  under the 17.06 GB/s DDR4 ceiling (§2). CPU-side DDR is an estimate (~100–150 MB/s); a defensible
  measure would need `perf`, which isn't on the board (adding it = a host-side PetaLinux rootfs rebuild,
  and the Vitis-AI image would need its matching BSP) — so we keep the estimate; the ~20× margin holds
  either way.
- **vs the mission objective** (full derivation → `docs/TerraSAR-X_objective.md`): TSX StripMap produces
  SLC at **211 MB/s** (working point) / 358 MB/s (worst case). One ZCU102 at its best config (warm+neon:
  FP 34 MB/s / ResSHyp 6 MB/s) is **~10× short of full-duty real-time**, but **meets the
  process-before-next-contact deadline** (FP, ~3× headroom) and the ~24× compressed product sits well
  inside the 270 Mb/s-net downlink.

---

## 9. Figures

Three exploratory scripts under `scripts/fpga/benchmark/` (output → `results/benchmark_stream/`):

- **`stream_gantt.py`** — schedule timelines (seq / s1 / p0 / row-block streaming) from measured
  per-stage means. Shows p0's *normalize-early → wait-on-DPU-mutex → DPU serialized*, and the streaming
  read‖compute overlap (ResSHyp compute-bound, FP read-bound).
- **`stream_roofline.py`** — DPU-kernel roofline (Williams), per subgraph, vaitrace-measured. `g_a`/`g_s`
  are compute-bound (70–97% of the 1229 GOP/s roof, far right of the 72 OP/byte ridge); `h_a`/`h_s` are
  tiny + weight-load-bound (~57 OP/byte, ~27%).
- **`stream_sysplot.py`** — two system views: (i) throughput vs the optimization ladder with the SD-read
  + warm ceilings; (ii) an OP/byte roofline placing FP on the SD-read roof (read-bound) and ResSHyp near
  the compute roof.

> ⚠️ **First-pass sketches — feel, not final.** They build intuition but likely oversimplify; the Gantt
> and roofline both need a rigor + polish pass before the manuscript (audit the axes, the roofs, and what
> each point actually represents). → §11.
>
> ⚠️ **All three are DPU + SD-read only** — none decomposes the CPU (entropy + normalize) load: the
> roofline is DPU by construction, the system roofline's only compute roof is the DPU, and the throughput
> plot bakes the CPU into the numbers without drawing it (for FP, the warm ceiling *is* the CPU entropy
> limit). The dedicated CPU view is the stacked-time figure (done: Fig. 3).

---

## 10. Overlap study (U5) — seam vs cost

**Complete.** Full-scene sweep FP + ResSHyp × λ{20,1000} × overlap {0,2,4,8,16} (cold+warm), scored vs
the project's own MERLIN full-tile GT (`merlin_full_gt.py`; whole-image-symmetrized, 64-px seam-free
blend). Reconstruction = board INT8 decode (`stream_pipeline --decode`) + host ramp-blend/stitch
(`stitch_ddc.py`, `src/utils/tiling.py`); grid offsets `stride = 256 − overlap`, snapped to the edge;
blend + metrics in **linear amplitude** (same basis as §5). Cost measured on the §8 pipeline (directly
comparable). Data → `results/benchmark_stream_overlap/overlap_table.{md,csv}`.

**Seam metric.** Full-tile means dilute the seam ~21× (seam band = ±3 px of the 256-grid ≈ 4.6 % of
pixels), so the sensitive number is `PSNR_interior − PSNR_seam`.

**Result — overlap 2 closes the seam at ~1 % cost; more buys nothing.**

| config | seam Δ @ ov0 (dB) | seam Δ @ ov2 | SSIM seam→interior @ ov0 | full-tile PSNR ov0→ov16 |
| --- | --- | --- | --- | --- |
| FP λ1000 | 1.16 | 0.01 | 0.786 → 0.853 | 24.99 → 25.04 |
| FP λ20 | 1.16 | 0.03 | 0.742 → 0.807 | 24.19 → 24.29 |
| ResSHyp λ1000 | 2.75 | 0.03 | 0.811 → 0.889 | 27.48 → 27.72 |
| ResSHyp λ20 | 2.06 | 0.01 | 0.737 → 0.810 | 25.37 → 25.53 |

*(seam Δ = PSNR_interior − PSNR_seam; PSNR clipped to `AMP_LIN_99`.)*

**Cost of ov2:** latency +0.7–1.0 % warm / free cold-FP (read-bound); downlink +0.78 % (patches
7,482→7,540, bpp flat); J/patch flat. Overlap 16 costs +14–19 % latency / +14.35 % downlink for no
further seam gain → **reconstruct at overlap 2**.

**INT8 bright-scatterer cap (metric-invisible).** The DPU `g_s` output is INT8 at fix-point 8, so recon
amplitude hard-caps at **2100.1** (code 127 → `exp(0.496·range+min)`); the float model reaches ~85 k
(FP) / ~68 k (ResSHyp). This clips the brightest **~0.7 %** of pixels (point scatterers). Invisible to
the reported metrics (all clip to `AMP_LIN_99 = 545` first), which also sets the SSIM `data_range` (the
fixed, cross-model-comparable basis; the convention lives in `src/utils/metrics.py`). A fix-point-7 requant would
lift the cap to ~44 k at half precision, if ever needed.

---

## 11. TODO

### Next experiments — DATE'27 priority

**Guiding principle:** the `main.tex` story is written *conveniently* — run what is most informative,
let the results (not the outline) drive the narrative, and be ready for any experiment to resolve
*against* the story. Each entry notes the manuscript slot it *would* unblock (**→ main.tex …**) purely
as navigation, never as a hole that must be filled. Do these before figure-polish.

**N6 — Four-architecture streaming ladder (cheap; validates the topology→schedule policy).** *Status:
partly done — all four archs now have the **fan-out lane sweep** (§11-N1, `fanout_table.md`), which
already validates the topology→schedule policy across archs (residual sets scaling strength, hyperprior
sets the oversubscription cliff).* Still open: the **cumulative** optimization ladder
(seq→s1→p0→prefetch→neon) for **SHyp** + **ResFP** — only FP + ResSHyp have it today (§8) — via the same
`stream_sweep.py`; runtime-only, no new code.

**N1 — DPU fan-out: streaming recovers the third core; the patch-only regression does *not* transfer.
Done.** Supersedes `P3 (deferred)` in `FPGA_benchmark.md` §10. **→ main.tex §coretrap + §eval.** Kept
behind the `--fanout` runtime flag ([[project_ablation_table]]).

**What was built.** `--fanout` is a `--p0` modifier (`inference_cpp/src/stream/`): instead of K workers
sharing one pipeline behind a DPU mutex (plain `p0`), it creates **K independent `BenchPipeline`s — one
DPU lane per worker, mutex dropped** — so `g_a` runs on up to 3 cores concurrently, data-parallel across
patches (the compress path has no `g_s`, so the third core is only fillable *across* patches; reuses the
`run_nn_only` fan-out pattern inside the streaming loop). **Byte-identical to `seq`** (the §7 gate:
seq / p0 / fanout / fanout+prefetch+neon all one sha256). Excludes `--s1` (both fight for the same 3
cores). VART exposes **no runner→core API** (checked the board headers — `vart::Runner`/`RunnerExt` have
none; `xir::DpuController::get_core_id` is unreachable from the handle), so each lane self-times `g_a`
and **`g_a` ms/call across lanes is the placement proxy**: uniform ⇒ clean core split, one lane ~2× ⇒ a
collision. Sweep: `stream_fanout_sweep.py` (lanes × cold/warm, cooldown-gated); table: `fanout_table.py`
→ `results/benchmark_stream/fanout_table.md`.

**Result.** Full metrics — `g_a` ms/call, per-patch compute latency (normalize+DPU+entropy; the
prefetched SD read is excluded), throughput, avg power, J/patch — for all four archs × lanes 1–4, each
cell **cold / warm**, pinned (deterministic) placement (regenerate with `fanout_full_table.py`; per-lane stage bars → `fanout_lane_timings.png`):

| arch (topology) | metric | 1 lane | 2 lanes | 3 lanes | 4 lanes |
| --- | --- | --- | --- | --- | --- |
| **FP** (factorized) | g_a [ms] | 5.2 / 5.2 | 5.4 / 5.4 | 5.7 / 5.7 | 6.3 / 6.4 |
|  | latency [ms] | 21 / 21 | 22 / 22 | 23 / 23 | 25 / 25 |
|  | throughput [patch/s] | 45.4 / 45.8 | 85.2 / 87.4 | 86.0 / 122.5 | 87.3 / 145.6 |
|  | avg power [W] | 10.6 / 10.5 | 12.2 / 12.3 | 12.2 / 13.7 | 12.3 / 14.6 |
|  | J/patch | 0.230 / 0.230 | 0.140 / 0.140 | 0.139 / 0.111 | 0.138 / 0.100 |
| **SHyp** (hyperprior) | g_a [ms] | 5.2 / 5.2 | 5.4 / 5.4 | 5.6 / 5.6 | 6.4 / 6.3 |
|  | latency [ms] | 29 / 29 | 31 / 31 | 32 / 32 | 37 / 37 |
|  | throughput [patch/s] | 33.2 / 33.7 | 61.3 / 62.6 | 86.3 / 90.3 | 85.3 / 103.0 |
|  | avg power [W] | 10.2 / 10.2 | 11.6 / 11.6 | 12.7 / 12.8 | 12.7 / 13.5 |
|  | J/patch | 0.306 / 0.303 | 0.185 / 0.184 | 0.144 / 0.142 | 0.145 / 0.130 |
| **ResFP** (factorized + residual) | g_a [ms] | 36.6 / 36.6 | 36.8 / 36.8 | 37.0 / 37.0 | 51.5 / 51.5 |
|  | latency [ms] | 84 / 84 | 84 / 84 | 85 / 85 | 114 / 114 |
|  | throughput [patch/s] | 11.8 / 11.9 | 23.3 / 23.5 | 33.5 / 34.0 | 35.6 / 36.1 |
|  | avg power [W] | 12.0 / 12.0 | 15.2 / 15.2 | 18.1 / 18.2 | 18.8 / 18.8 |
|  | J/patch | 1.014 / 1.012 | 0.648 / 0.648 | 0.535 / 0.535 | 0.523 / 0.521 |
| **ResSHyp** (hyperprior + residual) | g_a [ms] | 36.6 / 36.6 | 36.8 / 36.8 | 37.0 / 37.0 | 49.4 / 49.5 |
|  | latency [ms] | 92 / 92 | 93 / 93 | 93 / 93 | 143 / 143 |
|  | throughput [patch/s] | 10.8 / 10.8 | 21.0 / 21.2 | 30.8 / 31.1 | 27.0 / 27.2 |
|  | avg power [W] | 12.4 / 12.4 | 16.0 / 16.0 | 19.3 / 19.4 | 18.1 / 18.0 |
|  | J/patch | 1.146 / 1.148 | 0.753 / 0.752 | 0.622 / 0.621 | 0.663 / 0.663 |

Plain reading (measured; deeper causal stories are deliberately left out):

- Giving each worker its own DPU core raises throughput. For the DPU-heavy models it is the best config
  measured — ResSHyp 30.8 patch/s at 3 lanes vs 22.8 for the previous best (`p0+s1`).
- Models with residual blocks (heavy `g_a`: ResFP, ResSHyp) gain the most from added lanes. The lighter
  models (SHyp, FP) are limited more by CPU/entropy and the SD read, and gain most in the **warm**
  (fast-storage) case.
- On this 3-core board, 4 lanes slow **ResSHyp** down (its per-patch DPU time jumps — g_a/latency rows,
  and `fanout_lane_timings.png`); the other three tolerate or slightly benefit. Matching lanes to cores
  (3) is the safe default.
- More lanes lower energy per patch for every arch.
- **Placement was a run-to-run lottery — now fixed (deterministic).** Spotted via the per-lane Gantt:
  at 3 lanes = 3 cores the runner→core placement was *random* (~7/8 runs collided two lanes' `g_a` on one
  core) because each lane deserialized its own graph copy → different creation order
  → VART's round-robin scattered `g_a`. **Fixed** by deserializing once and creating
  runners **subgraph-major** so lane *k* pins to core *k* (dropping the never-run `g_s`); verified
  deterministic — pinned `g_a` clean 10/10 runs, byte-identical output. `--fanout` = pinned (default);
  `--lane-major` reproduces the naive baseline for the ablation. See the placement sketch below.
- For context, the patch-only DPU benchmark (`nn_only`) had ResSHyp fall back at 3 cores (1.81×) while
  the full streaming pipeline does not (2.85×). Why they differ is not established here.

**Placement sketch.** Empirically, VART places runners by a deterministic round-robin over
*runner-creation order* (1st→core 0, 2nd→1, 3rd→2, 4th→0, …), so the create order sets the mapping:

```text
                        create order → core (mod 3)
  naive lane-major:  L0.ga L0.ha L0.hs   L1.ga L1.ha L1.hs   L2.ga L2.ha L2.hs
     core:            0     1     2        0     1     2        0     1     2
     ⇒ all three g_a on core 0 (creates 0,3,6) → serialized
  fixed subgraph-major:  L0.ga L1.ga L2.ga   L0.ha L1.ha L2.ha   L0.hs L1.hs L2.hs
     core:                0     1     2        0     1     2        0     1     2
     ⇒ lane k entirely on core k → no g_a collision, every run
```

> **We never observe the physical core.** VART exposes no runner→core API (above), so the core numbers
> in this sketch are the *inferred* round-robin mapping, not a hardware readout. What is actually
> measured is the timing signature it predicts: subgraph-major keeps every lane's `g_a` uncontended
> (uniform-low ms/call, clean 10/10 runs), lane-major serializes them. The core indices are a mental
> model consistent with that timing — a Gantt track is a *lane*, not a proven core.

**`--lane-major` ablation** (naive vs pinned, cold, 3 lanes). The fix only bites when a lane creates
>1 DPU runner (hyperprior) — factorized archs create just `g_a`, so lane-major ≡ subgraph-major there:

| arch | pinned patch/s | lane-major patch/s (g_a) | pinned speedup |
| --- | --- | --- | --- |
| ResSHyp | 30.8 | 13.8 (96 ms) | **2.24×** |
| SHyp | 86.3 | 79.9 (7 ms) | 1.08× |
| ResFP | 33.5 | 33.6 (37 ms) | 1.00× |
| FP | 86.0 | 87.5 (5 ms) | 0.99× |

Naive lane-major serializes all three `g_a` on one core for the hyperprior models; the win is largest
for the DPU-heavy ResSHyp.

**N2 — Embedded-GPU baseline (NVIDIA Jetson).** *Status: not started; hardware reportedly attached to
this host — access route to be confirmed with a colleague.* Supplies the recognisable
`N× vs a named baseline` that DATE expects and frames an honest
onboard-payload question: **embedded GPU vs FPGA SoC**. **→ main.tex §eval + abstract** (the headline
`N×` vs a recognizable baseline).

- **Scope:** the *same end-to-end streaming pipeline* (read → normalize → NN → rANS → `.ddc`), not
  per-patch — a per-patch comparison would only reproduce the TGRS cross-platform table.
- **Precision:** probably **float32 throughout** on the Jetson (no INT8/TensorRT quantization work, depends on what the Jetson can run. To check first.).
- **Port cost:** the rANS coder is portable C++ and should move across directly; float32 checkpoints (before any Vitis AI adaptation) are present on Host and can be found using `manifest.json`.
- Keep the existing desktop-GPU (A4000) + Xeon CPU numbers from `results/benchmark_unified/` alongside
  as context — they cost nothing and are informative.

**N3 — Deadline/budget-driven rate allocation.** Vary λ *across the scene* under a bit-budget or
wall-clock deadline: **user-cancelled: tricky to implement cleanly*.

**N5 — CCSDS baseline (recognisable ratio).** **→ main.tex §eval + abstract** (alt/additional
recognizable baseline; disambiguates data- vs model-compression). CCSDS 122.0 is the de-facto onboard *image* codec
(wavelet + bit-plane — the space-grade JPEG2000-lite, widely in rad-hard hardware): the recognisable
baseline a DATE/space reviewer knows, and it disambiguates "compression" from model compression. However, it probably does not support SAR SLC data compression (effectively) out-of-the-box, need some deep checks.
A literature comparison is cheap; running it on the Hamburg tile is more work (open implementations
exist). **Metric caveat:** CCSDS-122 does not despeckle, so DDC wins on rate partly *because* it removes
high-entropy speckle — a fair comparison needs a common reference. First step: characterise CCSDS-122
RD behaviour from the literature, then decide paper-only vs reimplemented. Cost: low (paper) to medium (run).

**N4 (dropped as a standalone experiment).** The INT8 `g_s` output cap at 2100.1 (clips the brightest
~0.7 % of pixels) is metric-invisible; it lives as a one-line **limitation** in `main.tex` §Discussion,
not as a deepening study.

**A4 — Optimization-ladder figure: warm per rung.** **→ main.tex Fig. `optimization_ladder`.** Today only the last rung has a warm run. Re-run the
sweep with **warm for every rung** and show cold+warm per rung (paired bars — warm behind with cold in the
front and a pattern). Consider making **warm the primary series** (a fast/no-SD store is the expected onboard
case) with cold as the testbed overlay. Needs a board re-run. *(figure: `optimization_ladder.py`.)*

**A5 — Hardware platform details (HW-community venue).** **→ main.tex §Background/Setup (platform table).** Gather and report the accelerator's internal
design: DPU `3× B4096 @ 300 MHz` (see if DSP run at double clock frequency), PS DDR4 ≈17 GB/s, ZU9EG (base facts in §2),
**plus PL resource utilisation** (LUT/FF/BRAM/URAM/DSP) and clocks from the Vivado/DPU report — as a short
platform table in `main.tex` (Background or Setup).

### Deferred / optional

- **On-ground SHyp `.ddc` decoder** (optional, decoupled). Reproduce the board INT8 `h_s` on host so
  SHyp/ResSHyp `.ddc` decode in pure Python (FP already does); hard because the Gaussian decode needs
  every element's scale in the same `gc_scale_table` bucket and an FP32 `h_s` desyncs rANS. The overlap
  study (§10) sidesteps it via board decode; value if built = off-board verification + a real ground
  decoder.
- **Figures.** The §9 sketches (`stream_gantt`/`stream_roofline`/`stream_sysplot`) are *feel-only* and
  superseded by the DATE manuscript figures (`figures/scripts/`, LaTeX repo); the stacked time-per-patch
  figure is **done** (Fig. 3). A dedicated read-ceiling / CPU-view figure could still help.
- **Naming.** `stream_pipeline` runs a sequential path + a worker-pool (`--p0`), not a classic per-stage
  pipeline — rename (`stream_compress`?) or leave. Low priority.
