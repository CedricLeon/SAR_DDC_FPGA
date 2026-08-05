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
(§4), and the full-scene throughput/latency/energy sweep + memory/roofline analysis (§8) are done and
board-verified. Remaining (§10): **baseline grid snap-fix + §8 re-run** (the old grid dropped the far
edge sliver), the **overlap study** (in implementation — measured cost + board-decode/host-stitch
quality), a stacked time-per-patch figure, and figure rigor/polish. The on-ground `.ddc` decoder is
optional/decoupled. Last updated 2026-08-02.

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
- [ ] **U5**: Non-overlapping patch compression leaves seam/edge artifacts when the tile is recombined.
  How do we prevent that? *Being answered — the overlap study (§10, in implementation) sweeps a
  `--overlap {0,4,8,16}` px knob and measures reconstructed-image quality vs the latency/bitrate it
  costs, on the same streaming pipeline as §8; reconstruction via board INT8 decode + host stitch.*

---

## 1. Goal & scope

- **Deliverable:** a *configurable, reproducible experimental harness* (not a shipped binary) to
  explore pipeline/thread schedules for the onboard scenario, and measure **steady-state throughput
  (patch/s)** + **full-tile latency** (read → compress → write).
- **Models:** ResSHyp (DPU-bound) + FP (CPU-bound) primary; SHyp/ResFP come ~free (arch
  auto-detected from `manifest.json`).
- **Overlap:** deferred — non-overlapping per-patch bitstreams for now; the overlap sweep is future
  work (§10).
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

**Why:** `symmetrize()` is a whole-image FFT → would break streaming. But it is an **integer spectral
roll = spatial phase ramp → |amplitude| is exactly preserved**; it only redistributes energy between
real/imag. **Hypothesis:** despeckling is amplitude-dominated and the model processes real/imag
separately, so local symmetrization should cost little quality → it can live *inside* the
per-patch/per-block pipeline.

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

  *Δ(skip) = whole − none, from unrounded values (may differ ±0.01 from the rounded columns).*

- SSIM:

  | model         | whole  | none   | patch  | block  | Δ(skip) |
  |---------------|--------|--------|--------|--------|---------|
  | ResSHyp λ1000 | 0.9666 | 0.9627 | 0.9628 | 0.9628 | 0.0039  |
  | ResSHyp λ20   | 0.9210 | 0.9190 | 0.9189 | 0.9190 | 0.0020  |
  | ResSHyp λ2    | 0.6540 | 0.6539 | 0.6539 | 0.6537 | 0.0001  |
  | FP λ1000      | 0.9592 | 0.9555 | 0.9556 | 0.9556 | 0.0037  |
  | FP λ20        | 0.9348 | 0.9323 | 0.9327 | 0.9327 | 0.0025  |
  | FP λ2         | 0.8464 | 0.8456 | 0.8454 | 0.8457 | 0.0008  |

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
> each point actually represents). → §10.
>
> ⚠️ **All three are DPU + SD-read only** — none decomposes the CPU (entropy + normalize) load: the
> roofline is DPU by construction, the system roofline's only compute roof is the DPU, and the throughput
> plot bakes the CPU into the numbers without drawing it (for FP, the warm ceiling *is* the CPU entropy
> limit). The dedicated CPU view is the stacked-time figure → §10.

---

## 10. TODO

**Overlap study — reconstructed-tile quality vs cost (U5).** *Status: complete — full-scene sweep
(FP + ResSHyp × λ{20, 1000} × overlap {0, 2, 4, 8, 16}, cold + warm) scored vs the MERLIN full-tile GT
with coherent seam-band metrics; **Result** below. Remaining = manuscript-figure polish only.*
Non-overlapping patches leave seam artifacts when the despeckled tile is recombined; overlap +
ramp-blend removes them at a latency/bitrate cost. Measured on the **same streaming pipeline as §8**
(cost is measured, not estimated) and **decoupled from the host decoder below** — reconstruction uses
the board's real INT8 decode.

- **Grid rule (single source of truth).** Offsets = `make_offsets(dim, 256, stride)`, `stride = 256 −
  overlap`, last patch snapped to `dim − 256`. Board = `stream_pipeline.cpp::make_offsets`; host =
  `src/utils/tiling.py::make_offsets` (+ `blend_patches`), pinned byte-for-byte-close to the canonical
  `processing_utils.patch_infer` blend by `tiling._selftest` (max err 3.6e-7). The `.ddc` header carries
  `stride`, so `overlap = patch − stride` is recoverable — **no new `.ddc` field needed** (resolves U4).
- **Pass 1 — cost (board, timed).** ✅ `stream_pipeline --overlap N` (snap grid). Board-verified: grid
  counts, byte-identical gate across seq/s1/p0/windowed/prefetch at overlap 0 & 8, and the decode
  round-trip all pass. overlap=0 is the snap-covered baseline; expected patch inflation (to be measured in
  the sweep) **+3.3 % / +6.7 % / +14.4 %** at 4 / 8 / 16 px.
  - **Cost-measurement caveat.** Each overlap point is a single full-scene run (`iters=1` — the
    ~7,482-patch average is itself the measurement, stable) under the thermal cooldown gate (die ≤ 58 °C
    before each run, §8). A53 throttle is sampled per run and flagged in the JSON (`thermal.throttled`) but
    **not excluded**, so read sub-~1 % latency/energy deltas as within run-to-run noise, not signal.
- **Pass 2 — reconstruction (board).** ✅ `stream_pipeline --decode` decodes each patch with the real
  INT8 `h_s`+`g_s`+rANS and **stream-writes** it straight to the `[n,P,P]` NPY (via
  `npy_write_header_float32`) — no ~2.2 GB buffer at full-scene+overlap. Board-verified byte-identical to
  the prior buffered decode.
- **Pass 3 — stitch + score (host).** ✅ `scripts/evaluation/stitch_ddc.py` (+ `src/utils/tiling.py`):
  reads the `.ddc` header, reproduces the snap offsets, ramp-blends the decoded per-patch linA into the
  full `[H,W]` tile in host RAM (no board OOM), optional scoring vs the MERLIN GT. Validated on a board
  round-trip: overlap-0 stitch = plain reassembly bit-for-bit; overlap-8 coherent.
  - **Blend + score domain = linear amplitude.** The board's `g_s` denorm emits linA, so both the blend
    and the metrics live in linear amplitude — the project convention (metrics in linear amplitude,
    visualization in log; CLAUDE.md) and the same domain as §5. No log↔linA transform enters the loop, so
    there is no domain-mismatch seam and the §5 (symmetrization) and this study's numbers are on one basis.
  - **Seam-sensitive scoring.** `score_arrays` reports full-tile PSNR/SSIM **plus** a split into the
    **seam band** (±3 px of the non-overlap 256-grid boundaries, ≈4.6 % of pixels — where independent-patch
    seams live) and the **interior**. The full-tile mean dilutes the seam effect ~21×, so `PSNR_interior −
    PSNR_seam` is the sensitive number that isolates what overlap actually fixes.
- **MERLIN GT.** ✅ `scripts/evaluation/merlin_full_gt.py`: the project's own MERLIN despeckle on the
  whole-image-symmetrized scene, seam-free heavy-overlap (64 px) blend — consistent with §5. Validated on
  the 1024² crop vs the project's existing `linA_MERLIN.npy` (PSNR 54.8 dB, corr 0.991). The full-scene GT
  (`data/visualization/MERLIN/linA_MERLIN_full_Hamburg.npy`, `(32901, 14686)`) is generated and is the
  scoring reference for the sweep. (`…/MERLIN_DDS/linA_MERLIN_DDS_full_Hamburg.npy` exists but is the
  original-study checkpoint → a cross-check only.)
- **INT8 recon caps bright scatterers at 2100 — metric-invisible, and it sets the SSIM `data_range`.**
  *Observed* while scoring: full-scene SSIM sat at a saturated ~0.99 until the `data_range` was corrected.
  The on-board recon amplitude is hard-capped at **exactly 2100.1** (every tile / overlap / arch) because
  the DPU `g_s` output tensor is INT8 at **fix-point 8** — its largest code (127) maps to `x_hat =
  127/256 = 0.496`, which the denorm `exp(x_hat·(AMP_MAX−AMP_MIN)+AMP_MIN)` turns into 2100.1. The float32
  model has no such cap: on the same scene it reaches **~85 k (FP) / ~68 k (ResSHyp)**, near MERLIN's
  ~127 k, so the board clips the brightest **~0.7 %** of pixels (point scatterers) down to 2100.
  *Consequences:* (1) invisible to the reported metrics — PSNR/SSIM/MSE clip to `AMP_LIN_99 = 545` first,
  far below the cap; (2) the overlap study scores SSIM **coherently with PSNR/MSE**: clip recon+GT to
  `AMP_LIN_99` and use `data_range = AMP_LIN_99 = 545` (`stitch_ddc.score_arrays`). This is the fixed,
  cross-model-comparable basis — it avoids both `max(GT) ≈ 1e5` (saturates SSIM toward 1) and
  `max(recon) ≈ 2100` (non-comparable across models, since the recon max varies). This basis is now used
  by every distortion metric, host and board (SSIM, MS-SSIM and EPD included); re-evaluating the model set
  against it is tracked in `docs/ssim_data_range_issue.md`; (3) the board
  genuinely cannot represent bright targets — a fix-point-7 re-quantization would lift the cap to ~44 k at
  half the precision, if ever needed.

**Result — overlap 2 removes the seam at ~1 % cost; more buys nothing.** Full table (all archs, λ, cold +
warm, full/seam/interior PSNR & SSIM) → `results/benchmark_stream_overlap/overlap_table.{md,csv}`. The
seam is real but local: at **overlap 0** the reconstruction is **1.2–2.8 dB worse in the ±3 px seam band**
than in the interior, and its SSIM drops ~0.06–0.08 there — while the full-tile mean barely moves (the
seam is 4.6 % of pixels, so it is diluted ~21×). **Overlap 2 fully closes the seam** (deficit → ~0.02 dB);
overlaps 4/8/16 add nothing to the seam and only cost more.

| config | seam Δ @ ov0 (dB) | seam Δ @ ov2 | SSIM seam→interior @ ov0 | full-tile PSNR ov0→ov16 |
| --- | --- | --- | --- | --- |
| FP λ1000 | 1.16 | 0.01 | 0.786 → 0.853 | 24.99 → 25.04 |
| FP λ20 | 1.16 | 0.03 | 0.742 → 0.807 | 24.19 → 24.29 |
| ResSHyp λ1000 | 2.75 | 0.03 | 0.811 → 0.889 | 27.48 → 27.72 |
| ResSHyp λ20 | 2.06 | 0.01 | 0.737 → 0.810 | 25.37 → 25.53 |

*(seam Δ = PSNR_interior − PSNR_seam; all PSNR clipped to `AMP_LIN_99`.)*

**Cost of overlap 2** (measured on the §8 pipeline, so it is directly comparable):

- **Latency** — **+0.7–1.0 % warm** (compute-bound: FP λ1000 +0.68 %, ResSHyp λ1000 +0.89 %) and
  **free cold** on FP (read-bound — the extra compute hides behind the SD read; the whole ov sweep stays
  within ±2 % of ov0, no monotonic trend). By contrast overlap 16 costs **+14 % (ResSHyp / FP λ20 warm)
  to +19 % (FP λ1000 warm)**.
- **Downlink** — patch count grows **+0.78 %** at ov2 (7,482 → 7,540), vs +14.35 % at ov16; bpp-per-pixel
  is flat, so the `.ddc` grows in proportion to the patch count.
- **Energy** — J/patch flat across overlap (within run-to-run noise, all archs).

So the pipeline should reconstruct at **overlap 2**: it erases a genuine, visible per-patch seam for
about **1 % latency and 0.8 % downlink**, and anything beyond 2 px pays a growing latency/bitrate cost for
no further seam gain.

> **On-ground SHyp `.ddc` decoder** (optional, decoupled — *not* a blocker for the overlap study).
> *Idea:* reproduce the board's INT8 `h_s` on host so SHyp/ResSHyp `.ddc` decode in pure Python (FP
> already does). *Why it's hard:* the Gaussian decode needs the board's scales to land in the same
> `gc_scale_table` bucket (64 log-spaced, ~13 % wide) for **every** element — an FP32-checkpoint `h_s`
> crosses buckets and desyncs rANS. A faithful decoder needs the INT8 `h_s`: the quant artifacts aren't
> on disk (would re-quantize in Docker from `original_run_dir`) and bit-exactness vs the board is unproven
> (`export_xmodel(deploy_check=False)`). *Value if built:* off-board verification + a real ground-station
> decoder. The overlap study sidesteps it entirely via board decode (pass 2).

> **Stacked time-per-patch figure.**
> *Idea:* a stacked bar (read / DPU / normalize / entropy) per arch × schedule — the figure that shows
> **CPU load explicitly**, which the roofline/system plots miss (§9). ResSHyp's bar is DPU-dominated,
> FP's is entropy-dominated; NEON visibly shrinks the normalize slice.
> *Builds on:* §9. Data already exist (`benchmark_hardware` stage means + `benchmark_stream` read
> times) — no new board runs.

> **Figure rigor + polish pass.**
> *Idea:* the Gantt + roofline + system plots are feel-only sketches; before the manuscript, audit what
> each axis/roof/point means, then polish for legibility. Consider a dedicated read-ceiling figure.
> *Builds on:* §9.

> **Naming: `stream_pipeline`?**
> *Remark:* the binary does a sequential path + a worker-pool (`--p0`), not a classic per-stage pipeline
> — the name reads more pipeline-y than it is. Low priority; rename (`stream_compress`?) or leave.
