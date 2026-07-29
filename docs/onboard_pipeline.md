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
| ~~1.5 symmetrize~~ | — | **dropped** (E1 §6: ≤0.38 dB cost); optional one-time whole-image pre-pass if ever wanted |
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
| U2 | On-disk tile format (int16 `.cos` vs f32 `.npy`) | ⏳ open — `.cos` may be fragmented/awkward to parse; decide via E1 + an SD-read micro-bench. **Default f32 `[H,W,2]` first** |
| U3 | Harness shape | ✔ **locked** — extend `benchmark_hardware` with named presets + gates (§8) |
| U4 | `.ddc` container format | ✔ drafted (§7); still to verify round-trip once |
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

  | model | whole | none | patch | block | Δ(skip) |
  | --- | --- | --- | --- | --- | --- |
  | ResSHyp λ1000 | 31.65 | 31.10 | 31.10 | 31.11 | 0.54 dB |
  | ResSHyp λ20 | 28.61 | 28.36 | 28.41 | 28.42 | 0.25 dB |
  | ResSHyp λ2 | 22.94 | 22.91 | 22.93 | 22.93 | 0.03 dB |
  | FP λ1000 | 30.67 | 30.24 | 30.26 | 30.27 | 0.43 dB |
  | FP λ20 | 28.87 | 28.64 | 28.65 | 28.66 | 0.22 dB |
  | FP λ2 | 24.70 | 24.67 | 24.67 | 24.67 | 0.03 dB |

  **Conclusion → skip symmetrization** (definitive: 2 archs × 3 rates × full scene). Granularity is
  irrelevant everywhere (none ≈ patch ≈ block within ≤0.05 dB). Dropping whole-image symmetrization
  costs **at most 0.54 dB** (ResSHyp λ1000) and **shrinks with compression — only ~0.03 dB at λ2**,
  i.e. negligible at the operating points a compression paper cares about. The pipeline drops the
  symmetrization stage; an optional one-time whole-image pre-pass reclaims the ≤0.54 dB if ever
  wanted. (An earlier 4096² urban-region run agreed; absolute PSNR is higher on the full scene as it
  includes homogeneous terrain — the Δ is what matters.) Runs: `results/symmetrization_study/*_full.json`.

> **linA convention (single source of truth).** `x_hat` is 2-channel `[B,2,H,W]` (real, imag);
> `linA = sqrt(0.5·(exp(x0·Δ+AMP_MIN)² + exp(x1·Δ+AMP_MIN)²))`, Δ = AMP_MAX−AMP_MIN — as in
> `create_dataset.py::_predict_linA` (the code that generated the MERLIN/ADAM GT). `src/evaluate.py`
> mishandles the two channels in both its paths — **pending fix**.

---

## 7. `.ddc` downlink product (detailed)

The `.ddc` is the compressed tile as it would be queued for downlink: a small **header** (how to
decode) + one **record per patch** (the rANS bitstreams). Nothing model-specific (CDF tables,
weights) is embedded — those live in `entropy_params/` and are referenced by id.

**Header** (once, at file start; ~50 B, negligible):

| Field | Bytes | Meaning |
| --- | --- | --- |
| magic | 4 | ASCII `DDC1` — format + version |
| arch_id | 1 | 0=FP 1=ResFP 2=SHyp 3=ResSHyp → selects EB-only vs EB+GC decode path |
| N / M | 2+2 | main (128) / hyper (256) channels |
| patch | 2 | patch size (256) |
| grid_range / grid_azimuth | 2+2 | patch grid (57 × 128) → place patches back in the tile |
| model_hash | 16 | id/hash of the `entropy_params` + weights that produced it |
| AMP_MIN / MAX / EPS | 12 | denorm constants (f32 × 3) |

**Per-patch record** (× n_patches, row-major grid order):

| Field | Bytes | Meaning |
| --- | --- | --- |
| len_z | 4 | byte length of the hyper stream (SHyp only; 0 for FP) |
| z_bits | len_z | rANS bitstream for `z` (EntropyBottleneck) |
| len_y | 4 | byte length of the main stream |
| y_bits | len_y | rANS bitstream for `y` (GaussianConditional for SHyp, EB for FP) |

Optional trailer: an offset table (`n_patches × u32`) for random access / partial downlink.

**Why / gotchas:**

- **`model_hash` is the decodability guard** — rANS needs the *exact* CDF tables; without a reference
  to them the bytes are meaningless. #1 trap.
- **`len_*` prefixes** — rANS output is variable-length per patch, so each stream must be delimited.
- **Little-endian, fixed-width ints** — explicit and portable (board aarch64 + host x86 are both LE).
- **`arch_id` + grid dims** are all a decoder needs to route the decode path and re-tile.
- **Bitrate honesty:** `bpp = (Σ len_z + len_y) × 8 / (n_patches × 256²)`; report header/index
  overhead separately from payload.

**Verify (Q3):** `scripts/evaluation/ddc_decode.py` reads a `.ddc`, decompresses with the referenced
params, reconstructs, and checks round-trip PSNR vs the on-board `inference_hybrid` output — run once.

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
- [ ] SD sequential **read** throughput (tile load).
- [ ] SD **write** throughput (`.ddc`).
- [ ] E1 symmetrization study (§6).
- [ ] per-patch patchify + normalize timing on the raw scene.
- [ ] byte-identity gate harness.

---

## 10. Staged plan

1. **E1** (local GPU) — unblock symmetrization placement.
2. **`.ddc` format + Python verifier** — lock the product + decode path.
3. **`stream_seq`** on-board — read `.cos`/tile → full pipeline → write `.ddc`; gate vs `inference_hybrid`.
4. **`stream_p0`** (2-lane) — throughput + tile latency; power.
5. **`stream_fine`** + `--queue-depth`/`--entropy-threads` sweeps; full 7 296-patch scene streamed from SD.
6. Overlap + reconstructed-tile quality.
