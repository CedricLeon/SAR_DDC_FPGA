# Onboard Streaming Pipeline — Design & Measurements

> End-to-end **"receive focused SLC tile → despeckle + compress → write downlink bitstream"**
> streaming compressor on the Xilinx ZCU102 — the core contribution of the DATE'27 paper alongside the
> performance analysis. This file records the implemented design + measurements; it keeps the
> *decisions* and their rationale and points to the canonical docs for existing pieces.
>
> Builds on: `FPGA_inference.md` §5 (per-stage pipeline); `FPGA_benchmark.md` §10 (P0/P2
> patch-pipelining, now implemented + measured here); `Data.md` (`.cos` source, normalisation).

Status: **implemented + measured.** The streaming compressor, the parallel/I/O optimizations (§4), the
DPU fan-out core-scaling (§5–§6), the symmetrization (§7) and overlap (§8) studies, and the full-scene
throughput/latency/energy sweep (§10) are done and board-verified at the coherent setup (λ=20, overlap
2, snap grid, four architectures). The Orin Jetson embedded-GPU baseline (N2, §12) is done and board-verified
across the full 4-arch × 4-power-mode matrix; a per-arch quality sweep and Thor both remain (§12). The
classical SAR baseline (N5, §10) is resolved paper-only — no CCSDS standard targets SAR, so the paper
cites the closest literature instead of reimplementing.
Remaining (§13): an entropy-coding optimization pass (N6 — partially done, see
§13), the PL resource table (A5), and figure polish. Last updated 2026-08-25.

---

## 1. Goal & scope

- **Deliverable:** a *configurable, reproducible experimental harness* (not a shipped binary) to
  explore pipeline/thread schedules for the onboard scenario, and measure **steady-state throughput
  (patch/s)** + **full-tile latency** (read → compress → write).
- **Models:** ResSHyp (DPU-bound) + FP (CPU-bound) as the two spanning cases; SHyp/ResFP come ~free
  (arch auto-detected from `manifest.json`), so all four are characterized.
- **Setup:** λ=20 throughout (throughput is λ-independent — §10; compression is reported at this chosen
  operating point), overlap 2 (§8), the snap-covered grid (§3).
- **Language:** C++ on-board (only inference path). Python only for the offline symmetrization study
  (§7) and the `.ddc` verifier (§9).

---

## 2. Hardware facts (measured on the board unless noted)

| Memory | Capacity (this board) | Role in the pipeline | Source |
| --- | --- | --- | --- |
| **SD card** (rootfs) | 29.7 GB card; ext4 `/` = 27 GB, **17 GB free** | **Steps 1 & 7** — read tile, write `.ddc` | board `lsblk`/`df` |
| **PS DDR4** (RAM) | 4 GB spec; **3.84 GiB** seen, **~3.0 GiB free** | Working memory: in-flight patches, queues, weights | `/proc/meminfo` + UG1182 |
| DPU on-chip (BRAM/URAM) | ~84 % of ZU9EG BRAM (3× B4096) | DPU weight/activation buffers (VART) | `FPGA_benchmark.md` + PG338 |

- **4× A53 cores** (`nproc=4`) → parallel-worker budget, shared with DPU dispatch + OS.
- Queues between stages are a non-issue: **~50 MB at depth 16** vs ~3 GB free.
- **SD sequential read ≈ 23.5–23.8 MB/s** (measured cold: 1.93 GB ÷ 81 s) — the **Step-1 read ceiling**.
  To emulate a *faster persistent store* we use a warm read.
- **Page cache** (the Linux kernel's file cache, held in **DDR/RAM — not a CPU cache**; 4 KB page
  *granularity*, confirmed): our 1.93 GB tile ≈ **472k pages** in the ~3 GB of free DDR4; whole-tile
  *warm* read is fine, but whole-tile *f32 in-process load* (3.87 GB) OOMs → windowed streaming is
  mandatory.
- **PS DDR4 peak bandwidth = 17.06 GB/s** — 4 GB DDR4-2133 SODIMM (Kingston KVR21SE15S8/4), 64-bit
  (2133 MT/s × 8 B) [UG1182 + SODIMM part]. Sustained DDR traffic (SD read + DPU DMA + memcpy) sits
  ≈20× below this → DDR is **not** a bottleneck (vaitrace-measured; §10).
- **DPU = 3× DPUCZDX8G B4096 @ 300 MHz → 1229 GOP/s per core** (4096 ops/cycle × 0.30 GHz; the guide
  lists 1400 @ 350 MHz) [PG338, *DPUCZDX8G Peak Performance*; clock from `xdputil query`]. Roofline
  ridge vs DDR = 1229 ÷ 17.06 = 72 OP/byte.

Sources: [UG1182 ZCU102 Eval Board UG](https://docs.amd.com/v/u/en-US/ug1182-zcu102-eval-bd) ·
[DS891 Zynq UltraScale+ Data Sheet](https://www.mouser.com/datasheet/2/903/ds891_zynq_ultrascale_plus_overview-1662253.pdf) ·
[PG338 DPUCZDX8G Peak Performance](https://docs.amd.com/r/en-US/pg338-dpu/DPUCZDX8G-Peak-Performance).

---

## 3. Data & acquisition geometry

- Tile: `data/TSX_cos_files/Hamburg_…_strip_004.cos` = **14 686 (range) × 32 901 (azimuth)**, raw
  complex-int16 = **1.93 GB**. On the snap grid the base cover is **58 × 129 = 7 482** 256² patches (the
  last patch in each axis is flush to the edge, overlapping its neighbour by the remainder; a plain
  floor grid gives 57 × 128 = 7 296 and drops the far-edge sliver). At the **overlap-2 default** (stride
  254, §8) the grid grows to **7 540** patches — the count behind every §10 number.
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
| ~~1.5 symmetrize~~ | — | **dropped** (see §7/E1: ≤0.54 dB cost); optional one-time whole-image pre-pass if ever wanted |
| 2 patchify | CPU | strided per-row `memcpy` of `[256,256,2]` out of the DDR row-block |
| 3 normalize | CPU | log + min/max (~9 ms/patch; `--neon` = 2.42× faster, byte-identical) |
| 4 g_a×2 | DPU | already S1-parallel (1.95×) |
| 5 h_a/EB/h_s | DPU+CPU | hyperprior archs only (FP/ResFP skip) |
| 6 entropy | CPU | rANS → bits |
| 7 write bits | DDR → SD | generate `.ddc` product per tile (§9) |

**Harness design.** The harness picks a base executor with `--schedule {seq,p0}` and layers composable
modifiers on top (`--s1`, `--fanout`, `--prefetch`, `--neon`, `--threads K` → C++ `stream_pipeline`
flags); combos are validated at startup and **hard-error on nonsensical inputs** (errors-over-fallbacks).
The output is **byte-identical to `seq` by construction** — records are placed by absolute patch index,
so neither the thread count nor the schedule can change a single output byte — and this was *verified*
by hashing the `.ddc` across schedules (seq / p0 / fanout / fanout+prefetch+neon all one sha256), plus a
one-time `.ddc` round-trip PSNR check. So the knobs are purely about *speed and energy*, never quality.
The streaming executor lives in its own module (`inference_cpp/src/stream/` — `main_stream.cpp` CLI +
`stream_pipeline.{cpp,hpp}`, binary `stream_pipeline`), reusing the leaf stages.

**Optimizations.** Each layers on the sequential `seq` baseline; measured effects → §10.

- `seq` — single thread; correctness + latency baseline (reads the tile, writes the `.ddc`).
- **`--s1` — two cores per patch (scaffold).** Runs a patch's two halves, `g_a(re)` and `g_a(im)`, on
  two DPU cores at once. This was our first way to spread work across cores, but fan-out (§5) beats it for
  every arch and supersedes it, so it is kept only as a scaffold. Two reasons it loses: it can use at most
  two cores while the board has three, and it splits one patch across cores instead of running whole
  patches side by side, so a core waits on the slower half.
- **`--p0 --threads K` — worker pool.** K workers each run normalize → DPU (serialized by a mutex) →
  entropy → write, records placed by patch index. The main lever for the CPU-bound FP (scales to 4
  workers); the DPU-bound archs plateau earlier (DPU-serialized) — the fan-out lift is **§5**.
- **`--fanout` — DPU data-parallel lanes.** The DPU-bound lever beyond `--s1`; its own section (**§5**).
- **`--prefetch` — double-buffer I/O.** A producer thread reads row-block N+1 while the workers compress
  block N (bounded `RowBlockQueue`, depth 2). Byte-transparent; hides the SD read behind compute — fully
  for the DPU-bound archs, partially for FP (toward its read ceiling).
- **`--neon` — vectorized normalize/denorm.** NEON log/exp (Cephes/Pommier, `neon_mathfun.h`) behind a
  runtime flag, scalar path kept for A/B. Kernel error vs libm = 7e-8 → **byte-transparent encode** (≪
  the INT8 `g_a` step, so no quantisation flips). 2.42× faster normalize in isolation.
- **`--power` — energy instrumentation.** `PowerSampler` (INA226 sysfs + PMBus) wraps the compress
  phase → total J, **J/patch**, and the per-rail-group breakdown (PL / PS / DPU_fabric / PS_compute /
  MGT / MPSoC mean W) — the DPU-vs-CPU energy split.
- **cold/warm harness** (`stream_benchmark.py`, host-side over SSH). Drops the page cache before each
  run (`sync && echo 3 > /proc/sys/vm/drop_caches`) so **cold** is the honest SD-read number;
  `--keep-cache` runs **warm** (tile served from RAM) = the compute ceiling if storage were fast. Each
  result JSON stamps its git provenance (`git_sha`/`git_dirty`/`run_utc`) — the source it came from.

---

## 5. DPU fan-out — data-parallel lanes

Fan-out is the DPU-bound optimization beyond `--s1`, and the best schedule measured for every arch
(§10). **`--fanout` is a `--p0` modifier:** instead of K workers sharing one pipeline behind a DPU mutex
(plain `p0`), it creates **K independent pipelines — one DPU lane per worker, the mutex dropped** — so
`g_a` runs on up to 3 cores concurrently, data-parallel across patches. It **excludes `--s1`** (both
would fight for the same 3 cores), and is byte-identical to `seq` (§4 gate). Placement is deterministic
(lane *k* pins to core *k*); the how, and the per-core occupancy that explains the roofs, are **§6**.

**How to run.** Sweep lanes × cold/warm, cooldown-gated, then plot:

```bash
python scripts/fpga/benchmark/stream_fanout_sweep.py --archs FP,SHyp,ResFP,ResSHyp --lambdas 20 \
    --lanes 1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,32,48,68,96,128     # full lane sweep (iters=1 suffices — see below)
python scripts/fpga/benchmark/fanout_lane_plot.py    # -> results/benchmark_stream/fanout_lane_scaling_4arch.png
```

**Lane scaling.** Warm throughput / DPU + CPU occupancy / energy vs lane count (λ=20) are the three
panels of `fanout_lane_scaling_4arch.png`; the per-lane cells (patch/s, J/patch, power, `g_a` ms/call)
live in the result JSONs (`results/benchmark_stream/<arch>-relu_s0_L20_pt/p0_t{N}_fo_pf_neon_warm.json`)
and the tabulated `fanout_full_table.md`.
**Every arch roofs, and the roof height is set by the binding resource** (§6): the DPU-bound archs
(ResFP, ResSHyp) roof **low and early** — ~40 patch/s, flat from ~6 lanes; the CPU-bound archs (FP, SHyp)
roof **high and late** — FP to ~200, SHyp to ~145 patch/s. Fan-out beats the `--s1` ladder for all four
(e.g. ResSHyp 38.5 vs 23.4 patch/s, FP 200 vs 136); the earlier "hyperprior 4-lane cliff" was an artifact
of stopping at 4 lanes — ResSHyp dips at 4L, then climbs to ~38.5 by 24L.

**The XRT runner wall.** The two hyperprior archs stop at 96 lanes: the wall (`VART_XRT_NULL_PTR`) sits
**between 288 and 384 concurrent DPU runners** (96-lane hyperprior = 96 × 3 = 288 runs; 128-lane = 384
fails; the factorized single-`g_a` FP/ResFP never approach it). That XRT ceiling — not RAM — is the only
hard wall; CMA is the softer co-factor (§6).

**Operating point per arch** = each arch's warm throughput **peak**, clear of the XRT wall; the
reference table is **§10** (FP 64L / SHyp 48L / ResFP 6L / ResSHyp 24L). A stricter **knee** — the
smallest lane count within 1 % of that peak — is a defensible alternative, trading ≤1 % throughput for
far less oversubscription (fewer threads, less memory, more XRT margin): **32 / 24 / 6 / 20 lanes** for
FP / SHyp / ResFP / ResSHyp. It also sidesteps SHyp's noisy iters-1 48L peak.

**Why iters=1 suffices.** Each point is a full-scene average over 7 540 patches, so the law of large
numbers crushes run-to-run variance: the median-of-3 σ is **0.0–0.7 patch/s (<0.25 %)**, often identical
across iters (total time logged to 0.1 s). Future lane sweeps need no repeats.

---

## 6. DPU fan-out Placement & occupancy — under the hood

How the lanes land on cores, the per-subgraph DPU timing and per-core occupancy behind the DPU-bound
roofs, and the CPU occupancy + memory footprint behind the rest.

### Placement — why lane *k* → core *k*

**Why deterministic placement matters.** At 3 lanes = 3 cores, the naive scheme was a *run-to-run
lottery*: each lane deserialized its own graph copy in a different order, so VART's round-robin over
runner-creation order scattered `g_a` across cores and could results in collisions.
That fragility is what motivated pinning the placement.

**The fix.** Deserialize the graph **once** and create the lane runners **subgraph-major** (all `g_a`
first, then all `h_a`, then all `h_s`), so lane *k*'s `g_a` lands on core *k* every run (the never-run
`g_s` is dropped). `--fanout` uses this pinned placement by default; `--lane-major` reproduces the naive
pipeline-major baseline for the ablation.

```text
                        create order → core (mod 3)
  naive lane-major:  L0.ga L0.ha L0.hs   L1.ga L1.ha L1.hs   L2.ga L2.ha L2.hs
     core:            0     1     2        0     1     2        0     1     2
     ⇒ all three g_a on core 0 (creates 0,3,6) → serialized
  fixed subgraph-major:  L0.ga L1.ga L2.ga   L0.ha L1.ha L2.ha   L0.hs L1.hs L2.hs
     core:                0     1     2        0     1     2        0     1     2
     ⇒ lane k entirely on core k → no g_a collision, every run
```

The mapping is **directly observed**, not just inferred: `GLOG_logtostderr=1 DEBUG_DPU_RUNNER=1` logs
each runner's `device_core_id` at creation (only ever 0/1/2, reconfirming 3 cores), and
`fanout_validate.py` measures `g_a` concurrency (pinned → 3 cores in parallel, lane-major → serialized
to 1, never > 3). The public `vart::Runner` exposes no core accessor, so this is a one-time validation,
not an in-loop signal.

**`--lane-major` ablation** (naive vs pinned, cold, 3 lanes, λ=20). The fix only bites when a lane
creates >1 DPU runner (hyperprior) **and** the arch is DPU-bound — so it is a clean no-op elsewhere:

| arch | pinned patch/s | lane-major patch/s (g_a) | pinned speedup |
| --- | --- | --- | --- |
| ResSHyp | 31.9 | 13.8 (99 ms) | **2.31×** |
| SHyp | 86.1 | 86.0 (7 ms) | 1.00× |
| ResFP | 34.8 | 34.8 (37 ms) | 1.00× |
| FP | 86.8 | 86.6 (5 ms) | 1.00× |

SHyp *does* collide (`g_a` 5.9→7.0 ms) but is CPU/read-bound, so its throughput is unmoved; the
factorized archs create only `g_a`, so lane-major ≡ subgraph-major there. The win is real only for the
DPU-bound hyperprior (ResSHyp).

**Full-curve confirmation** (ResSHyp, warm, λ=20, lanes 1–32; `p0_t{N}_lanemaj_pf_neon_warm.json`). Naive
placement is **flat at ~13.8–13.9 patch/s from 2 lanes on** (≈ its 1-lane value): the round-robin collides
every lane's `g_a` onto one core, so extra lanes add nothing. Pinned climbs to 38.5, so the deficit **widens
with lanes — 2.3× at 3 lanes to ~2.8× at the operating point** (24L). The pinned curve also carries the
4-lane load-imbalance dip (32.4→27.6→36.1 at 3/4/6 lanes); lane-major, already serialized, shows none.

### DPU timing & occupancy

**Per-lane Gantt.** `fanout_gantt.py` draws a per-call execution timeline from a `--trace` CSV — one
track per DPU lane plus the prefetch reader, over a short steady-state window. Generate the trace on a
small subsample (a few row-blocks → a tiny CSV), then plot:

```bash
ssh ZCU102 "cd /home/root/SAR_DDC && ./build_cpp/stream_pipeline --xmodel active_model/*.xmodel \
  --params active_model/entropy_params --tile data/full_scene_i16.npy --out /tmp/t.ddc \
  --p0 --fanout --windowed --prefetch --neon --overlap 2 --max-rows 4 --trace /tmp/tr.csv"
```

Each call is a bar at its measured `[start, end]`. A DPU bar splits into compute (solid, `= min(span, e)`)
and an *inferred* wait (hatched, `= max(0, span − e)`; `e` = the kernel's median 1-lane exec, below),
drawn only when the call outruns `e`. Caveats to carry into the figure: the wait is inferred, not a measured
queue time; a track is a lane (its core is confirmed separately, above); and a DPU span includes the
in-`run()` int8 quantize/dequantize. Traces on disk (`results/benchmark_stream/traces/`): per-arch
across the scaling curve (`{arch}_{X}lane_L20.csv`, X through each operating point + oversubscribed
48/64/96 — out to **128** for the factorized archs (FP/ResFP) and **96** for the hyperprior ones
(SHyp/ResSHyp; 128L = 384 runners crashes)), the 3-lane pinned set (`{arch}_3lane_pinned.csv`), and
`ResSHyp_3lane_lanemajor.csv` / `ResSHyp_4lane.csv` for the collision / oversubscription cases. Each
oversubscribed trace has a companion `coreid/{arch}_{X}lane_L20.coreid.log`.

**Trace window.** `--trace` (fanout-only) logs every stage of every patch — each lane writing its own
lock-free timeline — across the compress phase, with `t = 0` set the instant the workers launch, so the
first row-block read sits inside the window (`stream_pipeline.cpp`). The traced region is a
**top-of-scene subsample**: `make_grid` capped to `--max-rows` row-blocks (58 patches each), which the
sweep sets to `max(3, ⌈lanes/6⌉)` ≈ **10 patches/lane** — 174 patches (≤16 lanes) up to 1 276 (128
lanes), a **1–32 s slice** vs the ~38 s full-scene FP sweep. That slice's effective patch/s tracks the
full scene within ~5 % (the gap is pipeline fill), so it is steady-state-representative. Because
`t = 0` precedes the fill, the **steady-state analysis window is chosen offline** — trim the leading
fill (and trailing drain) row-block before scoring, no re-capture needed. Every span is wall-clock: a
DPU span includes the synchronous `execute_async` queue-wait, a CPU span includes scheduler preemption
— neither is pure exec (hence the `e = median` correction below).

**Pure-exec proxy `e`.** From the 1-lane trace (no contention), `e` = the **median** exec of each kernel;
spread is the **coefficient of variation** (CoV = std/mean). The distributions are tight — plain `g_a`
5.23 ms (CoV 2.5 %), residual `g_a` 36.6 ms (CoV 0.4 %), `h_a`/`h_s` ~1.2–1.5 ms — so the median is a
sharp proxy and min/max give a narrow band. This same characterization *is* the measured per-arch kernel
time the rest of the doc otherwise cites approximately (plain vs residual `g_a` = 7×). Code:
`fanout_occupancy.py`.

**Per-subgraph DPU time & efficiency** (vaitrace hardware counter, per DPU core; our canonical DPU-time
reference — data `results/benchmark_stream/vaitrace/vt_*.txt`, tool details `docs/AMD_Vitis_AI.md`):

| subgraph | WL (GOP) | HW_RT (ms) | SW_RT (ms) | Effic | LdWB (MB) | AvgBw |
| --- | --- | --- | --- | --- | --- | --- |
| plain `g_a` (FP, SHyp) | 4.512 | 4.10 | 4.48 | 89.5 % | 1.175 | 1.61 GB/s |
| residual `g_a` (ResFP, ResSHyp) | 39.755 | 33.51 | 35.84 | 96.6 % | 3.520 | 0.65 GB/s |
| `h_a` (hyperprior) | 0.275 | 0.83 | 0.97 | 27.1 % | 4.688 | 5.74 GB/s |
| `h_s` (hyperprior) | 0.176 | 0.52 | 0.62 | 27.8 % | 3.001 | 6.01 GB/s |

**HW_RT** = pure DPU compute (hardware counter); **SW_RT** = the `run()` span (dispatch + int8 requant
around it); **Effic** = achieved GOP/s ÷ 1229 — the big convs fill the DPU, the tiny `h_a`/`h_s` cannot.
HW_RT is **lane-stable**: flat to 64 lanes for residual `g_a`, a bounded +8 % step for plain `g_a`; our
trace `e` matches SW_RT to ~2 %.

**`h_a`/`h_s` are weight-load-bound, not DDR-bound.** **LdWB** (external-memory weight+bias read, MB)
is 97–99 % of their total DDR traffic (vs 16 % for residual `g_a`, which is feature-map-dominated —
`LdFM`+`StFM` ≫ `LdWB` there); the two runs above (SHyp, ResSHyp) measure the same `h_a`/`h_s` weights
and agree to <0.2 %.

The roofline figure (`LaTeX/SAR_DDC_FPGA_DATE27/figures/scripts/roofline_subgraph.py`) plots this as a
third ceiling, **measured DPU weight-load bandwidth = 6.11 GB/s** — `achieved GOP/s ÷ AI` for `h_a`/`h_s`,
i.e. the same static xmodel byte-basis the figure's x-axis already uses (not vaitrace's raw `AvgBw` of
5.7–6.0 GB/s above, which uses a different, dynamic byte count and would put the points slightly above
the line). That's only ~34–36 % of the 17.06 GB/s DDR peak, and below even the tighter **9.6 GB/s
per-core AXI interface peak** (2× 128-bit `M_AXI_DATA` ports per DPUCZDX8G core at the 300 MHz DPU
clock, PG338 — the true single-core bound, since one core can't exceed its own interface width
regardless of DDR headroom). The figure shows both the DDR and AXI ceilings, pending a decision on
which to report.

This number is **single-core, uncontended (L1)** — under multi-lane fan-out it degrades and varies by
core: at 6 lanes, `h_a`'s bandwidth ranges 2.2–6.0 GB/s across the 3 cores (vs. 5.7 GB/s at L1), with
efficiency as low as 10.6 % (`results/benchmark_stream/vaitrace/vt_ResSHyp_L{1,3,6}.txt`). Not a fixed
hardware constant — the best-case rate for this access pattern (many small, poorly-reused weight
tensors) in isolation.

**Why the occupancy stays runner-span.** It uses `e` = the `run()` span (≈ SW_RT), not HW_RT. Charging
`e` = HW_RT would count the per-call CPU glue (SW_RT−HW_RT = **8.4 %** plain `g_a`, **6.5 %** residual
`g_a`, 14–17 % for the tiny `h_a`/`h_s`) as *idle/wait* — but that glue is necessary dispatch + requant,
not idle. So pure-DPU-compute busy is only ~6.5 % (residual archs) / ~8.4 % (plain-`g_a` archs) below the
reported runner-span occupancy.

**Per-core DPU occupancy.** For each core (lane→core = subgraph-major creation-order round-robin —
lane *k*'s `g_a` → core *k* mod 3, validated against the `device_core_id` logs), busy = the union of the
reconstructed compute intervals `[t1 − e, t1]` on it over the **steady-state window** (last lane to
start → first lane to finish, trimming fill + drain). Result (occupancy panel of the lane-scaling figure,
all archs × lanes; `fanout_occupancy.py`): the residual archs saturate the 3 cores (~97–100 %), the light
archs plateau ~⅓ idle (~63–66 %) — so their roof is not the DPU. Two mechanisms surface: **~2 lanes/core**
are needed to hide the CPU-feed gap (ResFP 86 %→99.7 % from 3→6 lanes, its peak), and the **4-lane dip**
is a load imbalance (the round-robin stacks two heavy `g_a` on one core when lanes ≠ 3·k).

### CPU occupancy & memory footprint

**What binds the light archs — measured on-board.** `mpstat -P ALL` + `pidstat` (kernel scheduler
accounting, no `perf` needed) at each arch's operating point resolve the CPU side that the DPU occupancy
leaves open. The split is direct in the occupancy panel (dashed CPU %usr vs solid DPU busy): **FP/SHyp are
CPU-bound** (CPU ~79/82 % user-space > DPU ~66/63 %), **ResFP/ResSHyp are DPU-bound** (DPU ~100/96 % > CPU
~14/19 %). For the CPU-bound archs the entropy rANS (~6.4 ms/patch) + normalize (~3.7 ms) dominate; kernel
time is only ~5 %.

**Why FP roofs at ~60 % of the naïve 4-core ceiling** — two measured effects, not a thread shortage:
per-patch compute **inflates ~30 %** under 64-thread load (12.1 → 15.8 ms, cache/DDR contention), and
**~17 % core idle persists** at every lane count from 16L on — a *balanced* CPU↔DPU pipeline where neither
side fully saturates, so extra threads cannot fill the idle and cost more than they add (128L < 64L).
Waterfall: ideal 331 → after inflation 253 → actual 199 patch/s. Figure: `fp_cpu_binding.png` (§11);
per-arch CPU %usr via `fanout_occupancy.cpu_occupancy_series`.

**Thread affinity — a considered, unmeasured lever.** That inflation is cache/DDR contention under heavy
oversubscription (far more workers than cores), where the scheduler may migrate a worker between cores and
lose its warm cache each time. Pinning each worker to a fixed core would keep caches warm and could
reclaim part of it — but it is unpursued here and unverified: it would likely help only *paired with* the
knee (few, pinned workers), since at the peak the extra lanes are partly hiding latency, and it does
nothing for the balanced-pipeline idle. Confirming it would need its own affinity × lane-count sweep.

**Process memory footprint** (measured `/proc/<pid>/status` + `/proc/meminfo`, at 4 → 64 fan-out lanes):

| arch | VmHWM (peak RAM) | CMA (DPU DMA) | 64L VmSize | threads |
| --- | --- | --- | --- | --- |
| FP | 214 → 566 MiB | 165 → 533 MiB | 2.9 GiB | 6 → 66 |
| SHyp | 223 → 657 MiB | 176 → 600 MiB | 3.0 GiB | 6 → 66 |
| ResFP | 213 → 527 MiB | 162 → 477 MiB | 2.9 GiB | 6 → 66 |
| ResSHyp | 213 → 591 MiB | 168 → 569 MiB | 3.0 GiB | 6 → 66 |

Footprint is set by **fan-out lanes, not architecture**: each lane adds a worker thread + in-flight patch
buffers + its DPU runner's CMA buffers (~6 MiB/lane), so the hyperprior archs (3 DPU runners/lane) use a
little more than the factorized (1). Peak ≈ 0.6 GiB resident RAM + ≈ 0.6 GiB CMA ≈ **1.2 GiB, well inside
the 3.84 GiB board**. The ~3 GiB `VmSize` is almost all *reserved* thread stacks (~8 MiB × 66), not
resident RAM — `VmHWM` is the real number.

- **VmRSS** resident RAM in use now · **VmHWM** its lifetime peak · **VmSize/VmPeak** reserved virtual
  address space (≫ RAM, mostly thread stacks) · **VmData** heap · **CMA** (Contiguous Memory Allocation)
  = the kernel pool of *physically-contiguous* RAM the DPU DMAs weights/feature-maps from (1.5 GiB
  reserved here); it grows per DPU runner, so it — not total RAM — is the tighter ceiling at extreme lane
  counts (contributing, with the XRT runner wall, to the very-high-lane hyperprior crashes).

---

## 7. Experiment E1 — symmetrization granularity (local GPU, no retraining)

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

## 8. Overlap study (U5) — seam vs cost (motivates the overlap-2 default)

Independent per-patch compression leaves a seam/edge artifact when the tile is recombined; a small patch
overlap with ramp-blending removes it. This study sets the **overlap-2 default** used everywhere in §10.

**Complete.** Full-scene sweep FP + ResSHyp × overlap {0,2,4,8,16} (cold+warm), scored vs the project's
own MERLIN full-tile GT (`merlin_full_gt.py`; whole-image-symmetrized, 64-px seam-free blend).
Reconstruction = board INT8 decode (`stream_pipeline --decode`) + host ramp-blend/stitch (`stitch_ddc.py`,
`src/utils/tiling.py`); grid offsets `stride = 256 − overlap`, snapped to the edge; blend + metrics in
**linear amplitude** (same basis as §7). Data → `results/benchmark_stream_overlap/overlap_table.{md,csv}`.

**Seam metric.** Full-tile means dilute the seam ~21× (seam band = ±3 px of the 256-grid ≈ 4.6 % of
pixels), so the sensitive number is the seam deficit `Δseam = PSNR_interior − PSNR_seam`.

**Result — overlap 2 closes the seam at ~1 % cost; more buys nothing.** PSNR per overlap as *seam /
full-tile* (dB, clipped to `AMP_LIN_99`), with the seam deficit at overlap 0 and 2:

| config (λ=20) | ov0 seam/full | ov2 | ov4 | ov8 | ov16 | Δseam@ov0 | Δseam@ov2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 23.09 / 24.19 | 24.19 / 24.23 | 24.23 / 24.25 | 24.28 / 24.27 | 24.41 / 24.29 | **1.16** | 0.03 |
| ResSHyp | 23.43 / 25.37 | 25.43 / 25.44 | 25.46 / 25.48 | 25.48 / 25.51 | 25.45 / 25.53 | **2.06** | 0.01 |

The λ=1000 runs show the same shape (larger seam at ov0, closed by ov2 — the deficit grows with
quality: ResSHyp λ1000 Δseam@ov0 = 2.75) — the conclusion is rate-robust; data retained in
`overlap_table.md`.

**Cost of ov2:** latency +0.7–1.0 % warm / free cold-FP (read-bound); downlink +0.78 % (patches
7 482→7 540, bpp flat); J/patch flat. Overlap 16 costs +14–19 % latency / +14.35 % downlink for no
further seam gain → **reconstruct at overlap 2**.

**INT8 bright-scatterer cap (metric-invisible).** The DPU `g_s` output is INT8 at fix-point 8, so recon
amplitude hard-caps at **2100.1** (code 127 → `exp(0.496·range+min)`); the float model reaches ~85 k
(FP) / ~68 k (ResSHyp). This clips the brightest **~0.7 %** of pixels (point scatterers). Invisible to
the reported metrics (all clip to `AMP_LIN_99 = 545` first), which also sets the SSIM `data_range` (the
fixed, cross-model-comparable basis; the convention lives in `src/utils/metrics.py`). A fix-point-7
requant would lift the cap to ~44 k at half precision, if ever needed.

---

## 9. `.ddc` downlink product (v1 — locked)

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
access matches, decode(file) ≈ decode(direct) within float32 ε; container overhead ≈ header + 8·n bytes
(negligible). The C++ streaming writer (step 7) must emit these exact bytes; the Python codec is the oracle.

---

## 10. Results

**The sweep.** `stream_sweep.py` deploys each model and runs the cumulative optimization ladder on the
full **7 540-patch** Hamburg scene (overlap 2, snap grid, §3), **cold with a paired warm run per rung**,
through the harness (§4); the fan-out configs (§5) use the same harness with `--fanout`. All four archs,
λ=20 — throughput is **λ-independent** (L20 = L1000 within ~1 %, since rANS time scales with the *number
of latents*, not bpp). Results in `results/benchmark_stream/<model>/*.json` + `ablation_table.md` /
`fanout_full_table.md`.

**Best configuration per architecture** (λ=20, seed s0, warm):

| arch | configuration | patch/s | SLC MB/s | J/patch |
| --- | --- | --- | --- | --- |
| FP | p0 + fan-out 64L + prefetch + neon | 199.6 | 51.1 | 0.081 |
| SHyp | p0 + fan-out 48L + prefetch + neon | 145.5 | 37.2 | 0.103 |
| ResFP | p0 + fan-out 6L + prefetch + neon | 41.0 | 10.5 | 0.487 |
| ResSHyp | p0 + fan-out 24L + prefetch + neon | 38.5 | 9.9 | 0.557 |

Fan-out is the best config for **every** arch. The DPU-bound archs plateau early as the 3 cores
saturate (ResFP by ~6, ResSHyp by ~12 lanes, ~10 MB/s), peaking at 6 / 24 lanes; the CPU-bound ones
keep gaining from independent lanes far past the cores, peaking at 64 (FP) / 48 (SHyp). The operating
points above are those warm peaks; on the real SD card the light archs are read-bound at ~22 MB/s (cold).

**The full ladder** (warm = compute ceiling, tile from RAM; cold = SD-card testbed). Rows are cumulative
(each = the row above + the named flag): `+p0` = 4-worker pool, `+prefetch` = double-buffered row-block
read, `+neon` = vectorized normalize. The **fan-out (roof)** row takes the `+neon` stack and swaps `s1`
→ N independent DPU lanes (no shared mutex), at each arch's operating lane count (§5/§10: FP 64 / SHyp 48 /
ResFP 6 / ResSHyp 24).

*Throughput — warm patch/s (cold in parens where the SD read binds); **bold** = best per arch:*

| optimization | FP | SHyp | ResFP | ResSHyp |
| --- | --- | --- | --- | --- |
| seq | 37.2 (26.7) | 29.5 (22.5) | 11.2 (10.0) | 10.3 (9.3) |
| +s1 | 44.2 (30.2) | 33.9 (24.9) | 18.5 (15.4) | 16.4 (14.0) |
| +p0 | 119.0 (52.8) | 85.2 (44.8) | 25.7 (20.2) | 22.1 (17.9) |
| +prefetch | 125.5 (85.7) | 90.8 (86.5) | 26.3 | 23.0 |
| +neon | 135.6 (86.5) | 95.3 (85.7) | 26.3 | 23.4 |
| fan-out (roof) | **199.6 (85.9)** | **145.5 (85.3)** | **41.0** | **38.5** |

*SLC MB/s = patch/s × 0.256 (the 1.93 GB tile over 7 540 patches). Cold shown only where it binds:
FP/SHyp are read-bound (parens throughout), ResFP/ResSHyp compute-bound (cold ≈ warm from `+prefetch`
on).*

*Energy — warm J/patch; **bold** = the best-throughput config's energy:*

| optimization | FP | SHyp | ResFP | ResSHyp |
| --- | --- | --- | --- | --- |
| seq | 0.271 | 0.336 | 1.046 | 1.163 |
| +s1 | 0.234 | 0.298 | 0.739 | 0.855 |
| +p0 | 0.112 | 0.145 | 0.610 | 0.731 |
| +prefetch | 0.108 | 0.140 | 0.604 | 0.714 |
| +neon | 0.103 | 0.134 | 0.603 | 0.706 |
| fan-out (roof) | **0.081** | **0.103** | **0.487** | **0.557** |

*Energy = MPSoC (PS+PL) INA226, cooldown-gated to 58 °C; J/patch is the comparable metric (avg W drifts
with thermal). The full per-rail-group breakdown is in each result JSON.*

**Findings.**

- **Parallelism is arch-dependent, and fan-out is the universal best.** The DPU-bound archs
  (ResFP/ResSHyp) are unlocked by fan-out lanes (`--s1`, §4, is an early scaffold fan-out supersedes); the
  CPU-bound archs (FP/SHyp) by `--p0` +
  `--prefetch`, and fan-out's independent pipelines lift them further still. Cumulative warm, seq→best
  (fan-out roof): FP **37.2→199.6 patch/s (5.4×)**, ResSHyp **10.3→38.5 (3.7×)**.
- **Storage was hiding the CPU parallelism.** At `+p0` (no prefetch yet) FP jumps **52.8→119.0 patch/s**
  cold→warm — the 4-worker pool is present in both, but cold it stalls on the SD read; `+prefetch` then
  recovers most of it cold (85.7). The cold/warm gap at the best config is the storage-boundedness
  signal: large for FP/SHyp (read-bound), ~zero for ResFP/ResSHyp (compute-bound, read fully hidden).
- **Parallelism costs power but saves energy.** FP seq→best **0.271→0.081 J/patch (3.3× better)**;
  ResSHyp **1.163→0.557 (2.1×)**. The draw is PL(DPU)-dominated for every arch (operating point: FP
  13.5 W PL / 2.9 W PS, ResSHyp 18.9 / 2.5) — the residual archs push far more of it; cross-arch ResSHyp
  costs **~7× the energy/patch** of FP.
- **DDR is not a bottleneck.** vaitrace: the dominant DPU traffic (`g_a`/`g_s`) is ~651–660 MB/s, ~20×
  under the 17.06 GB/s DDR4 ceiling (§2). CPU-side DDR is an estimate (~100–150 MB/s; no `perf` on the
  board) — the ~20× margin holds either way.
- **The CPU-bound archs are limited by compress *compute*, not the DPU or the OS.** At FP's roof the 4
  A53 cores fill to ~79 % user-space (rANS + normalize) while the DPU stays ~⅓ idle and kernel time is
  ~5 %; the roof mechanism (compute inflation + a persistent balanced-pipeline idle, waterfall
  331→253→199 patch/s) is **§6**. See `fp_cpu_binding.png` (§11).
- **Compression** (byte-identical across every config; **λ=20, the chosen operating point**): FP bpp
  0.156 → **~202× vs raw int16** (9.6 MB `.ddc`); ResSHyp bpp 0.133 → **~237×**. A deliberate rate
  point: λ=1000 buys ~+1.8 dB (FP) / +3.0 dB (ResSHyp) PSNR (§7) at roughly **10× less compression**
  (FP 21×, ResSHyp 26×). Peak DDR ~0.3 GB windowed vs ~3 GB free (§2).
- **Classical SAR baseline (N5) — paper-only, no reimplementation** (`main.tex` §Background +
  §Evaluation "SAR Image Compression Baseline"). No CCSDS standard targets SAR, raw or focused: the one
  study to try applied CCSDS 123.0-B to raw, pre-focusing SIR-C/X-SAR echoes (split into independent
  real/imag channels), landing in a similar low-single-digit-× regime to BAQ — TerraSAR-X's own onboard
  codec, 1.3–4× from its 8-bit ADC [Pitz & Miller 2010] — per [Prette, Magli & Bianchi 2019]. Neither
  transfers cleanly here: that regime is set by raw-echo fidelity requirements the already-focused SLC
  doesn't have, and neither codec despeckles. The closer reference is Amao-Oliva et al. (2024/2025) —
  same MERLIN despeckling base — whose MERLIN+JPEG2000 cascade needs ~0.3 bpp on their own TerraSAR-X
  scene; the operating point above sits at roughly half that bitrate (order-of-magnitude only — different
  scene, not a controlled comparison).
- **vs the mission objective** (full derivation → `docs/TerraSAR-X_objective.md`): TSX StripMap produces
  SLC at **211 MB/s** (working point) / 358 MB/s (worst case). One ZCU102 at its best config (FP
  fan-out roof, **51.1 MB/s warm**) is **~7× short of full-duty real-time**, but **meets the
  process-before-next-contact deadline** (FP, ~4.4× headroom) and the ~200× compressed product sits far
  inside the 270 Mb/s-net downlink.

---

## 11. Figures

Manuscript figures live in the LaTeX repo (`LaTeX/SAR_DDC_FPGA_DATE27/figures/scripts/`: system
dataflow, stacked time-per-patch, overlap, optimization ladder). The fan-out figures are built here:

- **`fanout_lane_plot.py`** → the lane-scaling figure `fanout_lane_scaling_4arch.png` (throughput /
  occupancy / energy vs lanes; occupancy panel: **solid = DPU busy, dashed = CPU %usr** ★ = operating point).
  CPU curves via `fanout_occupancy.cpu_occupancy_series` (mpstat, `results/benchmark_stream/cpu_probe/`).
- **`fanout_occupancy.py`** → per-core DPU occupancy from the `--trace` CSVs (§6 method + the
  `[t1 − e, t1]` reconstruction); `__main__` prints the per-core table.
- **`fanout_cpu_fp.py`** → the FP CPU-binding figure `fp_cpu_binding.png` (throughput / CPU-vs-DPU
  occupancy / user-CPU-per-patch vs lanes); reads `results/benchmark_stream/cpu_probe/{mpL,spL}_*.log`
  (`mpstat`/`pidstat`, §6 + §10).
- **`fanout_full_table.py`** (`--lam 20`) → the per-lane cell table `fanout_full_table.md` (patch/s |
  J/patch across lanes — the tabulated data behind the §5 figure); reads
  `results/benchmark_stream/<model>/p0_t{1..4}_fo_*.json`.
- **`fanout_gantt.py`** → per-lane execution Gantt from a `--trace` CSV (§6); reads
  `results/benchmark_stream/traces/*.csv`.
- **`fanout_lane_diagram.py`** → per-lane stage-time bars across lane counts.

The older `stream_gantt` / `stream_roofline` / `stream_sysplot` sketches (`scripts/fpga/benchmark/`) are
first-pass feel-only plots, superseded by the manuscript figures.

---

## 12. Jetson embedded-GPU baseline (N2)

**Status: implemented, board-verified across the full 4-arch × 4-power-mode matrix, quality spot-
checked.** A naive, unoptimized baseline by design (mirrors the FPGA's `seq` mode: no threading, no
fan-out, no DPU-style placement — not a re-run of the systems-engineering ladder on different silicon).
Supplies the recognisable `N× vs a named baseline` DATE expects. **→ main.tex §eval + abstract.**

Code, environment setup, deployment recipe, and known quirks (incl. a cross-GPU decode gotcha) →
`inference_edge/README.md`. Package = `inference_edge/` (`ddc-edge` CLI). Sweep orchestration →
`scripts/evaluation/jetson_power_arch_sweep.py` (handles the nvpmodel mode-switch reboot —
`MODE_30W`/`MODE_15W` disable CPU cores relative to `MAXN`/`MODE_50W`, which this L4T's nvpmodel build
requires a reboot to apply); table built by `scripts/evaluation/jetson_power_arch_table.py`.

**Results — full sweep** (Orin, all 4 λ=20/relu archs, overlap=2, full scene, 7,540 patches — grid count
matches §3 exactly). `g_a ms/patch` covers both real+imag calls per patch (single timer scope). `latency
ms` = 1000/patch_s, the full per-patch figure since the pipeline is strictly sequential (no overlap
between patches). `W (total)` is the whole-board figure (every tegrastats rail — the headline figure for
now, project decision 2026-08-24); `W (compute)` excludes the board-I/O/DRAM rail (`VIN_SYS_5V0`),
comparable to the FPGA's own peripherals-excluded convention — this is the figure main.tex's cross-
platform table uses. `mJ/patch` is computed from the total.

| Mode | Arch | g_a ms/patch | patch/s | latency ms | W (total) | W (compute) | mJ/patch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAXN | FP | 2.94 | 109.23 | 9.15 | 19.52 | 14.85 | 177.7 |
| MAXN | ResFP | 11.08 | 57.18 | 17.49 | 32.02 | 26.52 | 558.5 |
| MAXN | SHyp | 2.93 | 39.84 | 25.10 | 16.30 | 11.87 | 408.4 |
| MAXN | ResSHyp | 11.13 | 29.97 | 33.36 | 24.00 | 19.10 | 799.9 |
| MODE_50W | FP | 4.13 | 76.14 | 13.13 | 12.80 | 8.37 | 167.5 |
| MODE_50W | ResFP | 16.45 | 39.19 | 25.52 | 18.12 | 13.16 | 461.9 |
| MODE_50W | SHyp | 4.19 | 27.89 | 35.86 | 10.86 | 6.78 | 388.8 |
| MODE_50W | ResSHyp | 16.49 | 20.76 | 48.17 | 14.15 | 9.71 | 680.9 |
| MODE_30W | FP | 5.37 | 74.44 | 13.43 | 11.89 | 7.48 | 159.1 |
| MODE_30W | ResFP | 21.45 | 33.84 | 29.55 | 15.60 | 10.77 | 460.2 |
| MODE_30W | SHyp | 5.36 | 29.28 | 34.15 | 10.53 | 6.38 | 359.0 |
| MODE_30W | ResSHyp | 21.49 | 19.94 | 50.15 | 13.04 | 8.58 | 653.5 |
| MODE_15W | FP | 7.96 | 49.37 | 20.25 | 8.66 | 4.77 | 175.2 |
| MODE_15W | ResFP | 31.86 | 22.63 | 44.19 | 10.85 | 6.66 | 479.0 |
| MODE_15W | SHyp | 7.94 | 19.38 | 51.61 | 7.65 | 3.99 | 394.2 |
| MODE_15W | ResSHyp | 31.96 | 13.16 | 76.00 | 9.09 | 5.19 | 690.5 |

Source: `results/benchmark_jetson/orin/power_sweep/<arch>_<mode>.json` (`summary.csv` alongside).
Compress-only sweep — no per-mode quality/verify pass (quality below is SHyp only).

Two patterns worth flagging, not fully explained: **(1)** SHyp is consistently slower than ResFP despite
ResFP's `g_a` costing ~3.8× more — the hyperprior's extra `h_a`/`h_s` plus a second entropy-coding pass
(EB for `z`, GC for `y`, vs. FP/ResFP's single EB pass) apparently outweighs one heavy `g_a`. **(2)**
`mJ/patch` is U-shaped for most archs, bottoming around `MODE_30W` rather than falling monotonically
with the power cap — at `MODE_15W` throughput drops faster than power draw, so the most power-
constrained mode is *not* the most energy-efficient one.

**Quality** (SHyp only — no per-arch quality sweep yet): verified against MERLIN GT (overlap=0, full
7,482-patch coverage, decoded on-device — see the cross-GPU decode note in the README): **PSNR 28.05 ±
5.24 dB, SSIM 0.8144 ± 0.1055**, consistent with the neighboring FP/ResSHyp λ=20 reference numbers.
Visual crop comparison → `results/benchmark_jetson/orin/jetson_vs_fpga_vs_merlin_crop.png`.

Timing/power methodology independently audited (`StageTimer` cross-checked against `torch.cuda.Event`,
timer overhead and `--power`'s own perturbation both confirmed negligible) — full report:
[[project_jetson_edge_pipeline]] memory.

**Remaining:**

- Per-arch quality (PSNR/SSIM) sweep — currently SHyp only.
- Thor: blocked on a torch/NCCL aarch64 ABI gap, deferred — see `inference_edge/README.md`.
- TensorRT/FP16 — explicitly out of scope for this baseline, a separate future conversation if wanted.

---

## 13. TODO

**Guiding principle:** run what is most informative, let the results (not the outline) drive the
narrative, and be ready for any experiment to resolve *against* the story. Each entry notes the
manuscript slot it *would* unblock (**→ main.tex …**) purely as navigation, never as a hole that must be
filled.

**N6 — Optimize the entropy coding (partially done).** Profiling is done and one optimization from it is
already implemented + committed (`4ddbcc8`: flattened CDF table + a precomputed reciprocal per table
entry, replacing a division). Temporary notes still exist: **`docs/tmp_entropy-coding_opt_opportunities.md`**.
**Before submission — fix the toggle, not the algorithm**: entropy-on/off is currently a *commit*-level
switch (off = build the pre-`4ddbcc8` state, e.g. `324744f`), not a runtime one like every other pipeline
optimization (`--fanout`/`--prefetch`/`--neon`). Wrap the reciprocal-symbol path behind a real flag (e.g.
`--entropy`) so both states are reachable from one build — cost real time during the DATE27 ladder-figure
data collection (surgical revert of `entropy_models.{cpp,hpp}` + `rans/rans_interface_cxx.{cpp,hpp}`,
rebuild, run, restore, rebuild again, per entropy-off batch), exactly what a flag would make trivial.

**A5 — Hardware platform details (HW-community venue).** **→ main.tex §Background/Setup (platform
table).** Report the accelerator's internal design: DPU `3× B4096 @ 300 MHz` (check whether the DSPs run
at double clock), PS DDR4 ≈17 GB/s, ZU9EG (base facts in §2), **plus PL resource utilisation**
(LUT/FF/BRAM/URAM/DSP) and clocks from the Vivado/DPU report — as a short platform table in `main.tex`.

**N7 — Full result re-verification pass (pre-submission).** Every number that goes in the paper gets
recomputed from a clean, current-`HEAD` sweep before submission — development happened too
unsequentially (interleaved commits, stashes, branch swaps) to trust that today's `results/` trees are
all mutually consistent with each other or with the current codebase. Not urgent: the current push is
the draft/skeleton with figures and tables; this is the last pass, right before submission.

### Deferred / optional

- **N3 — Deadline/budget-driven rate allocation** (vary λ across the scene under a bit-budget or
  wall-clock deadline). **Cancelled** — tricky to implement cleanly.
- **N4 — INT8 `g_s` output cap** at 2100.1 (clips the brightest ~0.7 % of pixels). Metric-invisible;
  lives as a one-line **limitation** in `main.tex` §Discussion, not a deepening study.
- **On-ground SHyp `.ddc` decoder** (optional, decoupled). Reproduce the board INT8 `h_s` on host so
  SHyp/ResSHyp `.ddc` decode in pure Python (FP already does); hard because the Gaussian decode needs
  every element's scale in the same `gc_scale_table` bucket and an FP32 `h_s` desyncs rANS. The overlap
  study (§8) sidesteps it via board decode; value if built = off-board verification + a real ground
  decoder.
