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
The N7 re-verification sweep is done (§13) — canonical numbers now live in `results/date27/`
(`MANIFEST.md`); remaining work is the DATE'27 figure/writing plan (`DATE27_paper_plan.md` §4).
Last updated 2026-09-01.

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
- **Achievable PS-DDR4 read ceiling (literature) ≈ 13.7 GB/s** — microbenchmarks reach ~75–81 % of
  the theoretical peak: **Lu 2022** measures **13.7 GB/s @ 300 MHz** on a ZCU104 whose controller
  runs the DIMM at DDR4-2133 — the same config as our SODIMM [TRETS, p.19 / §5.3.4 / Fig. 17(b)];
  **Manev 2019** measures 14.4 GB/s on a ZCU102 carrying a **DDR4-2400** SODIMM, best with 3
  concurrent HP ports (4 ports is slower) [ICFPT, Table I + §IV] — their 19.2 GB/s peak is correct
  *for that SODIMM revision*, ours is 2133 → 17.06. Paper use: cite-only (the roofline draws the
  theoretical 9.6 / 17.06 ceilings; the ~13.7 GB/s is quoted in text — it implies `h_a`/`h_s` hit
  DDR even sooner than the drawn line). BibTeX: `luDemystifyingSoftHardened2022`,
  `manevUnexpectedDiversityQuantitative2019`.
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

**Board builds: always `make clean` first.** The ZCU102 clock is unset, so `make` clock-skew
mis-triggers incremental builds — any A/B comparison or measurement campaign must full-rebuild
(`make clean && make -j4`), verified the hard way during the N6 Stage-1 A/B.

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
  the INT8 `g_a` step, so no quantisation flips). **2.42× faster normalize in isolation**.
- **`--entropy` — rANS reciprocal-table coder.** Flattened CDF table + a precomputed reciprocal per entry,
  replacing a per-symbol division in the flush loop. Activable with a runtime flag: the pre-optimization
  CDF-lookup + divide path is kept alongside it for A/B (both live in `entropy_models.{cpp,hpp}` /
  `rans/rans_interface_cxx.{cpp,hpp}`, selected once per `compress()` call).
  **Table sizes, so the coder's constants don't get conflated:** the rANS probability scale is
  `precision = 16` (`rans_interface_cxx.cpp`), i.e. frequencies sum to $2^16$ = 65 536 — that 65 k is a
  *normalization total, not a table length*. The CDF tables themselves are per-channel and much smaller:
  entropy bottleneck **256 × 28 = 7 168 int32 (28 KB)**, Gaussian conditional **64 × 3 133 = 200 512
  int32 (802 KB)** (`entropy_params/{eb,gc}_quantized_cdf.npy`). The GC table being ~28× the EB one is
  the structural reason `gc_compress` costs ~9.7 ms/patch against `eb_enc`'s ~4.7 (§6).
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

**How to run** the fan-out lane grid: Per arch: deploy once, then loop `stream_benchmark.py` over the
lane counts, cooldown-gated, writing into `results/date27/lanes/<arch>/`.

```bash
ARCH=ResSHyp                                    # repeat for FP, SHyp, ResFP
python scripts/fpga/deploy/deploy.py --model-name ${ARCH}-relu_s0_L20_pt \
    --skip-compile --skip-infer --skip-fetch    # push xmodel + entropy_params to the board
for N in 1 2 3 4 5 6 8 10 12 16 20 24 32 48 64; do
  python scripts/fpga/benchmark/stream_benchmark.py \
      --schedule p0 --fanout --neon --prefetch --entropy --threads $N \
      --keep-cache --overlap 2 --cooldown --power --iters 1 \
      --trace /tmp/occ_t${N}.csv \
      --out results/date27/lanes/${ARCH}/t${N}_fo_neon_pf_ent_warm.json
  scp ZCU102:/tmp/occ_t${N}.csv results/date27/lanes/${ARCH}/t${N}_occtrace.csv
done

# If you want the plots to:
python scripts/figures/fanout_lane_plot.py      # -> LaTeX/SAR_DDC_FPGA_DATE27/figures/images/lane_scaling.{pdf,png}
```

`iters=1` is enough on full tile; `--cooldown` gates each run to ≤58 °C so accumulated heat doesn't bias
later lanes (§8). `stream_benchmark.py` passes `--trace` through to `stream_pipeline` on the board but
does not fetch the CSV — the `scp` line does that.

**Lane scaling.** Warm throughput / DPU + CPU occupancy / energy vs lane count (λ=20) are the three
panels of `lane_scaling.png`; the per-lane cells (patch/s, J/patch, power, `g_a` ms/call)
live in the result JSONs (`results/date27/lanes/<arch>/t{N}_fo_neon_pf_ent_warm.json`, the lane
grid — full CPU-opt stack, entropy-on).
**Every arch roofs, and the roof height is set by the binding resource** (§6): the DPU-bound archs
(ResFP, ResSHyp) roof **low and early** — ~40 patch/s, flat from ~6 lanes; the CPU-bound archs (FP, SHyp)
roof **high and late** — FP to ~205, SHyp to ~147 patch/s. Fan-out is the best schedule for all four
archs (§10), well past the `--s1` ladder; the earlier "hyperprior 4-lane cliff" was an artifact
of stopping at 4 lanes — ResSHyp dips at 4L, then climbs to ~38 by ~12L.

**The XRT runner wall.** The two hyperprior archs create **3 DPU runners per lane** (`g_a`, `h_a`,
`h_s`), so the wall (`VART_XRT_NULL_PTR`) is a runner-count ceiling: 112 L (336 runners) works, 113 L
(339) fails — the exact boundary, bisected with single-row `--fanout` probes. It is not an XRT/DPU
hardware ceiling: it's the ZCU102 shell's default per-process open-file-descriptor limit
(`ulimit -n` = 1024), which each DPU runner eats into; raising it with `prlimit --nofile` for the one
process clears the wall entirely (verified past 360 runners). The factorized single-`g_a` FP/ResFP
create one runner per lane and never approach it. CMA is the softer co-factor (§6). The 128 L point is
dropped from the lane figure.

**Operating point per arch** = the **knee**: the smallest lane count within ~1 % of the warm-throughput
peak, clear of the XRT wall. Trading ≤1 % throughput for far less oversubscription (fewer threads, less
memory, more XRT margin), it is what the paper reports and what the r0–r7 ladder pins across every rung:
**FP 12 / SHyp 21 / ResFP 6 / ResSHyp 15 lanes**. The unconstrained warm
peak sits higher and later for the CPU-bound archs (FP 206.7 @ 48 L, SHyp 147.6 @ 48 L) but buys nothing
over the knee; the DPU-bound archs peak essentially at their knee (ResFP 41.1 @ 6 L, ResSHyp 38.4 @ 15 L).
The knee also sidesteps SHyp's noisy iters-1 48 L peak. Reference table: **§10**.

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

**Placement ablation** (naive lane-major vs pinned subgraph-major, warm, 3 lanes, λ=20 — the ladder's
r2→r3 step, `results/date27/ladder/<arch>/r2_fo_t3_lanemaj_warm.json` vs `r3_fo_t3_warm.json`). The fix
only bites when a lane creates >1 DPU runner (hyperprior) **and** the arch is DPU-bound — so it is a
clean no-op elsewhere:

| arch | naive (lane-major) patch/s | pinned patch/s | pinned speedup |
| --- | --- | --- | --- |
| ResSHyp | 13.7 | 29.5 | **2.15×** |
| SHyp | 72.7 | 79.0 | 1.09× |
| ResFP | 31.7 | 31.7 | 1.00× |
| FP | 99.0 | 99.4 | 1.00× |

SHyp *does* collide (`g_a` 5.9→7.0 ms) but is CPU/read-bound, so it gains only ~9 %; the factorized
archs create only `g_a`, so lane-major ≡ subgraph-major there. The win is real only for the DPU-bound
hyperprior: naive round-robin leaves ResSHyp fan-out (13.7) barely above plain `mt` (12.4).

**Confirmation across lanes.** The campaign measured naive placement only at 3 lanes; the pinned curve
(fan-out lane grid) then climbs to ~38 patch/s by ~12 lanes while naive would stay pinned near its 1-lane
value (round-robin collides every lane's `g_a` onto one core), so the deficit widens with lanes. The
pinned curve also carries the 4-lane load-imbalance dip (32.8→27.6→36.4 at 3/4/6 lanes; `lanes/ResSHyp/`);
lane-major, already serialized, shows none.

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
in-`run()` int8 quantize/dequantize. Traces on disk (`results/date27/lanes/<arch>/t{N}_occtrace.csv`):
one subsampled occupancy trace per lane count of the lane grid (N ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20,
24, 32, 48, 64}), plus the full-scene r4/r7 traces in `results/date27/occupancy/<arch>/`.

**Trace window.** `--trace` (fanout-only) logs every stage of every patch — each lane writing its own
lock-free timeline — across the compress phase, with `t = 0` set the instant the workers launch, so the
first row-block read sits inside the window (`stream_pipeline.cpp`). The traced region is a
**top-of-scene subsample**: `make_grid` capped to `--max-rows` row-blocks (58 patches each), which the
sweep sets to `max(3, ⌈lanes/6⌉)` ≈ **10 patches/lane** — 174 patches (≤16 lanes) up to ~640 (64
lanes), a **1–20 s slice** vs the ~38 s full-scene FP sweep. That slice's effective patch/s tracks the
full scene within ~5 % (the gap is pipeline fill), so it is steady-state-representative. Because
`t = 0` precedes the fill, the **steady-state analysis window is chosen offline** — trim the leading
fill (and trailing drain) row-block before scoring, no re-capture needed. Every span is wall-clock: a
DPU span includes the synchronous `execute_async` queue-wait, a CPU span includes scheduler preemption
— neither is pure exec (hence the `e = median` correction below).

**Pure-exec proxy `e`.** From the 1-lane trace (no contention; `results/date27/lanes/<arch>/t1_occtrace.csv`),
`e` = the **median** `run()`-span of each kernel; spread is the **coefficient of variation** (CoV =
std/mean). The distributions are tight — plain `g_a` **5.00 ms** (CoV 2 %), residual `g_a` **36.3 ms**
(CoV <1 %), `h_a` **1.34 ms**, `h_s` **1.06 ms** (CoV ~4 %) — so the median is a sharp proxy and min/max
give a narrow band. This same characterization *is* the measured per-arch kernel time the rest of the
doc otherwise cites approximately (plain vs residual `g_a` ≈ 7×). Code: `fanout_occupancy.py`.

**Per-subgraph DPU time & efficiency** (vaitrace hardware counter, per DPU core; our canonical DPU-time
reference — data `results/date27/vaitrace/<arch>/vaitrace_1lane.txt`, tool details `docs/AMD_Vitis_AI.md`):

| subgraph | WL (GOP) | HW_RT (ms) | SW_RT (ms) | Effic | LdWB (MB) | AvgBw |
| --- | --- | --- | --- | --- | --- | --- |
| plain `g_a` (FP, SHyp) | 4.512 | 4.10 | 4.48 | 89.5 % | 1.175 | 1.61 GB/s |
| residual `g_a` (ResFP, ResSHyp) | 39.755 | 33.51 | 35.84 | 96.6 % | 3.520 | 0.65 GB/s |
| `h_a` (hyperprior) | 0.275 | 0.83 | 0.97 | 27.1 % | 4.688 | 5.74 GB/s |
| `h_s` (hyperprior) | 0.176 | 0.72 | 0.85 | 19.9 % | 3.001 | 4.31 GB/s |

**HW_RT** = pure DPU compute (hardware counter); **SW_RT** = the `run()` span (dispatch + int8 requant
around it); **Effic** = achieved GOP/s ÷ 1229 — the big convs fill the DPU, the tiny `h_a`/`h_s` cannot.
HW_RT is **lane-stable**: flat to 64 lanes for residual `g_a`, a bounded +8 % step for plain `g_a`. The
trace `e` above tracks SW_RT for the big kernels (residual `g_a` +1 %, plain `g_a` +12 %) but overshoots
the sub-ms `h_a`/`h_s` by ~25–40 % — the C++-side `execute_async`+`wait` round-trip and TensorBuffer
setup are noise on a 36 ms kernel, a third of a 1 ms one. The occupancy model uses `e` deliberately
(charging that dispatch cost as busy, not idle — see below); it inflates only the `h_a`/`h_s` slice,
which is a small share of the light archs' wall time.

The `h_s` row was **0.52 ms / 27.8 %** in the pre-campaign capture; `h_a` and both `g_a` forms
reproduced to the digit. The old `h_s` value was an artifact — 0.52 ms is **1.4× faster than the
zero-host-overhead `xdputil benchmark` synthetic peak** for the identical xmodel (1374 FPS = 0.728 ms),
i.e. physically impossible. The E4 value (0.72 ms) sits exactly at that synthetic peak, where an
uncontended kernel belongs; `h_a` likewise sits at its own 0.82 ms peak in both eras. vaitrace's
hardware counter over-reads short (sub-ms) transposed-conv kernels in small captures; the fix is the
5×-longer E4 sample. (Investigation 2026-09-03, `DATE27_paper_plan.md` P0.6.)

**`h_a`/`h_s` are weight-load-bound, not DDR-bound.** **LdWB** (external-memory weight+bias read, MB)
is 97–99 % of their total DDR traffic (vs 16 % for residual `g_a`, which is feature-map-dominated —
`LdFM`+`StFM` ≫ `LdWB` there); the two runs above (SHyp, ResSHyp) measure the same `h_a`/`h_s` weights
and agree to <0.2 %.

The roofline figure (`scripts/figures/roofline_subgraph.py`) places the `h_a`/`h_s` points far left of
every ridge — arithmetic intensity ~55 OP/byte on the static xmodel byte-basis — so they read as
weight-load-bound off the plot. It draws two ceilings only: the **1-core AXI interface peak, 9.6 GB/s**
(2× 128-bit `M_AXI_DATA` ports per DPUCZDX8G core at the 300 MHz DPU clock, PG338 — the true single-core
bound) and the **3-core shared-DDR peak, 17.06 GB/s**.

**Efficiency is a property of the operating point, not of the kernel.** `Effic = (WL / HW_RT) / 1229`
and `WL` is a static graph constant, so all the variation is `HW_RT` — how much DDR bandwidth a kernel
gets while its neighbours run. Two layers therefore answer two different questions, and P0.9
(2026-09-05) re-captured the second at the configuration the system actually ships:

| | `g_a` plain | `g_a` residual | `h_a` | `h_s` |
| --- | --- | --- | --- | --- |
| **1 lane** (uncontended) | 89.5 % | 96.6 % | 27.1 % | 19.9 % |
| 3 lanes, full stack | 86.3 / 86.6 % | 95.9 / 96.0 % | 20.4 / 22.9 % | 21.0 / 22.2 % |
| **deployed** (r7 @ knee) | 82.3 % | 95.9 % | 14.1 / 20.8 % | 15.5 / 19.7 % |

*(paired cells are SHyp / ResSHyp; `results/date27/vaitrace/<arch>/vaitrace_r7_knee<N>.txt`.)*

**The 1-lane layer is architecture-independent** — every arch measures identically, because efficiency
there is a property of the subgraph alone. That is the characterization the roofline exists to make.
**Residual `g_a` holds 96 % whatever the configuration**: compute-bound work scales across the three
cores essentially for free. Plain `g_a` gives up ~7 points, and the weight-bound side networks collapse
to 14–20 %, colliding with the shared DDR controller.

Two caveats worth keeping. **Side-network efficiency is not architecture-independent even though the
subgraph is**: `h_a` reads 14.1 % on SHyp but 20.0 % on ResSHyp, because the side networks are 22 % of
SHyp's per-patch DPU time against 3 % of ResSHyp's — contention tracks how *often* a kernel fires, not
overall DPU occupancy. And **core 3 is systematically starved** in every capture (deployed `h_a`:
27/22/10 on ResSHyp, 19/16/7 on SHyp), so a 3-core mean hides a ~2.5× spread.

The superseded E4 captures (`vaitrace_knee<N>.txt`, fan-out only, CPU optimizations off) sampled a
lower DPU duty cycle than the system ships; kept for provenance only.

**Why the occupancy stays runner-span.** It uses `e` = the `run()` span (≈ SW_RT), not HW_RT. Charging
`e` = HW_RT would count the per-call CPU glue (SW_RT−HW_RT = **8.4 %** plain `g_a`, **6.5 %** residual
`g_a`, 14–17 % for the tiny `h_a`/`h_s`) as *idle/wait* — but that glue is necessary dispatch + requant,
not idle. So pure-DPU-compute busy is only ~6.5 % (residual archs) / ~8.4 % (plain-`g_a` archs) below the
reported runner-span occupancy.

**Per-core DPU occupancy.** For each core (lane→core = subgraph-major creation-order round-robin —
lane *k*'s `g_a` → core *k* mod 3, validated against the `device_core_id` logs), busy = the union of the
reconstructed compute intervals `[t1 − e, t1]` on it over the **steady-state window** (last lane to
start → first lane to finish, trimming fill + drain). Result (occupancy panel of the lane-scaling figure,
all archs × lanes; `fanout_occupancy.py`): at each arch's knee the residual archs saturate the 3 cores
(ResFP 99.6 %, ResSHyp 95.8 %), the light archs plateau **~30–39 % idle** (FP 70.1 %, SHyp 61.1 % busy) —
so their roof is not the DPU. Two mechanisms surface: **~2 lanes/core** are needed to hide the CPU-feed
gap (ResFP 87.5 % → 99.6 % from 3 → 6 lanes, its knee), and a **4-lane dip** — a load imbalance when the
round-robin stacks two heavy `g_a` on one core (lanes ≠ 3·k): ResSHyp falls from 82.5 % at 3 lanes to
69.2 % at 4 (two of the three cores drop to ~54 %), recovering by 6.

### CPU occupancy & memory footprint

**What binds the light archs — measured on-board.** `mpstat -P ALL` + `pidstat` (kernel scheduler
accounting, no `perf` needed) at **each arch's knee** resolve the CPU side that the DPU occupancy leaves
open (P0.7; `results/date27/cpu_probe/<arch>/knee_<L>_mpstat.log`). Against per-core DPU busy from the
trace attribution at the same operating point:

| arch | knee | CPU %usr | %sys | %idle | DPU busy (mean of 3 cores) | binding side |
| --- | --- | --- | --- | --- | --- | --- |
| FP | 12 L | 67.4 | 8.3 | 24.3 | 70.1 | CPU (balanced) |
| SHyp | 21 L | 76.4 | 8.5 | 15.1 | 61.1 | **CPU** |
| ResFP | 6 L | 13.2 | 4.2 | 82.6 | 99.6 | **DPU** |
| ResSHyp | 15 L | 18.5 | 4.7 | 76.9 | 95.8 | **DPU** |

Kernel time stays 4–8 % throughout. For the CPU-bound archs the entropy coder dominates: uncontended
(1 lane) FP spends **4.72 ms/patch in `eb_enc` + 3.71 ms in `normalize`** (8.95 ms total), SHyp **9.65 ms
in `gc` + 3.72 ms normalize** (14.11 ms total) — the Gaussian-conditional coder is what makes the
hyperprior arch the more CPU-hungry of the two.

**Why FP roofs at ~46 % of the naïve 4-core ceiling** — two measured effects, not a thread shortage.
Per-patch CPU work **inflates with lane count** (cache/DDR contention under oversubscription), measured
from the `kind=cpu` spans of the E2 lane traces:

| FP lanes | 1 | 4 | 8 | **12 (knee)** | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU ms/patch | 8.95 | 9.78 | 12.16 | **12.23** | 14.75 | 14.94 |
| vs 1 lane | 1.00× | 1.09× | 1.36× | **1.37×** | 1.65× | 1.67× |
| throughput (patch/s) | 51.2 | 168.4 | 200.7 | **205.3** | 205.9 | 205.4 |

Waterfall at the knee: 4 cores at the *uncontended* cost would give **447 patch/s**; at the inflated
12 L cost, **327**; measured, **205**. The remaining gap is idle — 24.3 % of the four cores — a
*balanced* CPU↔DPU pipeline (CPU 67 % usr against DPU 70 % busy) where neither side saturates, so extra
threads cannot fill the idle. Accounting from %usr alone predicts 220 patch/s against 205 measured
(~7 %), the residual being user-space work the stage tracer does not instrument (patchify, record
write, queueing).

**This is what the knee *is*, measured.** Past 12 lanes the inflation keeps climbing while throughput
does not: 12 L → 64 L costs **22 % more CPU work per patch (12.23 → 14.94 ms) for 0 % more throughput**
(205.3 → 205.4). SHyp shows the same shape more sharply (13.8 ms at 1 L → 32.1 at its 21 L knee,
2.33×). Choosing the knee over the peak is therefore not a tie-break on noise — it is refusing to pay
contention for nothing.

*(Sources: `results/date27/cpu_probe/` for %usr/%sys/%idle, the E2 lane traces for per-patch CPU work
and DPU attribution. The CPU-composition view is `scripts/figures/cpu_composition.py` — generated for
inspection, not published as a float.)*

**Thread affinity — a considered, unmeasured lever.** That inflation is cache/DDR contention under heavy
oversubscription (far more workers than cores), where the scheduler may migrate a worker between cores and
lose its warm cache each time. Pinning each worker to a fixed core would keep caches warm and could
reclaim part of it — but it is unpursued here and unverified: it would likely help only *paired with* the
knee (few, pinned workers), since at the peak the extra lanes are partly hiding latency, and it does
nothing for the balanced-pipeline idle. Confirming it would need its own affinity × lane-count sweep.

**Process memory footprint.** Two measurements: (a) E6 sampled `/proc/<pid>/status` + `/proc/meminfo`
through a **64-lane** run — 3–10× past every knee, the stress point (`results/date27/checks/<arch>/mem_64L_raw.txt`);
(b) a pre-campaign lane sweep (4 → 64 lanes) gave the per-process slopes.

| arch | peak resident RAM (VmHWM) @ 64 L | free space left in the 1.5 GiB CMA pool @ 64 L | worker threads @ 64 L |
| --- | --- | --- | --- |
| FP | 623 MiB | 4 MiB (of 1536 — ~100 % full) | 64 + 2 |
| SHyp | 755 MiB | 17 MiB (~99 % full) | 64 + 2 |
| ResFP | 572 MiB | 99 MiB (~94 % full) | 64 + 2 |
| ResSHyp | 713 MiB | 28 MiB (~98 % full) | 64 + 2 |

Resident RAM peaks at **≈0.6–0.75 GiB, well inside the 3.84 GiB board**, and scales with **fan-out lanes,
not architecture** (each lane adds a worker thread + in-flight patch buffers; the hyperprior archs sit a
little higher for their extra `h_a`/`h_s` runners). The **1.5 GiB CMA pool** (kernel-reserved at boot;
what the DPU DMAs weights/feature-maps from) is the tighter ceiling: at 64 lanes it runs within a few
MiB of full. It splits into a **fixed ≈1 GiB driver/xclbin reservation** plus the streaming process's
own share, which grows **≈6.3 MiB per DPU runner**.

- Terms: **VmRSS** resident RAM in use now · **VmHWM** its lifetime peak · **VmSize/VmPeak** reserved
  virtual address space (≫ RAM, mostly thread stacks) · **VmData** heap · **CMA** (Contiguous Memory
  Allocation) = the kernel pool of *physically-contiguous* RAM the DPU DMA engine reads weights/feature
  maps from, sized once at boot (1.5 GiB here) and shared by the driver and every DPU runner.

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
access matches, decode(file) ≈ decode(direct) within float32 ε. The C++ streaming writer (step 7) must
emit these exact bytes; the Python codec is the oracle.

**Container overhead = header + 16·n bytes, not 8·n.** Each body record carries **two u32 length
prefixes** (`len_z`, `len_y` — both written even for FP, where `len_z` = 0) on top of its 8-byte trailer
entry, so counting the trailer alone halves the true figure. On the full 7,540-patch tile: 117 B header
+ 16 × 7 540 = **120.8 KB**, i.e. **1.24 % (FP) · 1.25 % (ResFP) · 1.31 % (SHyp) · 1.45 % (ResSHyp)** of
the file — quote the range as **1.2–1.5 %**. Two consequences worth stating once: the fraction *rises*
as the payload shrinks, so the best-compressing architecture pays the most overhead; and since the cost
is per-patch framing, it scales with patch count, not with scene size.

---

## 10. Results

**The sweep.** The DATE'27 campaign (`DATE27_paper_plan.md` §4.1a; canonical tree
`results/date27/`, per-run provenance in its `MANIFEST.md`) runs the cumulative **r0–r7** optimization
ladder (E1) and a 15-point fan-out lane grid (E2) on the full **7 540-patch** Hamburg scene (overlap 2,
snap grid, §3), warm, through the harness (§4). All four archs, λ=20 — throughput is **λ-independent**
(L20 = L1000 within ~1 %, since rANS time scales with the *number of latents*, not bpp). The r0–r7
ladder is the figure `ladder.py` renders; numbers below are recomputed from `results/date27/ladder/`
and `results/date27/lanes/` and are canonical in §4.0.

**The r0–r7 ladder** (cumulative — each rung = the one above + the named change; `mt` = 4-worker thread
pool, `fo3` = fan-out 3 lanes naive placement, `fo3p` = + pinned subgraph→core placement, `knee` =
fan-out at the per-arch knee, `+dbuf` = double-buffered row-block read, `+ent` = optimized rANS). Knee
lanes: **FP 12 / SHyp 21 / ResFP 6 / ResSHyp 15**.

*Throughput — warm patch/s (r0 cold in parens = SD-testbed floor); **bold** = the r7 operating point:*

| rung | FP | SHyp | ResFP | ResSHyp |
| --- | --- | --- | --- | --- |
| r0 seq | 36.9 (26.6) | 29.4 (22.4) | 11.1 (10.0) | 10.4 (9.3) |
| r1 mt | 83.7 | 58.0 | 13.4 | 12.4 |
| r2 fo3 (naive) | 99.0 | 72.7 | 31.7 | 13.7 |
| r3 fo3p (pinned) | 99.4 | 79.0 | 31.7 | 29.5 |
| r4 knee | 144.0 | 113.4 | 38.5 | 34.3 |
| r5 +neon | 172.5 | 128.7 | 38.8 | 35.3 |
| r6 +dbuf | 189.8 | 143.6 | 41.0 | 38.3 |
| r7 +ent | **204.4** | **146.4** | **41.1** | **38.4** |

*SLC MB/s = patch/s × 0.256 (the 1.93 GB tile over 7 540 patches) → r7: FP 52.4, SHyp 37.5, ResFP 10.5,
ResSHyp 9.8. On the real SD card the light archs are read-bound at ~23.5 MB/s (cold-read ceiling,
`results/date27/ladder/*/r0_seq_cold.json`); the residual archs are compute-bound (cold ≈ warm).*

Sequential per-patch time:

| arch | read | patchify | normalize | g_a | h_a | h_s | entropy | write | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP | 0.51 | 0.41 | 9.03 | 10.48 | — | — | 6.63 | 0.006 | 27.11 |
| SHyp | 0.52 | 0.41 | 9.00 | 10.46 | 1.36 | 0.84 | 11.36 | 0.006 | 34.02 |
| ResFP | 0.51 | 0.41 | 9.02 | 73.20 | — | — | 6.51 | 0.005 | 89.70 |
| ResSHyp | 0.51 | 0.41 | 9.06 | 73.18 | 1.39 | 0.84 | 11.22 | 0.004 | 96.68 |

I/O costs 0.92–0.94 ms/patch and is near-constant in absolute terms.

**Best configuration per architecture** = r7 at the knee (λ=20, seed s0, warm):

| arch | knee lanes | patch/s | SLC MB/s | J/patch |
| --- | --- | --- | --- | --- |
| FP | 12 | 204.4 | 52.4 | 0.079 |
| SHyp | 21 | 146.4 | 37.5 | 0.103 |
| ResFP | 6 | 41.1 | 10.5 | 0.480 |
| ResSHyp | 15 | 38.4 | 9.8 | 0.555 |

Fan-out is the best config for **every** arch. The DPU-bound archs (ResFP, ResSHyp) plateau early as
the 3 cores saturate (by ~6 / ~12 lanes, ~10 MB/s); the CPU-bound archs (FP, SHyp) keep gaining from
independent lanes past the cores, within ~1 % of peak by their 12 / 21-lane knee (unconstrained peak
206.7 / 147.6 @ 48 L — no gain over the knee).

*Energy — warm J/patch; **bold** = the r7 operating point:*

| rung | FP | SHyp | ResFP | ResSHyp |
| --- | --- | --- | --- | --- |
| r0 seq | 0.271 | 0.335 | 1.042 | 1.157 |
| r1 mt | 0.141 | 0.190 | 0.903 | 1.015 |
| r2 fo3 (naive) | 0.125 | 0.161 | 0.538 | 0.944 |
| r3 fo3p (pinned) | 0.125 | 0.151 | 0.539 | 0.613 |
| r4 knee | 0.096 | 0.119 | 0.493 | 0.583 |
| r5 +neon | 0.086 | 0.111 | 0.492 | 0.577 |
| r6 +dbuf | 0.083 | 0.104 | 0.480 | 0.556 |
| r7 +ent | **0.079** | **0.103** | **0.480** | **0.555** |

*Energy = MPSoC (PS+PL) INA226, cooldown-gated to 58 °C; J/patch is the comparable metric (avg W drifts
with thermal). The full per-rail-group breakdown is in each result JSON.*

> **J/patch reproduces to ~1 %, not to the third decimal — quote it accordingly.** Measured directly:
> the r0 rung was re-run twice on 2026-09-04 with byte-identical output and throughput stable to
> ≤0.27 %, yet J/patch rose on **every** arch (FP 0.2694→0.2714, SHyp 0.3329→0.3352,
> ResFP 1.0339→1.0420, ResSHyp 1.1533→1.1567, i.e. +0.3…+0.8 %). Cause is thermal, not noise: the
> second pass started 1.4–4.0 °C hotter on every arch and drew 0.05–0.08 W more, while run durations
> moved <0.3 % — leakage tracking die temperature. Rounding does not rescue it: at 2 dp three of the
> four values still flip (0.33→0.34, 1.03→1.04, 1.15→1.16) because they sit on a boundary, and 1 dp
> collapses FP and SHyp to the same "0.3". So **energy ratios are the stable quantity** (r7/r0 held at
> 3.4 / 3.3 / 2.2 / 2.1× across both passes); absolute J/patch is a ~1 % number that will move
> whenever the board is re-measured.
>
> **Caveat on the ratios themselves:** r0 is now the only rung measured on 2026-09-04 — r1–r7 are all
> from 2026-08-31/09-01 (see each JSON's `provenance.run_utc`). The ladder's energy curve therefore
> joins two thermal sessions at its first point, so the ×3.4/×2.1 endpoints mix sessions. Harmless at
> the precision quoted, but a full-ladder re-run in one session is what would make them internally
> consistent.

**Findings.**

- **Parallelism is arch-dependent, and fan-out is the universal best.** The DPU-bound archs
  (ResFP/ResSHyp) are unlocked by fan-out lanes (`--s1`, §4, is an early scaffold fan-out supersedes); the
  CPU-bound archs (FP/SHyp) by the thread pool + double-buffered read, and fan-out's independent
  pipelines lift them further still. Cumulative warm, r0→r7: FP **36.9→204.4 patch/s (×5.5)**,
  SHyp **×5.0**, ResFP **×3.7**, ResSHyp **10.4→38.2 (×3.7)**.
- **Storage was hiding the CPU parallelism.** The cold SD-read ceiling is ~92 patch/s (~23.5 MB/s) for
  every arch; the CPU-bound archs' warm throughput (FP 204, SHyp 147) runs 2–6× past it, so on the real
  SD card FP/SHyp are read-bound while ResFP/ResSHyp (warm ≈ cold from r3 on) are not. The r0 cold/warm
  gap (FP 26.6 → 36.9, SHyp 22.4 → 29.4) is the same signal at the sequential baseline.
- **Parallelism costs power but saves energy.** FP r0→r7 **0.271→0.079 J/patch (×3.4)**;
  ResSHyp **1.157→0.555 (×2.1)**. The draw is PL(DPU)-dominated for every arch — the residual archs
  push far more of it; cross-arch ResSHyp costs **~7× the energy/patch** of FP.
- **DDR is not a bottleneck.** vaitrace: the dominant DPU traffic (`g_a`) is ~0.65 GB/s (residual) to
  ~1.6 GB/s (plain), **~11–26×** under the 17.06 GB/s DDR4 ceiling (§2). CPU-side DDR is an estimate
  (~100–150 MB/s; no `perf` on the board) — the wide margin holds either way.
- **The CPU-bound archs are limited by compress *compute*, not the DPU or the OS.** At FP's 12-lane knee
  the 4 A53 cores fill to **67 % user-space** (rANS + normalize) against **70 % DPU busy** — a balanced
  pipeline with 24 % idle left over — and kernel time is 8 %. SHyp leans further: 76 % usr against 61 %
  DPU. The roof mechanism (per-patch CPU work inflating 1.37× by the knee, plus a persistent
  balanced-pipeline idle) is **§6**.
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
  SLC at **211 MB/s** (working point) / 358 MB/s (worst case). One ZCU102 at its best config (FP knee,
  **52.4 MB/s warm**) is **~6.9× short of full-duty real-time**, but **meets the
  process-before-next-contact deadline** (FP, ~4.5× headroom) and the ~200× compressed product sits far
  inside the 270 Mb/s-net downlink.

---

## 11. Figures

The DATE'27 figure code lives in **`scripts/figures/`** (this repo) — moved out of the LaTeX repo in
P1.0 so data + code + provenance all sit together and the figures are reproducible from a clone.
`scripts/figures/_figutils.py` is the shared module: palette (CPU / DPU / storage), per-arch colors,
display names, rung labels, knee lanes, the `results/date27/` loaders, the occupancy re-exports, and
`save_figure()` (writes `.pdf` + `.png` into `LaTeX/SAR_DDC_FPGA_DATE27/figures/images/`, env override
`DATE27_FIG_OUT`, hard-errors if that dir is absent). Every script reads from `results/date27/` only.
Run each under `conda activate DDC_FPGA`.

- **`stacked_time.py`** → s0 per-patch stacked bars (the bottleneck-migration figure); reads
  `results/date27/s0/<arch>/s0_compress_entoff.json`.
- **`ladder.py`** → the cumulative r0–r7 ladder, throughput (`optimization_ladder.{pdf,png}`) and
  energy (`energy_ladder.{pdf,png}`), all four archs; reads `results/date27/ladder/<arch>/r{0..7}_*_warm.json`.
  Also prints the per-rung numbers §10's tables quote.
- **`roofline_subgraph.py`** → per-subgraph Williams roofline; reads
  `results/date27/s0/<arch>/*_xmodel_info.json` + `results/date27/vaitrace/<arch>/vaitrace_{1lane,knee*}.txt`.
- **`fanout_lane_plot.py`** → the lane-scaling figure `lane_scaling.{pdf,png}` (throughput / occupancy /
  energy vs lanes; occupancy panel: **solid = mean per-core DPU busy, dashed = mean 4-core CPU busy**,
  ★ = knee). Occupancy series come from `fanout_occupancy.py` (via `_figutils`).
- **`cpu_composition.py`** → per-arch A53 CPU-time stack (%usr / %sys / %idle) at each knee, from the
  `cpu_probe/` mpstat logs. Not a manuscript float (P1.2) — numbers go to §6 / W5 prose.
- **`checkpoint_occupancy.py`** → prints (no figure) the DPU + CPU occupancy at ladder checkpoints
  r0 / r4 / r7; the r0 baseline and r4→r7 shift are a couple of sentences for W5.
- **`overlap_crop.py`** → the seam figure. The two reconstruction-crop panels need
  `results/benchmark_stream_overlap/_work/*.npy`, archived out of the tree on 2026-08-31 — the script
  hard-errors with the regeneration path and the last rendered `overlap_crop.{pdf,png}` is kept as-is.
  The seam-PSNR panel's `overlap_table.csv` is still in the tree.
- **`fanout_occupancy.py`** → per-core DPU occupancy from the trace CSVs (§6 method + the
  `[t1 − e, t1]` reconstruction) and mean 4-core CPU occupancy from the `kind=cpu` spans; `__main__`
  prints the per-core table. `_figutils` re-exports its public functions.
- **`fanout_gantt.py`** → per-call swim-lane Gantt from a trace CSV (`--csv … --solo-csv …`).
  **Diagnostic, not a manuscript figure** — for eyeballing lane/core scheduling when a lane-scaling
  number looks off; runs on the E2 `results/date27/lanes/*/t*_occtrace.csv` traces as-is.

`system_dataflow.py` was deleted in P1.0 — `main.tex` renders that figure from
`figures/tikz/system_dataflow` via `\includestandalone`. The pre-campaign
`scripts/fpga/benchmark/{stream_sysplot,stream_sweep,stream_gantt,stream_roofline,stream_table,
fanout_full_table,fanout_cpu_fp,fanout_lane_diagram,fanout_table}.py` plotters and
`scripts/evaluation/rescore_overlap_tiles.py` were **deleted** in P0.6 (2026-09-03) — all read the
archived `results/benchmark_stream/` / `results/benchmark_hardware/` trees and are superseded by the
`scripts/figures/` scripts above.

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

- **Per-stage instrumentation is not homogeneous across schedules and lacks small stages.** Only the sequential path
  (`stream_compress_tile`) times most stage: it is the sole place `read` / `patchify` / `write` are
  measured at all. Plain `--p0` records only `read` and `write` totals and **no compute stages**;
  `--fanout` records per-lane compute totals (`LanePerf`) but never `patchify`, and being per-lane sums
  across concurrent workers they cannot close on wall clock by construction (the occupancy caveat,
  §6). `benchmark_hardware` s0/s1 sit at the other extreme — full per-stage detail, but a 20-patch
  subset of a different binary with **no I/O stage at all**. Net: the number we headline, full-scene
  throughput at the knee, has no per-stage decomposition of its own; F1's decomposition is the
  *sequential* one, and that gap is worth closing for its own sake rather than for a figure.
  The fix is additive and byte-transparent — give the `p0`/fan-out worker the same per-stage
  accumulators the seq path now has — but it forces a ladder re-measure, which re-opens numbers
  already in the manuscript. Deferred on that basis, not because it is hard.
  **Naming is settled**, so a later pass has nothing to decide: one bucket per
  `BenchPipeline::stage_*` call, labelled exactly as `bench_configs.cpp`'s `StageTimer` marks
  (`normalize`, `g_a`, `h_a`, `eb_compress`, `eb_decompress`, `h_s`, `gc_compress`, `gc_decompress`,
  `g_s`, `denorm`), so a seq breakdown and an s0 one compare key-for-key. Aggregates (`dpu`,
  `entropy`) are *derived* from those buckets, never measured as one — **aggregation is the plotting
  script's decision, not the measurement's.** Two aggregations still outstanding under that rule:
  `stream_decode_ddc` folds `denorm` into `t_normalize_ms` and `g_s` into `t_dpu_ms`, and
  `LanePerf::entropy_ms` still lumps the three entropy calls.
  Worth checking if other small operations like channel split and channel interleave should also be timed. (For that, first answer if adding timers bears a cost in the final reported numbers, if not there is no reason not to add them because we can simply aggregate the data at will later.)

### Deferred / optional

- **N4 — INT8 `g_s` output cap** at 2100.1 (clips the brightest ~0.7 % of pixels). Metric-invisible;
  lives as a one-line **limitation** in `main.tex` §Discussion, not a deepening study.
- **On-ground SHyp `.ddc` decoder** (optional, decoupled). Reproduce the board INT8 `h_s` on host so
  SHyp/ResSHyp `.ddc` decode in pure Python (FP already does); hard because the Gaussian decode needs
  every element's scale in the same `gc_scale_table` bucket and an FP32 `h_s` desyncs rANS. The overlap
  study (§8) sidesteps it via board decode; value if built = off-board verification + a real ground
  decoder.
