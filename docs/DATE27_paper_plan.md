# DATE 2027 paper — planning scratchpad

> **Temporary working doc** for the DATE'27 submission. Dissolve into
> `LaTeX/SAR_DDC_FPGA_DATE27/` + the permanent docs once the manuscript takes shape.
> **The story we follow is the mechanism cut; the draft is `main.tex` and this doc plans around it.**
> Condensed DATE venue evidence is in §0.
>
> **DATE 2027**, 22–24 March, Dresden. Abstract **13 Sep 2026**, paper **20 Sep 2026 AoE**.
> 6 pages + 1 reference page, IEEE double column. **Track: E3** (ML solutions for embedded and
> cyber-physical systems) — its topic text explicitly names *space*, and its sessions carry
> deployment/scheduling papers. Presentation is a short pitch + poster/demo.

---

## 0. DATE venue — what it publishes (condensed)

**Logistics.** DATE 2027, 22–24 Mar, Dresden; abstract 13 Sep 2026, paper 20 Sep 2026 AoE; 6 pp + 1
ref, IEEE double-column; **Track E3** (ML for embedded/cyber-physical systems — its topic text names
*space*; its sessions carry deployment/scheduling papers). Format = short pitch + poster/demo (a
demo-able board helps).

**Corpus.** Findings below are measured from the DATE 2024–2026 proceedings (1,055 regular `TS*`
papers). Parser + data in `tmp/date_corpus/` (gitignored): `parse_date.py`, `date_corpus.json`, raw
programme HTML; regenerate with `curl -sL -o dateNN_prog.html https://dateNN.date-conference.com/programme && python3 parse_date.py`.

**Papers to read:**

| read? | paper | why it matters |
| --- | --- | --- |
| [x] | **DPUConfig** (2026 TS29.7) | same board + Vitis-AI DPU, accepted — existence proof + "mechanism on top of the overlay" template |
| [ ] | **Hearable neural beamformer** (2026 TS25.10) | closest shape: a deployment-methodology paper that meets a real-time constraint |
| [ ] | **Tiny-VBF** (2024 TS26.6) | signal-processing domain → embedded accelerator, with domain quality metrics |
| [ ] | **Nano-drone pose estimation** (2024 ASD01.3) | motivating a resource-constrained platform in two sentences; adaptation as the contribution |
| [ ] | **FAMERS** (2025 TS23.6) | canonical "vs RTX GPU" ratio framing on an FPGA |
| [ ] | **Spec-HD** (2024 TS07.7) | domain pipeline + near-storage preprocessing + FPGA (a data-movement story) |
| [ ] | **SolarML / DE²R / DAOP** (2025 TS18) | the E3 scheduling/deployment flavour — a *policy* as the contribution |

**Five findings (descriptive):**

- **F1** — Zero SAR / remote-sensing / Earth-observation papers in 1,055 (space appears only in
  focus/multi-partner sessions). No incumbent to beat, but nothing establishes the audience cares.
- **F2** — "Compression" at DATE means **model** compression (quantization/pruning); learned *image*
  compression is absent.
- **F3** — An off-the-shelf DPU overlay is acceptable (DPUConfig): the overlay is fine; having no
  mechanism on top of it is the problem.
- **F4** — DATE papers are **named techniques** — 46 % `ACRONYM:` titles, 14 % an `N×` ratio vs a named
  baseline in the abstract; <5 % use case-study / characterization language.
- **F5** — Winning application-paper recipe: two sentences of resource-constraint motivation → a named
  mechanism (2–4 components) → a hard constraint it *meets* → a headline ratio vs a recognizable
  baseline → quality shown preserved.

---

## 1. Situation

The repo holds a board-verified onboard streaming compressor for SAR SLC tiles on a ZCU102 plus four
measurement campaigns around it — almost none of it in the TGRS manuscript, which stopped at per-patch
characterization and an *extrapolated* full-tile projection. This paper replaces that extrapolation
with a **measured, end-to-end system** plus the **mechanism** the deployment reveals.

**Off-limits (spent in TGRS)** — cite, never re-report: the 4-arch RD ablation; hardware-aware
modifications (GDN→ReLU, output_padding, graph partition); the FP32/INT8 cross-precision study; the
qualitative reconstruction grid; the per-patch latency breakdown; the cross-platform CPU/GPU/FPGA
table; the full-tile extrapolation.

---

## 2. Assets

🟢 publication-ready · 🟡 data exists, analysis/figure needs work · 🔴 needs a run or a decision

**Deprecated should be updated and simplified**!
*(In this table, a bare §N refers to `onboard_pipeline.md` unless a doc is named.)*

| # | Asset | Where | State |
| --- | --- | --- | --- |
| A1 | Optimization ladder seq→s1→p0→prefetch→neon, cold+warm **per rung**, **all 4 archs**, full 7,540-patch scene | `results/benchmark_stream/`, `onboard_pipeline.md` §10 | 🟢 λ=20, overlap 2, snap grid |
| A2 | Symmetrization granularity (E1): whole/none/patch/block × 2 archs × 3 λ, whole scene | `results/symmetrization_study/`, §7 | 🟢 |
| A3 | Overlap study (U5): ov{0,2,4,8,16}, seam-band vs interior quality, cost | `results/benchmark_stream_overlap/`, §8 | 🟢 the cleanest co-design result |
| A4 | Memory/storage: 3.87 GB f32 tile vs ~3.0 GB DDR (whole-load OOMs), 0.3 GB windowed, SD read ~24 MB/s | §2, §10 | 🟢 |
| A5 | DPU fan-out lane scaling (4 archs × lanes 1–4) + pinned-vs-naive placement ablation; DPU-kernel roofline (OPs/bytes) | `results/benchmark_stream/` (fan-out), `onboard_pipeline.md` §5–§6; roofline `results/benchmark_hardware/_roofline/` | 🟢 |
| A6 | Mission objective (TerraSAR-X): three deadlines, page-cited | `docs/TerraSAR-X_objective.md` | 🟢 except take/contact numbers (objective §6/§9) |
| A7 | `.ddc` container + trailer offset table (O(1) access, prioritised downlink) | `src/utils/ddc_format.py`, §9 | 🟢 as artifact, **not** a contribution |
| A8 | INT8 `g_s` output cap at 2100.1 clips brightest ~0.7 % (point scatterers) | §8 | 🟡 prose-only unless a metric is built |
| A9 | Byte-identical correctness gate across every schedule | §4 | 🟢 credibility, one sentence |
| A10 | Cross-platform CPU/GPU energy (desktop A4000 + Xeon) | `results/benchmark_unified/` | 🟡 **spent in TGRS** — retained as *reference context* in a comparison table (informative even if published), never a contribution; rides alongside a new baseline (Jetson N2 / CCSDS N5) |

**Figure code:** the DATE manuscript figures are built in `LaTeX/…/figures/scripts/`.
The older `stream_gantt` / `stream_roofline` / `stream_sysplot` sketches (`scripts/fpga/benchmark/`) are superseded.

---

## 3. Risks

| # | Risk | Mitigation |
| --- | --- | --- |
| R1 | **Shaped as a measurement campaign** (DATE publishes named mechanisms, not often case studies). | *Partially addressed* — Still framed as a characterization study but stressing the DPU core placement fix, the topology-aware rule of thumb, the met deadline, etc. |
| R2 | Self-overlap with TGRS (same platform, models, scene). Double-blind submission so shouldn't be recognized. | §1 off-limits list as a checklist. |
| R3 | No SAR/EO/LIC prior art at DATE. | Motivate in systems terms (data rate, power, deadline); disambiguate "compression"/"architecture" (see `main.tex`); never assume the reader knows speckle or SLC. |

---

## 4. Execution plan — to a full prose draft for feedback round 2

**Goal**: transform `main.tex` from the bullet-point skeleton into full prose with the restructured
story and refreshed figures — not submission-ready, but clean enough for another Dirk/Martin pass.
**Clock**: abstract 13 Sep, paper 20 Sep AoE → Phase 0 this week (≤ Sep 4), figures by ~Sep 8, draft
to reviewers ~Sep 11, buffer to submission.
**Markers**: **🔄** = outcome may change this plan (changed numbers, broken story, new insight) —
when a 🔄 task lands, update §4.0 and downstream tasks *first*, then continue. All 🔄 tasks are
front-loaded into Phase 0. Mark tasks done by checking them off here; fold measurement outcomes into
`onboard_pipeline.md` as usual.
**Execution notes**: board tasks need the ZCU102 → serialize them (one session at a time owns the
board). Research and figure tasks are agent-delegable and parallel. Every number written into
text/figures gets computed by a `python3` command first (repo convention).

### 4.0 Settled decisions & facts (ledger for all writers)

**Paper skeleton** (replaces the current section cut):

- **I. Introduction** — contributions rewritten to match the new structure.
- **II. Background & Related Work** — SAR/SLC/speckle · BAQ + TerraSAR-X · LIC ¶ · DDC ¶ ·
  *condensed* classical-SAR-baseline ¶ (CCSDS/BAQ, moved from old §Eval) · architecture-selection ¶ ·
  platform ¶ (DPU-first) + Table 1 · delta vs prior work.
- **III. Analysis & Optimizations** — subsections: *System & setup* → *Characterization* →
  *DPU/scheduling optimizations* → *CPU optimizations* → *Combined results & rules of thumb*.
  (The old "Onboard Streaming Pipeline" section content lives in *System & setup*; ⚠ if III grows
  too big, System & setup can be promoted back to its own short section.)
- **IV. Evaluation** — *Relaxations* (old Sec. V, condensed) → *Cross-platform baseline* (prominent)
  → *TerraSAR-X deadlines* → *Energy*.
- **V. Discussion & Conclusion** — possibly split if content justifies it.

**The ladder** (one figure, cumulative, all 4 archs, replaces the two side-by-side ladders):
`seq → mt → fo3 → fo3p → knee → +neon → +dbuf → +ent`, i.e.

| rung | config | shows |
| --- | --- | --- |
| r0 `seq` | sequential baseline | reference |
| r1 `mt` | 4 CPU workers, shared serialized DPU (old `pool`) | obvious first step; big for CPU-bound archs |
| r2 `fo3` | fan-out, 3 lanes (1/core), default round-robin placement | the *structure* change, isolated from oversubscription (3 L not 4 L — 4 on 3 cores is the imbalance case the lane study explains later) |
| r3 `fo3p` | + pinned subgraph-to-core placement | the core fix |
| r4 `knee` | fan-out at per-arch knee lanes (FP 12/SH 24/ResFP 6/ResSH 20), pinned | lane-count selection |
| r5 `+neon` | + NEON log-approx normalization | CPU kernel opt |
| r6 `+dbuf` | + double-buffered row-block read | CPU kernel opt |
| r7 `+ent` | + optimized rANS (flattened CDF + reciprocal, `4ddbcc8`) | CPU kernel opt |

~~Fallback decision inside P0.2~~ **resolved 2026-09-01: not triggered** — the placement pathology
is fully visible at 3 lanes (ResSHyp r2→r3 = 2.15×; naive round-robin leaves fan-out barely above
`mt`), so the rung order stands. Two shaded bands on the figure: rungs r1–r4 = "scheduling",
r5–r7 = "CPU kernels"; each paper subsection points into its band.

**Entropy micro-opt**: no further algorithm work (N6's `4ddbcc8` optimization is final), but it gets
a **runtime flag** (`--entropy`, task P0.0) like every other optimization, so entropy-off configs
are reachable without commit swaps. In the ladder it is the last CPU-kernel rung `+ent`; rungs
r0–r6 run entropy-off. Its isolated speedup can additionally be quoted from the profiling notes
(`docs/tmp_entropy-coding_opt_opportunities.md`).

**Renames** (paper-level only; code/CLI flags unchanged — figure scripts map names at plot time):
`pool` → **`mt`** (thread pool / multithreading); `prefetch` → **`dbuf`** (double-buffered read).

**Terminology & facts every writer must respect**:

- "**SoC FPGA**" everywhere (never "FPGA MPSoC" / "FPGA SoC").
- DPU datatype = **signed INT8** (per-tensor fixpos); CPU path: complex int16 in → fp32 normalize →
  int8 to/from DPU → rANS bitstream out. **Batch = 1** everywhere, stated explicitly.
- Tab. 2 semantics: `Mem [MB]` = FP32 checkpoint (params × 4 B) — rename to "FP32 size" and/or add
  INT8 const-bytes column; `Max DPU FPS` = measured `xdputil benchmark` DPU-only peak (footnote).
  Dirk's 17.35 GB/s objection dissolves: 1374.7 FPS × 3.16 MB INT8 consts = 4.34 GB/s.
- Roofline ceilings: **9.6 GB/s** single-core (2 × 128-bit `M_AXI_DATA` @ 300 MHz, PG338) and
  **17.06 GB/s** chip DDR (DDR4-2133 × 64-bit; papers quoting 19.2 assume DDR4-2400). The 6.11 GB/s
  line is dropped (measured point, not a ceiling). 3-core roof: compute 3 × 1229 = 3687 GOP/s,
  memory capped at 17.06 (3 × 9.6 = 28.8 > 17.06) → ridge moves 128 → 216 OP/B.
- 3-core roofline insight (verify in P0.4, don't oversell): residual g_a pulls ~2 GB/s at 3 cores —
  scales cleanly; h_a/h_s pull ~6 GB/s *each per core* → 3 concurrent ≈ 18 GB/s > 17.06 → the
  weight-bound side networks collide with the shared DDR ceiling under fan-out (matches the measured
  2.2–6.0 GB/s spread at 6 lanes). Caveat in text: explains *their* efficiency drop, not the system
  bottleneck (they are a small share of wall-time).
- **No CPU roofline in the paper** (rANS is integer/lookup-bound; FLOP-based AI indefensible).
  Instead: measured per-kernel times + isolated speedups (neon 2.42×, entropy numbers), and "how
  close normalize gets to the CPU compute bound" if R3 produces defensible ceilings.
- XRT wall: one footnote — hyperprior archs need 3 runners/lane; 64 L (192 runners) works, 128 L
  (384) fails; limit estimated ≈300 runners. 128 L point dropped from the lane figure.
- Deadline table: plain percentages of the deadline (450 %, 15 %) instead of ×-ratios + marks.
- Cross-platform table: more prominent in the narrative; bold best board per row/metric; keep the
  Orin-runs-FP32-unoptimized caveat.
- Table 1 (PL util) stays; falls back to text only if space runs out.
- Energy lives in IV.Evaluation for now (flexible → Discussion if flow prefers).

**Post-sweep ledger updates (gate review, 2026-09-01 — these numbers override anything older):**

- Roof throughput (r7, patch/s): FP **204.4** (12 L) · SHyp **146.8** · ResFP **41.1** ·
  ResSHyp **38.2**.
  Seq baselines reproduce old within ~2.7 %; deadline margins and the "≈2× Orin" claim unchanged
  (FP ≈ 52 MB/s ⇒ ~4.5× before-contact headroom, ~6.9× short of real-time).
- Cumulative ladder speedups: FP **×5.5** · SHyp **×5.0** · ResFP **×3.8** · ResSHyp **×3.7**.
- **Placement fix = 2.15× on ResSHyp at 3 L** (r2 13.7 → r3 29.5 patch/s; SHyp +8.6 %, FP/ResFP
  ~0) — replaces the old "2.8×" everywhere. Prose hook: naive round-robin leaves ResSHyp fan-out
  (13.7) barely above plain `mt` (12.4).
- **Knee lanes (E2, all CPU opts on): FP 12 L** (was 32; peak 206.7 @ 48 L, 12 L within 1 %) ·
  SHyp 24 · ResFP 6 · ResSHyp 20 unchanged. **Resolved 2026-09-01: FP r4–r7 re-run at 12 L**
  (`results/date27/ladder/FP/r{4,5,6,7}_fo_t12_*_warm.json`; the 32 L files are kept, not deleted). r4→r7 @ 12 L: 144.0 → 172.5 → 189.8 → 204.4 patch/s
  (non-decreasing). r7@12 L agrees with the independent E2 lane-grid point at 12 L (205.3) to
  −0.41 %, confirming the knee. **Caveat, not a data problem**: r4–r6 @ 12 L are *not* all within 1 %
  of their 32 L counterparts (r4 −1.5 %, r5 +2.9 %, r6 −4.0 %; only r7 is, at +0.35 %) — because E2's
  grid runs the *full* stack (fanout+neon+prefetch+entropy) at every lane count, where 12 L and 32 L
  both sit within ~1 % of peak, but r4–r6 test fewer optimizations layered on, and an under-optimized
  pipeline's own saturation lane-count isn't necessarily the fully-optimized one's. The ladder
  deliberately pins one lane count across all cumulative rungs for a clean single progression; only
  the final (r7, full-stack) rung is guaranteed by construction to sit at the measured knee. Rule-of-
  thumb prose: FP knee = 4× #cores — the old "3–4×" guess now has a measured anchor (SHyp stays
  higher, 8×, GC entropy).
- **Sequential CPU share (FP/SHyp) = ~60/61 %** (FP: CPU 15.5 ms vs DPU 10.4 ms; DPU share 40 %) —
  replaces "~73 %", which included the now-dropped SD read. Story intact: DPU is 40–82 % of
  per-patch time across archs. ResFP/ResSHyp: DPU 82/79 %.
- 3-core roofline (E4, ResSHyp knee): residual g_a **95–96 % efficiency on all three cores**
  (compute-bound scales cleanly); h_a/h_s crash to **10–26 %** with per-core bandwidth 2.2–5.5 GB/s
  — the DDR-collision reading is supported and F3's aggregate dots are computable from
  `results/date27/vaitrace/`.
- Occupancy caveat for F7/W5: raw trace spans include DPU queue-wait — summing them exceeds 100 %
  busy; **always go through `fanout_occupancy.py`'s per-core attribution**, never raw span sums.
  r0 has no trace (tracer is fanout-only) → use E3 s0 shares for r0, as planned.

### 4.1 Phase 0 — measurement campaign & story-risk retirement (board; all 🔄)

- [x] **P0.0 `--entropy` runtime flag** (N6's remainder; blocks P0.2) — wrap the `4ddbcc8` rANS
  optimization behind a runtime flag like the other optimizations (surgical area:
  `entropy_models.{cpp,hpp}` + `rans/rans_interface_cxx.{cpp,hpp}`; host edit, board rebuild).
  Verify byte-identity both ways on a small patch subset: flag-on ≡ current `HEAD` output, flag-off
  ≡ the pre-`4ddbcc8` build (e.g. `324744f`).
- [x] **P0.C Results cleanup** — archive + delete the superseded result trees and create
  `results/date27/` + `MANIFEST.md`, exactly per §4.1a. Runs *before* the sweep so only traceable
  results exist afterwards.
- [x] **P0.1 Pre-flight** 🔄 — board reachable; all 4 archs (λ=20, seed 0) deployable; `make clean`
  full rebuild of current `HEAD` on the board (incremental builds mis-trigger — board clock unset,
  see `onboard_pipeline.md` §4); verify `--entropy` works and the naive-placement path is reachable
  (`lane_major` option in `stream_pipeline.hpp` — confirm its CLI spelling via `--help`; pinned is
  the default); one smoke run per binary. *(Resolved earlier: `stacked_time.py` reads stages from
  `benchmark_hardware/<arch>…L1000…/s0_compress.json` — a λ mix E3 fixes at λ=20.)* *(Found during
  pre-flight: `stream_benchmark.py` never forwarded `--entropy`/`--trace` to `stream_pipeline` despite
  the binary supporting both since P0.0, and `benchmark_hardware` had no way to disable the rANS
  optimization at all — added trivial CLI plumbing for all three, reusing the existing
  `set_entropy_opt()` setter (`--no-entropy-opt` on `benchmark_hardware`).)*
- [x] **P0.2 The sweep** 🔄 — this *is* N7 for every streaming number. Run E1–E6 from the §4.1a
  table, arch by arch (deploy → all runs for that arch → next arch), full scene, λ=20, overlap 2,
  snap grid, warm read, power sampling on. Outputs → `results/date27/` only, one `MANIFEST.md` line
  per run. *(Done 2026-09-01: all 4 archs × E1–E6, 740/740 independent validation checks passed —
  zero anomalies, byte-identity gate holds on every arch, E1 ladder non-decreasing r0→r7 on every
  arch, E2 knee within ~1% of peak everywhere. Delta vs the old committed numbers: seq baselines
  reproduce within ~2.7%; roof/peak throughput +0.2–3.5% (CPU-bound FP/SHyp gained from N6's entropy
  optimization, DPU-bound ResFP/ResSHyp ~unchanged, as expected).)*
- [x] **P0.3 Occupancy checkpoints** 🔄 — *(2026-09-01: traces collected for r4/r7; tracer is
  fanout-only so r0 uses E3 s0 shares. Proper busy% requires `fanout_occupancy.py` per-core
  attribution — raw span sums include queue-wait and exceed 100 %. Default = numbers in text; F7
  panel only if the proper computation yields a clean visual.)*
- [x] **P0.4 3-core roofline data** 🔄 — *(2026-09-01: confirmed — see gate-review block in §4.0;
  data in `results/date27/vaitrace/`. The old "2.8×" is superseded by the measured 2.15× at 3 L
  from E1 r2→r3.)*
- [x] **P0.5 Delta report** 🔄 — *(2026-09-01 gate review done: 740/740 checks passed; story-level
  changes — FP knee 32→12 L, placement 2.8×→2.15×@3 L, CPU share 73→60 % — written into the §4.0
  gate-review block, which overrides older numbers.
- [ ] **P0.6 Docs & code number-consistency pass** — **deferred to end of Phase 1** (so any
  experiment/bug surfacing during figure work lands in the same pass). Scope: reconcile the
  `results/date27/` numbers across `onboard_pipeline.md` (§5–§6, §10 tables and prose), the other
  docs that quote streaming numbers (`FPGA_benchmark.md`, `GPU_benchmark.md`), and a grep of
  code/scripts for hardcoded stale values (knee lanes, throughputs in defaults/comments). The
  minimal N7-resolution note is already folded (2026-09-01); this is the full pass.

### 4.1a Cleanup + sweep specification (validated 2026-08-31 — execute exactly, don't improvise)

**Cleanup.** Archive destination: `/mnt/vitisAI/DDC_results_archive/2026-08-31/` (outside the repo;
NOT `/tmp` — volatile). For each "archive+delete" tree: `tar czf` into the archive, verify the
tarball lists, then delete the tree from the repo.

| Tree | Treatment |
| --- | --- |
| `results/benchmark_stream/` (25 M) | archive + delete (superseded by E1/E2/E4/E5) |
| `results/benchmark_hardware/` (716 K) | archive + delete (superseded by E3; λ=1000/λ=20 mix) |
| `results/plots/`, `results/fpga_metrics_backup_pre_ssim/`, `results/wandb_summary_backup_pre_ssim_sweep_2026-08-04.csv`, `results/deleted_orphan_runs_2026-08-04.txt` | archive + delete (old outputs/backups) |
| `results/benchmark_stream_overlap/` (55 G) | keep metrics/JSONs in place; archive the raw bulk (`.ddc`, decoded tiles) + delete it from the tree |
| `results/benchmark_jetson/`, `results/symmetrization_study/`, `results/ssim_convention/`, `results/benchmark_unified/` | keep untouched (frozen canonical) |
| `results/fpga/` (17 G) | untouched (deploy infra; later prune, not this task) |
| Board: `/home/root/SAR_DDC/bench_results/`, stray stream outputs (`*.ddc`, `*.json` in the project root), board `/tmp` leftovers | delete before the campaign |

**New canonical tree** `results/date27/`: subdirs `ladder/<arch>/`, `lanes/<arch>/`, `s0/<arch>/`,
`vaitrace/<arch>/`, `occupancy/<arch>/`, `checks/<arch>/`, plus `MANIFEST.md` at the root.
**Filenames encode the exact flag set** (e.g. `fo_t16_pf_neon_ent_warm.json`) so later variant
sweeps (e.g. a no-CPU-opt lane grid) coexist without conflict. Each `MANIFEST.md` line: output file
· full CLI flags · host+board git SHA · model (arch/λ/seed) · date · `make clean` confirmed.
Figure scripts will read from this tree only.

**Sweep table.** Global setup for every run: λ=20, seed 0, overlap 2, snap grid, full scene, warm
read, power sampling on, batch 1; `make clean` rebuild once at campaign start; knee lanes = FP 32 /
SH 24 / ResFP 6 / ResSH 20 *(as run; FP's measured knee moved to 12 L — gate review, §4.0)*.

| ID | Experiment | Configs | Per arch | Feeds |
| --- | --- | --- | --- | --- |
| E1 | Ladder r0–r7 (cumulative) | r0 `seq` · r1 `--p0 --threads 4` · r2 fan-out 3 L *naive* placement · r3 fan-out 3 L pinned (default) · r4 fan-out knee-L pinned · r5 `+--neon` · r6 `+--prefetch` · r7 `+--entropy` | 8 warm runs + 1 cold `seq` (SD-read number) | F2 ladder, F5 energy, headline numbers |
| E2 | Lane grid, all CPU opts on | `--p0 --fanout --prefetch --neon --entropy --threads N`, N ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64} + subsampled occupancy trace per N (existing `--max-rows` methodology) | 15 runs + traces | F4 lane fig, F6 fp_cpu_stack, knee verification |
| E3 | Per-stage s0, **entropy OFF** (true sequential baseline) + xdputil peaks | `benchmark_hardware` s0/compress @ λ=20; `collect_roofline.py` | 1 + 1 | F1 stacked_time, Tab. 2 |
| E4 | vaitrace per-core counters | 1-lane (uncontended) + knee-lane operating point | 2 | F3 3-core roofline dots |
| E5 | Occupancy checkpoints | full traces at r0, r4, r7 (configs exactly as E1) | 3 | F7 / occupancy text |
| E6 | Cheap verifications | `.ddc` sha256 r0 ≡ r7 (byte-identity gate); peak RSS + CMA/lane at 64 L; optional SD cold-read timing | ~2 | A9 sentence, XRT/CMA footnote |

Frozen — **not** re-run: symmetrization (A2), overlap (A3), Jetson (N2), desktop CPU/GPU (A10);
TerraSAR-X inputs are a doc-check (R5). Estimated board time (rough): E1 ≈ 1.5–2 h, E2 ≈ 2.5–3 h,
E3–E6 ≈ 1 h → ~5–6 h + 4 model redeploys, one serialized session.

### 4.2 Phase 0b — research (agents; parallel with 4.1)

- [ ] **R1** Non-SoC-FPGA prior art: has any LIC codec been deployed on a *non-SoC* FPGA? Verify
  which platforms Mazouz 2025 and Sun 2024 actually used (we cite them as DPU-for-LIC precedents).
  Output: 3–5 sentences + BibTeX candidates → feeds II (related work) and the SoC-FPGA framing.
- [ ] **R2** Read the two memory papers Dirk cited — <https://dl.acm.org/doi/10.1145/3517131>
  (p. 19: 13.3 GB/s parallel read @ 300 MHz, ZCU104) and
  <https://ieeexplore.ieee.org/document/8977835> (14.4 GB/s 3 ports / 13.3 GB/s 4 ports) — decide:
  draw a measured ~13–14 GB/s ceiling on the 3-core roofline or cite-only (default: cite-only, keep
  the figure clean). Record numbers + page refs per the sourced-numbers convention.
- [ ] **R3** (exploratory, low priority — small agent) CPU-roofline feasibility: A53 NEON peak
  GFLOP/s (spec + microbenchmark; the 19.2 GFLOP/s = 4 × 1.2 GHz × 4 FLOP/cyc estimate is
  UNVERIFIED), STREAM-triad DDR bandwidth from the A53s (board experiment), FLOP/byte count for
  normalize. Deliverable: a short memo — can a defensible normalize-only roofline point be built?
  Default remains: not in the paper; salvage "normalize reaches X % of CPU peak" as a sentence.
- [ ] **R4** (optional) Skim the two most framing-relevant DATE papers (§0: hearable beamformer,
  FAMERS) for evaluation/deadline phrasing patterns.
- [ ] **R5** Verify the TerraSAR-X take/contact numbers used by the deadline table (9.9 GB contact
  budget, 64.5 GB worst-case orbit — `docs/TerraSAR-X_objective.md` §6/§9 flags them) before the
  percentage rework bakes them in.

### 4.3 Phase 1 — figures & tables (after P0 data; scripts in `LaTeX/…/figures/scripts/`)

Shared first step: define one palette/legend convention (CPU = one hue, DPU = another, *everywhere*;
combined-throughput bars in neutral colors that clash with neither) in a small shared module the
scripts import — this answers the figure-uniformity complaint structurally.

- [ ] **F1** `stacked_time.py` — drop the SD segment (warm-read baseline only); becomes *the*
  characterization figure; CPU/DPU palette.
- [ ] **F2** `optimization_ladder.py` — single ladder, rungs r0–r7, two shaded bands
  ("scheduling" r1–r4, "CPU kernels" r5–r7), per-arch cumulative × annotations, neutral bar colors,
  renamed rung labels (`mt`, `dbuf`, `ent`).
- [ ] **F3** `roofline_subgraph.py` — remove the 6.11 line; draw 1-core (solid) + 3-core (dashed)
  ceiling pairs; add aggregate dots from P0.4 (marker-distinguished from single-core dots); optional
  cited measured-BW ceiling per R2; short annotation at the weight-bound points.
- [ ] **F4** `fanout_lane_plot.py` — drop 128 L; XRT wall out of the figure (footnote in text);
  refresh with P0.2 lane grid.
- [ ] **F5** Energy figure for IV — J/patch per rung (or per key configs) from P0.2 power data;
  keep it small; decide bar-ladder vs table during design.
- [ ] **F6** `fp_cpu_stack.py` — refresh with new data; keep only if III's lane-study prose needs it
  and space allows (candidate cut).
- [ ] **F7** Occupancy checkpoint panel — only if P0.3 verdict = informative.
- [ ] **F8** Tables: Tab. 2 (column rename + INT8 const-bytes and/or footnotes: batch 1, xdputil
  peak definition); deadline table → percentages (recompute every cell with python); cross-platform
  table bold-best per row; Tab. 1 caption already carries tool versions.
- [ ] **F9** Dataflow figures (`system_dataflow`, `SAR_DDC_inference_dataflow`) — annotate datatypes
  along the path (int16 → fp32 → int8 → bitstream), align style with the palette convention. There
  is uncommitted WIP on these in the LaTeX worktree `date27-ddc-dataflow-fig` — reconcile/finish it
  rather than starting fresh.
- [ ] **F10** Export all to `figures/images/*.pdf`, compile check, eyeball pass in the PDF.

### 4.4 Phase 2 — writing (full prose; reuse the existing bullet phrasing wherever it is good)

Order: III first (hardest, most restructured), then IV, II, V, I, abstract last. Each task = turn
the section's bullets into prose under the new skeleton, keeping the §4.0 ledger in hand.

- [ ] **W1** III.*System & setup* — end-to-end path (with datatypes + batch 1), streaming as the
  acquisition model, `.ddc` product (condensed), SD-card note (consider footnoting), byte-identity
  gate (one paragraph, credibility).
- [ ] **W2** III.*Characterization* — stacked-time reading (F1), Tab. 2 with fixed semantics,
  roofline (F3) including the 3-core story + caveat; conclusion: the bottleneck migrates with
  topology and is predictable from it. h_a/h_s "cheap in absolute time" can shrink to one line.
- [ ] **W3** III.*DPU/scheduling optimizations* — the mechanism, as four named components in rung
  order: `mt` → fan-out structure → pinned placement (measured 2.15× on ResSHyp at 3 L) → lane-count
  rule (knee; multiples of #cores for DPU-bound, ≥2/core to hide CPU work; many more for CPU-bound).
  XRT footnote here. This is the section Dirk called the most important — spend the words here.
- [ ] **W4** III.*CPU optimizations* — `neon` (2.42× isolated), `dbuf`, `ent` (the rANS
  flattened-CDF + reciprocal optimization, now a measured rung; isolated numbers + the
  lookup-bound diagnosis from `docs/tmp_entropy-coding_opt_opportunities.md` — **delete that tmp
  doc once this task has extracted what it needs**); per-kernel numbers instead of a CPU roofline
  (+ R3 salvage sentence if defensible).
- [ ] **W5** III.*Combined & rules of thumb* — ladder reading (F2), occupancy deltas (P0.3), then
  the transferable rules stated *as the conclusion*: optimizations fork by binding resource;
  topology predicts the binding resource before any run; placement + lane rules.
  @user: "Based on our subgraph-to-core experiment and the number of lanes studies, we can say that <rule-of-thumb_explanations>. It should be noted that given that our observations rely on a small numbers of architectures this holds more from a rule-of-thumb than a predictive rule."
- [ ] **W6** IV — *Relaxations* (symmetrization table + overlap study, condensed from old Sec. V,
  framed as "hardware-forced relaxations, priced"); *Cross-platform baseline* (prominent, bolded
  table, FP32-Orin caveat); *TerraSAR-X deadlines* (percentage framing); *Energy* (F5 + short
  discussion: PL dominates draw; more optimization ⇒ less J/patch despite higher W).
- [ ] **W7** II — background blocks from the current draft + new: condensed CCSDS/BAQ baseline ¶
  (from old §Eval, trimmed), architecture-selection ¶ (the two binary choices, what *residual*
  means, "representative LIC topologies" — draft phrasing in this file's git history), platform ¶
  DPU-first with port widths + DDR4-2133 (+ check whether the DSPs run at double clock — PG338),
  Tab. 1, delta-vs-prior-work with double-blind-safe wording, R1 result if any.
- [ ] **W8** V — Discussion: restate the mechanism + rules of thumb, transferability to other
  models/applications, **gap analysis** grounded in the characterization (what closes the ~7×
  real-time gap: more DPU compute for Res archs — PL is at 85 % DSP, so bigger FPGA / Versal AIE;
  faster CPU or more cores for non-res archs; memory ceiling for weight-bound side nets; rough power
  cost framing); limitations (g_s INT8 cap one-liner, SD bandwidth, INT8-only quality delta cited
  from prior work); future work. Conclusion paragraph updated to the new structure.
- [ ] **W9** I — introduction + contributions list rewritten to the new skeleton ("SoC FPGA"
  wording); outline paragraph rewritten.
- [ ] **W10** Abstract + title pass — venue findings (§0 F4/F5) favor a named mechanism and a
  headline ratio; ⚠ decide with Cédric whether to name the scheduling mechanism.
- [ ] **W11** Polish for the feedback round — delete resolved `\callout`/`\CL`/`\DS` markers, run
  the P0.5 delta report against every number in the text, chktex/typo pass, full compile, PDF to
  Dirk + Martin.

### 4.5 Milestones

- **M-A** (≈ Sep 4): Phase 0 + 0b done → checkpoint with Cédric; §4.0 ledger updated if 🔄 fired.
- **M-B** (≈ Sep 8): figures compiled into the draft → quick visual review (send PNGs).
- **M-C** (≈ Sep 11): full prose draft → feedback round 2 (Dirk/Martin) — leaves ~1 week of buffer
  to the 20 Sep deadline (abstract due 13 Sep — register early with title + abstract from W10).
