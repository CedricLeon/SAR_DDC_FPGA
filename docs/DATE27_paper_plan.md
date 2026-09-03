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
| A6 | Mission objective (TerraSAR-X): three deadlines, page-cited | `docs/TerraSAR-X_objective.md` | 🟢 — R5-verified: rates solid, take/contact *durations* are flagged ESTIMATEs (→ footnote in deadline table); "9.9 GB" is really GiB → restate 10.1 GB |
| A7 | `.ddc` container + trailer offset table (O(1) access, prioritised downlink) | `src/utils/ddc_format.py`, §9 | 🟢 as artifact, **not** a contribution |
| A8 | INT8 `g_s` output cap at 2100.1 clips brightest ~0.7 % (point scatterers) | §8 | 🟡 prose-only unless a metric is built |
| A9 | Byte-identical correctness gate across every schedule | §4 | 🟢 credibility, one sentence |
| A10 | Cross-platform CPU/GPU energy (desktop A4000 + Xeon) | `results/benchmark_unified/` | 🟡 **spent in TGRS** — retained as *reference context* in a comparison table (informative even if published), never a contribution; rides alongside a new baseline (Jetson N2 / CCSDS N5) |

**Figure code:** `DDC_FPGA/scripts/figures/` (moved out of the LaTeX repo 2026-09-01 — rationale and
layout in §4.3). The LaTeX repo holds only rendered PDFs, the TikZ sources, and `main.tex`.
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
  Instead: measured per-kernel times + isolated speedups (neon 2.42–2.49×, entropy numbers).
  **R3 closed this 2026-09-01 and killed the salvage sentence too**: `normalize` reaches ≈3.3 % of
  the single-core NEON peak — memory-bound ruled out, limiter identified (unrolled serial dependency
  chain the in-order A53 cannot hide). A 3 % figure needs its whole mechanism explained to not read
  as a regression, which is disproportionate to one sentence. W4 keeps the isolated-speedup framing
  and drops the roofline angle entirely.
- **A53 NEON peak = 9.6 GFLOP/s per core, 38.4 GFLOP/s per chip** (LLVM `AArch64SchedA53.td` +
  microbenchmark, R3) — this **corrects the `19.2 GFLOP/s` estimate** this doc carried, which was off
  by exactly 2× (missing FMA fusion credit). Never quoted in `main.tex`: planning-doc correction only.
- XRT wall: one footnote — hyperprior archs need 3 runners/lane; 64 L (192 runners) works, 128 L
  (384) fails; limit estimated ≈300 runners. 128 L point dropped from the lane figure.
- Deadline table: plain percentages of the deadline (450 %, 15 %) instead of ×-ratios + marks.
- Cross-platform table: more prominent in the narrative; bold best board per row/metric; keep the
  Orin-runs-FP32-unoptimized caveat.
- Table 1 (PL util) stays; falls back to text only if space runs out.
- Energy lives in IV.Evaluation for now (flexible → Discussion if flow prefers).
- **Citation keys — Zotero is canonical.** Keys come from Cédric's Zotero Better-BibTeX export into
  `references.bib`; agents never invent keys.

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
- **Entropy time is near-independent of bitstream length** (verified 2026-09-01 against the archived
  λ=1000 s0 runs, prompted by the stacked-time stage moving 11.9 → 10.9 ms). SHyp `gc_compress`
  11.64 → 10.95 ms (−5.9 %) and ResSHyp 11.89 → 10.92 ms (−8.2 %) while bytes/patch fell
  12 183 → 903, i.e. **13.5× fewer output bytes buys 6–8 % less time**; FP `eb_compress` 6.75 → 6.59
  ms for 10× fewer bytes. Cost is per-symbol CDF work over a fixed-shape latent grid, not per output
  byte — the quantitative form of W4's lookup-bound diagnosis, and the reason the stage is stable
  across λ. *Also settles the stacked-time question: the shift is λ, not the entropy optimization —
  E3 ran `--no-entropy-opt` on all four archs by design (`entropy_opt: false` in every s0 JSON).*
- Stacked-time (F1) is now **λ=20 throughout**. The pre-migration figure mixed λ=1000 stage means
  with a λ=20 SD read; P1.0's repoint onto `results/date27/s0/` removed that mix.
- **CPU occupancy at the knee — mpstat, P0.7** (reduced 2026-09-01 from
  `results/date27/cpu_probe/<arch>/knee_<L>_mpstat.log`, `all`-CPU rows, first/last 10 % of samples
  trimmed as ramp/drain). **These are the numbers prose cites:**

  | arch | knee | %usr | %sys | %idle | busy |
  | --- | --- | --- | --- | --- | --- |
  | FP | 12 L | 67.4 | 8.3 | 24.3 | 75.7 |
  | SHyp | 24 L | 77.0 | 8.3 | 14.7 | 85.3 |
  | ResFP | 6 L | 13.2 | 4.2 | 82.6 | 17.4 |
  | ResSHyp | 20 L | 18.2 | 4.5 | 77.4 | 22.6 |

  The CPU-bound / DPU-bound split is stark and quotable: **67–77 % %usr for FP/SHyp against
  13–18 % for ResFP/ResSHyp**. *This validates the trace method rather than contradicting it*:
  trace-derived CPU busy for FP is 63–68 %, against mpstat's 67.4 % **%usr** — near-identical — and
  the gap to mpstat's 75.7 % total busy is the 8.3 % kernel time the trace cannot see (it only marks
  explicit pipeline stages: normalize, entropy, `g_a_cpu`). So "trace busy ≈ %usr" is the honest
  mapping. §6's old "~79 % %usr / ~17 % idle" is not wrong, it is **stale**: it described FP at the
  old 32 L knee, not the 12 L one (P0.6 re-grounding).

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
  - **`scripts/fpga/benchmark/fanout_cpu_fp.py`** (P1.0, 2026-09-01): repoint or delete. It reads the
    deleted `results/benchmark_stream/cpu_probe/` mpstat logs **and** imports `fanout_occupancy`, which
    P1.0 moved to `scripts/figures/` — so it now fails at import. Its figure (`fp_cpu_binding.png`) is
    superseded by `scripts/figures/fp_cpu_stack.py` + the `fanout_lane_plot.py` occupancy panel, both
    of which now derive CPU occupancy from the trace CSVs' `kind=cpu` spans. `onboard_pipeline.md` §6
    (waterfall 331→253→199, `fp_cpu_binding.png`, "~79 % %usr / ~17 % idle") rests on that mpstat
    source and needs re-grounding on the trace-derived numbers (FP 4-core CPU busy plateaus ≈ 63–68 %
    at the knee, vs DPU ≈ 70 %). Other stale plotters same as before: `fanout_full_table.py`,
    `fanout_gantt.py`, `fanout_lane_diagram.py`, `stream_gantt/roofline/sysplot`.
- [x] **P0.7 The last board session** (~30 min; added 2026-09-01 after the P1.0 review). Two
  unrelated gaps, bundled so the board is touched once more rather than twice. Same discipline as
  the campaign: campaign SHA, `make clean` rebuild, outputs into `results/date27/`, one MANIFEST
  line per run, λ=20/seed 0/overlap 2/snap grid/full scene/warm/power-on/batch 1.
  - **FP traces at the 12 L knee.** FP's `vaitrace/FP/vaitrace_knee32.txt` and
    `occupancy/FP/r{4,7}_full.csv` were taken at 32 L, before the knee moved to 12 (§4.0). Re-run:
    `vaitrace_knee12.txt` + `r4_full.csv` / `r7_full.csv` at `--threads 12`. Keep the 32 L files
    (provenance, as with the ladder). Blocks F3's FP aggregate dots and any F7 panel.
  - **`mpstat` CPU probe at the knee, all 4 archs** (4 runs). The trace-derived `kind=cpu` busy %
    stays the basis of F4's dashed series — it is self-consistent across all 15 lane counts and free
    — but **every CPU-occupancy number that lands in prose comes from `mpstat`**, so it stays
    comparable with the %usr/%sys/%idle framing `onboard_pipeline.md` §6 already uses. Knee only
    (FP 12 / SHyp 24 / ResFP 6 / ResSHyp 20); the full-grid re-run (~60 runs, 2.5–3 h) is
    **explicitly not worth it** — resolved 2026-09-01. Output → `results/date27/cpu_probe/<arch>/`.

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

- [x] **R1** Non-SoC-FPGA prior art: has any LIC codec been deployed on a *non-SoC* FPGA? Verify
  which platforms Mazouz 2025 and Sun 2024 actually used (we cite them as DPU-for-LIC precedents).
  Output: 3–5 sentences + BibTeX candidates → feeds II (related work) and the SoC-FPGA framing.
  *(2026-09-01: yes — Sun et al.'s lineage (VCIP'22/A-SSCC'22/JETCAS'24/ASPDAC'25) runs custom
  Verilog RTL on non-SoC KU115/VCU118 boards. Correction: `sunFPGACodecSystem2024` is NOT
  DPU-based — main.tex:137 miscites it alongside Mazouz 2025; it explicitly out-throughputs a
  DPU competitor (FPX-NIC, new candidate). Full findings + BibTeX →
  `LaTeX/SAR_DDC_FPGA_DATE27/references/R1_non-soc-fpga-LIC.md`.)*
- [x] **R2** *(2026-09-01: both papers read — **verdict: cite-only**; F3 keeps the two theoretical
  ceilings, the measured ceiling is cited in text. Lu 2022 (TRETS, ZCU104 run at DDR4-2133 — *our*
  config): peak read **13.7 GB/s @ 300 MHz** (80.6 %; Dirk's "13.3" is 13.7). Manev 2019 (ICFPT,
  ZCU102 with a **DDR4-2400** SODIMM): 14.4 GB/s; 3 HP ports beat 4; their 19.2 GB/s peak is
  correct *for their board revision*, not an error. Details + page refs → `onboard_pipeline.md`
  §2. ⚠ BibTeX for `luDemystifyingSoftHardened2022` / `manevUnexpectedDiversityQuantitative2019`
  still needs pasting into `references.bib` (W7/W11) — recreate from the DOIs if the agent's
  entries are lost.)*
- [x] **R3** CPU-roofline feasibility *(2026-09-01: verdict — confirms the default, NOT paper-worthy. Full memo:
  `docs/tmp_R3_cpu-roofline.md` — delete once W4 is written and has extracted what it needs, per this
  doc's own convention for `tmp_*` memos.)*
- **R4 — dropped 2026-09-01** (was optional: skim the hearable beamformer + FAMERS for
  evaluation/deadline phrasing). Revive only if W10's title/abstract pass stalls for want of a model.
- [x] **R5** *(2026-09-01)* Checked against `docs/references/` PDFs + web. **Verdict: mixed — one solid
  input, one weak input per headline number; both headline numbers are usable but rest on an ESTIMATE.**
  - **64.5 GB worst-case orbit** = 358 MB/s × **180 s**. Rate **solid** ([Pitz p.617]-cross-checked);
    **180 s take budget is an uncited ESTIMATE** (no primary source; [eoP]'s "<180 s" is a roll-slew
    time, not imaging — *do not cite it*).
  - **9.9 GB single-contact** = 33.75 MB/s × **5 min**. Downlink rate **solid** (net 270 Mb/s verbatim
    [Pitz p.617]); **5 min contact plausible but uncited**. ⚠️ Arithmetic: 33.75 × 300 s = **10.1 GB
    (decimal) = 9.9 GiB** — the deadline table's "9.9 GB" is really GiB; restate as **10.1 GB** or label
    GiB for consistency with the decimal 64.5 GB.
  - Bonus: [eoP] gives **320 Gbit BOL** SSMM (doc/Pitz say 384; EOL 256 agrees, immaterial — we size on
    EOL). Full outcome + citations folded into `TerraSAR-X_objective.md` §2/§6/§9.

### 4.3 Phase 1 — figures & tables

**Where the code lives (decided 2026-09-01).** Figure code moves *out* of the LaTeX repo into
`DDC_FPGA/scripts/figures/`. Today the dependency runs backwards — repo B's scripts climb
`../../../..` into repo A's results and `sys.path`-hack into `scripts/fpga/benchmark/` for the
occupancy math — while the LaTeX repo is a publication artifact that should not know how to compute
anything. After the move: repo A owns data + code + provenance, repo B receives rendered PDFs only.

- `scripts/figures/_figutils.py` — the shared module (leading underscore matches the `_plotkit` /
  `_benchmark_loader` house convention). Owns: the palette (CPU one hue, DPU another, storage/read
  neutral grey, *everywhere*; combined-throughput bars neutral against both), per-arch colors,
  display names, rung labels, knee lanes, loaders for every `results/date27/` subdir, the per-core
  occupancy attribution, and one save helper emitting PDF + PNG. This answers the figure-uniformity
  complaint structurally rather than by convention.
- Output dir resolves to `REPO_ROOT/LaTeX/SAR_DDC_FPGA_DATE27/figures/images/` with a
  `DATE27_FIG_OUT` env override, and **hard-errors if absent** — never a silent write to cwd.
- `results/date27/` is now **version-controlled** (`b784587`): the ignore is narrowed to `results/*`
  with a `!results/date27/` negation, and `^results/` is excluded from pre-commit so measurement
  records stay byte-exact. Figures are therefore reproducible from a clone of repo A alone.
- The other stale plotters in `scripts/fpga/benchmark/` (`fanout_gantt`, `fanout_cpu_fp`,
  `fanout_full_table`, `fanout_table`, `stream_gantt/roofline/sysplot/table`) also read the deleted
  trees. Only `fanout_occupancy.py` is load-bearing → moves in P1.0; **the rest are P0.6's problem**
  (repoint or delete).
- **P1.0 done (2026-09-01).** `scripts/figures/{_figutils,fanout_occupancy}.py` + the 7 figure scripts;
  6 run green, `overlap_crop.py` hard-errors by design (its `_work/` crops were archived — the last
  render is kept). `fanout_occupancy` now also derives CPU occupancy from the trace CSVs' `kind=cpu`
  spans (`mean min(active,4)/4` over the steady window) since `cpu_probe/` is gone. **One forced
  deviation from "current design":** `optimization_ladder.py` / `energy_ladder.py` were two panels with
  *arch-specific* rung sets; the campaign only measured the single cumulative r0–r7 sequence §4.0
  defines, so both are now one panel, all 4 archs — the F2/F5 redesign target minus the bands. All
  §4.0 self-check numbers reproduce (r7 tput, ×5.5/5.0/3.8/3.7, 2.15× placement, 40/82/79 % DPU share).

**Commissioning batches.** One agent session per row, reviewed before the next is issued.

| # | Batch | Covers | Seam rationale |
| --- | --- | --- | --- |
| P1.0 | Infrastructure & migration ✅ 2026-09-01 | `_figutils` + move 7 scripts (delete `system_dataflow.py` — it's the TikZ figure), repoint to `results/date27/`, every script green at its **current** design | Mechanical, zero design judgment; yields a gate-reviewable numbers table before any redesign |
| P1.1 | Characterization | F1, F3 | One story (bottleneck migrates, `g_a` decides); shared `s0/` + `vaitrace/` data |
| P1.2 | Mechanism — **interactive with Cédric** ✅ | F2+F5 merged into `ladder.py`; F4 refreshed; F6 → `cpu_composition.py`, **cut as a float**, numbers → W5; F7 = **no figure**, r0/r4/r7 table → W5 (`checkpoint_occupancy.py`) | Important section (Dirk) |
| P1.OV | Overlap crop salvage | Restore 5 crops from the archive so the overlap figure is editable again | No board time, touches nothing else — run it in parallel with P1.1 |
| P1.3 + P1.5 | **Manuscript numbers & floats** (merged 2026-09-03) | F8 tables + F10 float wiring, caption sync, compile | One focus — every edit is a number or a float in `main.tex`, verified by one compile. Splitting them would touch the same file twice |
| P1.4 | Dataflow tikz | F9 | Different medium (repo B, TikZ) |
| P1.R | Reproducibility kit | Campaign driver + `results/date27/README.md` | Independent of the figures; **after** the Sep 8 milestone |

Every batch reports the caption facts that changed, so P1.5 inherits a checklist instead of a diff
hunt. Batches P1.1–P1.3 must not edit captions or prose themselves.

**P1.OV spec**: `overlap_crop.py` needs decoded tiles that P0.C archived — 1.93 GB each, untrackable.
The archive (`benchmark_stream_overlap_work.tar.gz`, 23 GB, 61 entries) holds all five overlap
settings for ResSHyp at both λ=20 and λ=1000. Salvage **ResSHyp λ=1000, ov ∈ {0, 2, 4, 8, 16}**, cut
a **1024²** window from each (~4 MB each, ~20 MB total) into `results/date27/overlap/`, and repoint
the script. 1024² rather than the rendered 340² so the window can still be moved or zoomed; five
overlaps rather than the two the current figure shows so any pair — or a five-across strip — stays
reachable. λ=20 not salvaged. The seam-PSNR panel's `overlap_table.csv` survived in-tree untouched.

**P1.R spec** (when commissioned): a driver generated from the §4.1a campaign spec that writes
`results/date27/…` in the recorded layout and appends MANIFEST lines, with the board's SSH alias and
paths parameterized. It cannot be validated without 5–6 h of board time, so it ships with
`--dry-run` plus a check asserting its generated command set **equals the flag-sets MANIFEST.md
already records** — real verification at zero board cost. The companion `README.md` states plainly
that it is dry-run-verified, not re-executed.

**Figure specs** (F-numbers are referenced from the W tasks in §4.4 — keep them stable):

- [x] **F1** `stacked_time.py` — drop the SD segment (warm-read baseline only); becomes *the*
  characterization figure; CPU/DPU palette.
- [x] **F2 / F5 — one `ladder.py`** (P1.2, both scripts deleted). `--throughput` / `--energy` pick
  the figure; `--ratio` / `--absolute` pick the mode. They stay **two independent floats** (F2 in
  III, F5 in IV), never a two-panel figure. Decisions:
  - **F2 = per-arch normalized speedup (× over seq)**, not absolute patch/s — the linear absolute
    scale buried the DPU-bound mechanism (placement fix reads as a ~15 patch/s bump). Absolute
    patch/s labelled on the seq & +ent groups for FP (fastest) and ResSH (slowest) only; the full
    four are in the throughput table.
  - **F5 = absolute J/patch on a log axis** (default; `--ratio` gives the normalized mirror). Board
    power shown as a per-column range (`10–12 W` → `15–21 W`), not per-arch (collides).
  - Both: bands = mt–knee "scheduling" / +neon–+ent "CPU kernels"; unshaded gaps set off `seq` and
    the knee→+neon boundary; arch palette (blue/orange), not neutral; SD cold-read ceiling line
    dropped.
- [x] **F3** `roofline_subgraph.py` — remove the 6.11 line; draw 1-core (solid) + 3-core (dashed)
  ceiling pairs; add aggregate dots from P0.4 (marker-distinguished from single-core dots); short
  annotation at the weight-bound points. **The ~13.7 GB/s measured ceiling is NOT drawn** — R2's
  verdict (§4.2) is cite-only in text; the figure keeps the two theoretical ceilings. (This line
  used to say "optional"; R2 settled it.)
- [x] **F4** `fanout_lane_plot.py` — drop 128 L; XRT wall out of the figure (footnote in text);
  refresh with P0.2 lane grid (entropy-on for all four archs, which also dissolves the entropy-off /
  entropy-on mismatch flagged in the current caption). The dashed CPU series is **trace-derived**
  (`kind=cpu` spans; `cpu_probe/`'s mpstat basis died with P0.C) — legend reads "CPU busy", not
  "%usr". Numbers cited in *prose* come from P0.7's knee-point mpstat instead, so the doc's
  %usr/%sys/%idle framing survives. If the trace series looks noisy at high lane counts, suspect the
  `--max-rows` subsampling window before suspecting the method.
  @TOADD to text: "Each hyperprior lane needs 3 XRT runners; 64 lanes (192 runners) is the largest configuration tested, and 128 lanes (384 runners) fails to initialise. The runner ceiling is estimated at ≈300."
  P1.2: kept the 3 panels (throughput / occupancy / energy); confirmed the grid is the **full
  optimized stack** at every lane count (fanout+neon+dbuf+entropy, pinned), not fo3p; x-axis now
  ticks + labels every measured lane count.
- [x] **F5** — see the F2 entry above (same `ladder.py`).
- [x] **F6 → `cpu_composition.py`** (renamed from `fp_cpu_stack`; script + figure). **Not a
  manuscript float** (decided P1.2) — the numbers go into W5's prose instead. The mpstat lane sweep
  died with P0.C and P0.7 only re-probed the knee, so the "vs lanes" form is unrecoverable; the
  figure is now a 4-arch `%usr/%sys/%idle` stacked bar at each knee (kept for the record, original
  orange/red/grey colours). **W5 numbers** (steady-state, 4-core mean): SH uses the A53s hardest
  (77 % usr / 15 % idle), then FP (67 / 24), ResSH (18 / 77), ResFP least (13 / 83); `%sys` stays a
  manageable 4–8 % across all four. Reproduces the §4.0 mpstat table; loader
  `_figutils.cpu_probe_mpstat()`.
- [x] **F7 — NO figure** (P1.2 verdict). Numbers computed by
  `scripts/figures/checkpoint_occupancy.py` (`fanout_occupancy.occupancy()` now takes an optional
  `trace_path`, so the r4/r7 full-scene traces go through the *same* `[t1−e,t1]` reconstruction as
  the lane grid; r0 from the E3 sequential per-stage split, reprojected onto the 3-core DPU / 4-core
  A53 complex). **r7 reproduces F4's knee occupancy panel to <1 pt** (independent traces agree) — so
  a 3-checkpoint panel would be 2/3 restatement of F4. For W5's prose:

  | ckpt | FP | SH | ResFP | ResSH |
  | --- | --- | --- | --- | --- |
  | r0 seq DPU / CPU (% of complex) | 13 / 15 | 13 / 15 | 27 / 4 | 26 / 5 |
  | r4 DPU / CPU (3-/4-core) | 51 / 81 | 45 / 72 | 93 / 16 | 84 / 20 |
  | r7 DPU / CPU | 70 / 64 | 62 / 81 | 100 / 10 | 96 / 15 |

  **The insight F4 can't show — the r4→r7 crossover.** With scheduling alone (r4) FP is *still
  CPU-bound*: A53 cores 81 % busy vs DPU 51 %. The CPU-kernel rungs (r5–r7) rebalance it to DPU 70 /
  CPU 64. That is *why the ladder needs both bands* — for FP/SH the scheduling band only fills the
  DPU partway; the CPU-kernel band closes the gap. ResFP/ResSH are DPU-bound at every checkpoint
  (CPU ≤ 20 % throughout), so their CPU-kernel rungs barely move the needle — matches the ladder.
  Caveats: r0 is a construction (seq 1-thread duty ÷3/÷4), not a 3-core measurement; r0/r4 entropy
  unoptimised, r7 optimised, so the r4→r7 CPU delta bundles neon+dbuf+ent; 4-core CPU-busy mildly
  over-counts under oversubscription (direction reliable, absolutes soft — hard CPU numbers still
  come from P0.7 mpstat).
- [ ] **F8** Tables: Tab. 2 (column rename + INT8 const-bytes and/or footnotes: batch 1, xdputil
  peak definition); deadline table → percentages (recompute every cell with python); cross-platform
  table bold-best per row; Tab. 1 caption already carries tool versions.
- [ ] **F9** Dataflow figures (`system_dataflow`, `SAR_DDC_inference_dataflow`) — annotate datatypes
  along the path (int16 → fp32 → int8 → bitstream), align style with the palette convention.
  *Start from LaTeX `main`, not from a worktree.* The old `date27-ddc-dataflow-fig` branch was an
  Aug-12-based snapshot that `main` had already overtaken on every file but one; it was deleted
  2026-09-01 after its single surviving idea was recorded here: **the `system_dataflow` legend should
  read "Storage / ARM CPU (A53) / DPU" on row one and "factorized path / hyperprior only" on a second
  row below it** (`main`'s .tex still says "CPU" and "main path" on one row, and its caption already
  says *factorized* — so the .tex is the stale half). Re-centre the legend scope after widening.
- [ ] **F10** Float wiring, caption number-sync, compile check, eyeball pass in the PDF.
  **`main.tex` does not currently compile**: line ~280 still `\includegraphics`es
  `figures/images/fp_cpu_stack.pdf`, which P1.2 renamed to `cpu_composition.*` *and* cut as a float.
  Float changes: drop that figure block entirely; **un-comment the energy figure** (~line 351) and
  place it in IV.
  **Caption checklist from the P1.2 session** — every item below is a caption fact that changed:
  - **F2 ladder**: y-axis is now **normalized speedup (× over seq)**, not absolute patch/s. Absolute
    endpoints are labelled for **FP and ResSH only** (37→204, 10→38) so the caption must name which
    two archs carry them; all four live in the throughput table. Bands: "scheduling" = mt→knee,
    "CPU kernels" = +neon→+ent. Rung labels use the renames (`mt`, `dbuf`). Cumulative seq→+ent =
    ×5.5 / ×5.0 / ×3.8 / ×3.7. **The SD cold-read ceiling line is gone** — drop it from the caption.
  - **F5 energy**: absolute **J/patch on a log scale**. seq→+ent per arch: 0.271→0.079 ·
    0.333→0.102 · 1.047→0.480 · 1.146→0.551 J (3.4 / 3.3 / 2.2 / 2.1×). Board power 10–12 W → 15–21 W.
    Same bands as F2.
  - **F4 lanes**: every lane count now runs the **full optimized stack** (fanout+neon+dbuf+entropy,
    pinned), not `fo3p` — this **dissolves the old entropy-off/on `\CL{}` mismatch note**, delete it.
    Knee stars 205 / 146 / 41 / 38 patch/s. Dashed series legend = "CPU busy" (trace-derived);
    prose CPU numbers cite P0.7 mpstat instead. XRT wall leaves the figure for a footnote: *"Each
    hyperprior lane needs 3 XRT runners; 64 lanes (192 runners) is the largest tested, 128 (384)
    fails to initialise; ceiling estimated ≈300."*
  - **F6**: cut as a float — remove its `\label` and `\includegraphics`. Numbers go to W5: SH 77 %usr
    · FP 67 · ResSH 18 · ResFP 13, with %sys 4–8 % throughout. File is `cpu_composition.*` if ever
    reinstated.
  - **F1 / F3**: Cédric's rework notes above stay open for a final pass during writing; captions
    should not lock in wording those notes will change.

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
  doc once this task has extracted what it needs**); per-kernel numbers instead of a CPU roofline —
  **no salvage sentence**, R3 ruled it out (§4.0). Also delete `docs/tmp_R3_cpu-roofline.md` here.
- [ ] **W5** III.*Combined & rules of thumb* — ladder reading (F2), occupancy deltas (P0.3), then
  the transferable rules stated *as the conclusion*: optimizations fork by binding resource;
  topology predicts the binding resource before any run; placement + lane rules.
  @user: "Based on our subgraph-to-core experiment and the number of lanes studies, we can say that <rule-of-thumb_explanations>. It should be noted that given that our observations rely on a small numbers of architectures this holds more from a rule-of-thumb than a predictive rule."
- [ ] **W6** IV — *Relaxations* (symmetrization table + overlap study, condensed from old Sec. V,
  framed as "hardware-forced relaxations, priced"); *Cross-platform baseline* (prominent, bolded
  table, FP32-Orin caveat); *TerraSAR-X deadlines* (percentage framing; fix 9.9 GB → 10.1 GB and
  footnote that take/contact durations are operator-reported estimates while rates are page-cited —
  R5); *Energy* (F5 + short
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

#### Small figure adjustments while writing

*User maintained*:

- **F1**:
  - Re-order the stage back in their occurring order (like before) and remove the horizontal black line separating them
  - Y axis "latency [ms]", the info about per-patch and sequential goes in the caption + text.
  - Scale the "DPU XX%" annotation down (maybe no bold and in black)
  - Bring the "CPU-bound" and "DPU-bound" architectures closer together (no need to reinstaure the X axis, it's nice without)
  - Rename "CPU-bound" and "DPU-bound", it's too overselling, all these archs are not bound by anaything for the moment, just seq.
    We should either remove the caption entirely and leave it to the text or tone it down a bit like,
- **F3**:
  - There are way too many annotations, we need to remove a lot of them. A lot of the text should go in the caption (for example that there is 1 AXI per core but that the DDR is shared)
  - We need to investigate why hs dropped from 26 to 20% after we re-ran
  - Let's simplify the 1core to 3 core datapoint movements (maybe not colors?)
  - Are the 3-cores datapoints for the maximally optimized configs? i.e., with prefetch/nenon/etc.? I don't know if it changes the DPU efficiency. I don't know how we collect the DPU efficiency and throughput on 3 cores actually.

### 4.5 Milestones

- **M-A** (≈ Sep 4): Phase 0 + 0b done → checkpoint with Cédric; §4.0 ledger updated if 🔄 fired.
- **M-B** (≈ Sep 8): figures compiled into the draft → quick visual review (send PNGs).
- **M-C** (≈ Sep 11): full prose draft → feedback round 2 (Dirk/Martin) — leaves ~1 week of buffer
  to the 20 Sep deadline (abstract due 13 Sep — register early with title + abstract from W10).
