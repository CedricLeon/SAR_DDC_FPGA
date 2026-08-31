# DATE 2027 paper — planning scratchpad

> **Temporary working doc** for the DATE'27 submission. Dissolve into
> `LaTeX/SAR_DDC_FPGA_DATE27/` + the permanent docs once the manuscript takes shape.
> **The story we follow is the mechanism cut; the draft is `main.tex` and this doc plans around it.**
> Condensed DATE venue evidence is in §7.
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

## 4. Post-draft feedback — Dirk, 2026-08-26 (consolidated)

Three sources merged and deduplicated — **(E)** his email, **(O)** his Overleaf comments (`\DS{}` in
`main.tex` @ `2015a9e`), **(K)** points/reactions I added. Status marks: **✍** writing/figure work,
**✅** question resolved below, **⚠** decision still open, **🔍** research needed.
Overall verdict (E): *"quite difficult to understand at the moment"* — partly because the draft is
still bullet points, but the structure itself needs simplification.

### 4.1 Restructure (the big item) ✍

Adopt Dirk's narrative (E) — open with a quick overview of the structure, then the sections:

1. **Characterization** — stacked-time figure *without* SD card → DPU time is significant for all
   configurations → optimize the DPU first.
2. **DPU optimization** — fan-out scheduling + subgraph-to-core placement + matching CPU workers to
   DPU lanes → speedup X. Dirk: *"I think this is the most important part."* Show the new runtime in
   a stacked-time-style diagram.
3. **CPU optimization** — now that the DPU is faster, CPU dominates the non-residual nets → pool /
   prefetch / neon / entropy. Again show the resulting stacked-time.
4. **Combine** — final numbers, transferable rules of thumb, how the insights apply to other
   architectures/applications.
5. **Energy** — effect of the optimizations on energy (important for RS); in-section or Discussion.

Notes on mapping from the current draft:

- The current cut is "characterize → two ladders side by side"; the new cut serializes DPU-first,
  then CPU. **⚠ Check what we lose**: the "optimizations fork by binding resource" insight must
  survive — it can become the *conclusion* of steps 2+3 instead of their premise. This likely also
  dissolves my ladder-ordering worry (rung order changing the reading, "+entropy hidden behind the
  shared DPU"): with a DPU block and a CPU block, within-block ordering matters less.
- **Sec. V (relaxations)** → fold into background or directly into evaluation (E). ⚠ where.
- **Evaluation narrative** (E): (i) vs. a similar-sized embedded GPU we are faster with our
  optimizations; (ii) but still short of real-time; (iii) **gap analysis**: what would close the
  ~7× — bigger FPGA? faster CPU? faster memory? two FPGAs? Versal? and at what power cost / is it
  within the power budget? ✍ *new content to produce* (pairs with the conclusion's
  design-implications callout).
- **Discussion section** (E): restate the topology/scheduling findings, the rule of thumb for other
  applications, what new hardware architectures are required.

### 4.2 Figures ✍

- **Uniformity** (E): figures don't share a visual language; hard to see *what* got faster and how
  the CPU/DPU split changed (Fig. 3 shows per-kernel latency, but Fig. 5 only whole-app patch/s).
- **`fig:stacked_time`**: drop the SD-card segment (E,O,K); *repeat* the breakdown after each
  optimization block — e.g. subfigures: baseline / +DPU opt / +CPU+DPU opt (E,O,K).
- **`fig:ladder` colors** (E): blue/orange already mean CPU/DPU in Fig. 3; the ladder reuses them
  for combined throughput → change palette or align semantics.
- **Roofline** (E,O,K): drop the 6.11 GB/s line (a measured point, not a ceiling — see 4.3); keep
  9.6 GB/s as the single-core ceiling; add a 3-core roofline (compute ×3, memory capped at
  17.06 GB/s — since 3×9.6 = 28.8 > 17.06 the 3-core roof is DDR-capped; Dirk's picture, consistent
  with ours); try a CPU roofline with points for +neon/+prefetch; optionally show points moving with
  optimizations. h_a/h_s "not bound by anything and time-wise irrelevant" can shrink to one line of
  text if space is needed (E).
- **Fan-out lane figure**: drop the 128-lane point; XRT wall becomes one sentence/footnote (O,K —
  see 4.3).
- **Energy ladder**: bring back with a short discussion — Dirk finds it relevant (O,K); place per
  4.1 step 5.
- **Cross-platform table**: make it more prominent; bold the best board per row (O,K).
- **Deadline table**: replace headroom/short ratios + marks with plain percentages of the deadline
  (450 % instead of 4.5×, 15 % instead of 6.8× short) (O,K).
- **Table 1 (PL util)**: tool versions moved into the caption @`2015a9e` (O). ⚠ keep the table at
  all?

### 4.3 Fact checks & answered questions ✅

- **Dirk's 17.35 GB/s puzzle** (O: "1374 FPS × 12.63 MB/frame = 17 353 MB/s — do you use
  batch > 1?") — resolved; his premise is off, but it exposes a table-clarity bug. `Mem [MB]` in
  Tab. 2 is the **FP32 checkpoint size** (params × 4 B: 3.16 M × 4 = 12.63 MB), which the DPU never
  touches. Per inference (batch = 1, weights re-read every frame) the DPU loads the **INT8** consts:
  `const_bytes` = 3.16 MB for h_s (xmodel static info; vaitrace measures LdWB = 3.0 MB/frame,
  97–99 % of h_a/h_s DDR traffic). `Max DPU FPS` is the measured `xdputil benchmark` DPU-only peak.
  Implied bandwidth: 1374.7 × 3.16 MB = **4.34 GB/s** (h_a: 6.04 GB/s) — consistent with the
  roofline's 6.11 GB/s (= static bytes ÷ HW_RT), nothing excessive. **Actions ✍**: rename the
  column (e.g. "FP32 size") or add an INT8/per-frame-traffic column; footnote what Max DPU FPS
  means; state batch = 1 explicitly (O,K).
- **Memory ceilings** (E,O,K): all numbers check out. 9.6 GB/s **per core** = 2 × 128-bit
  `M_AXI_DATA` × 300 MHz (PG338) — Dirk agrees. 17.06 GB/s chip-wide = DDR4-2133 × 64-bit (the
  ZCU102's Kingston KVR21SE15S8/4 SODIMM is 2133 MT/s; papers quoting 19.2 GB/s assume DDR4-2400).
  6.11 GB/s is *not* a hardware ceiling — it's the measured best-case weight-load rate of h_a/h_s
  (single-core, uncontended; `onboard_pipeline.md` §6) → remove as a line, keep at most as an
  annotation on the points.
- **Practical DDR ceilings from Dirk's refs** 🔍: <https://dl.acm.org/doi/10.1145/3517131> (p. 19:
  13.3 GB/s max parallel read @300 MHz, ZCU104 ≈ ZCU102) and
  <https://ieeexplore.ieee.org/document/8977835> (14.4 GB/s for 3 ports, 13.3 GB/s for 4,
  @300 MHz). His own experiments agree. Read both; decide whether to draw a measured ~13–14 GB/s
  ceiling on the 3-core roofline or just cite them.
- **Datatypes** (O,K): the DPU computes in **INT8 (signed, per-tensor fixpos scaling)** — not uint8
  as Dirk wrote. CPU side: raw complex int16 in, fp32 normalization, int8 to/from the DPU, rANS
  bitstream out. ✍ state this along the end-to-end path + in the platform paragraph.
- **`pool` = multithreading?** (O,K): yes — N CPU worker threads sharing the 3 serialized DPU
  cores. ✍ rename (e.g. `mt`/`workers`) and call it a thread pool.
- **`prefetch` rename** (O,K): the prefetcher Dirk thought of is the CPU's *hardware cache
  prefetcher* (or `__builtin_prefetch`); ours is a double-buffered row-block read that overlaps I/O
  with compute. ✍ rename → `double-buffer` / `overlap-read` / "parallel load".
- **XRT wall** (O,K): hyperprior archs need 3 runners/lane → 64 L = 192 runners works, 128 L = 384
  fails; limit estimated ≈300 runners (i.e. ≈100 lanes for hyperprior). ✍ compress to one
  sentence/footnote, drop the figure's last point. (Exact XRT limit unverified — keep "≈".)

### 4.4 Small text edits ✍

- Say **"SoC FPGA"** everywhere (not "FPGA MPSoC" / "FPGA SoC") (O,K).
- **Architecture-selection paragraph** (O,K): explain the two binary choices properly — what
  *residual* means (the plain variant has no internal residual blocks in g_a/g_s), why these four,
  and that they represent typical LIC-codec topologies. Draft (K, improve): *"Similarly to Léonard
  et al., we experiment with 4 model architectures following the combinations of two binary
  choices: whether the entropy prior is factorized or a hyperprior, and whether the main
  encoder/decoder use internal residual connections to unlock larger representation capacities."*
- **Platform paragraph DPU-first** (O,K): "We use the default B4096 DPU (…, 2 × 128-bit AXI data
  ports per core) implemented at 300 MHz × 3 cores on a ZCU102 (4× Cortex-A53, 4 GB DDR4-2133
  64-bit)" — include memory speed and port width.
- **CCSDS / SAR-compression-baseline paragraph** (O): Dirk didn't get what to learn from it → move
  to background or next to the architecture-choice explanation. ⚠ interacts with the keep/drop
  decision (4.5).

### 4.5 Open decisions ⚠

- **SAR compression baseline (CCSDS/BAQ ¶)**: keep (moved to background, per Dirk) or drop? My
  callout worried about overselling; relocating it lowers the stakes.
- **Sec. V placement**: background vs. evaluation (4.1).
- **Table 1 (PL utilization)**: keep, shrink, or fold into text?
- **Which memory ceiling(s) on the roofline**: 9.6 single-core + 17.06 3-core (+ cite measured
  ~13–14)? → settle after reading Dirk's two refs.
- **Where energy lives**: inside the optimization section vs. Discussion.

### 4.6 Research todos 🔍

- **Non-SoC-FPGA prior art** (O): has a LIC codec been deployed on a *non-SoC* FPGA platform? Also
  check which platforms our cited works (Mazouz 2025, Sun 2024) actually used.
- Read the two memory-characterization papers (links in 4.3).
- DATE reading list (§0) still mostly unread.
