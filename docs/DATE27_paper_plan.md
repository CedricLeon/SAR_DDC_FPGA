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

## 1. Situation

The repo holds a board-verified onboard streaming compressor for SAR SLC tiles on a ZCU102 plus four
measurement campaigns around it — almost none of it in the TGRS manuscript, which stopped at per-patch
characterization and an *extrapolated* full-tile projection. This paper replaces that extrapolation
with a **measured, end-to-end system** plus the **mechanism** the deployment reveals. TGRS §Limits
names it as future work verbatim (*"a streaming pipeline in which patches are … processed continuously
across multiple compute stages, allowing … full-tile processing"*).

**Off-limits (spent in TGRS)** — cite, never re-report: the 4-arch RD ablation; hardware-aware
modifications (GDN→ReLU, output_padding, graph partition); the FP32/INT8 cross-precision study; the
qualitative reconstruction grid; the per-patch latency breakdown; the cross-platform CPU/GPU/FPGA
table; the full-tile extrapolation.

---

## 2. Assets

🟢 publication-ready · 🟡 data exists, analysis/figure needs work · 🔴 needs a run or a decision

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

*(In this table, a bare §N refers to `onboard_pipeline.md` unless a doc is named.)*

**Figure code:** the DATE manuscript figures are built (`LaTeX/…/figures/scripts/`: system dataflow,
stacked time-per-patch, overlap, optimization ladder; plus the ported DDC TikZ). The older
`stream_gantt` / `stream_roofline` / `stream_sysplot` sketches (`scripts/fpga/benchmark/`) are superseded.

---

## 3. Risks

| # | Risk | Mitigation |
| --- | --- | --- |
| R1 | **Shaped as a measurement campaign** (DATE publishes named mechanisms, not case studies). | *Being addressed* — reframed around the topology-aware mechanism + a met deadline + a recognizable baseline (§6–§7). Keep the mechanism up front. |
| R2 | Self-overlap with TGRS (same platform, models, scene). | §1 off-limits list as a checklist; confirm TGRS submission status before submitting. |
| R3 | λ / grid mixing across sweeps. | **Resolved** — the whole corpus is re-run at one coherent setup (λ=20, overlap 2, snap grid 7,540); throughput is λ-independent (measured). State the setup once. |
| R4 | Single scene, single board. | Be explicit — characterization scope, not a generalization claim. |
| R5 | No SAR/EO/LIC prior art at DATE. | Motivate in systems terms (data rate, power, deadline); disambiguate "compression"/"architecture" (see `main.tex`); never assume the reader knows speckle or SLC. |

---

## 4. Experiments to deepen the work

Single home: **`onboard_pipeline.md` §5–§6 (fan-out, done) + §12 (TODO)**. **Done:** N1 core-scaling
diagnosis→fix (the mechanism's second component — pinned placement recovers the third core, the
hyperprior oversubscribes at a fourth) and A4 (ladder warm-per-rung). **Open:** **N2** Jetson
embedded-GPU baseline, **N5** CCSDS-122 baseline, **A5** PL resource-usage table. **Dropped:** N3
(deadline/budget rate allocation — tricky to implement), N4 (INT8 range cap → one-line limitation in
`main.tex`), and the architecture-predicted overlap rule (2 px is too small to be a receptive-field
effect).

---

## 5. Open questions — resolved

- **Mission-ops numbers.** The 180 s take limit is a **power ceiling** [Fritz]; the ground-contact
  window and the Neustrelitz ~90 GB/day figure stay colleague-reported — tracked as assumptions in
  `TerraSAR-X_objective.md` §6/§9, not blockers.
- **"Latent enables onboard analysis."** Kept **architectural** — the pipeline emits a despeckled
  compact latent at X patch/s (the input rate any downstream head would see); no detection head, no
  performance claimed. Embodied in `main.tex` §Discussion.

---

## 6. Characterization story (the natural framing — journal / systems-venue cut)

**Headline framing:** *we built the full onboard SLC→`.ddc` streaming compressor and measured it
end-to-end — here is where it is bound, what the hardware forces on the algorithm, and how far it is
from the operational deadlines.* This is the natural shape of the work (it is what we did), kept here as
the **fallback / journal seed**. **The story we actually submit is the mechanism cut in `main.tex`** ---
it sharpens one thread of this (the architecture-dependent bottleneck) into a *named mechanism* for the
venue; both draw on the same measurements.

### Story (mini-intro)

Future SAR missions strain the downlink: finer resolution, more polarizations and bands, and larger
constellations all grow the data faster than downlink capacity improves, so onboard compression becomes
a pressure valve — and the same compact, despeckled representation is the natural substrate for the
onboard exploitation those missions want. A learned model that jointly despeckles and compresses (DDC)
delivers both: a superior rate-distortion tradeoff *and* removal of the speckle that limits readability
and machine interpretation. What nobody has established is what such a model costs *as a full system* on
satellite-class hardware — the published evidence stops at per-patch kernel timings~\cite{TGRS}.
We deploy DDC on a Xilinx ZCU102 MPSoC and build the complete
streaming path — a full TerraSAR-X SLC tile streamed off storage, compressed patch by patch across the
DPU and ARM cores, and written as a downlink-ready `.ddc` product — then measure it end-to-end. The
bottleneck is not where per-patch numbers suggest: it moves to storage bandwidth and CPU-side entropy
coding, and it lands in a different place depending on the model architecture. The deployment also
forces two changes to the algorithm, each of which we price. The result clears a full acquisition well
before the next ground contact, but remains ~10× short of real-time in worst case scenario.

### Contributions

1. An **end-to-end measured** onboard SLC despeckle-and-compress pipeline on a satellite-class FPGA SoC
   — full-scene, from SLC-on-storage to `.ddc`-on-storage, not extrapolated from per-patch timings.
   (Scope stated plainly: onboard focusing is assumed; the downlink is emulated by writing the product
   to storage.)
2. A **bottleneck characterization**: the limit is architecture-dependent — CPU/entropy-bound for the
   factorized-prior model, DPU-bound for the scale-hyperprior — established with a byte-identical
   optimization ladder, plus the DPU fan-out core-scaling (deterministic placement recovers the third
   core; the hyperprior oversubscribes at a fourth).
3. **Two hardware-forced relaxations, each priced** on the same pipeline: dropping whole-image
   symmetrization (which otherwise makes streaming impossible; ≤0.54 dB) and minimal patch overlap
   (which removes a real reconstruction seam; overlap-2 at ~1% latency / 0.8% downlink).
4. A **quantified gap to real mission deadlines**: meets process-before-next-contact (on every basis),
   ~10× short of real-time (warm worst case) — with the honest cold/warm × working-point/worst-case
   matrix, not a single number.

### Sections (systems structure — not classic Background→Method→Result; the System *is* the method)

**I. Introduction** — future-mission downlink pressure *and* the onboard-autonomy pull; DDC as the joint
answer (one systems sentence each on speckle and SLC); the gap (per-patch numbers do not establish
system feasibility; nobody has measured the full path); contributions; the "compression = data, not
model" disambiguation.

**II. Background and related work** — SAR acquisition → why SLC, why speckle matters (readability + ML
features); learned image compression in one paragraph, DDC/MERLIN as the mechanism. **Fig: DDC
inference dataflow diagram (g_a, h_a, h_s, entropy).** Onboard FPGA SoCs + the Vitis-AI DPU as a fixed
overlay, `DPUConfig` precedent cited. Explicit delta paragraph vs the TGRS paper.

**III. System: the onboard pipeline** — the SLC→`.ddc` path (row-block stream on the azimuth axis,
patchify, normalize, DPU, rANS, `.ddc`). **Fig: system dataflow with storage bands + optimization
callouts — likely split into (a) dataflow and (b) memory/streaming.** Windowed streaming is mandatory
(f32 tile > DDR, whole-load OOMs) — one sentence. The `.ddc` product (header/body/trailer, O(1) access →
prioritized partial downlink) in 3 sentences, artifact not contribution. The correctness gate (every
schedule byte-identical to sequential) in one sentence.

**IV. What the hardware forces on the algorithm** — framing: constraints propagating *backward* from
deployment into the method. **Symmetrization** — why it exists (MERLIN needs i.i.d. real/imag), why it
breaks streaming (whole-tile FFT), why skipping is defensible (the Re/Im correlation it removes for
Noise2Noise training is small, so skipping is only slightly suboptimal), cost ≤0.54 dB across archs/rates. **Overlap** —
independent patches leave a seam; overlap-2 removes it; beyond 2 px buys nothing. **Fig: qualitative
crop ov0 vs ov2 at a patch boundary** (one visible artifact > one plot); optionally the seam-vs-interior + cost curve.

**V. Evaluation** — setup (board, two archs spanning CPU/DPU-bound, scene, λ=20 with the λ-independence
justification, warm = representative / cold = SD-testbed, power sampling). **Fig: optimization ladder**
(throughput per rung, two archs, SD-read + warm ceilings — the archs climb through different rungs and
stop against different ceilings). **Fig: stacked time per patch** (read/normalize/DPU/entropy — makes
CPU- vs DPU-bound visible). Energy: J/patch (parallelism costs power, saves energy). DPU fan-out lane
scaling + the pinned-placement ablation and the 4-lane oversubscription cliff. Baseline: embedded GPU vs FPGA SoC (N2) and/or CCSDS (N5),
desktop CPU/GPU as reference context. **Table: throughput vs mission deadlines** (real-time /
before-contact / downlink-fit; two archs; warm + cold) — the matrix, not a figure.

**VI. Discussion and limits** — remaining gap and what closes it (more cores/boards, the core-scaling
fix, PL-side normalize); honest scope (onboard focusing assumed, downlink emulated, single scene/board,
INT8 range cap on bright scatterers); what the compact latent enables downstream (architectural only, no
performance claimed).

**VII. Conclusion.**

### Results to present

Measured today unless flagged. Warm = the "if storage weren't the limit" final rung, not a further
optimization.

1. Full-scene throughput/latency/energy per ladder rung, two archs, cold (+ warm as the final
   "faster-store" rung).
2. Cumulative speedup per architecture and the ceiling each stops against.
3. Compression ratio and bitrate over the full scene, with the byte-identity statement.
4. Per-stage time decomposition per architecture and schedule.
5. Energy per patch across the ladder and architectures.
6. Quality cost of dropping symmetrization, across architectures and rate points.
7. Seam-band vs interior quality across overlap, plus latency and downlink cost.
8. DPU fan-out lane scaling per architecture (1–4 lanes), the pinned-vs-naive placement ablation, and the 4-lane oversubscription cliff.
9. *[N2/N5]* Embedded GPU vs FPGA SoC on the same pipeline and/or CCSDS, desktop CPU/GPU as context.
10. Measured throughput vs each mission deadline (real-time / before-contact / downlink-fit): met, and
    by how much missed. Side note: peak DDR footprint, windowed vs whole-tile.

**Deadline matrix** (throughput vs the three mission deadlines, cold/warm × working-point/worst-case)
→ `TerraSAR-X_objective.md` §5 (Table B) is the single source; mirrored in `main.tex` Table II.

---

## 7. DATE venue — what it publishes (condensed)

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
| [ ] | **DPUConfig** (2026 TS29.7) | same board + Vitis-AI DPU, accepted — existence proof + "mechanism on top of the overlay" template |
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
