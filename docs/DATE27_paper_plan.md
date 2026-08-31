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

## 4. Feedback and actions to take after first draft (2026-08-26)

### Dirk's email and propositions

"Overall it's *quite difficult to understand* at the moment"

I also have some more high level comments on figures and how i would structure certain sections, maybe this helps.
Hopefully I understood the optimizations you did.

#### Figures

- the figures are a bit confusing, they are not uniform and make it difficult to understand what actually gets faster, which optimizations were made
- For example fig.3 reports Latency broken into separate kernels, but the next fig.5 reports Patch/s only for the total application. Difficult to say which part was accelerated, how did the distribution between CPU and DPU change.
- I would pick different colours in fig.5 as after fig.3 I associate blue with kernels running on CPU and orange with kernels on the DPU. But if I understand it correctly it is both combined.
- Roofline: I would remove 6.11 GB/s line also added some comments in overleaf. Would be nice to have a roofline of CPU kernels as well.

#### Optimizations/Analysis

- Start by a quick overview of the structure
    1. characterization (Latency diagram fig.3 without SD card) -> DPU time significant for all configurations -> we start with optimizations for the DPU
    2. DPU optimization -> Even though fixed, by changing scheduling and matching CPU threads to DPUs we accelerate the runtime by X
    3. CPU optimization -> After 2. CPU is a significant part for Non res nets we now look at CPU optimization
    4. Now we combine optimizations, and come up with some transferable rules of thumb
    (5. Finally, we also measure energy of different optimizations, because this is very important for RS)
- Then go into all the sections in detail

1. Initial Characterization
    - Fig. 3, explain stuff
    - Roofline is good, but has to include more information. Maybe multiple DPUs and a CPU roofline ?(Maybe plot how points change with optimizations). Currently it just tells us g_s and g_a are compute bound, and h_s/h_a are not bound by anything but also time wise pretty irrelevant. Could be a one liner in the text as well if you want to save space.

2. DPU
    - Substantial part of the pipeline (maybe ~40%-80%) fig.3 , which we discovered through characterization.
    - List all the optimizations to scheduling the DPU you did, I think this is the most important part.
    - Showcase the new runtime in a similar diagram to fig.3

3. CPU
    - CPU is important for non-res networks, especially now that the DPU is faster
    - List your optimizations (Pool, neom, prefetch, etc.),
    - Showcase the new runtime in a similar diagram to fig.3 (Fig.3 could also include multiple subfigures, after DPU opt, CPU + DPU opt, final Opt)

4: (Conclusion) Combine both things together
    - we are super fast
    - List your rule of thumbs (Or leave it to the end/ Discussion)
    - How would you use these optimizations for other architectures/applications

1. Maybe mention energy here as well, and/or what could improve throughput or put it at the end into discussion

#### Evaluation/Real-World

    - I would put limitations or algorithmic changes (Sec V) into background or directly into evaluation.
    - First we compare against a similar sized GPU, we are faster with our optimizations
    - However for real-time applications we are still too slow
    - What is missing for the FPGA, do we need a bigger FPGA faster CPU or faster Memory? Is it possible within the power budget, how much more power would we use. Could we use 2 FPGAs? Could newer FPGA help, i.e Versal?

##### Discussion

    - Restate the topology, scheduling stuff
    - Rule of thumb (How to use the insights for other applications)
    - New architectures required

### Some key points I extracted from the comments he left in Overleaf

- refer to the board as SoC FPGA, check if anyone did something similar than us on non-SoC FPGA?
- Move the version footnote to the table caption (if we keep the table)
- Explain in more details why we select these 4 architectures, how they differ (what residual mean) and that they represent ome typical topologies that can be expected for LIC codecs. "Similarly to Leonard et al, we experiment with 4 different model architectures following the combinations of 2 binary choices: whether the entropy prior is factorized or hyperprior and whether the main encpder/decoder use internal residual connections to deepen unlock larger representation capacities." (improve phrasing)
- Add more details about the DPU (speed and port width)
- Specify that the DPU uses uint8 datatype while CPu works on fp32
- Remove the SD card read from the detailed stage latencies figure
- Think about how we can show this same per-stage latencies after each optimizations
- Try to work on the roofline: Think about what it would mean for the 3 DPUs, try to plot it for the CPU ceilings.
- Check both papers he recommended and see if we can use their memory characterization for our plots and just mention them
- Verify the memory per frame for h_a/h_s, something is off in our report of his computation
- Specify batch size 1
- Is the \emph{pool} optimization multithreading? If yes rename.
- rename prefetch in parallel or double load. What is the prefetcher he refers to?
- Signiifcantly simplify the XRT wall bullet point. I'd say we remove the 128-lanes point from the figure and just add a footnote that XRT has a limit of runners around 300 that stops us from creating more lanes than 100 for hyperprior archs.
- Bring the energy plot back, with a quick discussion
- Transform the speedup (headroom/short) in the deadline percent in a simple percentage.
- HIghlight the cross-platform comparison a bit more, and bold for best scores.

### Next steps?

Overall, I think he makes a good point about the confusing structure. Well, the fact that the draft is currently made of bullet points and not prose has surely contributed to the confusion, but I think we should simplify the structure and disentangle the divers contributions and messages we try to present.
His proposition of "We characterize -> We optimize the DPU -> we optimize the CPU -> Overall results and insights" is way simpler and probably easier to read. We should see if we lose some ways to introduce some results/infos like we do now.
There is some work left on the figures (simplification, colors, new info).
We should open for a hardware crowd and insights about the FPGA and design.
