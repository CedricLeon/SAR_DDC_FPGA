# main.tex — current structure vs. target

Scratch reference for the restructuring pass. Target skeleton is `DATE27_paper_plan.md` §4.0;
writing tasks are §4.4. Delete this file once the restructure is done.

State as of 2026-09-04: `main.tex` = 585 lines, 7 sections, bullet-form. Numbers are verified
(number pass complete). Figures and tables are final. Nothing blocks the restructure.

---

## A. CURRENT — what is actually in the file

- **I. Introduction** (L104)
  - Motivation (future missions): resolution/polarisations/constellations grow SAR data faster
    than downlink improves; onboard compression as the pressure valve; the same compact despeckled
    representation is the substrate onboard exploitation needs
  - Disambiguation: *data* compression not model compression; "architecture"/"topology" = the
    network, not the hardware
  - Domain in systems terms: SLC, speckle, DDC learned codec in one forward pass
  - The gap: per-patch kernel timings do not tell whether the full system is feasible
  - Contributions: **literally `\CL{@TODO}` — empty**
- **II. Background and Related Work** (L129)
  - ¶ SAR Data and Speckle: active imaging, azimuth/range, focusing → SLC, speckle, MERLIN
  - ¶ SAR Data Compression: onboard raw-data compression, TerraSAR-X BAQ, LIC autoencoders,
    joint despeckling
  - **Commented-out block**: the old "SAR Image Compression Baseline" subsection (CCSDS / BAQ /
    MERLIN+JPEG2000 comparison, bpp and ratio figures) — already parked here, per Dirk's callout
    that it belongs in background rather than evaluation
  - Missing relative to target: architecture-selection ¶, platform ¶, delta-vs-prior-work
- **III. Onboard Streaming Pipeline** (L190)
  - Four codec architectures span the compute range: prior (FP vs SH) × residual (~9× g_a ops);
    λ=20 / seed 0 rationale (λ changes weights and entropy tables, not structure, so no hardware
    effect) — *carries an open Dirk callout asking what "residual" means and why these two axes*
  - ZCU102 platform and base facts: ZU9EG, 4× A53, 4 GB PS DDR4, 3× DPU B4096 @300 MHz
    (~1229 GOP/s/core), 85 % DSP, Table 1 — *open Dirk callout: lead with the DPU configuration,
    not the FPGA; add port width and speed*
  - End-to-end path (Fig. system_dataflow): stream row-blocks → patchify 256² → normalize
    (log + min/max) → DPU (g_a×2, plus h_a/h_s) → rANS → `.ddc`
  - Streaming is the acquisition model, not a memory trick: 14 686 × 32 901 scene, row-block =
    256 azimuth lines × full range = 58 patches, batch = 1 everywhere
  - The `.ddc` downlink product: header + per-patch body + optional trailer offset table, O(1)
    random access, prioritised/partial downlink; overhead 1.2–1.5 %, per-patch so it scales with
    patch count not scene size, largest fraction for the best-compressing arch
  - SD-card side note: ~23.5 MB/s ≈ 91.9 patch/s, becomes the bottleneck once optimised, hence
    warm-read (DDR) basis throughout — *open DS callout about Fig. 3 still showing the SD read;
    that is now stale, F1 dropped the SD segment*
  - Byte-identity correctness gate: every optimisation byte-identical to sequential, so the study
    is purely about throughput and energy, and because a bit offset in the latents shifts the
    entropy CDF
- **IV. Characterization: the Topology-Dependent Bottleneck** (L246)
  - **IV.A The bottleneck migrates with topology** (L250)
    - Per-patch latency decomposition (Fig. stacked_time): FP/SH — CPU work (normalize + entropy,
      ~15–20 ms) outweighs a small g_a (~10 ms); ResFP/ResSH dominated by residual g_a (~73 ms)
    - Per-subgraph DPU analysis via Table 2 + roofline (Fig. roofline)
    - Consequence: FP/SH spend ~60 % of compute on the CPU; Res* are DPU-dominated
  - **IV.B Two optimization ladders, and a deployment rule of thumb** (L294)
    - "The optimizations fork by binding resource" — **stale: still describes TWO ladders in two
      subplots. F2 is now ONE cumulative ladder, normalized speedup, all four archs**
    - CPU-focused optimisations in a nutshell: mt (4 workers sharing serialized DPU), dbuf, neon
      (×2.42 isolated), entropy (flattened per-channel CDF + precomputed reciprocal) — *open DS
      callouts: is mt just multithreading; rename dbuf to avoid implying the HW prefetcher*
    - Optimizing DPU usage: fan-out vs the prior channel-parallel `s1` scheme; N is per-arch and
      harder to pick than expected → forward-reference to IV.D
    - Figure reading: ×5.5 (FP) and ×5.0 (SH) overall; mt and fan-out carry most of it
  - **IV.C The subgraph-to-core placement fix** (L318)
    - The problem: VART assigns cores round-robin at runner creation (lane-major); harmless for
      factorized archs with one DPU subgraph, damaging for hyperprior archs with three unbalanced
      subgraphs — all lanes' g_a collide on one core
    - The fix: force subgraph-major creation so lane k's g_a (and h_a/h_s) lands on core k.
      ResSH at 3 lanes: 13.7 → 29.5 patch/s (**2.15×**); SH gains only ~9 % (CPU-bound); no
      arch is hurt, so it is applied everywhere afterwards
  - **IV.D Identifying the best schedule and fan-out** (L325)
    - A deployment rule of thumb: distinguish CPU-bound (non-residual g_a) from hyperprior archs
      to decide whether CPU-focused optimisation pays
    - Lane-count selection: DPU-bound need ≥ ~2 lanes/core to hide CPU work; more for hyperprior
      archs juggling more DPU calls
    - Lane study (Fig. lane): throughput / occupancy / energy vs lanes; long flat roofs, so the
      operating point is the **knee** (smallest lane count within 1 % of peak) — FP 12, SH 24,
      ResFP 6, ResSH 20; trades ≤1 % throughput for far less oversubscription
    - Observations: DPU-bound saturate early (ResFP 99.6 %, ResSH 90.9 % DPU); CPU-bound fill the
      A53s late, 67–77 % %usr at the knee
    - No hard wall: hyperprior archs (3 runners/lane) hit `ulimit -n`=1024 at 113 lanes (339
      runners), liftable with `prlimit`; memory never binds (~1.2 GiB peak resident, CMA ~6 MB per
      runner) — *open DS callout: compress to one sentence, drop the last data point*
- **V. What the Hardware Forces on the Algorithm** (L364)
  - Dropping whole-tile symmetrization: MERLIN's scene-wide FFT makes Re/Im i.i.d. for
    Noise2Noise-style training, but a whole-image FFT breaks streaming
  - Minimal patch overlap: patch recombination leaves edge artifacts, a common remote-sensing
    problem
- **VI. Evaluation: Energy, Deadlines, and Baselines** (L411)
  - VI.A Energy consumption (L418)
  - VI.B TerraSAR-X mission deadlines (L434)
  - VI.C Cross-platform baseline (L467)
- **VII. Conclusion** (L497)
  - Remaining real-time gap and what closes it: more DPU cores / parallel boards (byte-identical
    and embarrassingly data-parallel across patches); move log-normalisation into the PL
  - Honest scope: onboard focusing assumed; downlink emulated by writing `.ddc` to storage; single
    scene; single board; INT8 g_s cap clips the brightest ~0.7 %
  - The rule of thumb is a rule of thumb — four topologies only
  - Unexplored: what the compact latent enables onboard (ship detection, disaster monitoring …)

---

## B. TARGET — five sections, and what feeds each

- **I. Introduction**
  - Keep: motivation, disambiguation, domain-in-systems-terms, the gap
  - Write: the contributions list, matching the final structure (currently `@TODO`)
  - Venue steer: DATE rewards a named mechanism plus a headline ratio against a recognisable
    baseline — decide whether the scheduling mechanism gets a name
- **II. Background & Related Work**
  - Keep: SAR/SLC/speckle ¶, BAQ + TerraSAR-X ¶, LIC ¶
  - Add a DDC ¶ (currently folded into the LIC paragraph)
  - Uncomment and condense the CCSDS / BAQ / JPEG2000 baseline block already parked here
  - **Move in from III**: the architecture-selection ¶ (the two binary choices, what *residual*
    means, "representative LIC topologies" — answers Dirk's callout)
  - **Move in from III**: the platform ¶, reframed DPU-first with port widths and DDR4-2133,
    plus Table 1
  - Add: delta-vs-prior-work, double-blind-safe wording
- **III. Analysis & Optimizations**
  - **III.A System & setup** ← the rest of current III: end-to-end path with datatypes and
    batch = 1, streaming as acquisition model, `.ddc` product (condensed), SD-card note (consider
    footnoting), byte-identity gate as one credibility paragraph
  - **III.B Characterization** ← current IV.A: stacked-time reading, Table 2 with fixed semantics,
    roofline including the 3-core story and its caveat. Conclusion to state explicitly: the
    bottleneck migrates with topology and is predictable from it
  - **III.C DPU/scheduling optimizations** ← the mechanism, as named components in rung order:
    mt → fan-out structure → pinned placement (2.15× on ResSH at 3 L) → lane-count rule (knee;
    multiples of #cores for DPU-bound, ≥2/core to hide CPU work, many more for CPU-bound).
    Absorbs current IV.C entirely and the fan-out parts of IV.B and IV.D. XRT footnote lands here.
    *This is the section Dirk called the most important — spend the words here*
  - **III.D CPU optimizations** ← neon (2.42× isolated), dbuf, ent (flattened CDF + reciprocal).
    Currently one bullet inside IV.B; needs real expansion. Per-kernel numbers, no CPU roofline
  - **III.E Combined results & rules of thumb** ← the single ladder (F2), occupancy deltas, then
    the transferable rules *as the conclusion*: optimisations fork by binding resource; topology
    predicts the binding resource before any run; placement and lane rules
- **IV. Evaluation**
  - **IV.A Relaxations** ← current V, condensed, framed as "hardware-forced relaxations, priced"
    (symmetrization table + overlap study)
  - **IV.B Cross-platform baseline** ← current VI.C, **promoted to first and made prominent**;
    bolded table, keep the FP32-unoptimized-Orin caveat
  - **IV.C TerraSAR-X deadlines** ← current VI.B; percentage framing, 10.1 GB, footnote that
    rates are page-cited while take/contact durations are operator-reported estimates
  - **IV.D Energy** ← current VI.A, moved last; F5 plus the short discussion (PL dominates draw;
    more optimisation ⇒ less J/patch despite higher W)
- **V. Discussion & Conclusion**
  - Restate the mechanism and the rules of thumb, transferability to other models/applications
  - **Gap analysis** grounded in the characterization: what closes the ~7× real-time gap — more
    DPU compute for Res archs (PL already at 85 % DSP → bigger part or Versal AIE), faster/more
    CPU cores for non-residual archs, the memory ceiling for weight-bound side networks
  - Limitations (g_s INT8 cap one-liner, SD bandwidth, INT8-only quality delta cited from prior
    work) and future work

---

## C. THE MOVES — restructuring checklist

1. III "Four codec architectures" bullet → II as the architecture-selection ¶
2. III "ZCU102 platform" bullet + Table 1 → II as the platform ¶, reframed DPU-first
3. Remaining III bullets → III.A System & setup
4. IV.A → III.B, unchanged in substance
5. IV.B splits three ways: CPU-optimisation bullet → III.D; fan-out bullet → III.C;
   ladder-reading bullet → III.E
6. IV.C → III.C (the placement fix, intact)
7. IV.D splits: lane-count rule → III.C; rules of thumb and observations → III.E
8. V → IV.A, condensed
9. VI reorders: VI.C → IV.B (first), VI.B → IV.C, VI.A → IV.D (last)
10. Uncomment and trim the CCSDS baseline block into II
11. VII → V, expanded with the gap analysis
12. Contributions (I) written near the end; abstract last

---

## D. WATCH-OUTS while moving text

- **IV.B is stale in structure, not just wording**: it describes "two cumulative optimization
  ladders" in "two subplots". F2 is now one cumulative ladder with normalized speedup on the
  y-axis and absolute patch/s annotated only for FP and ResSH. That whole bullet needs rewriting
  as it moves, not copying.
- **IV.C's first bullet forward-references the section it is already in** ("a fix that we introduce
  in Section~\ref{sec:placement}") — an artifact of the two-ladder structure. Drop it.
- **Contributions are empty.** Nothing else in I depends on the restructure, so it can be written
  as soon as the section cut is settled.
- **Open reviewer callouts to resolve during the move**, not after: Dirk on what *residual* means
  and why those two axes (→ II architecture ¶), Dirk on leading with the DPU configuration (→ II
  platform ¶), Dirk on compressing the no-hard-wall bullet (→ III.C footnote), DS on mt vs
  multithreading and on the dbuf name (→ III.D).
- **DS's "Fig. 3 still includes the SD card read" is already resolved** — F1 dropped the SD
  segment. The marker can go when W11 sweeps markers.
- **Keep §1's off-limits list open as a checklist.** Moving text is exactly when material spent in
  the TGRS paper creeps back in: the 4-arch RD ablation, hardware-aware modifications, the
  FP32/INT8 study, the qualitative grid, the per-patch latency breakdown, the cross-platform
  CPU/GPU/FPGA table, the full-tile extrapolation.
- **7 sections → 5, on 6 pages + 1 reference page.** The compression is the point; III grows and
  everything else has to give.
