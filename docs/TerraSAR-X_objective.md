# TerraSAR-X Onboard-DDC Objective — Derivation

> **What this is.** A from-first-principles derivation of the *data-throughput objective* for the
> onboard despeckle-and-compress (DDC) demo: **how much SAR data a real mission produces, how fast we
> would have to process it, and whether our compression makes the result downlinkable.**
>
> **Scope:** **StripMap** mode (TSX's most common imaging mode), **single polarization**. When a spec
> gives several values we take the StripMap one, then the default, else the worst case.
>
> **Status:** numbers pinned + cross-checked against primary sources (2026-07-31). One deliberate
> modelling choice (worst-case incidence 45°) and the contact-window figures are flagged inline.

Reference keys used below (full details in [§8](#8-references)): **[Fritz]** = TSX product spec
TX-GS-DD-3302 · **[Pitz]** = Pitz & Miller, *The TerraSAR-X Satellite* · **[W&B]** = Werninghaus &
Buckreuss, *The TerraSAR-X Mission and System Design* · **[eoP]** = eoPortal · **[BAQ-pat]** = the BAQ
patent US6255987B1. Page numbers are the PDF pages of each source.

---

## 0. The question, in one picture

The onboard scenario is a chain:

```
   sensor  ──►  raw echoes  ──►  (onboard) focusing  ──►  SLC image  ──►  DDC (despeckle+compress)  ──►  .ddc  ──►  SSMM  ──►  X-band downlink
   PRF pulses    8-bit I/Q,        range-Doppler          16-bit I/Q      our FPGA pipeline            compressed   recorder    to ground station
   × Nr range    BAQ-compressed    (HYPOTHETICAL,         4 B/px          (this project)               product      (buffer)
   samples                          see §1.5)
```

We want three numbers, each a different **deadline**:

1. **Real-time** — can we compress *as fast as the sensor acquires*? (the hardest bar)
2. **Before next contact** — can we finish a whole acquisition before the next ground pass? (the *realistic* bar)
3. **Downlink-fit** — is the compressed product small enough for the X-band link?

Everything below exists to turn TSX specs into those three numbers. The single hardware fact from our
side: one ZCU102 running the factorized-prior (FP) model at its best schedule compresses SLC at
**≈ 34.7 MB/s warm** (the compute ceiling with storage non-limiting — representative of a focuser- or
SSMM-fed stream) and **≈ 22.7 MB/s cold** (SD-card testbed, read-bound). Measured, λ=20, snap grid.
`onboard_pipeline.md` §8 is the authoritative source for these numbers — **do not hardcode them
elsewhere; they evolve with the design.**

---

## 1. SAR imaging in five minutes

### 1.1 How a SAR samples the world — pulses × echoes

A SAR satellite flies along its orbit (the **azimuth** or along-track direction) and fires radar
**pulses** at a fixed rate. After each pulse it listens and records the returning echo, sampling it in
time; those samples span the **range** (across-track) direction.

So the raw data is a 2-D grid: **one azimuth line per pulse**, and **$N_r$ range samples per line**. Two
knobs therefore set the data volume — how many pulses per second (PRF, see §1.2) and how many range samples
per line ($N_r$, §1.3–1.4).

### 1.2 Pulse Repetition Frequency (PRF) - the azimuth sample rate

**PRF = pulses per second = azimuth lines per second.** It matters because it directly multiplies the
data rate, and it is boxed in from both sides:

- **Lower bound — azimuth ambiguities.** The PRF must exceed the **Doppler bandwidth** of the azimuth
  signal (explained just below), or the azimuth spectrum aliases into *azimuth ambiguities* (ghost
  targets folded into the image).
- **Upper bound — range ambiguities / timing.** The radar cannot receive while it transmits, so each
  echo must land in a *receive window* between pulses; a second strong echo arriving one pulse-interval
  later aliases in range (*range ambiguity*). A wider swath / higher incidence lengthens the echo span,
  which generally forces a **lower** PRF (the coupling used in §3.3).

TSX can command **PRF 2.0–6.5 kHz** across all modes [Fritz p.15]; the high end is dual-pol/Spotlight.
For single-pol StripMap it sits near the low end (§3.2).

### 1.3 Range bandwidth → resolution → number of range samples

The pulse is a **chirp** of bandwidth *B*. Bandwidth sets **range resolution**: $\delta_{slant} = \frac{c}{2B}$.
So more bandwidth = finer detail. TSX StripMap uses **B = 150 MHz** [Fritz p.15, p.28] (a 300 MHz
experimental mode exists [Pitz p.617], which we ignore), giving **slant-range resolution 1.2 m**
[Fritz p.28].

The echo is digitised at the **range sampling frequency** (RSF ≈ 1.1·B), so the **slant-range pixel
spacing** is $\frac{c}{2RSF}$ ≈ **0.9 m** for StripMap SSC [Fritz p.50]. The **number of range samples** is
then $N_r$ = (slant-range swath) / (pixel spacing).

*Bandwidth → resolution → pixel spacing → sample count → data volume.*

### 1.4 Incidence angle — the geometry knob

The **incidence angle θ** is the angle between the radar line-of-sight and the vertical at the target.
It links slant range and ground range: a ground-range interval $\Delta R_g$ maps to a slant-range interval
$\Delta R_s = \Delta R_g · sin \theta$. Two consequences for a **fixed 30 km ground swath**:

- **Slant swath = 30 km · sin θ.** Higher θ (more grazing) → wider slant swath → **more range samples**.
- But higher θ also lengthens the echo window → forces a **lower** PRF (§1.2). So incidence trades
  *more range samples* against *fewer azimuth lines* (We handle this in §3.3.)

### 1.5 Raw vs focused (SLC) — two products, two sizes

The sensor records **raw** (unfocused) echoes. **Focusing** (the range-Doppler algorithm) correlates
them into a **Single Look Complex (SLC)** — a sharp, complex-valued image. TSX focuses **on the
ground**, not on the satellite. **Our scenario assumes on-board focusing**, which is hypothetical but
demonstrated feasible on FPGA by [Mandapati et al., EUSAR 2024]; our contribution is *what DDC enables
given* an on-board SLC. We compress the **SLC**.

### 1.6 Data formats and BAQ — why the SLC is bigger than the raw

- **Raw sample:** the ADC produces **8-bit I + 8-bit Q** → 16 bit = **2 B/px** [Fritz p.79: "BAQ 8:8"
  is the uncompressed reference; Pitz p.617 lists the ratios].
- **BAQ (Block Adaptive Quantization)** compresses the raw *before* it is stored [Pitz p.617–618].
  How it works [BAQ-pat]: it groups **128 sample-pairs per block**, computes one **scale (exponent)**
  per block from the block's *average magnitude* (Σ|I|+|Q|), then requantises **I and Q separately** to
  a **fixed** 2/3/4/6-bit mantissa. The selectable ratios are **8:6, 8:4, 8:3, 8:2** [Pitz p.617, Fritz
  p.79, eoP]. ⚠️ The adaptation is the per-block **scale at a fixed bit-rate** — *not* a spatially
  varying bit count. Variable per-pixel bits ("6 bits over cities, 2 over ocean") is **FDBAQ**
  a different, later scheme used in Sentinel-1.
- **SLC sample:** full **16-bit I + 16-bit Q** → 32 bit = **4 B/px** [Fritz p.30, p.35]. ✅ this is
  exactly our `int16` complex assumption.

So the raw is *smaller* than the SLC only because it is coarsely quantised. Per complex sample:

| form | bits | B/px | vs SLC |
| --- | --- | --- | --- |
| ADC raw (BAQ 8:8) | 16 | 2.0 | ÷2 |
| **BAQ 8:4** (our baseline) | 8 | **1.0** | **÷4** |
| BAQ 8:2 | 4 | 0.5 | ÷8 |
| **SLC / SSC** | 32 | **4.0** | — |

This is the origin of the often-quoted "**SLC is 4× the raw**": it compares the 32-bit SLC to the
BAQ-8:4 raw (8 bit). BAQ 8:4 itself only compresses the raw **2×** (16→8 bit); the extra factor is the
SLC being full-precision.

---

## 2. StripMap parameters (cited)

Primary numbers for **StripMap, single-pol**:

| quantity | value | source |
| --- | --- | --- |
| Ground swath | **30 km** (15 km dual-pol) | [Fritz p.16], [Pitz Table II p.617] |
| Ground resolution | 3 m (both axes); az 3.3 m, gr-range 1.70–3.49 m | [Pitz p.617], [Fritz p.16] |
| Range bandwidth | **150 MHz** (100 MHz far-range; 300 MHz experimental) | [Fritz p.15, p.28], [Pitz p.617] |
| Slant-range resolution | **1.2 m** | [Fritz p.28] |
| SSC pixel spacing | **range 0.9 m, azimuth 2.0 m** | [Fritz p.50] |
| Doppler bandwidth (processed, single-pol) | **2765 Hz** | [Fritz p.29] |
| Incidence — full performance | **20°–45°** (15°–60° data collection) | [Pitz Table II p.617], [Fritz p.16, p.50] |
| SLC format | 16-bit int complex, **4 B/px** | [Fritz p.30, p.35] |
| **Instrument data rate** (mean, 8:4 BAQ) | **StripMap 580 Mbps** (Spotlight 680, ScanSAR 580) | [Pitz Table II p.617] |
| Orbit height | 514 km | [Fritz p.15] |
| Continuous imaging limit (antenna thermal limit) | up to 10 min | [Pitz p.617] |

Space-segment / ground-segment:

| quantity | value | source |
| --- | --- | --- |
| **X-band downlink** | **270 Mb/s net** (300 Mb/s gross channel) | net: [Pitz p.617]; gross: [eoP], [W&B] |
| **SSMM** (solid-state mass memory = the on-board recorder) | **384 Gbit BOL / 256 Gbit EOL** | [Pitz p.618]; EOL also [eoP] |
| **Orbit period** | **~94.9 min** | two independent derivations, below |
| Ground contact frequency | ~1 contact per orbit | [eoP] — *weak, pin before citing* |
| Acquisition constraint | **≤ 180 s monostatic per orbit** (physical ceiling set by the power budget), now ~¼ of that, battery ageing | via colleague [Fritz] |
| Ground contact window | ~5–10 min; Neustrelitz ~90 GB/day | colleague — *to confirm* |

> **Orbit period — ~94.9 min, cross-checked two ways.** (i) Repeat-cycle: 11 d × 1440 min ÷ 167 orbits
> = 94.85 min — *loose*, since a repeat ground track closes over 11 **nodal** days, not 11 solar days.
> (ii) Independent orbital mechanics: circular orbit at the cited 514.8 km mean altitude,
> $T = 2\pi\sqrt{a^3/\mu}$ with $a = R_e + h$, $\mu = 398\,600.44$ km³/s² → **94.92 min**. The two agree
> to 0.07 min, so **~94.9 min is safe**; over the quoted 505–533 km altitude range it spans
> 94.72–95.30 min. Route (ii) is the one to cite.

> **SSMM** = *Solid-State Mass Memory*: the on-board data recorder that buffers SAR data between
> acquisition and the brief, infrequent ground contacts (you cannot downlink in real time).
> **BOL / EOL = Begin-/End-of-Life:** memory cells degrade under the cumulative space-radiation dose, so
> the *guaranteed* capacity shrinks over the mission — TSX is specified at **384 Gbit at launch (BOL)**
> and **≥ 256 Gbit at the end of its design life (EOL)**, i.e. **48 GB → 32 GB**. We size against the
> pessimistic **EOL 32 GB**.

---

## 3. Deriving the numbers

### 3.1 Sensor velocity (needed for PRF)

From orbit height $h = 514$ km [Fritz p.15], Earth radius $R = 6371$ km and $GM = 3.986\times10^{14}\ \mathrm{m^3 s^{-2}}$:

$$v_\mathrm{orb} = \sqrt{\frac{GM}{R+h}} = 7.61\ \mathrm{km/s}, \qquad v_\mathrm{gt} = v_\mathrm{orb}\,\frac{R}{R+h} = 7.04\ \mathrm{km/s}$$

where $v_\mathrm{orb}$ is the orbital speed and $v_\mathrm{gt}$ the ground-track (beam-footprint) speed that sets the azimuth sampling.

### 3.2 PRF — pinned two independent ways

We are not given the StripMap PRF directly, but two independent routes agree.

**(i) From the azimuth pixel spacing.** In a single-look complex each azimuth sample *is* one pulse, so
the azimuth pixel spacing $\Delta x = v/\mathrm{PRF}$. With $\Delta x = 2.0$ m [Fritz p.50]:

$$\mathrm{PRF} = \frac{v}{\Delta x} = \frac{v}{2.0\ \mathrm{m}} = 3520\ \mathrm{Hz}\ (v_\mathrm{gt})\ \dots\ 3804\ \mathrm{Hz}\ (v_\mathrm{orb})$$

**(ii) From the Doppler bandwidth.** The PRF must oversample $B_D = 2765$ Hz [Fritz p.29] by the usual
factor $\alpha \approx 1.1\text{–}1.4$:

$$\mathrm{PRF} = \alpha\,B_D = 1.27 \times 2765 = 3520\ \mathrm{Hz}$$

Both give the same band ⇒ **PRF ≈ 3500–3800 Hz** for StripMap single-pol. (The 6.5 kHz ceiling
[Fritz p.15, p.17] is the *total* PRF — dual-pol doubles it, Spotlight needs it — **not** the StripMap
single-pol point.) We take **3800 Hz as the worst case**.

### 3.3 Number of range samples $N_r$ (working point + worst case)

With slant-range pixel spacing $\Delta r = 0.9$ m [Fritz p.50] and slant swath $= W_g \sin\theta$ (ground swath $W_g = 30$ km):

$$N_r = \frac{W_g \sin\theta}{\Delta r} = \frac{30\,000\ \mathrm{m} \cdot \sin\theta}{0.9\ \mathrm{m}}$$

$N_r$ grows with incidence $\theta$, so the *most-samples* case is the highest $\theta$ in the specified StripMap SSC range (**20°–45°**, [Fritz p.50]):

| incidence θ | slant swath | $N_r$ | note |
| --- | --- | --- | --- |
| ~26° | 13.2 km | **14 686** | our real Hamburg SM scene (working point) |
| **45°** (full-perf edge) | 21.2 km | **23 570** | **worst case (headline)** |
| 60° (data-collection extreme) | 26.0 km | 28 868 | outside SSC full-perf; and PRF drops here (§1.2) |

We anchor the working point on our real scene (14 686 range × 32 901 azimuth — a genuine TSX StripMap
SSC) and headline the **45° worst case**. We stop at 45°, not 60°, because (a) 60° is outside the
specified SSC range and (b) at 60° the longer echo window forces a *lower* PRF, cancelling part of the
sample gain — so **45° with PRF 3800 is the honest peak**.

### 3.4 Data rate — and the primary-source check

Two rates matter: the **SLC rate** is what *we* ingest and compress; the **raw rate** is what the
instrument emits (and what the primary source quotes). Both use the **int16 SSC storage format**
(4 B/px — see the ⚠️ note):

$$R_\mathrm{SLC} = N_r \cdot \mathrm{PRF} \cdot 4\ \mathrm{B/px}, \qquad R_\mathrm{raw}^{8:4} = N_r \cdot \mathrm{PRF} \cdot 1\ \mathrm{B/px} = \frac{R_\mathrm{SLC}}{4}$$

| reference point | $N_r \times$ PRF | SLC rate | raw rate (8:4) |
| --- | --- | --- | --- |
| working point (real scene) | 14 686 × 3600 | 211 MB/s | 423 Mbps |
| **[Pitz p.617] StripMap mean** | — | **290 MB/s** | **580 Mbps** |
| **worst case (headline)** | 23 570 × 3800 | **358 MB/s** | 717 Mbps |

> ⚠️ **Pixel size — int16, not fp32.** The ×4 factor holds because the SSC is **16-bit I + 16-bit Q =
> 4 B/px** [Fritz p.30, p.35] — exactly the format we read from storage. Our pipeline expands each patch
> to **fp32 (8 B/px) only in DRAM** for the DPU/normalize step; that is an internal processing cost (and
> the reason a whole-tile fp32 load OOMs), **not** the ingest rate. Quoting fp32 would wrongly *double*
> the SLC rate (×8 vs raw) — so we consistently use the int16 rate on both sides of the comparison.

**The check that matters:** [Pitz Table II p.617] gives the StripMap **mean raw rate 580 Mbps** (8:4
BAQ). ×4 ⇒ a **mean SLC rate of 290 MB/s**, sitting between our working point (211) and worst case
(358); our worst-case raw (717 Mbps) lands just above the Spotlight mean (680) — a sensible upper
bound. So the whole $N_r \times \mathrm{PRF} \times$ bit-depth chain is consistent with an independent
primary-source figure (we could not find the source rate written as a formula anywhere — deriving it is
what let us use it).

### 3.5 Volume per take and the duty cycle

At 100% duty the sensor images for the full **180 s** budget, so the take volume is

$$V = R_\mathrm{SLC} \times 180\ \mathrm{s} = \begin{cases} 358\ \mathrm{MB/s} \times 180\ \mathrm{s} = \textbf{64.5 GB} & \text{(worst case)}\\[2pt] 211\ \mathrm{MB/s} \times 180\ \mathrm{s} = \textbf{38 GB} & \text{(working point)} \end{cases}$$

**Duty cycle scales *duration*, not rate.** A 25% duty cycle is a 45 s take (≈ 16 GB), *not* a slower
acquisition — the instantaneous rate is still ~358 MB/s while imaging. So the real-time bar (§4) is
unchanged by duty cycle; only the total volume (and the before-contact bar) shrinks.

### 3.6 Downlink, SSMM, and the contact window

The X-band link carries **270 Mb/s net = 33.75 MB/s** [Pitz p.617] (300 Mb/s gross [eoP]). That is far
below the ~580 Mbps *raw* rate, which is exactly why the **SSMM buffers**: image now, downlink over the
next contacts. A 64.5 GB worst-case take is ~2× the 32 GB EOL recorder — so even *storing* an
uncompressed focused take on-board is marginal.

### 3.7 Compression

Our DDC compresses the SLC ~**24×** (measured, `onboard_pipeline.md` §8). Worst-case take
64.5 GB → **2.7 GB**; at 33.75 MB/s that downlinks in **~80 s**, well inside a 5–10 min contact. Note
the raw is *already* downlinkable (BAQ + store-and-forward), so the value of DDC is **not** "enables
downlink" but a ~6× smaller *exploitation* product, on-board semantic latents, and making on-board SLC
storage feasible.

---

## 4. The three deadlines vs. our current implementation

Requirements are the **worst case** (a 358 MB/s stream / a 64.5 GB take). "Current implementation" = the
measured single-ZCU102 best schedule (`p0+s1+prefetch+neon`), **warm** basis (compute ceiling, storage
non-limiting — representative of a focuser/SSMM-fed stream), **cold** SD-testbed in parentheses; λ=20
(throughput is λ-independent). We state **both models explicitly**, as these numbers evolve with the design:

- **FP** (factorized-prior, CPU-bound): **34.7 MB/s warm** (22.7 cold), ~21.3× compression (λ1000)
- **ResSHyp** (residual scale-hyperprior, DPU-bound): **6.0 MB/s** (SD read fully hidden → warm ≈ cold),
  ~25.6× compression (λ1000)

- **(a) Real-time** — compress as fast as acquired (**needs 358 MB/s**): FP → **10.3× short warm**
  (15.8× cold); ResSHyp → **59× short**.
- **(b) Before next contact** — finish the 64.5 GB take before the next pass ~92 min away (**needs
  11.7 MB/s**): FP → **met, 3.0× headroom warm** (1.9× cold — met on every basis) ✅; ResSHyp →
  **1.9× short**.
- **(c) Downlink-fit** — compressed output must fit the 33.75 MB/s net link: FP → **16.5 MB/s, fits
  2.0×** ✅; ResSHyp → **13.7 MB/s, fits 2.5×** ✅ (set by compression ratio, not throughput).

> **Scaling note:** the real-time gap (a) can be close with more DPU cores or several boards in parallel, see `onboard_pipeline.md`.

---

## 5. Summary tables (the hand-off)

Cite these from other docs. **Table A** is what the mission/specs give (independent of our design);
**Table B** is how our current implementation measures up (and will evolve).

**Table A — mission-derived objective** (StripMap single-pol, 100% duty, SLC = int16 4 B/px, BAQ 8:4):

| quantity | working point (real scene) | **worst case (headline)** | primary-source anchor | key sources |
| --- | --- | --- | --- | --- |
| incidence θ | ~26° | **45°** (full-perf edge) | 20°–45° range | [Pitz p.617], [Fritz p.50] |
| PRF | 3600 Hz | **3800 Hz** | 3500–3800 pinned | [Fritz p.50, p.29] |
| range samples $N_r$ | 14 686 | **23 570** | — | real scene / [Fritz p.50] |
| **SLC acq rate** (int16) | 211 MB/s | **358 MB/s** | 290 MB/s (mean) | derived; [Pitz p.617] |
| raw rate (8:4 → SSMM) | 423 Mbps | 717 Mbps | **580 Mbps (mean)** | derived; **[Pitz p.617]** |
| 180 s take (SLC) | 38 GB | **64.5 GB** | ~52 GB | derived |

Platform constants: downlink **270 Mb/s net / 300 gross**; SSMM **384 Gbit BOL / 256 EOL (48/32 GB)**;
ground swath **30 km**; SLC **int16 4 B/px**.

**Table B — current implementation vs. the worst-case objective** (single ZCU102, best schedule
`p0+s1+prefetch+neon`, λ=20; **warm** basis, **cold** SD-testbed in parentheses; *evolves with the design*):

| metric | FP (CPU-bound) | ResSHyp (DPU-bound) | requirement |
| --- | --- | --- | --- |
| SLC throughput | 34.7 (22.7 cold) MB/s | 6.0 MB/s (read hidden) | — |
| compression ratio (λ1000) | 21.3× | 25.6× | — |
| compressed worst-case take | 3.0 GB | 2.5 GB | ≤ contact budget |
| (a) real-time | 10.3× short (15.8× cold) | 59× short | 358 MB/s |
| (b) before-contact (92 min) | ✅ 3.0× headroom (1.9× cold) | 1.9× short | 11.7 MB/s |
| (c) downlink-fit | ✅ fits 2.0× | ✅ fits 2.4× | ≤ 33.75 MB/s net |

---

## 6. Assumptions & open items

- **Hypothetical on-board focusing** — TSX focuses on the ground; we assume an on-board focuser exists
  (existence proof: [Mandapati et al., EUSAR 2024]). State this explicitly in the paper.
- **Worst-case = 45°** (not 60°): the incidence–PRF coupling (§3.3) makes 45°×3800 the honest peak;
  60° would add range samples but lose PRF.
- **BAQ 8:4** assumed as the raw baseline; the operational per-scene setting is not pinned.
- **Contact-window / duty-cycle figures** (180 s, ~90 GB/day, 5–10 min) are colleague-reported — to be
  confirmed against a mission-operations source.

---

## 7. References

- **[Fritz]** T. Fritz et al., *TerraSAR-X Ground Segment — Basic Product Specification Document*,
  TX-GS-DD-3302, Issue 1.9, DLR 2013.
  `docs/references/TX-GS-DD-3302_TerraSAR-X_Basic_Product_Specification.pdf` (or
  <https://sss.terrasar-x.dlr.de/docs/TX-GS-DD-3302.pdf>). Pages cited: 15, 16, 17, 18, 28, 29, 30, 35, 50, 79.
- **[Pitz]** W. Pitz & D. Miller, *The TerraSAR-X Satellite*, IEEE TGRS 48(2), 2010, pp. 615–622.
  `docs/references/pitz_The_TerraSAR-X_satellite_2010.pdf`. Pages cited: 617 (Table II — modes, data
  rates, access ranges), 618 (SSMM 384/256 Gbit, BAQ), 622 (HRWS / scan-on-receive).
- **[W&B]** R. Werninghaus & S. Buckreuss, *The TerraSAR-X Mission and System Design*, IEEE TGRS 48(2),
  2010. `docs/references/The_TerraSAR-X_Mission_and_System_Design.pdf` (confirms 300 Mb/s gross downlink;
  ops/organisation focus — no source rate). *Note: this is the same paper as the "Werninghaus" PDF.*
- **[eoP]** eoPortal, *TerraSAR-X*, <https://www.eoportal.org/satellite-missions/terrasar-x> (SSMM
  256 Gbit EOL, 300 Mbit/s downlink, BAQ 8/{6,4,3,2}).
- **[BAQ-pat]** Lancashire, Barnes & Udall, *Block Adaptive Quantization*, US Patent US6255987B1, 2001,
  <https://patents.google.com/patent/US6255987B1/en>.
- **[Mandapati et al.]** Mandapati, Balss & Breit, *Real Time Floating Point SAR Focusing on FPGA*,
  EUSAR 2024, pp. 60–65, <https://ieeexplore.ieee.org/document/10659675>.
- **[Buckreuss, 2018]** *Ten Years of TerraSAR-X Operations*, REMOTE SENSING. `docs/references/Ten_Years_of_TerraSAR-X_Operations_Buckreuss_2018.pdf` Not a lot here except that battery life deteriorated, so can cite for claims like "data-take length is battery-limited" @TODO: clean

---

## 9. Mission context & colleague notes (to confirm)

Background for the paper's framing, from a colleague — **not yet tied to a primary source**:

- **Acquisition & timing.** TSX had a ~180 s monostatic acquisition budget per satellite per orbit
  (360 s for the TSX/TDX pair); battery ageing has since cut it to ~¼. What matters operationally is
  finishing before the **next ground contact**, not the next orbit — contacts recur from a few orbits
  to a few minutes apart.
- **Ground contacts.** Duration depends on the ground-station antenna and pass geometry: typically
  **5–10 min**, down to seconds for low passes near the horizon. DLR **Neustrelitz** (~120 km N of
  Berlin) receives **~90 GB/day** (≈ 1 morning + 1 evening pass, max 2+2). At 270 Mb/s net that is
  ~16 GB (5 min) to ~32 GB (10 min) per contact.
- **Power & thermal (why the duty cycle is limited).** The SAR transmits ~2.5 kW but the solar panels
  harvest only ~800 W, so each take draws down a battery that must recharge. The X-band T/R modules sit
  close together and overheat, needing cool-down between takes (the ≤ 10 min continuous-imaging limit,
  [Pitz p.617]). L-band systems (NISAR, ROSE-L) use larger, cooler, lower-power modules and can image
  almost continuously — at coarser resolution. Comparable X-band/high-power missions: ICEYE, Capella.
