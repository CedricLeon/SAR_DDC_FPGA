# SSIM / MS-SSIM `data_range` issue — re-evaluation briefing

> **Purpose.** Hand-off for a dedicated session that will (a) decide the fix, (b) re-evaluate the
> affected models, (c) update W&B, (d) regenerate the manuscript plots/tables, and (e) check whether
> any TGRS-paper conclusion changes. This doc is the entry point: it states the problem, the exact
> code, the evidence, the likely conclusion-impact, the proposed fix, and the scope. Read it, then
> verify on the real test set + models (the numbers here are spot-checks, not the full sweep).

## TL;DR

`ssim()` and `ms_ssim()` use **`data_range = max(predicted)`** (the per-image recon max). Because the
SSIM stabilisers scale with `data_range²`, this makes SSIM **non-comparable across models** and
**saturates it for high-dynamic-range reconstructions**. The most damaging consequence: the float32
recon reaches ~85 k while the INT8 (PTQ) recon is hard-capped at ~2100 (see below), so the two are
scored with a **40× different `data_range`** — inflating float SSIM toward 1 and depressing int8 SSIM.
The apparent "float→int8 SSIM drop" is therefore mostly an artifact.

- **Affected:** `ssim`, `ms_ssim` (and every metric built on them: `ssim_merlin`, `ssim_adam_noc`,
  `ssim_noisy`, `ms_ssim_*`).
- **Not affected:** `psnr`, `mse` (both clip to `AMP_LIN_99` and use it as the peak — coherent);
  `epd`, `enl`, `ratio_mean`, `ratio_enl` (no `data_range`).

## The mechanism (why `data_range` matters so much)

SSIM compares two images in a sliding Gaussian window and averages; each local value is

`SSIM = [(2μₓμ_y + C1)(2σₓ_y + C2)] / [(μₓ² + μ_y² + C1)(σₓ² + σ_y² + C2)]`,
`C1 = (0.01·L)²`, `C2 = (0.03·L)²`, `L = data_range`.

`C1, C2` are stabilisers (avoid 0/0 in flat regions) and are meant to be small vs typical `μ²/σ²`.
They scale with `L²`. SAR scenes are mostly dark (amplitude ~10–100 → `μ²/σ² ~1e2–1e4`) with rare
bright scatterers. If `L` is set by a bright pixel (e.g. 1e5), `C1/C2 ~1e6` swamp `μ²/σ²` across the
dark majority → every local term collapses to `C/C ≈ 1` → **SSIM saturates near 1**, insensitive to
real differences. So a larger `L` → higher, less meaningful SSIM.

Since `L = max(predicted)` varies per model/recon, two models are scored on different `L` — SSIM is
not comparable between them.

## Where it is (code)

| function | file:line | `data_range` | status |
| --- | --- | --- | --- |
| `ssim` | `src/utils/metrics.py:83` | `max(predicted)` if `None` | **bug** |
| `ms_ssim` | `src/utils/metrics.py:92` | `max(predicted)` if `None` | **bug (same)** |
| `psnr` | `src/utils/metrics.py:72` | peak = `AMP_LIN_99` (fixed) | coherent |
| `mse` | `src/utils/metrics.py:60` | clips both to `AMP_LIN_99` | coherent |
| test-set eval | `src/models/sar_ddc_module.py::test_step` (~L256–288) | `peak = max(clean_im)` passed to SSIM/MS-SSIM | **bug (propagates)** |

`AMP_LIN_99 = 545.2` (99th-percentile linear amplitude) — `src/utils/constants.py`. It's the peak/clip
target PSNR and MSE already use, so it's the natural coherent choice.

## Why float vs INT8 differ so much (the 2100 cap)

The on-board INT8 recon amplitude is **hard-capped at exactly 2100.1** — the DPU `g_s` output tensor is
INT8 at fix-point 8, so its top code is `x_hat = 127/256 = 0.496`, which the denorm turns into 2100.1.
The float32 model has no such cap: on the same scene it reaches ~85 k (FP) / ~68 k (ResSHyp), near
MERLIN's ~127 k. Full write-up + verification: `docs/onboard_pipeline.md` §10 ("INT8 recon caps bright
scatterers at 2100"). Consequence for SSIM: `max(predicted)` is ~2100 for INT8 but ~85 k for float →
the 40× `data_range` gap that fabricates most of the SSIM difference.

## Evidence (spot-check, FP λ20, 4096² urban region, recon vs MERLIN GT)

| convention | float SSIM | int8 SSIM | apparent drop |
| --- | --- | --- | --- |
| current: `data_range = max(pred)` (float 84880 / int8 2100) | 0.9951 | 0.7712 | **0.224** |
| coherent: `data_range = AMP_LIN_99 = 545` for both | 0.6954 | 0.6133 | **0.082** |

So ~0.14 of the ~0.22 gap is pure `data_range` artifact. The *real* INT8 SSIM cost is ~0.08. (Also
note the current-convention values differ from a per-patch test-set average by spatial coverage —
full-tile SSIM is diluted by dark areas; the test-set is per-256²-patch. Reproduce on the real test
set before trusting exact magnitudes.) Reproduce with `scripts` analogous to
`$CLAUDE_JOB_DIR/tmp/ssim_ptq.py` (float model via `symmetrization_study.resolve_model` + `_load_module`
+ `predict_linA`; INT8 recon from the board/quant model).

## Likely impact on conclusions (verify against the manuscript)

The paper reportedly concludes PTQ (INT8) **raises PSNR** (regulariser effect) but **drops SSIM ~0.3**.
- PSNR is on the coherent `AMP_LIN_99` basis → that half is probably fine.
- The SSIM drop is mostly the `data_range` artifact → after the fix it likely **shrinks to ~0.08**,
  i.e. INT8 is close to float on both metrics, and the "large SSIM penalty" narrative weakens.
- Any **cross-model** SSIM comparison (across λ / seeds / archs) is also affected whenever recon maxes
  differ — not just float-vs-int8. Re-check every SSIM-based claim, ranking, or plot.

## Proposed fix (decide first)

Make SSIM/MS-SSIM coherent with PSNR/MSE: **clip predicted+target to `AMP_LIN_99` and pass
`data_range = AMP_LIN_99`** (a fixed constant, same for every model). Rationale: identical basis as
PSNR/MSE, comparable across all models, no saturation, no dependence on the recon's brightest pixel.
Alternatives to weigh: a fixed nominal like `exp(AMP_MAX)=46270` (data-independent but large → still
saturates); keep `max(predicted)` (the current problem). Recommend `AMP_LIN_99`. Whatever is chosen,
apply it identically to SSIM and MS-SSIM and document it next to the PSNR clip.

## Scope of re-evaluation

- **Models:** the full set used in the paper — ~4 archs × (λ set) × 6 seeds. Enumerate exactly (float
  checkpoints under `logs/train/...`; INT8/compiled under `results/fpga/compiled_models/`, currently
  λ∈{1,2,5,10,20,50,100,200,500,1000}). Confirm how INT8 SSIM was produced (quant model in Docker vs board) — that drives
  the cost.
- **Eval code:** `test_step` (`sar_ddc_module.py`) already computes all of this.
- **W&B:** update the SSIM/MS-SSIM summary (and `ssim_noisy/adam/merlin`, `ms_ssim_*`) per run without
  clobbering unaffected history. `scripts/evaluation/update_wandb_runs.py` is a precedent for W&B edits., it should re-evaluate all models matching a filter.
- **FPGA scores**: After the float32 scores, it might be necessary to re-evaluate the SSIM of the INT8 models on the FPGA. Then all models need to be re-deployed and their inference re-run, their results fetched. Maybe recompilation is not necessary and simply updating the results/ folder of each compiled_models/ is faster.
- **Plots/tables:** regenerate every manuscript figure/table that uses SSIM/MS-SSIM (find them in the
  analysis notebooks + `LaTeX/` draft).
- **Rewrite**: the sections referring SSIM or that derive from the previous conclusion. Most of the work will be done by the user, but you will help identify parts that need an update (see the current manuscript at `LaTeX/SAR_DDC_FPGA_TGRS_2026/main.tex`).

## Effort estimate

- Fix + convention decision: small (1 hour, mostly deciding + a coherent 2-line change + a self-test).
- Float re-eval: automatable sweep, ~1–2 min/model on GPU → a few GPU-hours for a few hundred models.
- INT8 re-eval: cost depends on the pipeline (Re-compile? Deploy per model and re-evaluation). This is the main unknown — scope it first.
- W&B update + plot/table regen + conclusion re-assessment (needs the manuscript): the careful part, ≈1 day.
- **Overall: ~1–2 focused days**, dominated by the W&B/plots/interpretation, not the code.

## What the new session should do

1. Clearly identify the problem and reproduce the artifact on the real test set for a couple of representative models (float + INT8).
2. Read the manuscript draft; list every conclusion that rests on SSIM/MS-SSIM (esp. the PTQ
   float-vs-int8 claim and any cross-model SSIM ranking).
3. Decide + implement the `data_range` fix; add a self-test/known-answer.
4. Re-evaluate the model set and update W&B
5. Jointly with the user, regenerate the SSIM plots/tables and slowly tackle the manuscript changes

Pointers: `src/utils/metrics.py` (metric defs), `src/models/sar_ddc_module.py::test_step` (eval),
`src/utils/constants.py` (`AMP_LIN_99`, `AMP_MIN/MAX`), `docs/onboard_pipeline.md` §10 (the 2100 INT8
cap), `scripts/evaluation/update_wandb_runs.py` (W&B edits), `scripts/fpga/deploy/`.
