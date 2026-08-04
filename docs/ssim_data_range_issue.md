# The AMP_LIN_99 metric basis (SSIM / MS-SSIM / EPD re-evaluation)

> **Status.** Convention decided and implemented (host + board); evidence collected on the Hamburg
> tile for all 240 matched model pairs. **Pending:** the float32 W&B re-evaluation sweep, the INT8
> board sweep, the figure/table regeneration, and the manuscript rewrite (§7).

## TL;DR

Every reference-based distortion metric now scores on one fixed basis: **inputs clipped to
`AMP_LIN_99 = 545.2`, and `data_range = AMP_LIN_99` for SSIM/MS-SSIM**. Previously MSE/PSNR clipped
but SSIM/MS-SSIM used a per-image `data_range = max(predicted)` and EPD did not clip at all — so
both depended on the brightest pixel of each reconstruction and were **not comparable across
models**. The damage concentrates on the float32-vs-INT8 comparison, because the DPU caps the INT8
recon at 2100 while float32 reaches ~1e5.

Measured over all 240 matched (arch × λ × seed) Hamburg tiles, the reported float32→INT8 drop is
mostly measurement convention:

| gap (float32 − INT8), mean over 240 pairs | old convention | AMP_LIN_99 basis | artifact share |
| --- | --- | --- | --- |
| SSIM | +0.201 | +0.031 | 84 % |
| EPD | +0.431 | +0.051 | 88 % |

The old-convention values reproduce the manuscript's claims (SSIM −0.19 to −0.23, EPD −0.36 to
−0.40) almost exactly, which is what identifies them as the artifact.

## The mechanism

SSIM compares two images in a sliding Gaussian window and averages; each local value is

`SSIM = [(2μₓμ_y + C1)(2σₓ_y + C2)] / [(μₓ² + μ_y² + C1)(σₓ² + σ_y² + C2)]`,
`C1 = (0.01·L)²`, `C2 = (0.03·L)²`, `L = data_range`.

`C1, C2` are stabilisers (they avoid 0/0 in flat regions) and are meant to be small relative to
typical `μ²/σ²`. SAR scenes are mostly dark (amplitude ~10–100 → `μ²/σ² ~1e2–1e4`) with rare bright
scatterers. If `L` is set by a bright pixel (e.g. 1e5) then `C1/C2 ~1e6` swamp `μ²/σ²` across the
dark majority, every local term collapses to `C/C ≈ 1`, and **SSIM saturates near 1** — insensitive
to real differences. A larger `L` therefore buys a higher, less meaningful score, and since
`L = max(predicted)` varies per reconstruction, two models are scored on different `L`.

EPD has no `data_range`, but it is a ratio of gradient-magnitude sums, so unclipped it is dominated
by the steep gradients around bright scatterers. It then measures how well a backend represents
point targets rather than how well it preserves edges — the same confound, via a different route.

**Why float32 and INT8 differ so much.** The on-board INT8 recon amplitude is hard-capped at exactly
2100.1: the DPU `g_s` output tensor is INT8 at fix-point 8, so its top code is `x_hat = 127/256 =
0.496`, which the denorm turns into 2100.1. Float32 has no such cap — on the same scene it reaches
~85 k (FP) / ~68 k (ResSHyp), near MERLIN's ~127 k. Full write-up: `docs/onboard_pipeline.md` §10.

## What each code path used before the fix

Four different conventions were live at once, and the manuscript's cross-precision figure compared
two of them against each other:

| path | consumer | old `data_range` | clipped? |
| --- | --- | --- | --- |
| `src/utils/metrics.py::ssim/ms_ssim` | `get_all_distortion_metrics`, notebooks, `src/evaluate.py` | `max(predicted)` | no |
| `sar_ddc_module.py::test_step` | W&B `test_sub500/*` → **GPU curves** | `max(clean_im)` | no |
| `inference_cpp/src/metrics.cpp::compute_ssim` | board `metrics.json` → **FPGA curves** | fixed **255** (OpenCV `QualitySSIM` hard-codes `C1=6.5025, C2=58.5225`) | no |
| `stitch_ddc.py::score_arrays` | overlap study (§10) | `AMP_LIN_99` | yes |

The board convention was verified empirically rather than assumed: for `FP-relu_s0_L20` the board
wrote `ssim_MERLIN = 0.61372`, and re-scoring its saved tile off-board gives 0.61395 at `L = 255`
versus 0.7919 at `max(recon)` and 0.6705 at `AMP_LIN_99`. OpenCV exposes no `data_range` parameter,
which is why the value had drifted from every host convention.

## The fix

`src/utils/metrics.py::_clip_to_amp99` is the single home for the convention; `mse`, `ssim`,
`ms_ssim` and `epd` all route through it, and `ssim`/`ms_ssim` no longer take a `data_range`
argument at all (no caller can reintroduce a per-image range). `test_step` calls those functions
instead of torchmetrics directly.

On the board, `compute_ssim` cannot pass a `data_range` to OpenCV. It instead clips to `AMP_LIN_99`
and rescales both images by `255/AMP_LIN_99`: SSIM is homogeneous, so
`SSIM(x, y, L) = SSIM(s·x, s·y, s·L)`, and scoring the rescaled pair at OpenCV's fixed 255 returns
exactly SSIM at `data_range = AMP_LIN_99`. `compute_epd` clips the same way.

`tests/test_metrics_convention.py` locks this down: scores are invariant to a peak placed above the
clip (2100 vs 85 000 must score identically), `ssim` matches torchmetrics on clipped inputs at
`data_range = AMP_LIN_99`, the scale-invariance identity the C++ relies on holds, and MSE/PSNR are
unmoved by the refactor. The board path itself has no host unit test — `opencv_quality` is not
available off-board — so the identity test is what stands in for it.

## Re-evaluating the model set

**Float32 (740 W&B runs).** `scripts/evaluation/update_wandb_runs.py` re-runs `model.test()` and
rewrites the `test_sub500/*` summary per run. Its `FILTERS_CONFIG` already selects the λ set and
seeds 0–5; widen it to all activations / `no_output_padding` variants so the ablation figures are
not left on the old basis. Roughly 1–2 min/run on GPU.

**INT8 (240 compiled models).** The 500-patch `metrics.json` is produced on the board, and per-patch
reconstructions were not saved, so the numbers need a board sweep with the rebuilt binary:

```bash
python scripts/fpga/deploy/batch_deploy.py --config <cfg>.yaml --tag ssim_fix \
    --skip-compile --save-recons        # xmodels are unchanged; only the metrics code moved
```

`--save-recons` (new) dumps every test-subset reconstruction in linA; `deploy.py` stores it
compressed as `results/reconstructions_test_set/recon_test_set_linA.npz` (~69 MB/model, ~17 GB for
the full set, lossless). This is what makes the *next* metric change a free off-board re-score
instead of another overnight sweep. Precedent batch logs put a full 240-model pass at 10–23 h.

**Hamburg tiles — already re-scorable off-board.** Both backends' tile reconstructions are on disk
(`<run_dir>/recon_<tile>_linA.npy` and `<model_dir>/results/<tile>_recon_linA.npy`), so
`scripts/evaluation/rescore_hamburg_tiles.py` produces the whole float-vs-INT8 comparison under both
conventions at no deploy cost. Its output —
`results/ssim_convention/hamburg_tile_rescore.csv`, 240 pairs — is the evidence table above. It is
read-only with respect to the canonical board-written `metrics.json`.

## Manuscript impact

Everything sits in the "The Cost of Quantization" subsection (`sec:results_crossprec`):

- **"the quantization has a significant cost: −0.19 to −0.23 points in SSIM"** — on the tile
  evidence this becomes ≈ −0.03. The headline changes from a significant perceptual cost to INT8
  being close to float32 on both metrics.
- **"EPD drops by 0.36–0.40 … mirroring the SSIM trend"** — becomes ≈ −0.05. It is not independent
  corroboration: it is the same bright-scatterer artifact reached through the gradient sums, so it
  cannot be used to support the loss-of-structure reading.
- **"PTQ acts as a regularizer … erases fine structure"** — the PSNR half (+0.82 to +2.08 dB) rests
  on the already-coherent basis and stands; the structure-erasure half loses its evidence.
- **"FPGA models tend to produce less saturated reconstructions in high-scatterer areas … we
  attribute this behavior to PTQ leading to smoother images"** — the observation is real, but the
  cause is not smoothing: it is the fix-point-8 `g_s` output cap at 2100 (§10 of
  `docs/onboard_pipeline.md`). This is a concrete, verifiable mechanism and an upgrade over the
  speculative attribution.
- Any **cross-model** SSIM ranking (across λ, seeds, archs) is affected wherever recon maxima
  differ, so every SSIM-based claim, ranking, and plot needs re-checking, not only the float-vs-INT8
  one.

Figures to regenerate once the sweeps land: `fig_crossprecision_RD` (SSIM row) from
`notebooks/compare_gpu_fpga.ipynb`, `fig_qualitative_grid` (per-tile SSIM annotations) from
`notebooks/reconstruction_visualization.ipynb`, and any SSIM panel in
`notebooks/RD-curve_ablation.ipynb`. The GPU side of those notebooks reads
`notebooks/SAR_DDC_FPGA_all_runs_WandB.csv`, so re-export it with `notebooks/fetch_wandb_runs.py`
after the W&B update.
