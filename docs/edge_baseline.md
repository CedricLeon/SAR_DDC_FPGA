# Edge embedded-GPU baseline — Jetson AGX Orin (N2)

Cross-platform baseline for the onboard DDC compressor on the NVIDIA Jetson AGX Orin — the recognisable
`N× vs a named baseline` the DATE'27 paper reports against the ZCU102 (→ `main.tex` §eval + abstract).
This doc holds the **measurements**; the code, environment setup, deployment recipe, and known quirks
(incl. the cross-GPU decode gotcha) live in `inference_edge/README.md` (package `inference_edge/`,
`ddc-edge` CLI). For where the FPGA numbers this compares against come from, see `onboard_pipeline.md` §10.

**Status: implemented, board-verified across the full 4-arch × 4-power-mode matrix, quality spot-checked.**
A naive, unoptimized baseline by design (mirrors the FPGA's `seq` mode: no threading, no fan-out, no
DPU-style placement — not a re-run of the systems-engineering ladder on different silicon).

Sweep orchestration → `scripts/evaluation/jetson_power_arch_sweep.py` (handles the nvpmodel mode-switch
reboot — `MODE_30W`/`MODE_15W` disable CPU cores relative to `MAXN`/`MODE_50W`, which this L4T's nvpmodel
build requires a reboot to apply). The batch × precision × fuse grid and its per-arch bpp fail-fast are
driven by the same script's `--batch-sizes`/`--precisions`/`--fuse` flags; the table below is built by
`scripts/evaluation/jetson_batch_precision_table.py`.

## Results — batch × precision sweep

Orin, all 4 λ=20/relu archs, overlap=2, full scene, 7,540 patches, clocks pinned, `--warmup-rows 8`,
power on. Each cell is **`patch/s · W · mJ/patch`**, all compute-only (`VDD_GPU_SOC`+`VDD_CPU_CV` via
tegrastats — the FPGA's peripherals-excluded convention, matching main.tex's cross-platform table);
`mJ/patch = W / (patch/s)`. Precisions run best-first (bf16 ≈ fp16 here — both quality-neutral, ≈0 dB,
and leave bpp unchanged). Real/imag fusion (`--fuse-reim`) was swept too but moved throughput ≤0.2 % at
the operating point, so it is omitted. <mark>Highlight</mark> marks the best throughput / lowest power /
lowest energy per architecture; **bold** cells are hyperprior (`SHyp`/`ResSHyp`) configs at batch>1,
whose numbers are *indicative only* — that batched `.ddc` is not decodable (see *Why these GPU numbers
are indicative* below).

<table>
<thead>
<tr><th rowspan="2" align="center">mode</th><th rowspan="2" align="center">batch</th><th rowspan="2" align="center">prec</th><th colspan="3" align="center">FP</th><th colspan="3" align="center">ResFP</th><th colspan="3" align="center">SHyp</th><th colspan="3" align="center">ResSHyp</th></tr>
<tr><th align="center">patch/s</th><th align="center">W</th><th align="center">mJ</th><th align="center">patch/s</th><th align="center">W</th><th align="center">mJ</th><th align="center">patch/s</th><th align="center">W</th><th align="center">mJ</th><th align="center">patch/s</th><th align="center">W</th><th align="center">mJ</th></tr>
</thead>
<tbody>
<tr><td rowspan="9" align="center"><b>MAXN</b></td><td rowspan="3" align="center">32</td><td align="center">bf16</td><td>159.5</td><td>12.7</td><td>80</td><td>106.1</td><td>22.4</td><td>211</td><td><b>73.6</b></td><td><b>10.8</b></td><td><b>146</b></td><td><b>59.5</b></td><td><b>17.0</b></td><td><b>285</b></td></tr>
<tr><td align="center">fp16</td><td><mark>160.4</mark></td><td>12.6</td><td>79</td><td><mark>107.0</mark></td><td>23.0</td><td>215</td><td><b><mark>73.7</mark></b></td><td><b>10.7</b></td><td><b>145</b></td><td><b><mark>59.8</mark></b></td><td><b>17.4</b></td><td><b>290</b></td></tr>
<tr><td align="center">fp32</td><td>142.4</td><td>14.6</td><td>103</td><td>72.8</td><td>28.7</td><td>395</td><td><b>69.7</b></td><td><b>11.7</b></td><td><b>168</b></td><td><b>47.3</b></td><td><b>22.6</b></td><td><b>478</b></td></tr>
<tr><td rowspan="3" align="center">8</td><td align="center">bf16</td><td>156.9</td><td>12.9</td><td>82</td><td>103.9</td><td>22.4</td><td>216</td><td><b>70.5</b></td><td><b>10.9</b></td><td><b>155</b></td><td><b>57.3</b></td><td><b>16.7</b></td><td><b>292</b></td></tr>
<tr><td align="center">fp16</td><td>157.4</td><td>12.8</td><td>81</td><td>104.6</td><td>23.0</td><td>220</td><td><b>70.7</b></td><td><b>10.8</b></td><td><b>152</b></td><td><b>57.6</b></td><td><b>17.2</b></td><td><b>299</b></td></tr>
<tr><td align="center">fp32</td><td>120.0</td><td>18.8</td><td>157</td><td>67.4</td><td>29.7</td><td>440</td><td><b>62.0</b></td><td><b>14.1</b></td><td><b>228</b></td><td><b>44.2</b></td><td><b>23.5</b></td><td><b>531</b></td></tr>
<tr><td rowspan="3" align="center">1</td><td align="center">bf16</td><td>119.2</td><td>12.5</td><td>105</td><td>80.4</td><td>21.6</td><td>268</td><td>40.2</td><td>10.4</td><td>258</td><td>35.3</td><td>15.1</td><td>429</td></tr>
<tr><td align="center">fp16</td><td>118.9</td><td>12.4</td><td>104</td><td>80.5</td><td>21.9</td><td>272</td><td>40.5</td><td>10.4</td><td>256</td><td>35.2</td><td>15.2</td><td>433</td></tr>
<tr><td align="center">fp32</td><td>119.4</td><td>14.4</td><td>120</td><td>64.1</td><td>26.4</td><td>413</td><td>41.3</td><td>11.2</td><td>271</td><td>32.3</td><td>18.3</td><td>565</td></tr>
<tr><td rowspan="6" align="center"><b>MODE_15W</b></td><td rowspan="3" align="center">32</td><td align="center">bf16</td><td>75.9</td><td>4.3</td><td>56</td><td>45.5</td><td>6.0</td><td><mark>132</mark></td><td><b>36.3</b></td><td><b>3.8</b></td><td><b>104</b></td><td><b>27.4</b></td><td><b>5.1</b></td><td><b><mark>186</mark></b></td></tr>
<tr><td align="center">fp16</td><td>76.8</td><td>4.2</td><td><mark>55</mark></td><td>45.9</td><td>6.1</td><td>133</td><td><b>36.4</b></td><td><b>3.8</b></td><td><b><mark>103</mark></b></td><td><b>27.5</b></td><td><b>5.1</b></td><td><b>187</b></td></tr>
<tr><td align="center">fp32</td><td>66.4</td><td>4.6</td><td>69</td><td>29.2</td><td>7.2</td><td>246</td><td><b>33.9</b></td><td><b>4.0</b></td><td><b>119</b></td><td><b>20.4</b></td><td><b>6.2</b></td><td><b>305</b></td></tr>
<tr><td rowspan="3" align="center">1</td><td align="center">bf16</td><td>58.5</td><td><mark>4.0</mark></td><td>68</td><td>34.4</td><td><mark>6.0</mark></td><td>174</td><td>20.9</td><td><mark>3.6</mark></td><td>172</td><td>16.7</td><td><mark>4.8</mark></td><td>286</td></tr>
<tr><td align="center">fp16</td><td>58.7</td><td><mark>4.0</mark></td><td>68</td><td>34.6</td><td>6.0</td><td>173</td><td>21.0</td><td><mark>3.6</mark></td><td>171</td><td>16.7</td><td><mark>4.8</mark></td><td>287</td></tr>
<tr><td align="center">fp32</td><td>55.6</td><td>4.6</td><td>83</td><td>26.3</td><td>6.8</td><td>256</td><td>20.8</td><td>3.9</td><td>190</td><td>14.7</td><td>5.3</td><td>362</td></tr>
</tbody>
</table>

Source: `results/benchmark_jetson/orin/batch_precision_sweep/` (one JSON per run,
`<arch>_<mode>_b<batch>_<prec>_<fuse>.json`). Compress-only; quality is separate (below).

Findings: **(1)** precision helps the residual archs most (`ResFP` b32 ×1.47 fp16-over-fp32) and the
entropy-bound `SHyp` least (×1.06, dominated by CPU rANS not convs); batching helps the hyperprior
archs most (`SHyp` ×1.82, `ResSHyp` ×1.70 from b1→b32 fp16) as their extra `h_a`/`h_s` conv work
parallelizes, vs. ×1.33–1.35 for `FP`/`ResFP`. Combined (b1/fp32 → b32/fp16) the win is ×1.34 (`FP`),
×1.67 (`ResFP`), ×1.79 (`SHyp`), ×1.85 (`ResSHyp`) throughput and ~1.5–2.0× less energy. **(2)** `SHyp`
stays slower than `ResFP` (b1/fp32: 41 vs 64 patch/s) despite `ResFP`'s ~3.8× heavier `g_a` — its
second entropy pass (EB for `z`, GC for `y`) outweighs the one heavy `g_a`.

**Quality** (SHyp only — no per-arch quality sweep yet): verified against MERLIN GT (overlap=0, full
7,482-patch coverage, decoded on-device — see the cross-GPU decode note in the README): **PSNR 28.05 ±
5.24 dB, SSIM 0.8144 ± 0.1055**, consistent with the neighboring FP/ResSHyp λ=20 reference numbers.
Visual crop comparison → `results/benchmark_jetson/orin/jetson_vs_fpga_vs_merlin_crop.png`.

Timing/power methodology independently audited (`StageTimer` cross-checked against `torch.cuda.Event`,
timer overhead and `--power`'s own perturbation both confirmed negligible) — full report:
[[project_jetson_edge_pipeline]] memory.

**Why these GPU numbers are indicative, not a deployable-configuration headline.** The obvious way to
raise GPU throughput is to batch several patches through each forward pass. That is a valid, equal-bpp,
equal-quality speedup for the **factorized-prior** archs (a batched `.ddc` still decodes correctly), but
**not** for the **hyperprior** archs. A hyperprior decoder does not store the per-patch entropy tables —
it recovers them by *re-running* `h_s` at decode time to reproduce the scales the encoder used, and the
entropy coder is exact only if that reproduction is bit-identical. Running `h_s` on a batch of patches
versus one patch at a time changes its output by a hair (floating-point sums reorder with batch size),
which is enough to snap a few borderline scales into a neighbouring table and desync the entropy decode —
the reconstruction then blows up rather than degrading gracefully. So a batched hyperprior `.ddc`
measures the compute cost but is not a usable artifact (same failure class as decoding on a different GPU
or precision). We therefore report per-patch, per-configuration hardware costs as **indicative** of the
platform's capability, not as a single optimized operating point uniform across the four architectures.

## Remaining

- Per-arch quality (PSNR/SSIM) sweep — currently SHyp only.
- Thor: blocked on a torch/NCCL aarch64 ABI gap, deferred — see `inference_edge/README.md`.
- TensorRT/FP16 — explicitly out of scope for this baseline, a separate future conversation if wanted.
