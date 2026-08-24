# ddc-edge

A device-agnostic Python/PyTorch replica of the onboard streaming pipeline's **default** behavior
(`inference_cpp/src/stream/stream_pipeline.cpp`'s `seq` mode), for benchmarking embedded/desktop CUDA
GPUs (Jetson Orin, Thor, Nano, or any CUDA box) as an "N2" baseline against the FPGA numbers in
`docs/onboard_pipeline.md` §12.

## Scope (deliberate)

Mirrors the current C++ pipeline's **algorithmic** default only — same patch grid (`overlap=2` by
default, via `src/utils/tiling.make_offsets` — byte-identical rule to the C++ grid), no symmetrization,
same normalization, same per-patch compress sequence (`g_a → h_a → EB → h_s → GC.compress`, real +
imag). It deliberately drops everything about the C++ implementation that is a *systems* optimization
rather than a behavior: no CPU thread pools, no `--s1`/`--p0`/`--fanout`/`--prefetch`/`--neon`-style
flags, no windowed/streamed tile reads (the whole tile is loaded at once — trivial on Jetson-class RAM),
no DPU-style subgraph/runner placement (there is no DPU here). One patch at a time, straight through.

This is intentional: the goal is a clean baseline number, not a re-implementation of the FPGA's
systems-engineering ladder on a different chip.

## Why a separate package, not an extension of `benchmark_gpu.py`

`scripts/evaluation/benchmark_gpu.py` predates several pipeline changes (dropped symmetrization,
`overlap=2` default, the SSIM/checkpoint fixes — see `project_jetson_benchmark_scope` memory) and only
ever cycles a small synthetic patch subset, never a real full scene.
`ddc-edge` is the implementation that actually mirrors current pipeline behavior end-to-end
over a real tile, and is what any reported N2 number should come from.

## Genericity / packaging

Own `pyproject.toml`, own minimal dependency set, device
picked at runtime (`cuda` if available else `cpu`) — same code should run on Orin, Thor, a Nano, or a desktop
GPU. Power sampling degrades gracefully: `tegrastats` (Jetson, whole-SoC) → `nvidia-smi` (any NVIDIA
GPU, board power only) → `null` if neither exists (never a silent 0).

**Not yet fully vendored**: it still imports model definitions and a few utilities (`src.utils.tiling`,
`src.utils.constants`, `src.utils.ddc_format`, optionally `src.utils.sar_utils`/`metrics`) from the main
DDC_FPGA repo via `rootutils`, so a target device needs a checkout of this repo alongside `inference_edge/`
(not a fully standalone wheel). Full vendoring is a possible future step, not done yet.

## Environment setup (Jetson)

Verified on an AGX Orin (2026-08-24), JetPack 7.2.1 / L4T r39.2 / CUDA 13.2 (`cat /etc/nv_tegra_release`,
`dpkg -l nvidia-jetpack`). No Jetson-specific package index was needed — as of this JetPack generation
Orin can use the **standard upstream PyTorch wheels** directly:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install --no-cache-dir lightning torchmetrics "compressai==1.2.8" "hydra-core==1.3.2" rootutils
pip install -e .   # from inference_edge/
```

`compressai==1.2.8` has no aarch64 wheel on PyPI and builds from source (a small C++ extension —
seconds, not an obstacle; the devkit image already has `gcc`/`g++`/`cmake`).

**Before benchmarking**: check `nvpmodel -q` — Jetson boards often default to a reduced power mode (Orin
defaulted to `MODE_30W`, not `MAXN`, after a fresh flash). Set with `sudo nvpmodel -m 0` (MAXN is
usually mode 0 — confirm via `/etc/nvpmodel.conf`), then `sudo jetson_clocks` to lock clocks at that
mode's ceiling (Jetson boards can still clock down within a mode when idle otherwise).

**Deploying to a new board** (no script yet — this is the manual recipe, worth scripting if it becomes
routine): rsync `.project-root`, `src/`, `context/`, `inference_edge/`,
`results/fpga/<chosen_model>/manifest.json`, the target checkpoint's `.hydra/config.yaml` +
`checkpoints/last.ckpt`, and the tile(s) to compress, into one directory on the board (e.g.
`~/ddc_edge/repo/`), matching this repo's relative layout so `rootutils`/hydra's config resolution
still works. Then `pip install -e inference_edge/` inside the venv.

## Usage

```bash
ddc-edge compress --model-dir <path with manifest.json> --tile <tile.npy or .cos> --out out.ddc \
    [--overlap 2] [--max-rows N] [--power] [--json results.json]
ddc-edge verify --model-dir <path with manifest.json> --ddc <compressed.ddc> --gt <merlin_gt.npy> \
    [--sample 200] [--json verify.json]
```

`verify` decodes a `.ddc` and scores it (PSNR/SSIM, `src/utils/metrics.py`, AMP_LIN_99 basis) against a
`[n,256,256]` MERLIN ground-truth patch stack matching the `.ddc`'s grid — in practice, run `compress`
at `--overlap 0` for the verification pass specifically, so patch index `k` lines up 1:1 with a
pre-computed GT stack (e.g. `data/cache/symstudy/*_full_merlin_gt.npy`, built for the E1 study) without
needing to regenerate GT or remap indices for an overlapping grid. The `overlap=2` production run is a
separate invocation from the one you verify.

### Decode must run on the same GPU that compressed (hyperprior archs)

**`GaussianConditional`'s entropy decode needs every element's `scale` in the exact same
`gc_scale_table` bucket it was encoded with — and `h_s`'s FP32 output is not bit-reproducible across
GPU architectures.** A small numeric drift between two different GPUs is enough to push some elements
across a bucket boundary, desyncing the entropy decode for those specific elements: not imprecise, but
*wrong* (observed: a decoded value of 5,136,333 where O(1) is normal, propagating to `inf` after
denormalization). Same mechanism `docs/onboard_pipeline.md`'s "on-ground SHyp decoder" note already
flagged for a different context (FPGA vs. host); the cross-GPU-architecture version bit us decoding an
Orin-compressed `.ddc` on a desktop GPU. `decompress_record` raises `ValueError` on any non-finite
reconstruction rather than returning it — if you see that error, you're decoding on the wrong device;
re-run `verify` on the same machine that ran `compress`. FP32 desync-prone entropy coding is not unique
to cross-GPU decode either — it's the same class of problem as the FPGA's own INT8 board-vs-host
decode story, just triggered by a different precision boundary.

## Status (Orin, `SHyp-relu_s0_L20_pt`)

Production run (`MODE_30W`, overlap=2, full scene) lands within ~1% of the FPGA's own `seq`-mode
throughput, power, and energy/patch — see `docs/onboard_pipeline.md` §12 for the table. Quality verified
against MERLIN GT (overlap=0, full 7,482-patch coverage, decoded on-device per the note above): **PSNR
28.05 ± 5.24 dB, SSIM 0.8144 ± 0.1055** — consistent with the neighboring FP/ResSHyp λ=20 reference
numbers. Visual crop comparison (`scripts/evaluation/jetson_vs_fpga_crop.py`) confirms the same
qualitatively — see `results/benchmark_jetson/orin/jetson_vs_fpga_vs_merlin_crop.png`.

**Not yet done**: re-run at `MAXN` (current numbers are `MODE_30W`, not peak); Thor; a 4-arch ×
power-mode sweep; an independent audit of what exactly is timed and whether the power sampling is
methodologically sound (proposed, not yet run — see `docs/onboard_pipeline.md` §12).
