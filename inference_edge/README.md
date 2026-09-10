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

### Optional throughput/precision knobs

The default (`--batch-size 1 --precision fp32`, no `--fuse-reim`) is exactly the baseline above. Three
opt-in knobs enable a throughput/precision sensitivity study on top of it, without changing that default:

- `--batch-size N` — N patches per forward pass (GPU parallelism). Counts patches; real and imag stay
  two separate `g_a` calls unless `--fuse-reim` folds them into one `[2N,…]` call (a separable extra).
- `--precision {fp32,fp16,bf16}` — `torch.autocast` on the conv subgraphs only; the entropy coder stays
  FP32. fp16/bf16 are quality-neutral (≈0 dB) and leave bpp unchanged.
- `--warmup-rows N` — a discarded warmup pass that absorbs the one-time CC-8.7 JIT (see Known quirks).

**These are compress/throughput levers only for hyperprior archs:** a batched (or cross-precision)
hyperprior `.ddc` is *not decodable* — see "Decode must match the encode" below. So anything you decode
or `verify` must use `--batch-size 1` at the matching precision. Full analysis → `docs/onboard_pipeline.md` §12.

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

Switching to/from a mode that changes the online-CPU-core count (`MODE_30W`/`MODE_15W` vs. `MAXN`/
`MODE_50W` — see `/etc/nvpmodel.conf`) triggers a **reboot** on this L4T's nvpmodel build (1.1.4):
plain `nvpmodel -m <id>` will prompt `DO YOU WANT TO REBOOT NOW?` interactively (fine in a live shell —
just answer `yes`); over SSH non-interactively it hangs/errors instead. Use `nvpmodel -m <id> --force`
to auto-reboot without the prompt (board back in ~50s) — see
`scripts/evaluation/jetson_power_arch_sweep.py` for a script that already handles this.

**Deploying to a new board** (no script yet — this is the manual recipe, worth scripting if it becomes
routine): rsync `.project-root`, `src/`, `context/`, `inference_edge/`,
`results/fpga/<chosen_model>/manifest.json`, the target checkpoint's `.hydra/config.yaml` +
`checkpoints/last.ckpt`, and the tile(s) to compress, into one directory on the board (e.g.
`~/ddc_edge/repo/`), matching this repo's relative layout so `rootutils`/hydra's config resolution
still works. Then `pip install -e inference_edge/` inside the venv.

## Known quirks

- **First patch pays a one-time ~132ms JIT-compile spike.** Not a gradual warmup — patch 0 measured
  132.4ms, patch 1 onward flat at ~4.55ms (a clean 29× one-shot cost, confirmed by per-patch logging).
  Root cause: Orin's compute capability (8.7) isn't in this torch build's precompiled kernel list
  (`torch.cuda` prints this warning every run — *"No published PyTorch CUDA builds for release
  2.13.0+cu132 support this GPU"*), so CUDA kernels PTX-JIT-compile on first use. Negligible over a
  full-scene run (~130ms in ~280s), but matters for short/`--max-rows` runs and for comparing against
  any other short smoke test.

## Usage

```bash
ddc-edge compress --model-dir <path with manifest.json> --tile <tile.npy or .cos> --out out.ddc \
    [--overlap 2] [--max-rows N] [--batch-size N] [--precision fp32|fp16|bf16] [--fuse-reim] \
    [--warmup-rows N] [--power] [--json results.json]
ddc-edge verify --model-dir <path with manifest.json> --ddc <compressed.ddc> --gt <merlin_gt.npy> \
    [--sample 200] [--precision fp32|fp16|bf16] [--json verify.json]
```

`verify` decodes a `.ddc` and scores it (PSNR/SSIM, `src/utils/metrics.py`, AMP_LIN_99 basis) against a
`[n,256,256]` MERLIN ground-truth patch stack matching the `.ddc`'s grid — in practice, run `compress`
at `--overlap 0` for the verification pass specifically, so patch index `k` lines up 1:1 with a
pre-computed GT stack (e.g. `data/cache/symstudy/*_full_merlin_gt.npy`, built for the E1 study) without
needing to regenerate GT or remap indices for an overlapping grid. The `overlap=2` production run is a
separate invocation from the one you verify.

### Decode must match the encode — same device, precision, and batch 1 (hyperprior archs)

**`GaussianConditional`'s entropy decode needs every element's `scale` in the exact same
`gc_scale_table` bucket it was encoded with — and `h_s`'s FP32 output is not bit-reproducible across
GPU architectures, across `--precision`, or between a batched compress and the per-record decode.** A small numeric drift between two different GPUs is enough to push some elements
across a bucket boundary, desyncing the entropy decode for those specific elements: not imprecise, but
*wrong* (observed: a decoded value of 5,136,333 where O(1) is normal, propagating to `inf` after
denormalization). Same mechanism `docs/onboard_pipeline.md`'s "on-ground SHyp decoder" note already
flagged for a different context (FPGA vs. host); the cross-GPU-architecture version bit us decoding an
Orin-compressed `.ddc` on a desktop GPU. `decompress_record` raises `ValueError` on any non-finite
reconstruction rather than returning it — if you see that error, you're decoding on the wrong device;
re-run `verify` on the same machine that ran `compress`. FP32 desync-prone entropy coding is not unique
to cross-GPU decode either — it's the same class of problem as the FPGA's own INT8 board-vs-host
decode story, just triggered by a different precision boundary.

## Thor — blocked, deferred (2026-08-24)

The "should carry to Thor largely unchanged" claim earlier in this doc does not hold today. Three
independent problems, found by the audit, in the order you'd hit them:

1. **No internet access** on Thor's network segment (`sche_ao@10.0.0.5`) — confirmed at the TCP level
   (raw connect to `8.8.8.8:53` / `1.1.1.1:443` both fail, not just DNS), despite a configured default
   route and a real resolver. The `pip install ... --index-url ...` recipe above cannot run as written;
   needs an offline wheelhouse built elsewhere and rsynced over.
2. **`python3 -m venv` fails outright** — `ensurepip`/`python3.12-venv` isn't installed and there's no
   `apt` access to add it (see #1). Workaround: `venv --without-pip` + manually bootstrap a pip wheel
   (itself needing to come from somewhere with internet, because of #1 again).
3. **The real blocker: Orin's exact torch build does not import on Thor.** `torch==2.13.0+cu132`'s
   pinned `nvidia-nccl-cu13==2.29.7` has no published aarch64 wheel (PyPI only has `0.0.0a0` and
   `2.27.7` for aarch64). Substituting `2.27.7` resolves the dependency but then `import torch` fails at
   the ELF loader level — `undefined symbol: ncclCommResume`, a real ABI gap (that symbol only exists in
   NCCL ≥2.28), not a version-pinning mistake. `torch==2.9.1+cu130` looks compatible by wheel metadata
   (its NCCL pin does have an aarch64 wheel) but **was not empirically confirmed** — worth trying first
   when this is picked back up, but don't assume it works without testing.

**Consequence for later:** any Orin-vs-Thor comparison needs a torch version that actually works on
*both* boards, or it carries a silent version confound alongside the hardware difference. Decide the
pin before re-attempting, don't default back to Orin's.

Smaller, already-handled: Thor's `tegrastats` output format differs from Orin's (2-value `cur/avg`
fields, not 3; different rail names, including a `VIN` rail that genuinely is a superset of the others,
unlike any of Orin's rails) — `power.py`'s parser and rail policy now handle this generically (see
`_RAIL_POLICY` in that file), but Thor's specific policy entry is informed by only one live capture and
is **not independently verified** the way Orin's is — don't trust it blindly when Thor work resumes.

## Status

Full 4-arch × 4-power-mode sweep done and board-verified (all 4 λ=20/relu archs × `MAXN`/`MODE_50W`/
`MODE_30W`/`MODE_15W`, full scene, `overlap=2`) — results table, patterns, and remaining TODOs (Thor,
per-arch quality sweep) → `docs/onboard_pipeline.md` §12. Quality (PSNR/SSIM vs MERLIN GT) verified for
`SHyp-relu_s0_L20_pt` only so far — same section.
