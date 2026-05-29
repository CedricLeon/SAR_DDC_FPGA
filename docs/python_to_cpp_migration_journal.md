# Python → C++ Inference & Benchmark Migration — Journal

> **What this document is.** The record of *why* and *how* the on-board FPGA inference pipeline (and
> its benchmarking) was ported from Python to C++: the deployment/packaging changes, the before/after
> numbers, the new benchmark capabilities, and the nastier bugs worth remembering. It absorbs and
> replaces the former `cpp_inference_design.md`.
>
> **Scope.** Inference (`inference_hybrid`) **and** benchmarking (`benchmark_hardware`).
> This is a *transition* record — it does **not** describe the current C++ implementation in detail
> (see `FPGA_inference.md`) nor track forward-looking C++ work (see `FPGA_inference.md` and
> `benchmark_hardware_design.md`).
>
> **Companion docs.** `FPGA_inference.md` = how the C++ pipeline works *now* (reference + future work).
> `benchmark_hardware_design.md` = benchmark development journal + remaining milestones.
> `performance_benchmark_implementation.md` = ZCU102 hardware + benchmark methodology.

## 1. Motivation — why port to C++

The DPU (DPUCZDX8G **B4096 @ 300 MHz**, 3 cores) runs the four NN subgraphs as INT8, and there is no
straightforward way to accelerate them further within this FPGA design. The bottleneck is therefore
on the **CPU side** (ARM A53 ×4): entropy coding, normalization, tensor reshaping.

The Python pipeline could not hide that CPU cost behind DPU work because of the **GIL**: only one
thread executes Python bytecode at a time, so CPU entropy coding (EntropyBottleneck / GaussianConditional)
held the GIL while the DPU sat idle. `threading.Thread` does not give real overlap; `multiprocessing`
adds Inter-Process-Communication overhead. C++ removes the GIL entirely and exposes the VART
async API (`execute_async` + `wait`) for true `std::thread` parallelism — the prerequisite for any
DPU‖CPU pipelining study.

Secondary win, discovered during the port: a large part of the Python entropy cost was **numpy
index-building / `.tolist()` marshalling**, not the rANS coding itself. Native C++ collapses that
overhead (see §3).

## 2. What changed — deployment, build & packaging

This journal does **not** re-describe the pipeline; for the current C++ implementation (architecture,
modules, run reference) see `FPGA_inference.md`. What the *migration* changed:

- **Interpreted → compiled.** Python scripts were `scp`'d to the board and run with `python3`. The
  C++ binary is **built natively on the ZCU102** (CMake + `make -j4`, no cross-compile) and run
  directly. `deploy.py --rebuild-cpp` / `batch_deploy.py` push sources and rebuild on the board.
- **`ans.so` cpython import is gone.** Python imported `ans.cpython-*.so`, a **pybind11 extension**
  reachable only through the Python C API — unlinkable from pure C++. Replaced by a **pybind11-free
  rANS fork** (`inference_cpp/src/rans/`, `py::bytes` → `std::vector<uint8_t>`), bit-compatible with
  the Python encoder (verified by a roundtrip unit test). The old `deploy_cpp_entropy_coder/`
  cross-compile toolchain for that `.so` is retired.
- **Entropy-params packaging.** `deploy.py` exports the entropy CDF tables as individual `.npy`
  files (`eb_*`, `gc_*`) into `active_model/entropy_params/`, which the C++ binary loads directly.
  The old single `entropy_params.npz` archive is dropped — it was read only by the Python pipeline.
- **No Python on the board.** Inference scripts are no longer copied into each compiled-model dir; a
  single C++ binary serves all models, and `metrics.json` (FPGA quality evaluation) is now produced
  by the C++ `inference_runner` rather than Python.

Before / after invocation:

```bash
# before — Python, per-model script on the board
python3 inference_hybrid.py --xmodel ... --data ...

# after — C++, built once on the board
build_cpp/inference_hybrid --xmodel active_model/*.xmodel \
    --params active_model/entropy_params --data data/test_sub500_seed42.npy --subset 100
```

## 3. Performance — before / after

**Setup.** 4 architectures (FP, SH = ScaleHyperprior, ResFP, ResSH; λ=1000, seed 0). Python = `benchmark_fpga.py`; C++ = `benchmark_hardware`. warmup 5, iters 50, `--power`, ZCU102. **S0** = sequential; **S1** = channel-parallel (g_a/g_s real‖imag on 2 DPU cores).

> *Caveat:* Python repeats the first patch per iter; C++ cycles a 20-patch subset — latency-per-patch comparable, entropy-stage timing may differ slightly (data-dependent bitstream length).

Source notebook + JSONs: `notebooks/cpp_migration_legacy.ipynb` + `notebooks/cpp_migration_data/`
(tagged `cpp-migration-notebook` before notebook deletion).

### 3.1 Latency & throughput (`compress` scenario)

![latency & throughput, compress](images/migration_latency_throughput_compress.png)

C++ is **2.3–4.8×** faster (compress, total latency); throughput improves **1.4–4.7×**. The win is
biggest for the small models (FP/SH) because their Python pipeline is **entropy-dominated**; it is
smaller for the DPU-heavy residual models. `full` scenario shows the same pattern
(`images/migration_latency_throughput_full.png`).

### 3.2 Per-stage breakdown (`full` scenario)

![per-stage, full / S1](images/migration_stages_full_s1.png)

The DPU stages (`g_a`/`g_s`, blues) are the **same silicon** → essentially unchanged Python↔C++.
The **entropy stages** (`eb_*` reds, `gc_*` oranges) collapse from tens/hundreds of ms to a few ms.

Per-stage entropy speedups (full/s0):

| Stage | FP / ResFP | SH / ResSH |
| --- | --- | --- |
| eb_compress | ~7.3× | ~40–43× |
| eb_decompress | ~10.7× | ~44–46× |
| gc_compress | | ~9.4× |
| gc_decompress | | ~13× |

**Why so large:** the speedup is *not* faster rANS — it is the elimination of Python numpy
index-building / `.tolist()` marshalling around each `ans` call. DPU stages match within ~1.03×
(wrapper overhead only).

**Bottleneck shift (key finding from M1).** For ResSHyp the C++ pipeline is now **78% DPU**, 12%
entropy, 10% normalize+denorm — the bottleneck moved *back to the DPU*. For FP the DPU is only ~37%;
normalize+denorm (scalar `std::log`/`exp`, ~18.7 ms) and entropy dominate — so FP benefits more from
NEON vectorization than from scheduling.

### 3.3 Energy per patch

![energy, compress / S0](images/migration_energy_compress_s0.png)

Active MPSoC power is **nearly identical** Python↔C++ (~11.5 W) — DPU hardware draw is
language-independent. Energy/patch drops **~2–3×** purely from the latency reduction (≈2.1× compress,
≈2.2× full for ResSHyp).

## 4. Correctness validation (high level)

The C++ port is the production implementation; during the port its output was compared **patch-by-patch
against the Python pipeline** (since removed) to catch bugs. The goal was **bug detection**, not
bit-exact replication of a float64 Python reference.

- **Validation gate:** per patch, `|PSNR_cpp_vs_MERLIN − PSNR_py_vs_MERLIN| < 0.1 dB`.
- **Result (ResSHyp-relu_s0_L1000, 100 patches):** 100/100 pass. EB z-stream byte-identical on all
  100; GC y-stream byte-identical on 91, within ≤4 bytes on 9 (half-integer rounding boundaries,
  not a bug — see §6). Mean Δ(PSNR vs MERLIN) = **+0.083 dB** (C++ marginally *better* than Python).
- FP path also validated.

## 5. Benchmarking — what the migration added

The Python benchmark (`benchmark_fpga.py`) measured five **scenarios** — `full`, `compress`,
`decompress`, `nn_only`, `entropy_only` — each optionally with `--no-parallel` (sequential) vs the
default thread-parallel `g_a`/`g_s`, plus INA226/PMBus power read in Python. The C++ `benchmark_hardware`
reproduces and extends this:

**Reproduced (same intent, new naming).**

- Parallelism is now an explicit **config** rather than a boolean flag: **`s0`** (sequential, =
  `--no-parallel`) and **`s1`** (channel-parallel — exactly 2 `g_a` + 2 `g_s` runners). `compress`
  and `full` remain **scenarios** (`decompress` dropped as a standalone — it lives inside `full`).
- Per-stage latency breakdown via a native `StageTimer` (mirrors the Python `StepTimer`), so the
  Python↔C++ comparison in §3 is stage-aligned.

**Added / changed.**

- **`nn_only` / `entropy_only` repurposed as data-parallel ceilings.** In Python they isolated a
  component's latency; in C++ they are **fan-out sweeps** — `nn_only --dpu-cores {1,2,3}` measures the
  DPU throughput ceiling, `entropy_only --entropy-threads {1,2,3,4}` the CPU/entropy ceiling. This is
  the headline addition: the benchmark now quantifies *how far the hardware could go*, not just what
  one pipeline does.
- **Native C++ power sampler** (`power_sampler.{hpp,cpp}`): INA226 via sysfs (~100 Hz) + PMBus via
  I²C (~25 Hz) in background threads, `--power`-gated, with an idle-baseline window — replacing the
  Python sysfs/ioctl reads. Group aggregates (PL/PS/MPSoC/DPU_fabric/…) match the Python reference.
- **Roofline / static metrics** per subgraph (workload OPs, INT8 footprint, peak fps) collected via
  `collect_roofline.py` → `_xmodel_info.json` — no Python equivalent.
- **Throughput-first JSON schema** (`config`, `dpu_cores`, `entropy_threads`, `throughput_fps`
  headline) so results feed the analysis notebooks.

**Methodological differences to keep in mind** (matter when comparing Python and C++ numbers):
the C++ benchmark cycles a **20-patch subset** while Python repeated the first patch; timing is over
warmup + N iters in both; power is the native sampler vs Python reads. M3 (S0/S1 + ceilings) is
board-verified; the pipelined configs that *use* these ceilings are future work (see
`benchmark_hardware_design.md`).

## 6. Bugs found & fixed (archive — the nasty / recurring ones)

Kept because they are subtle, cost real debugging time, or can recur on new hardware/models.

| Bug | Symptom | Root cause | Fix / lesson |
| --- | --- | --- | --- |
| **Core-order footgun** | S1 / parallel runs show *no* speedup | Two concurrent runners landed on the **same** DPU core (VART round-robin at creation time) | Creation order is the only lever; create concurrent-pair runners consecutively & first. `init_s1` is **model-aware** (SHyp vs FP differ in primary-runner count → different next free core). Verify by timing. |
| **vai_c_xir div-by-0** | Compile crash: `Division by 0 or minus in div_ceil, 3 / 0`, no location | Triggered inside the EntropyBottleneck subgraph during `vai_c_xir`; cryptic, no stack | Historical Vitis-AI compiler gotcha — if it recurs, suspect entropy/hyperprior shapes; see `Vitis-AI_journey.md`. |
| **float64→float32 reinterpret (Bug 6)** | Hamburg tile = pure noise, 256×256 grid; metrics `null`; ALL 240 models | `sym_Noisy.npy` saved float64; `NpyArray::as_float32()` was a raw `reinterpret_cast` → garbage + NaN | `to_float32_vec()` does dtype-aware cast. **Lesson: always check NPY dtype before casting** (test set was float32 so it hid the bug). |
| **Banker's rounding / FPU mode (Bug 3/3a/4)** | GC y-stream off by a few bytes / ~1% pixels differ on some patches | `std::round` (half-away-from-zero) ≠ numpy `round` (half-to-even) at half-integer DPU dequant boundaries; `std::rint` depends on FPU mode VART may change | Explicit `round_half_to_even()` (FPU-mode-independent). The residual ≤4-byte / pixel diffs are **not a bug** — C++ is marginally closer to MERLIN. |
| **y channel layout (Bug 2/5)** | SHyp: ~+1500 bytes/patch; FP: ≤8 bytes | `y` built as `[real_block│imag_block]` instead of **NHWC interleaved** (`C=2·C_MAIN`) → wrong channel↔CDF/scale pairing | Build `y` interleaved; deinterleave `y_hat` by channel. Both paths now consistent. |
| **INT8 quantize rounding (Bug 1)** | FP model ~6% pixel ratio error | C++ used `std::round`; Python `.astype(np.int8)` **truncates toward zero** (+0.5 LSB bias) | `static_cast<int8_t>(v)` with saturating clamp. |

DPU model constraints that caused early failures (now codified in `CLAUDE.md`): no `GDN` (use ReLU),
no `LowerBoundFunction` (use clamp), `ConvTranspose2d` must have `output_padding=0`.

## 7. Doc maintenance note

This journal supersedes `cpp_inference_design.md` (deleted). Current-state C++ pipeline description
lives in `FPGA_inference.md` (rewritten as the C++ reference). Forward-looking work is tracked in
`FPGA_inference.md` (inference: NEON normalize/denorm, full-image streaming) and
`benchmark_hardware_design.md` (benchmark: M4/M5 pipelining, P3, unified GPU/CPU/FPGA runner) — kept
in those docs rather than duplicated here so the journal stays a fixed transition record.
