# /deploy-board — Deploy a model to the ZCU102 board and build the C++ inference binary

## What this skill does

1. Selects a compiled model from `results/fpga/compiled_models/`
2. Transfers the model to the ZCU102 board via `deploy.py` (compile + transfer by default; inference and fetch skipped)
3. Pushes updated C++ sources and rebuilds the board binary (`build_cpp/inference_hybrid`)
4. **(compare mode, optional)** Runs both C++ and Python inference with `--compare-out`, fetches results, runs `compare_py_cpp.py` and saves the log

---

## Input syntax

```
/deploy-board [model-filter] [--skip-compile] [--compare [N] [description]]
```

- **model-filter** — any substring(s) of a model name, e.g. `ResSHyp L1000`, `FP s0 L100`, `L1000`. Used to narrow down `compiled_models/`.
- **--skip-compile** — pass `--skip-compile` to `deploy.py` (skip Vitis-AI Docker compile; model must already be compiled)
- **--compare** — enable comparison mode (see Step 4)
  - optional **N**: subset patch count (default: 100)
  - optional **description**: short snake_case label for output dirs, e.g. `block_layout_test` (asked interactively if absent)

---

## Step 1 — Resolve the model

Run:
```bash
ls results/fpga/compiled_models/
```

Filter the list by any tokens the user gave as the model-filter argument (case-insensitive substring match on each token). Examples:
- `ResSHyp L1000` → keep names containing both `ResSHyp` and `L1000`
- `FP s0 L100` → keep names containing `FP`, `s0`, and `L100`

**If exactly one model matches**: proceed with it. State which model was selected.

**If multiple models match**: print the filtered list (numbered) and ask the user to pick one by number or by typing a more specific filter. Do not proceed until a single model is unambiguously identified.

**If no model-filter was given at all**: print the full list of compiled models and ask the user to specify one.

**If the model does not exist** in `compiled_models/`: tell the user and stop. Do not attempt to compile it from scratch (that requires editing `RUN_DIR` in `deploy.py`, which is a separate task).

---

## Step 2 — Transfer model to board

Construct and run the `deploy.py` command. Always include:
- `--model-name <resolved-model-name>` — selects the model
- `--skip-infer` — skip FPGA inference (we use the C++ binary directly)
- `--skip-fetch` — skip result fetch

Add `--skip-compile` only if:
- The user passed `--skip-compile` in the skill args, OR
- The model already exists in `compiled_models/` AND the user explicitly asked to skip compile

By default (no `--skip-compile` flag from user): do NOT skip compile.

```bash
python scripts/fpga/deploy.py \
    --model-name <model> \
    [--skip-compile] \
    --skip-infer \
    --skip-fetch
```

Wait for this to complete before proceeding.

---

## Step 3 — Push C++ sources and build on board

Always do this step regardless of mode.

### 3a. Push sources (from host to board)

```bash
scp -r inference_cpp/src/ ZCU102:/home/root/SAR_DDC/inference_cpp/src/
```

This copies the full `src/` tree including `rans/` subdirectory.

### 3b. Build on board

```bash
ssh ZCU102 "cd /home/root/SAR_DDC/build_cpp && make -j4"
```

If the build fails, print the error output and stop. Do not proceed to Step 4.

The resulting binary is at `ZCU102:/home/root/SAR_DDC/build_cpp/inference_hybrid`.
It is **not** copied to `active_model/` — it is invoked directly as `build_cpp/inference_hybrid`.

---

## Step 4 — Comparison mode (only if --compare was given)

### 4a. Resolve output labels

Extract from the skill args:
- **N** (subset): integer after `--compare`, default `100`
- **description**: snake_case label, e.g. `block_layout_test`. If not given in args, ask the user: *"Short description for this comparison run (e.g. block_layout_test)?"*

Derive the model architecture prefix from the model name (everything before the first `-`):
- `ResSHyp-relu_s0_L1000_pt` → `ResSHyp`
- `FP-relu_s0_L100_pt` → `FP`
- `ResFP-relu_s0_L200_pt` → `ResFP`

Build output directory names:
```
cpp_dir = /tmp/cpp_compare_<arch>_<N>_<description>
py_dir  = /tmp/py_compare_<arch>_<N>_<description>
```

Example: `arch=ResSHyp`, `N=100`, `description=block_layout_test`
→ `/tmp/cpp_compare_ResSHyp_100_block_layout_test`
→ `/tmp/py_compare_ResSHyp_100_block_layout_test`

### 4b. Run C++ inference (on board)

```bash
ssh ZCU102 "cd /home/root/SAR_DDC && \
    build_cpp/inference_hybrid \
    --xmodel active_model/*.xmodel \
    --params active_model/entropy_params \
    --data data/test_sub500_seed42.npy \
    --subset <N> \
    --compare-out <cpp_dir>"
```

### 4c. Run Python reference (on board)

```bash
ssh ZCU102 "export PYTHONPATH=\$PYTHONPATH:/home/root/SAR_DDC && \
    cd /home/root/SAR_DDC/active_model && \
    python3 inference_hybrid.py \
    --xmodel ./*.xmodel \
    --data ../data/test_sub500_seed42.npy \
    --subset <N> \
    --compare-out <py_dir>"
```

### 4d. Fetch results to host

```bash
scp -r ZCU102:<cpp_dir> tmp/
scp -r ZCU102:<py_dir>  tmp/
```

This creates `tmp/cpp_compare_<arch>_<N>_<description>/` and `tmp/py_compare_<arch>_<N>_<description>/` on the host.

### 4e. Run comparison script and save log

```bash
python scripts/fpga/compare_py_cpp.py \
    --py  tmp/py_compare_<arch>_<N>_<description> \
    --cpp tmp/cpp_compare_<arch>_<N>_<description> \
    | tee tmp/compare_<arch>_<N>_<description>.log
```

Report the key summary lines to the user (BPP pass/fail, Pixel pass/fail, PSNR vs MERLIN section).

---

## Paths reference

| Item | Path |
| --- | --- |
| Compiled models (host) | `results/fpga/compiled_models/<model-name>/` |
| Active model (host) | `results/fpga/active_model/` (symlink) |
| Board project root | `ZCU102:/home/root/SAR_DDC/` |
| Board active model | `ZCU102:/home/root/SAR_DDC/active_model/` |
| Board C++ sources | `ZCU102:/home/root/SAR_DDC/inference_cpp/src/` |
| Board build dir | `ZCU102:/home/root/SAR_DDC/build_cpp/` |
| Board C++ binary | `ZCU102:/home/root/SAR_DDC/build_cpp/inference_hybrid` |
| Board test data | `ZCU102:/home/root/SAR_DDC/data/test_sub500_seed42.npy` |
| Host compare logs | `tmp/compare_<arch>_<N>_<description>.log` |
