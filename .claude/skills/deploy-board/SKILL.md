# /deploy-board — Deploy a model to the ZCU102 board and build the C++ inference binary

## What this skill does

1. Selects a compiled model from `results/fpga/compiled_models/`
2. Transfers the model to the ZCU102 board via `deploy.py` (compile + transfer by default; inference and fetch skipped)
3. Pushes updated C++ sources and rebuilds the board binary (`build_cpp/inference_hybrid`)

---

## Input syntax

```
/deploy-board [model-filter] [--skip-compile]
```

- **model-filter** — any substring(s) of a model name, e.g. `ResSHyp L1000`, `FP s0 L100`, `L1000`. Used to narrow down `compiled_models/`.
- **--skip-compile** — pass `--skip-compile` to `deploy.py` (skip Vitis-AI Docker compile; model must already be compiled)

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

Since the skill always selects an already-compiled model by name, always add `--skip-compile`.
(Running compile without `--skip-compile` would compile the hardcoded `RUN_DIR` model in `deploy.py`,
not the selected model — which is wrong and wastes ~1 min of compile time.)

```bash
python scripts/fpga/deploy/deploy.py \
    --model-name <model> \
    [--skip-compile] \
    --skip-infer \
    --skip-fetch
```

Wait for this to complete before proceeding.

---

## Step 3 — Push C++ sources and build on board

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
