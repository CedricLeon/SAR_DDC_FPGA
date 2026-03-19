#!/usr/bin/env python3
"""End-to-end FPGA deployment orchestrator.

Runs from the DDC_FPGA project root with the SAR_DDC conda environment:

    python scripts/fpga/deploy.py [options]

Phases:
    0 — Container health  : ensure vai_container is running and GPU-healthy
    1 — Compile           : quantize, compile, export entropy params, organize output
    2 — Transfer          : scp compiled model to FPGA
    3 — Infer             : run inference_hybrid.py on FPGA (output streamed)
    4 — Fetch             : scp results back to host

Any phase can be skipped with --skip-<phase>. Phases 2-4 always read the active model
name from results/fpga/active_model/manifest.json, so they also work standalone after
a previous compile (--skip-compile).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from omegaconf import OmegaConf

# ============================================================
# - USER CONFIGURATION — edit RUN_DIR for each deployment
# ============================================================

# RUN_DIR = "DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-07_00-51-42/0" # ResSHyp-relu_s0_L1_pt
# RUN_DIR = "DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-07_00-51-42/1" # ResSHyp-relu_s0_L5_pt
# RUN_DIR = "DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-07_00-51-42/4" # ResSHyp-relu_s0_L100_pt
RUN_DIR = "DDC_FPGA/logs/train/sar_ddc/hyperprior/multiruns/2026-02-06_14-57-22/0"  # ResSHyp-relu_s0_L1000_pt


# ============================================================
# CONSTANTS
# ============================================================

VAI_IMAGE = "xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde"
CONTAINER_NAME = "vai_container"

# Output prefix produced by vai_c_xir (-n flag)
DPU_WRAPPER_NAME = "ResidualScaleHyperpriorDPUWrapper"

ARCH_JSON_LOOKUP = {
    "ZCU102": "DDC_FPGA/scripts/fpga/DPU_archs/ZCU102_DPUCZDX8G_ISA1_B4096_arch.json",
    "Leopard": "DDC_FPGA/scripts/fpga/DPU_archs/KP-Labs_Leopard_DPUCZDX8G_ISA1_B1024_arch.json",
}
TARGET_LOOKUP = {
    "ZCU102": "DPUCZDX8G_ISA1_B4096",
    "Leopard": "DPUCZDX8G_ISA1_B1024",
}

FPGA_HOST = "ZCU102"
FPGA_BASE_DIR = "/home/root/SAR_DDC"
FPGA_DATA_PATH = "../data/test_sub500_seed42.npy"
FPGA_DATASET_SIZE = 500

CALIB_SUBSET_LEN = 200
EVAL_SUBSET_LEN = 200

CONTAINER_PIP_DEPS = "h5py omegaconf compressai torchmetrics"

# Verbosity toggles — set False to silence noisy phases in the terminal.
# Output is ALWAYS written to the log file regardless of this setting.
PHASE1_VERBOSE = False  # Docker / Vitis-AI compile output (very verbose)
PHASE3_VERBOSE = True  # FPGA inference output

# ---- Paths (absolute, resolved relative to this file's location) ----
# deploy.py lives at DDC_FPGA/scripts/fpga/deploy.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../DDC_FPGA/
VITIS_AI_ROOT = PROJECT_ROOT.parent  # .../Vitis-AI/

ACTIVE_MODEL_LINK = PROJECT_ROOT / "results" / "fpga" / "active_model"
START_CONTAINER_SCRIPT = PROJECT_ROOT / "scripts" / "vitis-ai-automation" / "start_container_bg.sh"

# ============================================================
# TEE LOGGING HELPER
# ============================================================

# Matches tqdm progress lines: " 42%|████      | 84/200 ..."
_TQDM_RE = re.compile(r"^\s*(\d+)%\|")


class Tee:
    """Dual-output writer: everything written to stdout also goes to a log file.

    Works transparently with both print() and subprocess output (via run()).
    Set verbose=False to silence terminal output while still writing to the file.
    The verbose property can be toggled mid-context to mute/unmute specific sub-phases.

    Progress bar handling (tqdm):
      - Terminal: uses \\r overwrite so bars look and behave normally.
      - Log file: intermediate progress lines are dropped; only the final 100% line is kept.
    """

    def __init__(self, path: Path, verbose: bool = True):
        self._file = open(path, "w", encoding="utf-8")
        self._orig = sys.stdout
        self._verbose = verbose

    @property
    def verbose(self) -> bool:
        """Property access for verbose."""
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool) -> None:
        """Setter for verbose property."""
        self._verbose = v

    def write(self, s: str) -> None:
        """Write string s to both the log file and the terminal (if verbose).

        Handles tqdm lines specially.
        """
        line = s.rstrip("\n")
        m = _TQDM_RE.match(line)
        if m:
            pct = int(m.group(1))
            is_complete = pct == 100
            # Log: skip intermediate updates; record only the 100% line
            if is_complete:
                self._file.write(line.lstrip() + "\n")
                self._file.flush()
            # Terminal: overwrite current line; newline only when done
            if self._verbose:
                self._orig.write("\r" + line.lstrip())
                if is_complete:
                    self._orig.write("\n")
                self._orig.flush()
        else:
            self._file.write(s)
            self._file.flush()
            if self._verbose:
                self._orig.write(s)
                self._orig.flush()

    def flush(self) -> None:
        """Flush both outputs."""
        self._file.flush()
        if self._verbose:
            self._orig.flush()

    def __enter__(self) -> "Tee":
        """Redirect sys.stdout to this Tee instance."""
        sys.stdout = self
        return self

    def __exit__(self, *_) -> None:
        """Restore original sys.stdout."""
        sys.stdout = self._orig
        self._file.close()


# ============================================================
# HOST-SIDE COMPILE HELPERS
# ============================================================


def _export_entropy_params(ckpt_path: Path, output_path: Path) -> None:
    """Export entropy model parameters to a .npz file for FPGA inference.

    Instantiates ResidualScaleHyperpriorPatched from the checkpoint config, calls .update() to
    populate the CDF/quantile tables, then saves the tables required by entropy_models_inference.py
    on the FPGA.
    """
    # Deferred src imports — only valid in SAR_DDC env
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from src.models.components.res_scale_hyperprior_dpu import (  # noqa: E402
        ResidualScaleHyperpriorPatched,
    )

    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"Saving entropy parameters to: {output_path}")

    # ----- 1. Load Config & Instantiate Model -----
    # We need to instantiate the model to call .update() method which populates the tables
    config_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"Reading config from {config_path}...")
    conf = OmegaConf.load(config_path)
    model_params = conf.model.net
    if "_target_" in model_params:
        model_params.pop("_target_")
    model_params["export_dpu"] = True
    print(f"Instantiating model with: {model_params}")
    model = ResidualScaleHyperpriorPatched(**model_params)

    # ----- 2. Load Weights -----
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = {k.replace("net.", "", 1): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # ----- 3. Force Update -----
    print("Force entropy models update (populate tables)...")
    model.update(force=True)

    # ----- 4. Extract parameters -----
    eb = model.entropy_bottleneck
    eb_medians = eb.quantiles[:, 0, 1].detach().cpu().numpy()
    eb_quantized_cdf = eb._quantized_cdf.detach().cpu().numpy().astype(np.int32)
    eb_offset = eb._offset.detach().cpu().numpy().astype(np.int32)
    eb_cdf_length = eb._cdf_length.detach().cpu().numpy().astype(np.int32)
    if eb_quantized_cdf.size == 0:
        raise ValueError("EntropyBottleneck CDF is empty — update() failed.")

    gc = model.gaussian_conditional
    gc_scale_table = gc.scale_table.detach().cpu().numpy().astype(np.float32)
    gc_quantized_cdf = gc.quantized_cdf.detach().cpu().numpy().astype(np.int32)
    gc_cdf_length = gc.cdf_length.detach().cpu().numpy().astype(np.int32)
    gc_offset = gc.offset.detach().cpu().numpy().astype(np.int32)
    if gc_scale_table.size == 0:
        raise RuntimeError("GaussianConditional scale_table is empty — update() failed.")

    print("Checks:")
    print(f"  EB medians:    {eb_medians.shape}")
    print(f"  EB CDF:        {eb_quantized_cdf.shape}")
    print(f"  GC scale tbl:  {gc_scale_table.shape}")
    print(f"  GC CDF:        {gc_quantized_cdf.shape}")

    np.savez(
        output_path,
        eb_quantized_cdf=eb_quantized_cdf,
        eb_offset=eb_offset,
        eb_cdf_length=eb_cdf_length,
        eb_medians=eb_medians,
        gc_scale_table=gc_scale_table,
        gc_quantized_cdf=gc_quantized_cdf,
        gc_cdf_length=gc_cdf_length,
        gc_offset=gc_offset,
    )
    print(f"Entropy parameters saved to {output_path}.")


def _make_compiled_model_name(cfg: Any) -> str:
    """Derive the FPGA artifact folder name from a Hydra config.

    Convention: ``<model>-<activation>_s<seed>_L<lambda>_pt``
    Intentionally different from the WandB run name in utils.py.
    """

    def _get(obj, key, default=None):
        """Helper to safely get nested config values with a default."""
        return obj.get(key, default) if isinstance(obj, dict) else obj.get(key, default)

    net = cfg.model.net
    target = _get(net, "_target_", "")
    if "ResidualScaleHyperprior" in target:
        model = "ResSHyp"
    elif "Merlin" in target:
        model = "Merlin"
    else:
        model = "UnknownModel"

    activation = _get(net, "activation", "gdn")
    seed = cfg.get("seed", None)
    lmbda = _get(cfg.model.criterion, "lmbda", None)
    # Because only models supported by the FPGA will be compiled we do not need to track no_output_padding.
    return f"{model}-{activation}_s{seed}_L{lmbda}_pt"


def _get_wandb_run_id_from_run_dir(run_dir_host: Path) -> str:
    """Infer the W&B run ID from a Hydra run directory.

    W&B writes a ``wandb/latest-run`` symlink pointing to a directory named
    ``run-{timestamp}-{run_id}``.  We read that symlink and extract the run ID
    (the last ``-``-separated token in the target name).

    Critique of the approach:
    - Pro: deterministic, no network call needed, always present after a W&B run.
    - Con: ``latest-run`` points to the *most recent* W&B run started from that
      directory.  In practice each Hydra output dir has exactly one run, so this
      is a non-issue.
    - Fallback: returns ``""`` when the symlink is absent (logger=null runs) or
      the target name is malformed.
    """
    latest_run_link = run_dir_host / "wandb" / "latest-run"
    if not latest_run_link.is_symlink():
        return ""
    try:
        target_name = Path(os.readlink(latest_run_link)).name  # e.g. run-20260211_213632-kel1uzlo
        run_id = target_name.rsplit("-", 1)[-1]  # 'kel1uzlo'
        return run_id if run_id else ""
    except OSError:
        return ""


def _create_manifest(
    dest_dir: Path, cfg: Any, model_name: str, original_run_dir: str, wandb_run_id: str = ""
) -> None:
    """Write manifest.json and a MODEL_IS_*.txt marker into dest_dir."""
    # 1. Extract Training Timestamp from Run Directory path
    # Expected format: .../YYYY-MM-DD_HH-MM-SS/N or .../YYYY-MM-DD_HH-MM-SS/
    trained_at = "Unknown"
    try:
        for part in reversed(Path(original_run_dir).parts):
            # Simple check for timestamp-like string
            if len(part) == 19 and "_" in part and "-" in part:
                try:
                    datetime.strptime(part, "%Y-%m-%d_%H-%M-%S")
                    trained_at = part
                    break
                except ValueError:
                    continue
    except Exception:
        pass

    # 2. Build Metadata Dictionary
    meta: dict = {
        "model_name": model_name,
        "wandb_run_id": wandb_run_id,  # W&B run ID; empty string if deployed outside batch_deploy
        "compiled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_at": trained_at,
        "original_run_dir": original_run_dir,
        "seed": cfg.get("seed", "Unknown"),
        "lambda": cfg.model.criterion.get("lmbda", "Unknown"),
        "activation": cfg.model.net.get("activation", "Unknown"),
    }
    loss_target = cfg.model.criterion.get("_target_", None)
    if loss_target:
        meta["loss"] = loss_target.split(".")[-1]
    if "no_output_padding" in cfg.model.net:
        meta["no_output_padding"] = cfg.model.net.no_output_padding

    manifest_path = dest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Created manifest at {manifest_path}")
    (dest_dir / f"MODEL_IS_{model_name}.txt").touch()


def _organize_compiled_output(
    compiled_dir_abs: Path, run_dir_host: Path, wandb_run_id: str = ""
) -> None:
    """Generate manifest, move compiled dir to compiled_models/, update active_model symlink."""
    output_root = PROJECT_ROOT / "results" / "fpga"
    compiled_models_dir = output_root / "compiled_models"
    compiled_models_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir_host / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. Necessary for manifest metadata."
        )
    cfg = OmegaConf.load(config_path)
    final_name = _make_compiled_model_name(cfg)
    OmegaConf.save(cfg, compiled_dir_abs / "train_config.yaml")
    _create_manifest(compiled_dir_abs, cfg, final_name, str(run_dir_host), wandb_run_id)

    destination = compiled_models_dir / final_name
    if destination.exists():
        results_dir = destination / "results"
        print(f"Warning: {destination} exists. Overwriting.")
        if results_dir.exists():
            print("FPGA inference results existed and were overwritten.")
        shutil.rmtree(destination)

    print(f"Moving {compiled_dir_abs} -> {destination}")
    shutil.move(str(compiled_dir_abs), str(destination))

    symlink_path = output_root / "active_model"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    os.symlink(Path("compiled_models") / final_name, symlink_path)
    print(f"Updated symlink: {symlink_path} -> compiled_models/{final_name}")
    print(f"Model ready at: {symlink_path}")


# ============================================================
# HELPERS
# ============================================================


def print_header(msg: str) -> None:
    """Print a formatted header for a new phase."""
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def run(cmd: List, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, printing it first, raising CalledProcessError on failure.

    When stdout is patched to a Tee instance, subprocess output is streamed line-by-line through it
    so it reaches both the log file and the terminal.
    """
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")
    if isinstance(sys.stdout, Tee):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return proc  # type: ignore[return-value]
    return subprocess.run(cmd, check=True, **kwargs)


def run_in_container(cmd: str) -> None:
    """Run a shell command inside vai_container with vitis-ai-pytorch env active.

    Every docker exec starts a fresh shell without .bashrc, so we source conda manually and set
    LD_PRELOAD before executing the requested command.
    """
    preamble = (
        "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && "
        "conda activate vitis-ai-pytorch && "
        "export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6:$LD_PRELOAD && "
        "cd /workspace && "
    )
    run(["docker", "exec", CONTAINER_NAME, "bash", "-c", preamble + cmd])


# ============================================================
# PHASE 0: Container lifecycle
# ============================================================


def _is_container_running() -> bool:
    """Check if a container with CONTAINER_NAME is currently running."""
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name=^/{CONTAINER_NAME}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return CONTAINER_NAME in result.stdout


def _is_gpu_healthy() -> bool:
    """Check if the GPU inside the container is healthy by running nvidia-smi."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "nvidia-smi"],
        capture_output=True,
        text=True,
    )
    unhealthy = result.returncode != 0 or "NVML" in result.stdout or "NVML" in result.stderr
    return not unhealthy


def _start_container() -> None:
    """Start the Vitis-AI container in detached mode and install Python dependencies."""
    print("Starting vai_container in detached mode...")
    run([str(START_CONTAINER_SCRIPT)])
    print("Installing container Python dependencies (one-time pip install)...")
    run_in_container(f"pip install --quiet {CONTAINER_PIP_DEPS}")


def ensure_container_healthy() -> None:
    """Ensure that the Vitis-AI container is running and the GPU is healthy.

    If not, restart the container.
    """
    print_header("Phase 0: Container Health")
    if _is_container_running():
        print(f"Container '{CONTAINER_NAME}' is running. Checking GPU health...")
        if _is_gpu_healthy():
            print("GPU healthy. Reusing existing container.")
            return
        print("GPU dead (NVML error). Killing and restarting container...")
        run(["docker", "rm", "-f", CONTAINER_NAME])
    else:
        print(f"Container '{CONTAINER_NAME}' not found.")
    _start_container()
    print("Container ready.")


# ============================================================
# PHASE 1: Compile (all sub-steps)
# ============================================================


def phase_compile(
    run_dir: str,
    arch: str,
    inspect: bool,
    eval_float: bool,
    eval_quant: bool,
    fast_finetune: bool,
    image_graph: bool,
    wandb_run_id: str = "",
) -> None:
    """Compile the model for FPGA deployment, running all sub-Phase 1 steps in sequence."""
    arch_json = ARCH_JSON_LOOKUP[arch]
    target = TARGET_LOOKUP[arch]
    ff = " --fast_finetune" if fast_finetune else ""

    # Paths relative to /workspace (used for container commands)
    model_quant_cmd = f"python DDC_FPGA/scripts/fpga/model_quant.py --run_dir {run_dir}"
    xmodel_int = f"quantize_result/{DPU_WRAPPER_NAME}_int.xmodel"
    compiled_dir_rel = f"{DPU_WRAPPER_NAME}_pt"

    # Absolute host paths (used for host-side commands)
    compiled_dir_abs = VITIS_AI_ROOT / compiled_dir_rel
    run_dir_host = VITIS_AI_ROOT / run_dir

    print_header("Phase 1: Compile")
    print(f"  RUN_DIR : {run_dir}")
    print(f"  Arch    : {arch}  ({arch_json})")
    print(f"  Target  : {target}")

    # 1.1. Inspect (optional)
    if inspect:
        print("\n--- 1.1: Inspect ---")
        run_in_container(f"{model_quant_cmd} --quant_mode float --inspect --target {target}")

    # 1.2. Evaluate float model (optional)
    if eval_float:
        print("\n--- 1.2: Evaluate float model ---")
        run_in_container(f"{model_quant_cmd} --quant_mode float --subset_len {EVAL_SUBSET_LEN}")

    # 1.3. Calibrate (always)
    print("\n--- 1.3: Calibrate ---")
    run_in_container(f"{model_quant_cmd} --quant_mode calib --subset_len {CALIB_SUBSET_LEN}{ff}")

    # 1.4. Evaluate quantized model (optional)
    if eval_quant:
        print("\n--- 1.4: Evaluate quantized model ---")
        run_in_container(f"{model_quant_cmd} --quant_mode test --subset_len {EVAL_SUBSET_LEN}{ff}")

    # 1.5. Export xmodel (always — generates the deployable .xmodel file)
    print("\n--- 1.5: Export xmodel ---")
    run_in_container(
        f"{model_quant_cmd} --quant_mode test --subset_len 1 --batch_size 1 --deploy{ff}"
    )

    # 1.6. Compile xmodel with vai_c_xir (always)
    print("\n--- 1.6: Compile xmodel (vai_c_xir) ---")
    run_in_container(
        f"vai_c_xir"
        f" -x {xmodel_int}"
        f" -a {arch_json}"
        f" -o {compiled_dir_rel}"
        f" -n {DPU_WRAPPER_NAME}_pt"
    )

    # 1.7. Generate SVG graph (optional)
    if image_graph:
        print("\n--- 1.7: Generate SVG graph ---")
        run_in_container(
            f"xdputil xmodel {xmodel_int}" f" -s quantize_result/{DPU_WRAPPER_NAME}_graph.svg"
        )

    # Fix permissions: vai_c_xir runs as root inside the container, so the compiled
    # directory is root-owned. Chmod it so the host user can write into it.
    print("\n--- Fix permissions on compiled dir ---")
    run_in_container(f"chmod -R a+rwX /workspace/{compiled_dir_rel}")

    # 1.8. Export entropy parameters (on host, SAR_DDC env)
    print("\n--- 1.8: Export entropy parameters (host) ---")
    _export_entropy_params(
        ckpt_path=run_dir_host / "checkpoints" / "last.ckpt",
        output_path=compiled_dir_abs / "entropy_params.npz",
    )

    # Copy inference scripts into the compiled directory (self-contained deployment)
    print("\n--- Copying inference scripts into compiled dir ---")
    for script in [
        "inference_hybrid.py",
        "inference_utils.py",
        "entropy_models_inference.py",
    ]:
        src = PROJECT_ROOT / "scripts" / "fpga" / script
        shutil.copy2(src, compiled_dir_abs / script)
        print(f"  Copied {script}")

    # 1.9. Organize output: generate manifest, move to compiled_models/, update symlink
    print("\n--- 1.9: Organize output (host) ---")
    # Auto-detect W&B run ID when it was not passed explicitly (standalone deploy.py use).
    if not wandb_run_id:
        wandb_run_id = _get_wandb_run_id_from_run_dir(run_dir_host)
        if wandb_run_id:
            print(f"  Auto-detected W&B run ID from run dir: {wandb_run_id}")
        else:
            print(
                "  Note: W&B run ID not found (no wandb/latest-run symlink). manifest will have empty wandb_run_id."
            )
    _organize_compiled_output(compiled_dir_abs, run_dir_host, wandb_run_id)


# ============================================================
# MODEL NAME RESOLUTION (shared by phases 2-4)
# ============================================================


def get_model_name() -> str:
    """Read the compiled model name from active_model/manifest.json."""
    manifest_path = ACTIVE_MODEL_LINK / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found.")
        print("  Run phase 1 first, or check that active_model symlink is valid.")
        sys.exit(1)
    return json.loads(manifest_path.read_text())["model_name"]


def _set_active_model_symlink(model_name: str) -> None:
    """Point active_model symlink at an already-compiled model in compiled_models/.

    Called when --skip-compile --model-name <name> is passed (e.g. by batch_deploy.py). Raises
    SystemExit if the target directory does not exist.
    """
    compiled_dir = PROJECT_ROOT / "results" / "fpga" / "compiled_models" / model_name
    if not compiled_dir.exists():
        print(f"ERROR: Compiled model not found: {compiled_dir}")
        sys.exit(1)
    if ACTIVE_MODEL_LINK.exists() or ACTIVE_MODEL_LINK.is_symlink():
        ACTIVE_MODEL_LINK.unlink()
    os.symlink(Path("compiled_models") / model_name, ACTIVE_MODEL_LINK)
    print(f"  active_model -> compiled_models/{model_name}")


# ============================================================
# PHASE 2: Transfer compiled model to FPGA
# ============================================================


def phase_transfer(model_name: str) -> None:
    """Transfer the compiled model from the host to the FPGA using scp."""
    print_header(f"Phase 2: Transfer to FPGA ({FPGA_HOST})")
    print(f"  Model     : {model_name}")
    print(f"  Local src : {ACTIVE_MODEL_LINK}")
    print(f"  Remote dst: {FPGA_HOST}:{FPGA_BASE_DIR}/active_model/")

    # Wipe old model from FPGA to prevent stale artifacts from previous runs.
    # inference_hybrid.py cleans its own results/ dir, but model files (xmodel,
    # entropy_params.npz, etc.) must also be fresh.
    print(f"  Wiping {FPGA_BASE_DIR}/active_model on FPGA...")
    run(["ssh", FPGA_HOST, f"rm -rf {FPGA_BASE_DIR}/active_model"])

    # scp -r follows symlinks on the source, so the actual compiled model directory
    # is transferred, not the symlink itself.
    run(["scp", "-r", str(ACTIVE_MODEL_LINK), f"{FPGA_HOST}:{FPGA_BASE_DIR}/"])
    print("  Transfer complete.")


# ============================================================
# PHASE 3: Run inference on FPGA
# ============================================================


def phase_infer(subset: int) -> None:
    """Run inference on the FPGA by SSHing in and executing inference_hybrid.py with the
    appropriate arguments."""
    print_header("Phase 3: Inference on FPGA")
    # ~/.bashrc is NOT sourced by non-interactive SSH sessions, so PYTHONPATH (which
    # points to the ans.so C++ rANS extension at /home/root/SAR_DDC/) must be set
    # explicitly in the command string.
    infer_cmd = (
        f"export PYTHONPATH=$PYTHONPATH:{FPGA_BASE_DIR} && "
        f"cd {FPGA_BASE_DIR}/active_model && "
        f"python3 inference_hybrid.py"
        f" --xmodel ./*.xmodel"
        f" --data {FPGA_DATA_PATH}"
        f" --subset {subset}"
    )
    print(f"  SSH command: {infer_cmd}")
    # Output streams live to this terminal (no capture).
    run(["ssh", FPGA_HOST, infer_cmd])


# ============================================================
# PHASE 4: Fetch results back to host
# ============================================================


def phase_fetch() -> None:
    """Fetch the inference results from the FPGA back to the host using scp, saving them into the
    active_model directory."""
    print_header("Phase 4: Fetch results from FPGA")
    remote_results = f"{FPGA_HOST}:{FPGA_BASE_DIR}/active_model/results"
    local_dest = str(ACTIVE_MODEL_LINK) + "/"

    # Remove stale local results so scp doesn't nest into an existing results/results/
    local_results = ACTIVE_MODEL_LINK / "results"

    print(f"  Fetching {remote_results} -> {local_dest}")
    run(["scp", "-r", remote_results, local_dest])
    print(f"  Results saved to: {local_results}")


# ============================================================
# ENTRYPOINT
# ============================================================


def main() -> None:
    """Main entry point for the deployment script."""
    # Force line-buffered output so print() interleaves correctly with subprocess
    # output even when stdout is redirected to a file (e.g. &> log.txt).
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target run
    g_run = parser.add_argument_group("target run")
    g_run.add_argument(
        "--run-dir",
        default=RUN_DIR,
        metavar="RUN_DIR",
        help=(
            "Hydra run directory relative to VITIS_AI_ROOT "
            "(default: module-level RUN_DIR constant)."
        ),
    )
    g_run.add_argument(
        "--model-name",
        default=None,
        metavar="NAME",
        help=(
            "Compiled model name (e.g. ResSHyp-relu_s0_L100_pt). "
            "When --skip-compile is set, updates active_model symlink before phases 2-4. "
            "Typically set automatically by batch_deploy.py."
        ),
    )
    g_run.add_argument(
        "--wandb-run-id",
        default="",
        metavar="ID",
        help="W&B run ID to record in manifest.json. Set automatically by batch_deploy.py.",
    )

    # Phase skip flags
    g_skip = parser.add_argument_group("phase skips")
    g_skip.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip Phase 1 (compile). Uses current active_model/manifest.json for name.",
    )
    g_skip.add_argument("--skip-transfer", action="store_true", help="Skip Phase 2 (SCP to FPGA).")
    g_skip.add_argument("--skip-infer", action="store_true", help="Skip Phase 3 (inference).")
    g_skip.add_argument("--skip-fetch", action="store_true", help="Skip Phase 4 (fetch results).")

    # Compile sub-options
    g_compile = parser.add_argument_group("compile options (Phase 1)")
    g_compile.add_argument(
        "--arch",
        default="ZCU102",
        choices=list(ARCH_JSON_LOOKUP.keys()),
        help="Target DPU architecture (default: ZCU102).",
    )
    g_compile.add_argument(
        "--inspect",
        action="store_true",
        help="Run DPU Inspector before calibration (1a).",
    )
    g_compile.add_argument(
        "--eval-float",
        action="store_true",
        help="Evaluate float model before calibration (1b).",
    )
    g_compile.add_argument(
        "--eval-quant",
        action="store_true",
        help="Evaluate quantized model before xmodel export (1d).",
    )
    g_compile.add_argument(
        "--fast-finetune",
        action="store_true",
        help="Enable fast finetuning during calibration.",
    )
    g_compile.add_argument(
        "--image-graph",
        action="store_true",
        help="Generate SVG graph of the compiled xmodel (1g).",
    )

    # Inference options
    g_infer = parser.add_argument_group("inference options (Phase 3)")
    g_infer.add_argument(
        "--subset",
        type=int,
        default=FPGA_DATASET_SIZE,
        help=f"Number of test samples for inference (default: {FPGA_DATASET_SIZE}).",
    )

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"  deploy.py — RUN_DIR: {args.run_dir}")
    print(f"  Arch: {args.arch}  |  Subset: {args.subset}")
    skip_flags = [
        f"compile={'SKIP' if args.skip_compile else 'YES'}",
        f"transfer={'SKIP' if args.skip_transfer else 'YES'}",
        f"infer={'SKIP' if args.skip_infer else 'YES'}",
        f"fetch={'SKIP' if args.skip_fetch else 'YES'}",
    ]
    print(f"  Phases: {', '.join(skip_flags)}")
    print(f"{'#' * 60}")

    # Temporary path for the compile log (moved into the model folder after Phase 1i)
    _compile_log_tmp = VITIS_AI_ROOT / ".deploy_compile_tmp.log"

    # Phase 0 + 1: compile — all output goes to compile.log (and optionally terminal)
    if not args.skip_compile:
        with Tee(_compile_log_tmp, verbose=PHASE1_VERBOSE):
            ensure_container_healthy()
            phase_compile(
                run_dir=args.run_dir,
                arch=args.arch,
                inspect=args.inspect,
                eval_float=args.eval_float,
                eval_quant=args.eval_quant,
                fast_finetune=args.fast_finetune,
                image_graph=args.image_graph,
                wandb_run_id=args.wandb_run_id,
            )

    # If skipping compile with a known model name (e.g. from batch_deploy.py), redirect symlink.
    if args.skip_compile and args.model_name:
        _set_active_model_symlink(args.model_name)

    model_name = get_model_name()
    print(f"\nActive model: {model_name}")

    # Move compile log into the model folder now that we know its final path
    if not args.skip_compile and _compile_log_tmp.exists():
        compile_log_dest = ACTIVE_MODEL_LINK / "compile.log"
        shutil.move(str(_compile_log_tmp), compile_log_dest)
        print(f"  Compile log: {compile_log_dest}")

    # Phases 2-4: transfer, infer, fetch — all output goes to deploy.log
    deploy_log_path = ACTIVE_MODEL_LINK / "deploy.log"
    with Tee(deploy_log_path, verbose=True) as tee:
        if not args.skip_transfer:
            phase_transfer(model_name)

        if not args.skip_infer:
            tee.verbose = PHASE3_VERBOSE  # toggle: may mute FPGA inference output
            phase_infer(args.subset)
            tee.verbose = True  # restore for phase 4 and Done summary

        if not args.skip_fetch:
            phase_fetch()

        print_header("Done!")
        print(f"  Model      : {model_name}")
        print(f"  Deploy log : {deploy_log_path}")
        if not args.skip_infer:
            if not args.skip_fetch:
                print(f"  Results    : {ACTIVE_MODEL_LINK / 'results'}")
            else:
                print("  Results and deploy logs on Target.")


if __name__ == "__main__":
    main()
