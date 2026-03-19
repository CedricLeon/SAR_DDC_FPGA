#!/usr/bin/env python3
"""Collect per-patch reconstruction statistics for a GPU model on the test_sub500 dataset.

Saves per-patch min / max / mean / std / p25 / p75 of linA for:
  - GPU reconstruction (compress → decompress pipeline, matching test_step)
  - MERLIN GT
  - Noisy input

Statistics are accumulated over the first N_PATCHES patches of DATA_PATH and
written as an NPZ file to::

    results/plots/stat_{model_name}/patch_stats.npz

The model_name follows the deploy.py convention: ``{model}-{activation}_s{seed}_L{lmbda}_pt``

Usage (from DDC_FPGA root, SAR_DDC env):
    python scripts/compare_FPGA_to_GPU/collect_patch_stats_gpu.py
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.constants import AMP_MAX, AMP_MIN, EPS  # noqa: E402

# ── User configuration ─────────────────────────────────────────────────────────
# Edit these two constants for each model you want to evaluate.
RUN_DIR = Path(
    # "logs/train/sar_ddc/hyperprior/multiruns/2026-02-07_00-51-42/0" # ResSHyp-relu_s0_L1_pt
    # "logs/train/sar_ddc/hyperprior/multiruns/2026-02-07_00-51-42/1" # ResSHyp-relu_s0_L5_pt
    "logs/train/sar_ddc/hyperprior/multiruns/2026-02-07_00-51-42/4"  # ResSHyp-relu_s0_L100_pt
    # "logs/train/sar_ddc/hyperprior/multiruns/2026-02-06_14-57-22/0" # ResSHyp-relu_s0_L1000_pt
)
DATA_PATH = Path("data/processed_hdf5/TSX_spatial_splits_5_256x256/test_sub500_seed42.npy")
N_PATCHES = 100
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[2]


# ── Model-name helper (mirrors deploy.py _make_compiled_model_name) ────────────
def _make_model_name(cfg) -> str:
    """Derive model name following deploy.py convention.

    Returns: ``{model}-{activation}_s{seed}_L{lmbda}_pt``
    """

    def _get(obj, key, default=None):
        return obj.get(key, default)

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
    return f"{model}-{activation}_s{seed}_L{lmbda}_pt"


# ── Stat helper ────────────────────────────────────────────────────────────────
def _compute_patch_stats(patch_linA: np.ndarray) -> Dict[str, float]:
    """Compute statistics of a single patch in linear amplitude (flattened)."""
    flat = patch_linA.ravel()
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    run_dir_abs = ROOT_DIR / RUN_DIR
    data_path_abs = ROOT_DIR / DATA_PATH
    ckpt_path = run_dir_abs / "checkpoints" / "last.ckpt"

    # ── Load config ────────────────────────────────────────────────────────────
    config_path = run_dir_abs / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    model_name = _make_model_name(cfg)

    print(f"Model name : {model_name}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Data       : {data_path_abs}")

    # ── Instantiate model and load weights ─────────────────────────────────────
    import hydra  # noqa: PLC0415 — lazy import to keep startup fast

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Populate entropy bottleneck CDF tables (required before compress/decompress)
    model.net.update(force=True)
    print(f"Device     : {device}")

    # ── Load data — first N_PATCHES patches ────────────────────────────────────
    if not data_path_abs.exists():
        raise FileNotFoundError(f"Data file not found: {data_path_abs}")
    raw = np.load(str(data_path_abs)).astype(np.float32)  # [N, H, W, 4]
    if len(raw) < N_PATCHES:
        raise ValueError(f"Dataset has only {len(raw)} patches, expected >= {N_PATCHES}.")
    raw = raw[:N_PATCHES]
    print(f"Loaded {len(raw)} patches  shape={raw.shape}")

    # ── Per-patch inference and stat accumulation ──────────────────────────────
    stat_keys: List[str] = ["min", "max", "mean", "std", "p25", "p75"]
    accum: Dict[str, Dict[str, List[float]]] = {
        src: {k: [] for k in stat_keys} for src in ["gpu_recon", "merlin", "noisy"]
    }

    with torch.inference_mode():
        for i in range(len(raw)):
            patch = raw[i]  # [H, W, 4]

            # ── Build input tensor [1, 2, H, W] ───────────────────────────────
            real = torch.from_numpy(patch[..., 0]).unsqueeze(0).unsqueeze(0).to(device)
            imag = torch.from_numpy(patch[..., 1]).unsqueeze(0).unsqueeze(0).to(device)
            inp = torch.cat([real, imag], dim=1)  # [1, 2, H, W]

            # ── Normalise (matches SARDDCModule.forward / test_step) ───────────
            x = (torch.log(torch.square(inp) + EPS) - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)

            # ── Compress + decompress (matches test_step real-bitstream path) ──
            out_enc = model.net.compress(x)
            out_dec = model.net.decompress(out_enc["strings"], out_enc["shape"])
            x_hat = out_dec  # [1, 2, H, W] normalised log-scale

            # ── Denormalise → linA [H, W] ──────────────────────────────────────
            recon_denorm = x_hat * (AMP_MAX - AMP_MIN) + AMP_MIN
            recon_lin = torch.exp(recon_denorm)
            clean_linI = 0.5 * (torch.square(recon_lin[:, 0]) + torch.square(recon_lin[:, 1]))
            recon_linA = torch.sqrt(clean_linI).squeeze().cpu().numpy()  # [H, W]

            # ── Noisy linA [H, W] ──────────────────────────────────────────────
            noisy_linA = np.sqrt(patch[..., 0] ** 2 + patch[..., 1] ** 2)  # [H, W]

            # ── MERLIN GT linA [H, W] — channel 3 is already linA ─────────────
            merlin_linA = patch[..., 3]  # [H, W]

            # ── Accumulate per-source stats ────────────────────────────────────
            for src, arr in [
                ("gpu_recon", recon_linA),
                ("merlin", merlin_linA),
                ("noisy", noisy_linA),
            ]:
                s = _compute_patch_stats(arr)
                for k, v in s.items():
                    accum[src][k].append(v)

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(raw)}]")

    # ── Build output dict and save ─────────────────────────────────────────────
    arrays: Dict[str, np.ndarray] = {}
    for src, stats_dict in accum.items():
        for k, values in stats_dict.items():
            arrays[f"{src}_{k}"] = np.array(values, dtype=np.float32)
    arrays["n_patches"] = np.array(len(raw))

    out_dir = ROOT_DIR / "results" / "plots" / f"stat_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "patch_stats.npz"
    np.savez(str(save_path), **arrays)

    print(f"\nSaved patch stats ({len(raw)} patches) → {save_path}")
    print("Keys:", sorted(arrays.keys()))


if __name__ == "__main__":
    main()
