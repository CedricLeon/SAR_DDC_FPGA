import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to sys path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts", "fpga"))

from fpga.entropy_models_dpu import EntropyBottleneckDPU, GaussianConditionalDPU

from src.models.components.res_scale_hyperprior_dpu import (
    ResidualScaleHyperpriorPatched,
)
from src.utils.constants import AMP_MAX, AMP_MIN, EPS

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def preprocess_batch(data_npy, subset=5):
    """Normalize raw complex data [N, H, W, 2] -> [N, 2, H, W] Normalized Log Intensity."""
    # Ensure we only pick the first two channels (Real, Imag)
    # Check shape to avoid index errors
    if data_npy.ndim == 4:
        x_complex = data_npy[:subset, :, :, 0:2]  # (N, H, W, 2)
    else:
        # Handle case where it might be already formatted or unexpected
        x_complex = data_npy[:subset]

    # 1. Compute Intensity -> Log Intensity
    # Logic from inference_hybrid.py
    # x_sq = real^2 + imag^2
    x_sq = np.square(x_complex)
    x_log = np.log(x_sq + EPS)  # Sum channels for Intensity ? No, elementwise.
    # inference_hybrid.py: x_sq = np.square(x_complex) -> x_log = np.log(x_sq + EPS).
    # Then x_norm = ...

    x_norm = (x_log - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)

    # Transpose to NCHW
    x_norm = np.transpose(x_norm, (0, 3, 1, 2))
    return x_norm.astype(np.float32)


def compare_results():
    """Main comparison function."""
    CKPT_PATH = "logs/train/sar_ddc/hyperprior/runs/2026-01-16_13-34-45/checkpoints/last.ckpt"
    DATA_PATH = "data/processed_hdf5/TSX_spatial_splits_5_256x256/test_1000.npy"

    print(f"Loading Model from {CKPT_PATH}...")

    # 1. Load Config to instantiate model correctly
    ckpt_path = Path(CKPT_PATH)
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    print(f"Reading config from {config_path}...")
    conf = OmegaConf.load(config_path)
    model_params = conf.model.net
    if "_target_" in model_params:
        model_params.pop("_target_")

    # Override for DPU export
    model_params["export_dpu"] = True
    print(f"Instantiating ResidualScaleHyperpriorPatched with params: {model_params}")
    model = ResidualScaleHyperpriorPatched(**model_params)

    # 2. Load Weights
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove "net." prefix
        state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # FORCE UPDATE to populate CDFs and Tables used for Inference (and DPU export)
    # The checkpoint might not have them if it was saved without update or if we want to be sure.
    print("Updating entropy models (computing CDFs/Tables)...")

    # Check status before
    print(f"DEBUG: CDF before update: {model.entropy_bottleneck._quantized_cdf.shape}")

    # model.update(force=True)
    # Calling explicitly on components to be sure
    # ret_eb = model.entropy_bottleneck.update(force=True)
    # print(f"DEBUG: EB Update returned: {ret_eb}")
    # print(f"DEBUG: CDF after EB update: {model.entropy_bottleneck._quantized_cdf.shape}")

    # Try model.update(force=True) logic to ensure CDFs are populated
    try:
        model.update(force=True)
    except Exception as e:
        print(f"WARNING: model.update() raised exception: {e}")

    # Check if update was successful for EB
    if model.entropy_bottleneck._quantized_cdf.numel() == 0:
        print(
            "Information: model.update() did not populate EB CDFs (likely due to strict mode or buffer detachment). Forcing manual component updates..."
        )

        # 1. Update EntropyBottleneck
        model.entropy_bottleneck.update(force=True)

        # 2. Update GaussianConditional
        # Generate standard CompressAI scale table (0.11 to 256, 64 steps)
        scale_table = torch.exp(torch.linspace(np.log(0.11), np.log(256), 64))

        try:
            model.gaussian_conditional.update_scale_table(scale_table)
        except Exception:
            # Try without args if signature differs
            try:
                model.gaussian_conditional.update_scale_table()
            except Exception:
                pass

    print(f"Loading Data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    inputs_np = preprocess_batch(data, subset=1)  # One image
    inputs_torch = torch.from_numpy(inputs_np)

    print("----------------------------------------------------------------")
    print("Running Mode 1: Classic PyTorch")
    with torch.no_grad():
        # Trace the forward pass parts manually to get intermediates
        # Step 1: g_a
        # inputs: [1, 2, 256, 256]. Split.
        x_real = inputs_torch[:, :1]
        x_imag = inputs_torch[:, 1:]
        y_real = model.g_a(x_real)
        y_imag = model.g_a(x_imag)
        y_torch = torch.cat((y_real, y_imag), dim=1)

        # Step 2: h_a
        z_in_torch = torch.abs(y_torch)
        z_torch = model.h_a(z_in_torch)

        # Step 3: EntropyBottleneck
        z_hat_torch, z_lik_torch = model.entropy_bottleneck(z_torch)

        # Step 4: h_s
        scales_torch = model.h_s(z_hat_torch)

        # Step 5: GaussianConditional
        y_hat_torch, y_lik_torch = model.gaussian_conditional(y_torch, scales_torch)

        # Step 6: g_s
        y_hat_real = y_hat_torch[:, : y_hat_torch.shape[1] // 2]
        y_hat_imag = y_hat_torch[:, y_hat_torch.shape[1] // 2 :]
        x_hat_real = model.g_s(y_hat_real)
        x_hat_imag = model.g_s(y_hat_imag)
        x_hat_torch = torch.cat((x_hat_real, x_hat_imag), dim=1)

    print("PyTorch Inference Done.")

    print("----------------------------------------------------------------")
    print("Running Mode 2: Hybrid (Numpy Entropy)")

    # 1. Initialize Entropy Models
    # Extract params from loaded model to ensure sync
    eb_state = model.entropy_bottleneck.state_dict()
    gc_state = model.gaussian_conditional.state_dict()

    # EntropyBottleneck Params
    # quantiles (C, 1, 3) -> medians (C, 1, 1) is index 1
    if "quantiles" in eb_state:
        medians = eb_state["quantiles"][:, 0, 1].cpu().numpy()
    else:
        # Fallback if medians not reachable via state_dict (should not happen if loaded)
        raise ValueError("Could not find quantiles/medians in EntropyBottleneck state dict")

    # Debug Shapes
    cdf = eb_state.get("_quantized_cdf", torch.tensor([]))
    # print(f"DEBUG: Medians shape: {medians.shape}")
    # print(f"DEBUG: CDF shape: {cdf.shape}")

    eb_channels = medians.shape[0]

    eb_dpu = EntropyBottleneckDPU(
        channels=eb_channels,
        quantized_cdf=cdf.cpu().numpy(),
        cdf_length=eb_state["_cdf_length"].cpu().numpy(),
        offset=eb_state["_offset"].cpu().numpy(),
        medians=medians,
    )

    # GaussianConditional Params
    gc_dpu = GaussianConditionalDPU(
        scale_table=(
            gc_state.get("scale_table", None).cpu().numpy() if "scale_table" in gc_state else None
        )
    )

    # 2. Execution (Simulating DPU with PyTorch-to-Numpy calls)

    # g_a (DPU) - Use PyTorch output converted to Numpy
    y_np = y_torch.cpu().numpy()

    # h_a (DPU) - Use PyTorch output converted to Numpy
    z_np = z_torch.cpu().numpy()

    # EntropyBottleneck (CPU/Numpy)
    z_hat_np, z_lik_np = eb_dpu.forward(z_np)

    # Verify Step: Check z_hat compatibility
    max_diff_z_hat = np.max(np.abs(z_hat_torch.cpu().numpy() - z_hat_np))
    print(f"[Check] z_hat max diff: {max_diff_z_hat:.6f}")
    if max_diff_z_hat > 1e-4:
        print("WARNING: z_hat divergent!")

    # h_s (DPU)
    # We use the z_hat from the DPU path (z_hat_np).
    # Since we don't have a DPU, we run the PyTorch layer on z_hat_np
    # Need to convert back to tensor
    z_hat_for_dpu = torch.from_numpy(z_hat_np)
    with torch.no_grad():
        scales_sim_tensor = model.h_s(z_hat_for_dpu)
    scales_np = scales_sim_tensor.cpu().numpy()

    # GaussianConditional (CPU/Numpy)
    y_hat_np, y_lik_np = gc_dpu.forward(y_np, scales_np)

    # Verify Step: Check y_hat compatibility
    max_diff_y_hat = np.max(np.abs(y_hat_torch.cpu().numpy() - y_hat_np))
    print(f"[Check] y_hat max diff: {max_diff_y_hat:.6f}")
    if max_diff_y_hat > 1e-4:
        print("WARNING: y_hat divergent!")

    # g_s (DPU) - Simulated
    y_hat_for_dpu = torch.from_numpy(y_hat_np)
    y_hat_real_sim = y_hat_for_dpu[:, : y_hat_for_dpu.shape[1] // 2]
    y_hat_imag_sim = y_hat_for_dpu[:, y_hat_for_dpu.shape[1] // 2 :]

    with torch.no_grad():
        x_hat_real_sim = model.g_s(y_hat_real_sim)
        x_hat_imag_sim = model.g_s(y_hat_imag_sim)
        x_hat_sim = torch.cat((x_hat_real_sim, x_hat_imag_sim), dim=1)

    x_hat_np = x_hat_sim.cpu().numpy()

    # ----------------------------------------------------------------
    # Comparison
    print("----------------------------------------------------------------")
    print("COMPARISON RESULTS:")

    # X_HAT
    diff_x = np.abs(x_hat_torch.cpu().numpy() - x_hat_np)
    print(f"x_hat: Max Diff = {np.max(diff_x):.6f}, Mean Diff = {np.mean(diff_x):.6f}")

    # Likelihoods
    # Note: Torch uses interpolated likelihoods roughly, we use erf/discrete CDF.
    # Expect some differences.
    diff_z_lik = np.abs(z_lik_torch.cpu().numpy() - z_lik_np)
    print(f"z_lik: Max Diff = {np.max(diff_z_lik):.6f}, Mean Diff = {np.mean(diff_z_lik):.6f}")

    diff_y_lik = np.abs(y_lik_torch.cpu().numpy() - y_lik_np)
    print(f"y_lik: Max Diff = {np.max(diff_y_lik):.6f}, Mean Diff = {np.mean(diff_y_lik):.6f}")

    # BPP estimates
    bpp_torch = (
        np.log(z_lik_torch.cpu().numpy()).sum() + np.log(y_lik_torch.cpu().numpy()).sum()
    ) / (-np.log(2) * 256 * 256)
    bpp_np = (np.log(z_lik_np).sum() + np.log(y_lik_np).sum()) / (-np.log(2) * 256 * 256)
    print(f"BPP Torch: {bpp_torch:.4f}")
    print(f"BPP Numpy: {bpp_np:.4f}")

    if np.max(diff_x) < 1e-4:
        print("\nSUCCESS: Reconstruction matches exactly (within float32 precision).")
    else:
        print("\nFAILURE: Reconstruction mismatch.")


if __name__ == "__main__":
    compare_results()
