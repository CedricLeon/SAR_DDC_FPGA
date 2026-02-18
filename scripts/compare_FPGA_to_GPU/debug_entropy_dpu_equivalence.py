import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

# Add the current directory (scripts/compare_FPGA_to_GPU) to import the local ans module
sys.path.append(str(Path(__file__).parent.resolve()))

# Try importing the compiled C++ extension
try:
    import ans

    print(
        f"Successfully imported 'ans' module. Available functions: {', '.join(f for f in dir(ans) if not f.startswith('_'))}."
    )
except ImportError:
    print(
        "Error: 'ans' module not found. Please compile it first using 'python setup_ans_host.py build_ext --inplace' in scripts/compare_FPGA_to_GPU/"
    )
    sys.exit(1)


# Add project root to sys path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.fpga.entropy_models_inference import EntropyBottleneck, GaussianConditional
from src.models.components.compressai_dpu import get_scale_table
from src.models.components.res_scale_hyperprior_dpu import (
    ResidualScaleHyperpriorPatched,
)
from src.utils.constants import AMP_MAX, AMP_MIN, EPS

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


# ----- HYPERPARAMS -----
# Lmbda=1
CKPT_PATH = "logs/train/sar_ddc/hyperprior/multiruns/2026-02-11_14-42-17/0/checkpoints/last.ckpt"
# Lmbda=10
# CKPT_PATH = "logs/train/sar_ddc/hyperprior/runs/2026-01-16_13-34-45/checkpoints/last.ckpt"
# Lmbda=1000
# CKPT_PATH = "logs/train/sar_ddc/hyperprior/multiruns/2026-02-12_19-06-18/1/checkpoints/last.ckpt"

DATA_PATH = "data/processed_hdf5/TSX_spatial_splits_5_256x256/test_1000.npy"


def preprocess_batch(data_npy, subset=1):
    """Normalize raw complex data [N, H, W, 2] -> [N, 2, H, W] Normalized Log Intensity."""
    # Ensure we only pick the first two channels (Real, Imag)
    if data_npy.ndim == 4:
        x_complex = data_npy[:subset, :, :, 0:2]  # (N, H, W, 2)
    else:
        x_complex = data_npy[:subset]

    # 1. Compute Intensity -> Log Intensity
    x_sq = np.square(x_complex)
    x_log = np.log(x_sq + EPS)
    x_norm = (x_log - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)

    # Transpose to NCHW
    x_norm = np.transpose(x_norm, (0, 3, 1, 2))
    return x_norm.astype(np.float32)


def get_entropy_params(model):
    """Extracts entropy parameters from the model (logic from export_entropy_params.py)."""

    # Force updates
    print("  -> Updating EntropyBottleneck...")
    model.entropy_bottleneck.update(force=True)

    print("  -> Updating GaussianConditional...")
    gc = model.gaussian_conditional
    if gc.scale_table.numel() == 0:
        print("     No scale table found, using default.")
        scale_table = get_scale_table()
        gc.update_scale_table(scale_table, force=True)
    else:
        gc.update_scale_table(gc.scale_table, force=True)

    # Extract EB params
    eb = model.entropy_bottleneck
    eb_medians = eb.quantiles[:, 0, 1].detach().cpu().numpy()
    eb_quantized_cdf = eb._quantized_cdf.detach().cpu().numpy().astype(np.int32)
    eb_offset = eb._offset.detach().cpu().numpy().astype(np.int32)
    eb_cdf_length = eb._cdf_length.detach().cpu().numpy().astype(np.int32)

    # Extract GC params
    gc_scale_table = gc.scale_table.detach().cpu().numpy().astype(np.float32)
    gc_quantized_cdf = gc.quantized_cdf.detach().cpu().numpy().astype(np.int32)
    gc_cdf_length = gc.cdf_length.detach().cpu().numpy().astype(np.int32)
    gc_offset = gc.offset.detach().cpu().numpy().astype(np.int32)

    return {
        "eb_quantized_cdf": eb_quantized_cdf,
        "eb_cdf_length": eb_cdf_length,
        "eb_offset": eb_offset,
        "eb_medians": eb_medians,
        "gc_scale_table": gc_scale_table,
        "gc_quantized_cdf": gc_quantized_cdf,
        "gc_cdf_length": gc_cdf_length,
        "gc_offset": gc_offset,
    }


def compare_results():
    """Main comparison function."""
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

    # Override for DPU export logic usually, but here we run on CPU/GPU
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

    # 3. Extract Params for Real Inference
    print("Extracting Entropy Parameters (computing CDFs/Tables)...")
    params = get_entropy_params(model)

    # 4. Initialize Real Inference Models
    print("Initializing C++ Wrapped Models...")
    eb_real = EntropyBottleneck(
        channels=params["eb_cdf_length"].shape[0],  # or params["eb_medians"].shape[0]
        quantized_cdf=params["eb_quantized_cdf"],
        cdf_length=params["eb_cdf_length"],
        offset=params["eb_offset"],
        medians=params["eb_medians"],
    )

    gc_real = GaussianConditional(
        scale_table=params["gc_scale_table"],
        quantized_cdf=params["gc_quantized_cdf"],
        cdf_length=params["gc_cdf_length"],
        offset=params["gc_offset"],
    )

    # 5. Load Data
    print(f"Loading Data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    inputs_np = preprocess_batch(data, subset=1)  # One image
    inputs_torch = torch.from_numpy(inputs_np)

    print("----------------------------------------------------------------")
    print("Running Mode 1: Classic PyTorch (Likelihood Estimation)")

    t0 = time.time()
    with torch.no_grad():
        # Step 1: g_a
        x_real = inputs_torch[:, :1]
        x_imag = inputs_torch[:, 1:]
        y_real = model.g_a(x_real)
        y_imag = model.g_a(x_imag)
        y_torch = torch.cat((y_real, y_imag), dim=1)

        # Step 2: h_a
        z_in_torch = torch.abs(y_torch)
        z_torch = model.h_a(z_in_torch)

        # Step 3: EntropyBottleneck (Likelihoods)
        z_hat_torch, z_lik_torch = model.entropy_bottleneck(z_torch)

        # Step 4: h_s
        scales_torch = model.h_s(z_hat_torch)

        # Step 5: GaussianConditional (Likelihoods)
        y_hat_torch, y_lik_torch = model.gaussian_conditional(y_torch, scales_torch)

        # Step 6: g_s
        y_hat_real = y_hat_torch[:, : y_hat_torch.shape[1] // 2]
        y_hat_imag = y_hat_torch[:, y_hat_torch.shape[1] // 2 :]
        x_hat_real = model.g_s(y_hat_real)
        x_hat_imag = model.g_s(y_hat_imag)
        x_hat_torch = torch.cat((x_hat_real, x_hat_imag), dim=1)

    t_torch = time.time() - t0
    print(f"PyTorch Inference Done in {t_torch:.4f}s.")

    print("----------------------------------------------------------------")
    print("Running Mode 2: Hybrid Real (C++ Entropy Coding)")

    t0 = time.time()
    # 1. Inputs (using Numpy)
    # Usually on FPGA we run g_a, h_a, etc. on DPU. Here we reuse PyTorch outputs as "Simulated DPU"
    # But for "Authenticity" we should rely on what we can.
    # We will trust PyTorch for NN layers (g_a, h_a, h_s, g_s) and test the entropy parts.

    # NN: g_a (Simulated)
    y_np = y_torch.cpu().numpy()  # [1, 256, 16, 16] - Wait, Torch is NCHW. DPU is NHWC usually?
    # Our inference_hybrid.py expects NHWC or handles conversion.
    # The EB/GC wrappers in entropy_models_inference.py:
    # EB.compress(inputs): "inputs: Latent z (N, C, H, W) or (N, H, W, C)."
    # GC.compress(inputs): "inputs: Latent y (N, H, W, C)" (Note: GC docstring says NHWC)

    # Let's stick to NCHW from Torch and check if wrappers handle it.
    # EB Wrapper:
    #   if inputs.ndim == 4 and inputs.shape[1] == self.channels: inputs = inputs.transpose(0, 2, 3, 1)
    # GC Wrapper:
    #   Does NOT seem to check dim order in compress()!
    #   It does: "inputs = inputs - means" -> "symbols = round(inputs)" -> "indexes = ... scales".
    #   It assumes 'scales' structure matches.
    #   We MUST provide NHWC to GC wrapper if that's what it expects?
    #   Reading GC.compress again:
    #   "Latent y (N, H, W, C)" -> Yes.

    # So we must transpose PyTorch outputs to NHWC for our Real wrappers.

    # NN: h_a (Simulated)
    z_np = z_torch.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC

    # EntropyBottleneck (CPU/Real)
    # Compress
    z_strings = eb_real.compress(z_np)
    z_bytes = sum(len(s) for s in z_strings)

    # Decompress
    z_hat_real = eb_real.decompress(z_strings, (z_np.shape[1], z_np.shape[2]))  # Returns NHWC

    # Verify Step: Check z_hat compatibility
    # Compare with z_hat_torch (NCHW)
    z_hat_real_nchw = z_hat_real.transpose(0, 3, 1, 2)
    max_diff_z_hat = np.max(np.abs(z_hat_torch.cpu().numpy() - z_hat_real_nchw))
    print(f"[Check] z_hat max diff: {max_diff_z_hat:.6f}")
    if max_diff_z_hat > 0.5:  # Rounding differences can occur, but should be small?
        # Actually z_hat is quantized. If medians are close to .5, we might round differently?
        # Torch uses noise during training, but quantization during eval.
        print("WARNING: z_hat divergent!")

    # NN: h_s (Simulated)
    # Use z_hat_real (NHWC) -> Transpose to NCHW for Pytorch layer
    z_hat_for_hs = torch.from_numpy(z_hat_real_nchw)
    with torch.no_grad():
        scales_sim_tensor = model.h_s(z_hat_for_hs)
    scales_np = scales_sim_tensor.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC

    # GaussianConditional (CPU/Real)
    # Compress
    y_np_nhwc = y_np.transpose(0, 2, 3, 1)
    means_np = np.zeros_like(y_np_nhwc)

    y_strings = gc_real.compress(y_np_nhwc, scales_np, means_np)
    y_bytes = sum(len(s) for s in y_strings)

    # Decompress
    y_hat_real = gc_real.decompress(y_strings, scales_np, means_np)  # NHWC

    # Verify Step: Check y_hat compatibility
    y_hat_real_nchw = y_hat_real.transpose(0, 3, 1, 2)
    max_diff_y_hat = np.max(np.abs(y_hat_torch.cpu().numpy() - y_hat_real_nchw))
    print(f"[Check] y_hat max diff: {max_diff_y_hat:.6f}")

    # NN: g_s (Simulated)
    y_hat_for_gs = torch.from_numpy(y_hat_real_nchw)

    y_hat_real_sim = y_hat_for_gs[:, : y_hat_for_gs.shape[1] // 2]
    y_hat_imag_sim = y_hat_for_gs[:, y_hat_for_gs.shape[1] // 2 :]

    with torch.no_grad():
        x_hat_real_sim = model.g_s(y_hat_real_sim)
        x_hat_imag_sim = model.g_s(y_hat_imag_sim)
        x_hat_sim = torch.cat((x_hat_real_sim, x_hat_imag_sim), dim=1)

    x_hat_np = x_hat_sim.cpu().numpy()

    t_real = time.time() - t0
    print(f"Real Inference Done in {t_real:.4f}s.")

    # ----------------------------------------------------------------
    # Comparison
    print("----------------------------------------------------------------")
    print("COMPARISON RESULTS:")

    # X_HAT
    diff_x = np.abs(x_hat_torch.cpu().numpy() - x_hat_np)
    print(f"x_hat: Max Diff = {np.max(diff_x):.6f}, Mean Diff = {np.mean(diff_x):.6f}")

    # BPP
    bpp_torch = (
        np.log(z_lik_torch.cpu().numpy()).sum() + np.log(y_lik_torch.cpu().numpy()).sum()
    ) / (-np.log(2) * 256 * 256)

    bpp_real = (z_bytes + y_bytes) * 8 / (256 * 256)

    print(f"BPP Torch (Likelihood): {bpp_torch:.4f}")
    print(f"BPP Real (Bytes):       {bpp_real:.4f}")
    print(f"Difference:             {abs(bpp_torch - bpp_real):.4f}")

    if np.max(diff_x) < 1e-4:
        print("\nSUCCESS: Reconstruction matches exactly (within float32 precision).")
    else:
        print("\nFAILURE: Reconstruction mismatch.")

    # Optional: Save params for inspection
    # out_dir = Path("results/comparison_FPGA_GPU")
    # out_dir.mkdir(parents=True, exist_ok=True)
    # np.savez(out_dir / "comparison_params.npz", **params)


if __name__ == "__main__":
    compare_results()
