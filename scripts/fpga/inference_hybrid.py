#!/usr/bin/env python3
"""Hybrid Inference Script for Vitis-AI DPU + CPU Orchestration.

This script manages the execution of a "split" model where:
- Neural Network subgraphs (g_a, h_a, h_s, g_s) run on the DPU.
- Entropy operations (quantization, likelihoods) run on the CPU (using Numpy).

Usage:
    python3 inference_hybrid.py --xmodel model.xmodel --data test.npy --params entropy_params.npz

Requirements:
    - vart, xir (Vitis-AI runtime)
    - numpy
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import vart  # type: ignore
import xir  # type: ignore
from entropy_models_dpu import (
    EntropyBottleneckDPU,
    GaussianConditionalDPU,
    load_entropy_models_dpu,
)
from inference_utils import (
    AMP_MAX,
    AMP_MIN,
    EPS,
    MetricsTracker,
    print_tensor_stats,
    visualize_patches,
)

log_file = ""


def log(msg):
    """Manual logging function."""
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
IMAGE_SIZE = 256  # Input image size (always square)
S_MAIN = 16  # Downsampling factor of the main autoencoder
S_HYPER = 8  # Downsampling factor of the hyperprior autoencoder
H_LATENT = IMAGE_SIZE // S_MAIN  # height of latent representation y
W_LATENT = IMAGE_SIZE // S_MAIN  # width of latent representation y
H_HYPER = H_LATENT // S_HYPER  # height of latent representation z
W_HYPER = W_LATENT // S_HYPER  # width of latent representation z
C_MAIN = 128  # Number channels main autoencoder (g_a, g_s)
C_HYPER = C_MAIN * 2  # Number channels hyperprior (h_a, h_s)


# -----------------------------------------------------------------------------
# DPU RUNNER HELPER
# -----------------------------------------------------------------------------
class DPUSubgraphRunner:
    """Helper to wrap a single DPU subgraph runner."""

    def __init__(self, runner: vart.Runner, subgraph: xir.Subgraph, name: str):
        self.runner = runner
        self.name = name

        # ----- IO Shapes -----
        self.input_tensors = runner.get_input_tensors()
        self.output_tensors = runner.get_output_tensors()
        # We assume 1 input and 1 output for simplicity based on our wrapper
        self.input_shape = tuple(self.input_tensors[0].dims)  # [N, H, W, C]
        self.output_shape = tuple(self.output_tensors[0].dims)  # [N, H, W, C]
        # Get fixed-point scales for conversion
        input_fixpos = self.input_tensors[0].get_attr("fix_point")
        output_fixpos = self.output_tensors[0].get_attr("fix_point")
        self.input_scale = 2.0**input_fixpos
        self.output_scale = 2.0 ** (-output_fixpos)

        print(
            f"[{name}] In: {self.input_shape} (scale={self.input_scale}), Out: {self.output_shape} (scale={self.output_scale})"
        )

    def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on a batch of data.

        input_data: Float numpy array matching input shape (NCHW or NHWC).
        """
        # Note: VART expects NHWC but PyTorch is NCHW.
        # expected_dims = len(self.input_shape)
        # if expected_dims == 4:
        #     # Heuristic check for NCHW vs NHWC
        #     # DPU usually HWC.
        #     if input_data.shape != self.input_shape:
        #         # Try simple transpose (N, H, W, C) from (N, C, H, W)
        #         # assuming input_data is NCHW
        #         input_data = input_data.transpose(0, 2, 3, 1)

        # 1. Quantize Input (Float -> Int8)
        input_int8 = float_to_DPU_int(input_data, self.input_scale)

        # 2. Prepare Buffers
        # VART needs input/output buffers with exact shape from DPU, order="C" ensures data is laid out in row-major order (unlike Fortran order)
        input_buffer = np.ascontiguousarray(input_int8)
        output_buffer = np.empty(self.output_shape, dtype=np.int8, order="C")

        # 3. Execute (job_id is returned)
        job_id = self.runner.execute_async([input_buffer], [output_buffer])
        self.runner.wait(job_id)

        # 4. Dequantize Output (Int8 -> Float)
        output_float = DPU_int_to_float(output_buffer, self.output_scale)

        # # 5. Transpose back to NCHW if needed
        # if expected_dims == 4:
        #     # (N, H, W, C) -> (N, C, H, W)
        #     output_float = output_float.transpose(0, 3, 1, 2)

        return output_float


def float_to_DPU_int(data_float: np.ndarray, input_scale: float) -> np.ndarray:
    """Convert float data to DPU fixed-point INT8 using the given scale."""
    return (data_float * input_scale).astype(np.int8)


def DPU_int_to_float(data_int: np.ndarray, scale: float) -> np.ndarray:
    """Convert DPU fixed-point INT8 data back to float using the given scale."""
    return data_int.astype(np.float32) * scale


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------


def identify_subgraphs(graph: xir.Graph) -> Dict[str, xir.Subgraph]:
    """Identify which subgraph corresponds to g_a, h_a, h_s, g_s based on shapes.

    (This is a heuristic and might need adjustment based on real compiler names)
    """
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraphs = [
        s for s in subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"
    ]
    cpu_subgraphs = [
        s for s in subgraphs if s.has_attr("device") and s.get_attr("device") == "CPU"
    ]
    log(
        f"Found {len(dpu_subgraphs)} DPU subgraphs and {len(cpu_subgraphs)} CPU subgraphs, for a total of {len(subgraphs)}."
    )

    mapping = {}
    for sg in dpu_subgraphs:
        # Get input tensor shapes
        inputs = sg.get_input_tensors()
        if not inputs:
            continue
        # Shapes based on 256x256 input and are NHWC format
        shape = tuple(list(inputs)[0].dims)

        # g_a: In (1, 256, 256, 1) -> Out (1, 16, 16, 128)
        if shape[1:3] == (IMAGE_SIZE, IMAGE_SIZE) and shape[3] == 1:
            mapping["g_a"] = sg

        # h_a: In (1, 16, 16, 256) -> Out (1, 2, 2, 256)  (Abs(y) is 256 channels)
        elif shape[1:3] == (H_LATENT, W_LATENT) and shape[3] == C_HYPER:
            mapping["h_a"] = sg

        # h_s: In (1, 2, 2, 256) -> Out (1, 16, 16, 256) (Scales)
        elif shape[1:3] == (H_HYPER, W_HYPER) and shape[3] == C_HYPER:
            mapping["h_s"] = sg

        # g_s: In (1, 16, 16, 128) -> Out (1, 256, 256, 1) (One channel decode)
        elif shape[1:3] == (H_LATENT, W_LATENT) and shape[3] == C_MAIN:
            mapping["g_s"] = sg

        log(f"Identified DPU subgraph: {sg.get_name()} with input shape {shape}")

    # Ensure validity of the graph mapping
    required_keys = ["g_a", "h_a", "h_s", "g_s"]
    missing_keys = [k for k in required_keys if k not in mapping]
    if missing_keys:
        log(f"ERROR: Could not find subgraphs for: {missing_keys}")
        log(f"Found mapped subgraphs: {list(mapping.keys())}")

        log("\n--- Debug: All DPU Subgraphs ---")
        root = graph.get_root_subgraph()
        for sg in root.toposort_child_subgraph():
            if sg.has_attr("device") and sg.get_attr("device") == "DPU":
                inputs = list(sg.get_input_tensors())
                outputs = list(sg.get_output_tensors())
                in_shape = tuple(inputs[0].dims) if inputs else "None"
                out_shape = tuple(outputs[0].dims) if outputs else "None"
                log(f"Subgraph: {sg.get_name()}")
                log(f"  Input:  {in_shape}")
                log(f"  Output: {out_shape}")
        log("--------------------------------\n")
        raise ValueError("Could not identify all required subgraphs in the model.")

    return mapping


def load_npy_data(
    dataset_path: str, subset_len: int = 100
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load data from NPY file.

    Noisy data is in the first two channels (real, imag) and is unnormalized. Ground truths are in
    subsequent channels (2: ADAM-NOC and 3: MERLIN), they come in linear amplitude format.
    """
    data = np.load(dataset_path)
    if subset_len > 0 and subset_len < data.shape[0]:
        data = data[:subset_len]
    noisy = data[:, :, :, 0:2]  # [N, H, W, 2] real and imag channels

    ground_truths = {  # [N, H, W, 1] already in linear amplitude
        "adam_noc": data[:subset_len, :, :, 2:3],
        "merlin": data[:subset_len, :, :, 3:4],
    }

    return noisy, ground_truths


def run_hybrid_inference(xmodel: str, dataset_path: str, subset: int = 100):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ---- Paths and Logging ----
    entropy_params_path = os.path.dirname(xmodel) + "/entropy_params.npz"
    model_name = xmodel.split("/")[-1].replace(".xmodel", "")
    output_dir = f"results/inference_{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    global log_file
    log_file = os.path.join(output_dir, "inference.log")

    log(f"Starting Hybrid Inference at {timestamp}.")
    # log("Expected script execution with constants (N, H, W, C):")
    # log(f" - Input image shape: [1, {IMAGE_SIZE}, {IMAGE_SIZE}, 1]")
    # log(f" - Latent y shape: [1, {H_LATENT}, {W_LATENT}, {C_MAIN}]")
    # log(f" - Latent z shape: [1, {H_HYPER}, {W_HYPER}, {C_HYPER}]")

    # 1. Load Model
    log(f"Loading graph from {xmodel}...")
    graph = xir.Graph.deserialize(xmodel)
    subgraph_map = identify_subgraphs(graph)
    log(f"Loading Entropy Models from {entropy_params_path}...")
    eb, gc = load_entropy_models_dpu(entropy_params_path)

    # 2. Create Runners
    log("Creating DPU Runners...")
    runners = {}
    for key, subgraph in subgraph_map.items():
        runners[key] = DPUSubgraphRunner(vart.Runner.create_runner(subgraph, "run"), subgraph, key)

    # 3. Load Data
    log(f"Loading a subset {subset} of data from {dataset_path}...")
    noisy, ground_truths = load_npy_data(dataset_path, subset)
    n_samples = len(noisy)
    log(f"Loaded {n_samples} samples.")
    # Metrics
    metric_list = ["bpp", "mse", "psnr"]
    tracker_noisy = MetricsTracker(metric_list)
    tracker_adam = MetricsTracker(metric_list)
    tracker_merlin = MetricsTracker(metric_list)

    start_time = time.time()

    # Store for visualization
    vis_indices = np.linspace(0, n_samples - 1, 5, dtype=int)
    vis_noisy = []
    vis_recon = []
    vis_adam = []
    vis_merlin = []

    # 4. Inference Loop
    print("\nStarting Inference Loop...")
    for i in range(n_samples):
        t0_sample = time.time()

        # --- Step 0. Prepare Input (CPU) ---
        noisy_i = noisy[i]  # [256, 256, 2]
        # SARDDCModule performs normalization on input, so we need to do it manually when feeding the model directly.
        noisy_sq = np.square(noisy_i)
        noisy_logI = np.log(noisy_sq + EPS)
        noisy_logI_norm = (noisy_logI - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)
        # g_a expects [1, 1, 256, 256] (NCHW)
        noisy_real = noisy_logI_norm[:, :, 0][np.newaxis, np.newaxis, :, :]
        noisy_imag = noisy_logI_norm[:, :, 1][np.newaxis, np.newaxis, :, :]
        print_tensor_stats(f"   Sample {i} - Noisy Real Input", noisy_real)
        print_tensor_stats(f"   Sample {i} - Noisy Imag Input", noisy_imag)

        # --- Step 1: Main Encoder (DPU g_a) ---
        y_real = runners["g_a"].run(noisy_real)
        y_imag = runners["g_a"].run(noisy_imag)

        # --- Step 2: Combine & Prep for Hyper (CPU) ---
        y = np.concatenate((y_real, y_imag), axis=-1)
        y_abs = np.abs(y)
        print_tensor_stats(f"   Sample {i} - Latent y (combined)", y)

        # --- Step 3: Hyper Encoder (DPU h_a) ---
        z = runners["h_a"].run(y_abs)
        print_tensor_stats(f"   Sample {i} - Latent z", z)

        # --- Step 4: Entropy Bottleneck / Quantization (CPU) ---
        z_hat, z_lik = eb.forward(z)
        print_tensor_stats(f"   Sample {i} - Latent z_hat", z_hat)
        print_tensor_stats(f"   Sample {i} - Latent z Likelihood", z_lik)

        # --- Step 5: Hyper Decoder (DPU h_s) ---
        scales = runners["h_s"].run(z_hat)
        print_tensor_stats(f"   Sample {i} - Scales", scales)

        # --- Step 6: Gaussian Conditional / Quantization (CPU) ---
        y_hat, y_lik = gc.forward(y, scales)
        print_tensor_stats(f"   Sample {i} - Latent y_hat", y_hat)
        print_tensor_stats(f"   Sample {i} - Latent y Likelihood", y_lik)

        # TMP: Calculate BPP
        likelihoods = {"y": y_lik, "z": z_lik}
        bpp_i = MetricsTracker.estimate_bpp(noisy_i[np.newaxis, ...], likelihoods)
        print(f"   Sample {i} - BPP : {bpp_i:.4f}")

        # Split y_hat back to real/imag for decoder
        y_hat_real = y_hat[..., :C_MAIN]
        y_hat_imag = y_hat[..., C_MAIN:]

        # --- Step 7: Main Decoder (DPU g_s) ---
        recon_real = runners["g_s"].run(y_hat_real)  # -> [1, 1, 256, 256]
        recon_imag = runners["g_s"].run(y_hat_imag)

        # --- Step 8: Reconstruct & Denormalize (CPU) ---
        # g_s output is typically NHWC (1, 256, 256, 1) on DPU
        recon_real = recon_real[0, :, :, 0]
        recon_imag = recon_imag[0, :, :, 0]
        recon = np.stack((recon_real, recon_imag), axis=-1)  # -> [256, 256, 2]
        print_tensor_stats(f"   Sample {i} - Reconstruction", recon)

        dt = time.time() - t0_sample
        if i % 10 == 0:
            log(f"Sample {i}: {dt * 1000:.1f}ms")
        print()

        # --- Denormalize for Metrics ---
        # recon is normalized Log-Intensity, we need Linear Amplitude.
        recon_denorm = recon * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = np.exp(recon_denorm)
        recon_linI = 0.5 * (np.square(recon_lin[..., 0]) + np.square(recon_lin[..., 1]))
        recon_linA = np.sqrt(recon_linI)[np.newaxis, ..., np.newaxis]  # [1, H, W, 1]
        print_tensor_stats(f"   Sample {i} - Reconstruction Linear Amplitude", recon_linA)

        # Metric Targets
        noisy_linI = noisy_sq[..., 0] + noisy_sq[..., 1]
        noisy_linA = np.sqrt(noisy_linI)[np.newaxis, ..., np.newaxis]  # [1, H, W, 1]
        print_tensor_stats(f"   Sample {i} - Noisy Linear Amplitude", noisy_linA)

        adam_linA = ground_truths["adam_noc"][i][np.newaxis, ...]
        merlin_linA = ground_truths["merlin"][i][np.newaxis, ...]
        print_tensor_stats(f"   Sample {i} - ADAM-NOC Linear Amplitude", adam_linA)
        print_tensor_stats(f"   Sample {i} - MERLIN Linear Amplitude", merlin_linA)
        print()
        # Update Trackers
        tracker_noisy.update(recon_linA, likelihoods, noisy_linA)
        tracker_adam.update(recon_linA, likelihoods, adam_linA)
        tracker_merlin.update(recon_linA, likelihoods, merlin_linA)

        # Store for Viz, all visualization must be in log-Intensity format
        if i in vis_indices:
            noisy_logI = np.log(noisy_sq[..., 0] + noisy_sq[..., 1] + EPS)
            recon_logI = np.log(recon_linI + EPS)
            adam_logI = np.log(np.square(adam_linA[0, ...]) + EPS).squeeze()
            merlin_logI = np.log(np.square(merlin_linA[0, ...]) + EPS).squeeze()

            vis_noisy.append(noisy_logI)
            vis_recon.append(recon_logI)
            vis_adam.append(adam_logI)
            vis_merlin.append(merlin_logI)

    total_time = time.time() - start_time
    log(
        f"Inference Loop Finished in {total_time:.2f}s ({total_time / n_samples * 1000:.1f}ms/sample)"
    )

    # Save Metrics
    log("\nFinal Results:")
    summary = {}
    for name, tracker in [
        ("Noisy", tracker_noisy),
        ("ADAM", tracker_adam),
        ("MERLIN", tracker_merlin),
    ]:
        res = tracker.summary()
        log(f"  Vs {name}:")
        for k, v in res.items():
            log(f"    {k}: {v:.4f}")
        summary[name] = res

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Save Visualization arrays
    vis_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(vis_dir, exist_ok=True)
    log(f"Saving reconstruction (log-I) arrays to {vis_dir}...")
    np.save(os.path.join(vis_dir, "vis_noisy.npy"), np.array(vis_noisy))
    np.save(os.path.join(vis_dir, "vis_recon.npy"), np.array(vis_recon))
    np.save(os.path.join(vis_dir, "vis_adam.npy"), np.array(vis_adam))
    np.save(os.path.join(vis_dir, "vis_merlin.npy"), np.array(vis_merlin))

    # Try Visualization (if matplotlib exists)
    log("Attempting Visualization...")
    visualize_patches(
        np.array(vis_noisy),
        np.array(vis_recon),
        np.array(vis_adam),
        np.array(vis_merlin),
        os.path.join(output_dir, "visual_comparison.png"),
    )
    log("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmodel", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--subset", type=int, default=100)
    args = parser.parse_args()

    run_hybrid_inference(args.xmodel, args.data, args.subset)
