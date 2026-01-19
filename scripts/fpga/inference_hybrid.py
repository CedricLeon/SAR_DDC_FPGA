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
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from inference_utils import (
    AMP_MAX,
    AMP_MIN,
    MetricsTracker,
    visualize_patches,
)

# Mocking Vitis-AI modules for linting if not on target
try:
    import vart
    import xir
except ImportError:
    print("WARNING: Vitis-AI libraries (xir, vart) not found. This script will fail if run.")
    xir = None
    vart = None


# -----------------------------------------------------------------------------
# NUMPY IMPLEMENTATION OF ENTROPY MATH (CPU)
# -----------------------------------------------------------------------------


def simple_quantize(x: np.ndarray) -> np.ndarray:
    """Simulate uniform quantization (round)."""
    return np.round(x)


def gaussian_likelihood(
    inputs: np.ndarray, scales: np.ndarray, means: np.ndarray = None
) -> np.ndarray:
    """Compute likelihoods for Gaussian Conditional using Numpy. Mathematical equivalent of
    CompressAI's GaussianConditional._likelihood.

    L = 0.5 * (erfc((0.5 - (x-mu))/scale * c) - erfc((-0.5 - (x-mu))/scale * c))
    where c = -1/sqrt(2)
    """
    if means is not None:
        values = inputs - means
    else:
        values = inputs

    # Standard deviation constant: -1 / sqrt(2)
    # We try scipy.special.erfc, else fallback to math.erfc (vectorized)
    try:
        from scipy.special import erfc
    except ImportError:
        # Fallback: math.erfc is part of stdlib since 3.2
        # But it takes scalars. We must vectorize.
        import math

        erfc = np.vectorize(math.erfc)

    # We need a lower bound on scales to avoid div by zero, similar to LowerBound
    # Assume scales already passed through LowerBound or do it here
    scales = np.maximum(scales, 0.11)  # Default scale bound

    values = np.abs(values)
    const = -(2**-0.5)

    # Vectorized erfc
    upper = 0.5 * erfc(const * (0.5 - values) / scales)
    lower = 0.5 * erfc(const * (-0.5 - values) / scales)

    return upper - lower


# -----------------------------------------------------------------------------
# DPU RUNNER HELPER
# -----------------------------------------------------------------------------


class DPUSubgraphRunner:
    """Helper to wrap a single DPU subgraph runner."""

    def __init__(self, runner: "vart.Runner", subgraph: "xir.Subgraph", name: str):
        self.runner = runner
        self.name = name

        # IO Shapes
        self.input_tensors = runner.get_input_tensors()
        self.output_tensors = runner.get_output_tensors()

        # We assume 1 input and 1 output for simplicity based on our wrapper,
        # but robust code handles list.
        self.input_shape = tuple(self.input_tensors[0].dims)
        self.output_shape = tuple(self.output_tensors[0].dims)

        # Scaling factors (Float -> Int8 -> Float)
        # fix_point attribute returns the position of the point.
        # scale = 2^fix_point? Or 2^-fix_point?
        # Usually: real = integer * 2^(-fix_point)
        # Wait, get_attr("fix_point") usually returns 'p'.
        # val_float = val_int * (2 ** -p)
        # val_int = val_float * (2 ** p)

        inp_fix = self.input_tensors[0].get_attr("fix_point")
        out_fix = self.output_tensors[0].get_attr("fix_point")

        self.input_scale = 2.0**inp_fix
        self.output_scale = 2.0 ** (-out_fix)

        print(
            f"[{name}] In: {self.input_shape} (scale={self.input_scale}), Out: {self.output_shape} (scale={self.output_scale})"
        )

    def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on a batch of data.

        input_data: Float numpy array matching input shape (NCHW or NHWC).
        """
        # 1. Quantize Input (Float -> Int8)
        # Note: VART expects NHWC usually!
        # But PyTorch is NCHW.
        # The xir graph usually retains the shape from compilation.
        # Check dims. If self.input_shape is [1, 256, 256, 1], it is NHWC.
        # If input_data is [1, 1, 256, 256], we need transpose.

        expected_dims = len(self.input_shape)
        if expected_dims == 4:
            # Heuristic check for NCHW vs NHWC
            # DPU usually HWC.
            if input_data.shape != self.input_shape:
                # Try simple transpose (N, H, W, C) from (N, C, H, W)
                # assuming input_data is NCHW
                input_data = input_data.transpose(0, 2, 3, 1)

        input_int8 = (input_data * self.input_scale).astype(np.int8)

        # 2. Prepare Buffers
        # use C-order
        input_buffer = np.ascontiguousarray(input_int8)
        output_buffer = np.empty(self.output_shape, dtype=np.int8, order="C")

        # 3. Execute (job_id is returned)
        # run inputs: list of numpy arrays
        # run outputs: list of numpy arrays
        job_id = self.runner.execute_async([input_buffer], [output_buffer])
        self.runner.wait(job_id)

        # 4. Dequantize Output (Int8 -> Float)
        output_float = output_buffer.astype(np.float32) * self.output_scale

        # 5. Transpose back to NCHW if needed
        if expected_dims == 4:
            # (N, H, W, C) -> (N, C, H, W)
            output_float = output_float.transpose(0, 3, 1, 2)

        return output_float


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------


def identify_subgraphs(graph: "xir.Graph") -> Dict[str, "xir.Subgraph"]:
    """Identify which subgraph corresponds to g_a, h_a, h_s, g_s based on shapes.

    (This is a heuristic and might need adjustment based on real compiler names)
    """
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraphs = [
        s for s in subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"
    ]

    mapping = {}

    for sg in dpu_subgraphs:
        # Get input tensor shapes
        inputs = sg.get_input_tensors()
        if not inputs:
            continue
        shape = tuple(list(inputs)[0].dims)  # (B, H, W, C) or similar

        # Shapes based on 256x256 input
        # Note: DPU shapes are typically NHWC

        # g_a: In (1, 256, 256, 1) -> Out (1, 16, 16, 128)
        if shape[1:3] == (256, 256) and shape[3] == 1:
            mapping["g_a"] = sg

        # h_a: In (1, 16, 16, 256) -> Out (1, 2, 2, 256)  (Abs(y) is 256 channels)
        elif shape[1:3] == (16, 16) and shape[3] == 256:
            mapping["h_a"] = sg

        # h_s: In (1, 2, 2, 256) -> Out (1, 16, 16, 256) (Scales)
        elif shape[1:3] == (2, 2) and shape[3] == 256:
            mapping["h_s"] = sg

        # g_s: In (1, 16, 16, 128) -> Out (1, 256, 256, 1) (One channel decode)
        elif shape[1:3] == (16, 16) and shape[3] == 128:
            mapping["g_s"] = sg

    return mapping


def load_npy_data(
    dataset_path: str, subset_len: int = 100
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load data from NPY file.

    Matches inference.py logic.
    """
    data = np.load(dataset_path)
    if subset_len > 0 and subset_len < data.shape[0]:
        data = data[:subset_len]
    noisy = data[:, :, :, 0:2]  # [B, H, W, 2] real and imag channels

    # Check if GT exists in channels 2 and 3
    ground_truths = {}
    if data.shape[-1] >= 3:
        ground_truths["adam_noc"] = data[:, :, :, 2:3]
    if data.shape[-1] >= 4:
        ground_truths["merlin"] = data[:, :, :, 3:4]

    return noisy, ground_truths


def run_hybrid_inference(xmodel: str, dataset_path: str, subset: int = 100):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = xmodel.split("/")[-1].replace(".xmodel", "")
    output_dir = f"results/inference_{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "inference.log")

    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log(f"Starting Hybrid Inference at {timestamp}")
    log(f"Model: {xmodel}")
    log(f"Data: {dataset_path}")
    log(f"Subset: {subset}")

    # 1. Load Model
    log(f"Loading graph from {xmodel}...")
    graph = xir.Graph.deserialize(xmodel)
    subgraph_map = identify_subgraphs(graph)

    required_keys = ["g_a", "h_a", "h_s", "g_s"]
    missing_keys = [k for k in required_keys if k not in subgraph_map]
    if missing_keys:
        log(f"ERROR: Could not find subgraphs for: {missing_keys}")
        log(f"Found mapped subgraphs: {list(subgraph_map.keys())}")

        # Debug: Print all DPU subgraphs to help diagnosis
        log("\n--- Debug: All DPU Subgraphs ---")
        root = graph.get_root_subgraph()
        for sg in root.toposort_child_subgraph():
            if sg.has_attr("device") and sg.get_attr("device") == "DPU":
                inputs = sg.get_input_tensors()
                outputs = sg.get_output_tensors()
                in_shape = tuple(inputs[0].dims) if inputs else "None"
                out_shape = tuple(outputs[0].dims) if outputs else "None"
                log(f"Subgraph: {sg.get_name()}")
                log(f"  Input:  {in_shape}")
                log(f"  Output: {out_shape}")
        log("--------------------------------\n")
        return

    # 2. Create Runners
    log("Creating DPU Runners...")
    runners = {}
    for k, sg in subgraph_map.items():
        runners[k] = DPUSubgraphRunner(vart.Runner.create_runner(sg, "run"), sg, k)

    # 3. Load Data
    log(f"Loading data from {dataset_path}...")
    noisy, ground_truths = load_npy_data(dataset_path, subset)
    n_samples = len(noisy)
    log(f"Loaded {n_samples} samples.")
    has_gt = "merlin" in ground_truths and "adam_noc" in ground_truths

    # Metrics
    metric_list = ["bpp", "mse", "psnr", "merlin"]
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
    for i in range(n_samples):
        # x_complex: [256, 256, 2]
        x_complex = noisy[i]

        t0_sample = time.time()

        # 1. Normalize Input (Important! Model expects Normalized Log-Intensity)
        # SARDDCModule performs normalization on input.
        # Logic: x_in = (log(|x|^2) - 2*min) / (2*max - 2*min)
        # Note: x_complex is Raw Complex.
        x_sq = np.square(x_complex)
        x_log = np.log(x_sq + 1e-2)  # Epsilon
        x_norm = (x_log - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)

        # --- Channel Split (CPU) ---
        # g_a expects [1, 1, 256, 256] (NCHW)
        x_real_norm = x_norm[:, :, 0][np.newaxis, np.newaxis, :, :]
        x_imag_norm = x_norm[:, :, 1][np.newaxis, np.newaxis, :, :]

        # --- Step 1: Main Encoder (DPU g_a) ---
        y_real = runners["g_a"].run(x_real_norm)  # -> [1, 128, 16, 16]
        y_imag = runners["g_a"].run(x_imag_norm)

        # --- Step 2: Combine & Prep for Hyper (CPU) ---
        # y: [1, 256, 16, 16]
        y = np.concatenate((y_real, y_imag), axis=1)
        y_abs = np.abs(y)

        # --- Step 3: Hyper Encoder (DPU h_a) ---
        z = runners["h_a"].run(y_abs)  # -> [1, 256, 2, 2]

        # --- Step 4: Entropy Bottleneck / Quantization (CPU) ---
        z_hat = simple_quantize(z)

        # --- Step 5: Hyper Decoder (DPU h_s) ---
        scales = runners["h_s"].run(z_hat)  # -> [1, 256, 16, 16]

        # --- Step 6: Gaussian Conditional / Quantization (CPU) ---
        scales_clamped = np.maximum(scales, 0.11)
        y_hat = simple_quantize(y)

        # Calculate BPP
        likelihoods_y = gaussian_likelihood(y_hat, scales_clamped)
        likelihoods = {"y": likelihoods_y}

        # Split y_hat back to real/imag for decoder
        y_hat_real = y_hat[:, :128, :, :]
        y_hat_imag = y_hat[:, 128:, :, :]

        # --- Step 7: Main Decoder (DPU g_s) ---
        x_hat_real = runners["g_s"].run(y_hat_real)  # -> [1, 1, 256, 256]
        x_hat_imag = runners["g_s"].run(y_hat_imag)

        # --- Step 8: Reconstruct & Denormalize (CPU) ---
        # Output is NCHW [1, 1, 256, 256].
        x_hat_real = x_hat_real[0, 0]  # [256, 256]
        x_hat_imag = x_hat_imag[0, 0]
        x_hat_complex_norm = np.stack((x_hat_real, x_hat_imag), axis=-1)  # [H, W, 2]

        dt = time.time() - t0_sample
        if i % 10 == 0:
            log(f"Sample {i}: {dt * 1000:.1f}ms")

        # --- Denormalize for Metrics ---
        # x_hat is Normalized Log-Intensity.
        # We need Linear Amplitude.
        x_hat_log = x_hat_complex_norm * (2 * AMP_MAX - 2 * AMP_MIN) + 2 * AMP_MIN
        x_hat_sq = np.exp(x_hat_log)

        # Intensity I = 0.5 * (Real^2 + Imag^2) (Average Reconstruction)
        recon_I = 0.5 * (x_hat_sq[..., 0] + x_hat_sq[..., 1])
        recon_linA = np.sqrt(recon_I)[np.newaxis, ..., np.newaxis]  # [1, H, W, 1]

        # Metric Targets
        noisy_I = np.square(x_complex[..., 0]) + np.square(x_complex[..., 1])
        noisy_linA = np.sqrt(noisy_I)[np.newaxis, ..., np.newaxis]

        if has_gt:
            adam_val = ground_truths["adam_noc"][i][np.newaxis, ...]
            merlin_val = ground_truths["merlin"][i][np.newaxis, ...]
        else:
            adam_val = np.zeros_like(recon_linA)
            merlin_val = np.zeros_like(recon_linA)

        # Update Trackers
        tracker_noisy.update(recon_linA, likelihoods, noisy_linA)
        if has_gt:
            tracker_adam.update(recon_linA, likelihoods, adam_val)
            tracker_merlin.update(recon_linA, likelihoods, merlin_val)

        # Store for Viz
        if i in vis_indices:
            vis_noisy.append(x_complex)  # Raw complex
            # For recon, we store a pseudo-complex where abs() matches recon_linA
            # Or just pass magnitude in Real channel and 0 in Imag?
            # visualize_patches takes complex [H,W,2] and does sq(real)+sq(imag).
            # We have recon_linA (sqrt(I)).
            # Make a dummy complex where real=recon_linA, imag=0.
            # Then sq(real)+sq(imag) = I.
            dummy_complex = np.zeros_like(x_complex)
            dummy_complex[..., 0] = recon_linA[0, ..., 0]
            vis_recon.append(dummy_complex)

            vis_adam.append(adam_val[0])
            vis_merlin.append(merlin_val[0])

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

    # Save Visualization
    # log("Saving Visualization...")
    # Save Visualization Arrays (for offline plotting)
    vis_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(vis_dir, exist_ok=True)
    log(f"Saving reconstruction arrays to {vis_dir}...")
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
