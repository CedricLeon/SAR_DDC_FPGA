#!/usr/bin/env python3
"""Hybrid Inference Script for Vitis-AI DPU + CPU Orchestration.

This script manages the execution of a "split" model where:
- Neural Network subgraphs (g_a, h_a, h_s, g_s) run on the DPU.
- Entropy operations (quantization, likelihoods) run on the CPU (using Numpy).

Usage (/!\\ Only on FPGA /!\\):
    python3 inference_hybrid.py --xmodel model.xmodel --data test.npy --params entropy_params.npz
"""

import argparse
import json
import sys
import time
from datetime import datetime

# import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import vart  # type: ignore
    import xir  # type: ignore
except ImportError:
    print("ERROR: Vitis-AI libraries (vart, xir) not found")
    sys.exit(1)

from entropy_models_dpu import (
    EntropyBottleneckDPU,
    GaussianConditionalDPU,
    load_entropy_models_dpu,
)
from inference_utils import (
    AMP_MAX,
    AMP_MIN,
    EPS,
    DPUSubgraphRunner,
    MetricsTracker,
    extract_patches,
    pad_to_multiple,
    print_tensor_stats,
    reconstruct_from_patches,
)

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIG
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

log_file = ""


def log(msg):
    """Manual logging function."""
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------


def identify_subgraphs(graph: xir.Graph) -> Dict[str, xir.Subgraph]:
    """Identify which subgraph corresponds to g_a, h_a, h_s, g_s based on shapes."""
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
    required = ["g_a", "h_a", "h_s", "g_s"]
    missing = [k for k in required if k not in mapping]
    if missing:
        raise ValueError(
            f"Could not identify subgraphs for: {missing}. Found: {list(mapping.keys())}"
        )

    return mapping


def load_npy_test_set(
    dataset_path: Path, subset_len: int = 100
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


def process_single_tile(
    noisy: np.ndarray,  # [256, 256, 2] Raw Complex
    runners: Dict[str, DPUSubgraphRunner],
    eb: EntropyBottleneckDPU,
    gc: GaussianConditionalDPU,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run full inference on a single 256x256 tile.

    Returns:
        recon_norm_logI (np.ndarray): [256, 256, 2] (Normalized Log Intensity)
        likelihoods (Dict[str, np.ndarray]): Dict of likelihood arrays
    """
    # --- Step 0. Prepare Input (CPU) ---
    # SARDDCModule performs normalization on input, so we need to do it manually when feeding the model directly.
    noisy_sq = np.square(noisy)
    noisy_logI = np.log(noisy_sq + EPS)
    noisy_logI_norm = (noisy_logI - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)
    # g_a expects [1, 1, 256, 256] (NCHW)
    noisy_real = noisy_logI_norm[:, :, 0][np.newaxis, np.newaxis, :, :]
    noisy_imag = noisy_logI_norm[:, :, 1][np.newaxis, np.newaxis, :, :]
    if verbose:
        print_tensor_stats("   - Noisy Real Input", noisy_real)
        print_tensor_stats("   - Noisy Imag Input", noisy_imag)

    # --- Step 1: Main Encoder (DPU g_a) ---
    y_real = runners["g_a"].run(noisy_real)
    y_imag = runners["g_a"].run(noisy_imag)

    # --- Step 2: Combine & Prep for Hyper (CPU) ---
    y = np.concatenate((y_real, y_imag), axis=-1)
    y_abs = np.abs(y)
    if verbose:
        print_tensor_stats("   - Latent y (combined)", y)

    # --- Step 3: Hyper Encoder (DPU h_a) ---
    z = runners["h_a"].run(y_abs)
    if verbose:
        print_tensor_stats("   - Latent z", z)
    # --- Step 4: Entropy Bottleneck / Quantization (CPU) ---
    z_hat, z_lik = eb.forward(z)
    if verbose:
        print_tensor_stats("   - Latent z_hat", z_hat)
        print_tensor_stats("   - Latent z Likelihood", z_lik)

    # --- Step 5: Hyper Decoder (DPU h_s) ---
    scales = runners["h_s"].run(z_hat)
    if verbose:
        print_tensor_stats("   - Scales", scales)
    # --- Step 6: Gaussian Conditional / Quantization (CPU) ---
    y_hat, y_lik = gc.forward(y, scales)
    if verbose:
        print_tensor_stats("   - Latent y_hat", y_hat)
        print_tensor_stats("   - Latent y Likelihood", y_lik)

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
    likelihoods = {"y": y_lik, "z": z_lik}
    if verbose:
        print_tensor_stats("   - Reconstruction", recon)

    return recon, likelihoods
    #################################################


def run_hybrid_inference(
    xmodel: Path, dataset_path: Path, subset: int = 100, verbose: bool = False
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ---- Paths and Logging ----
    model_name_and_timestamp = xmodel.parent.name
    output_dir = Path(f"results/FPGA_inference_{model_name_and_timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    global log_file
    log_file = output_dir / "inference.log"

    log(f"Starting Hybrid Inference at {timestamp}.")

    # 1. Load Model
    entropy_params_path = xmodel.parent / "entropy_params.npz"
    log(f"Loading graph from {xmodel}...")
    graph = xir.Graph.deserialize(str(xmodel))
    subgraph_map = identify_subgraphs(graph)
    log(f"Loading Entropy Models from {entropy_params_path}...")
    eb, gc = load_entropy_models_dpu(entropy_params_path)

    # Copy train_config.yaml if available
    train_config_src = xmodel.parent / "train_config.yaml"
    if train_config_src.exists():
        import shutil

        shutil.copy(train_config_src, output_dir / "train_config.yaml")
        log(f"Copied train_config.yaml to {output_dir}")

    # 2. Create Runners
    log("Creating DPU Runners...")
    runners = {}
    for key, subgraph in subgraph_map.items():
        runners[key] = DPUSubgraphRunner(vart.Runner.create_runner(subgraph, "run"), subgraph, key)

    # 3. Load Data
    log(f"Loading a subset {subset} of data from {dataset_path}...")
    noisy, ground_truths = load_npy_test_set(dataset_path, subset)
    n_samples = len(noisy)
    log(f"Loaded {n_samples} samples.")
    # Metrics
    metric_list = ["bpp", "mse", "psnr"]
    tracker_noisy = MetricsTracker(metric_list)
    tracker_adam = MetricsTracker(metric_list)
    tracker_merlin = MetricsTracker(metric_list)

    start_time = time.time()

    # Store for visualization
    vis_indices_test_set = np.linspace(0, n_samples - 1, 5, dtype=int)
    vis_noisy_test_set = []
    vis_recon_test_set = []
    vis_adam_test_set = []
    vis_merlin_test_set = []

    # 4. Inference Loop
    log("\nStarting Inference Loop...")
    for i in range(n_samples):
        t0_sample = time.time()
        if verbose:
            log(f"\n--- Sample {i} ---")
        # --- Hybrid Inference Call ---
        recon_norm_logI, likelihoods = process_single_tile(
            noisy[i], runners, eb, gc, verbose=verbose
        )

        dt = time.time() - t0_sample
        if i % 10 == 0:
            log(f"Sample {i} processed in {dt * 1000:.1f}ms")

        # --- Denormalize for Metrics ---
        recon_logI = recon_norm_logI * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_linI = np.exp(recon_logI)
        recon_linI = 0.5 * (np.square(recon_linI[..., 0]) + np.square(recon_linI[..., 1]))
        recon_linA = np.sqrt(recon_linI)[np.newaxis, ..., np.newaxis]  # [1, H, W, 1]

        # Metric Targets
        noisy_sq = np.square(noisy[i])
        noisy_linI = noisy_sq[..., 0] + noisy_sq[..., 1]
        noisy_linA = np.sqrt(noisy_linI)[np.newaxis, ..., np.newaxis]  # [1, H, W, 1]

        adam_linA = ground_truths["adam_noc"][i][np.newaxis, ...]
        merlin_linA = ground_truths["merlin"][i][np.newaxis, ...]
        if verbose:
            print()
            print_tensor_stats(f"   Sample {i} - Reconstruction Linear Amplitude", recon_linA)
            print_tensor_stats(f"   Sample {i} - Noisy Linear Amplitude", noisy_linA)
            print_tensor_stats(f"   Sample {i} - ADAM-NOC Linear Amplitude", adam_linA)
            print_tensor_stats(f"   Sample {i} - MERLIN Linear Amplitude", merlin_linA)
            print()
        # Update Trackers
        tracker_noisy.update(recon_linA, likelihoods, noisy_linA)
        tracker_adam.update(recon_linA, likelihoods, adam_linA)
        tracker_merlin.update(recon_linA, likelihoods, merlin_linA)

        # Store for Viz, all visualization must be in log-Intensity format
        if i in vis_indices_test_set:
            noisy_logI = np.log(noisy_sq[..., 0] + noisy_sq[..., 1] + EPS)
            recon_logI = np.log(recon_linI + EPS)
            adam_logI = np.log(np.square(adam_linA[0, ...]) + EPS).squeeze()
            merlin_logI = np.log(np.square(merlin_linA[0, ...]) + EPS).squeeze()

            vis_noisy_test_set.append(noisy_logI)
            vis_recon_test_set.append(recon_logI)
            vis_adam_test_set.append(adam_logI)
            vis_merlin_test_set.append(merlin_logI)

    total_time = time.time() - start_time
    log(
        f"Inference Loop Finished in {total_time:.2f}s ({total_time / n_samples * 1000:.1f}ms/sample)"
    )

    # Save Metrics
    log("\n ----- Test set Results -----")
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

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Save Visualization arrays
    vis_dir = output_dir / "reconstructions_test_set"
    vis_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving reconstruction (log-I) arrays to {vis_dir}...")
    np.save(vis_dir / "vis_noisy.npy", np.array(vis_noisy_test_set))
    np.save(vis_dir / "vis_recon.npy", np.array(vis_recon_test_set))
    np.save(vis_dir / "vis_adam.npy", np.array(vis_adam_test_set))
    np.save(vis_dir / "vis_merlin.npy", np.array(vis_merlin_test_set))

    # --- 2. Large Tile Evaluation (Folder Scanning) ---
    root_dir = dataset_path.parent
    log(f"\nScanning {root_dir} for Large Tiles")
    # For each folder in root_dir
    for tile_path in root_dir.iterdir():
        if not tile_path.is_dir():
            continue
        # Check for required files
        noisy_tile_path = tile_path / "raw_input_symmetrized.npy"
        if not noisy_tile_path.exists():
            log(f"Skipping {tile_path}, no raw_input_symmetrized.npy found.")
            continue
        log(f"\n--- Running Large Tile Inference on {tile_path.name} ---")
        start_tile = time.time()
        # Load Data
        noisy_tile = np.load(noisy_tile_path)  # [H, W, 2]
        if verbose:
            print_tensor_stats(f" - Loaded Noisy Tile ({tile_path.name})", noisy_tile)

        # Pad & Patch
        noisy_tile, (h_pad, w_pad) = pad_to_multiple(noisy_tile, IMAGE_SIZE)
        noisy_patches = extract_patches(noisy_tile, IMAGE_SIZE)  # [N, 256, 256, 2]
        recon_patches = []
        tile_bpp = 0

        for i in range(len(noisy_patches)):
            recon_norm_logI, likelihoods = process_single_tile(
                noisy_patches[i], runners, eb, gc, verbose=False
            )
            recon_patches.append(recon_norm_logI)
            patch_bpp = MetricsTracker.estimate_bpp(noisy_patches[i][np.newaxis, ...], likelihoods)
            tile_bpp += patch_bpp
        tile_bpp /= len(noisy_patches)

        # Stitch (Normalized LogI Domain)
        recon_norm_logI = reconstruct_from_patches(
            np.array(recon_patches), (noisy_tile.shape[0], noisy_tile.shape[1]), IMAGE_SIZE
        )
        if h_pad > 0 or w_pad > 0:
            recon_norm_logI = recon_norm_logI[
                : noisy_tile.shape[0] - h_pad, : noisy_tile.shape[1] - w_pad
            ]

        # ----- Metrics -----
        # Recon to LinA
        recon_logI = recon_norm_logI * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_linI = np.exp(recon_logI)
        recon_linI = 0.5 * (np.square(recon_linI[..., 0]) + np.square(recon_linI[..., 1]))
        recon_linA = np.sqrt(recon_linI)
        if verbose:
            print_tensor_stats(f" - recon_linA ({tile_path.name})", recon_linA)
        # Noisy to linA
        noisy_sq = np.square(noisy_tile)
        noisy_linI = noisy_sq[..., 0] + noisy_sq[..., 1]
        noisy_linA = np.sqrt(noisy_linI)
        # Metrics vs Noisy
        tile_metrics = {"bpp": tile_bpp}
        tile_metrics["mse_noisy"] = MetricsTracker.compute_mse(recon_linA, noisy_linA)
        tile_metrics["psnr_noisy"] = MetricsTracker.compute_psnr(recon_linA, noisy_linA)

        # Metrics vs GT
        denoised_references = {"merlin": "GT_MERLIN_linA.npy", "adam": "GT_ADAM-NOC_linA.npy"}
        for ref_name, ref_filename in denoised_references.items():
            ref_path = tile_path / ref_filename
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"  Reference GT file {ref_path} not found for metrics comparison."
                )
            ref_linA = np.load(ref_path)  # [H, W]
            if verbose:
                print_tensor_stats(f" - {ref_name} linA ({tile_path.name})", ref_linA)

            tile_metrics[f"psnr_{ref_name}"] = MetricsTracker.compute_psnr(recon_linA, ref_linA)
            tile_metrics[f"mse_{ref_name}"] = MetricsTracker.compute_mse(recon_linA, ref_linA)

        log(f"Metrics: {tile_metrics}")
        log(f"Time: {time.time() - start_tile:.2f}s")

        # Save
        save_path = output_dir / f"{tile_path.name}_recon_linA.npy"
        np.save(save_path, recon_linA)

        meta_path = output_dir / f"{tile_path.name}_metrics.json"
        with open(meta_path, "w") as f:
            json.dump(tile_metrics, f, indent=4)
        log(f"Finished evaluating {tile_path.name}.")
    log(f"Evaluation Complete, results stored in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmodel", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    xmodel_path = Path(args.xmodel)
    if not xmodel_path.exists():
        log(f"Error: XModel file {xmodel_path} does not exist.")
        sys.exit(1)
    data_path = Path(args.data)
    if not data_path.exists():
        log(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    run_hybrid_inference(xmodel_path, data_path, args.subset, args.verbose)
