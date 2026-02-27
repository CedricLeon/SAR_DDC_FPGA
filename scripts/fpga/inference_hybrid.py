#!/usr/bin/env python3
"""Hybrid Inference Script for Vitis-AI DPU + CPU Orchestration.

This script manages the execution of a "split" model where:
- Neural Network subgraphs (g_a, h_a, h_s, g_s) run on the DPU.
- Entropy operations (compression and decompression) run on the CPU (using Numpy and C++ implementation of rANS).

Usage (/!\\ Only on FPGA /!\\):
    python3 inference_hybrid.py --xmodel model.xmodel --data test.npy --params entropy_params.npz
"""

import argparse
import json
import shutil
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

from entropy_models_inference import (
    EntropyBottleneck,
    GaussianConditional,
)
from inference_utils import (
    AMP_MAX,
    AMP_MIN,
    EPS,
    MetricsTracker,
    display_manifest,
    patch_infer_fpga,
    print_tensor_stats,
)

log_file = ""


def log(msg):
    """Manual logging function."""
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


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


def identify_subgraphs(graph: xir.Graph, meta_path: Path) -> Dict[str, xir.Subgraph]:
    """Uses meta.json to identify which subgraph corresponds to g_a, h_a, h_s, g_s."""
    mapping = {}

    # 1. Try to load meta.json for precise name mapping
    meta_path = xmodel_path.parent / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta file not found at {meta_path}. Cannot identify subgraphs without it."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    kernels = meta.get("kernel", [])
    log(f"Loading subgraph mapping from meta.json: found {len(kernels)} kernels.")

    # Map based on substrings in the kernel name
    name_to_role = {}
    for k_name in kernels:
        if "g_a" in k_name:
            name_to_role[k_name] = "g_a"
        elif "g_s" in k_name:
            name_to_role[k_name] = "g_s"
        elif "h_a" in k_name:
            name_to_role[k_name] = "h_a"
        elif "h_s" in k_name:
            name_to_role[k_name] = "h_s"
        else:
            log(f"WARNING: Unrecognized kernel name in meta.json: {k_name}")

    # Find the actual subgraphs in the graph object
    root = graph.get_root_subgraph()
    for sg in root.toposort_child_subgraph():
        if sg.get_name() in name_to_role:
            role = name_to_role[sg.get_name()]
            mapping[role] = sg
            log(f"Mapped {role} -> {sg.get_name()} (via meta.json)")

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
    eb: EntropyBottleneck,
    gc: GaussianConditional,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """Run full inference on a single 256x256 tile.

    Returns:
        recon_norm_logI (np.ndarray): [256, 256, 2] (Normalized Log Intensity)
        num_bytes (int): Total bytes used to compress this tile
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

    # --- Step 4: Entropy Bottleneck /  Compression (CPU) ---
    # z is [1, 16, 16, 256] (NHWC)
    z_strings = eb.compress(z)
    z_bytes = sum(len(s) for s in z_strings)

    # Decompress to get z_hat for hyper-decoder
    # z_hat from decompress is float, suitable for DPU input if matched correctly
    # Note: C++ decompress returns flattened list, wrapper handles it?
    # Wrapper returns flat list? No, let's check wrapper.
    # The wrapper's decompress takes 'shape' and returns ndarray. But shape needs to be (H, W).
    z_hat = eb.decompress(z_strings, (z.shape[1], z.shape[2]))

    if verbose:
        print_tensor_stats("   - Latent z_hat", z_hat)
        log(f"   - z_bytes: {z_bytes}")

    # --- Step 5: Hyper Decoder (DPU h_s) ---
    scales = runners["h_s"].run(z_hat)
    if verbose:
        print_tensor_stats("   - Scales", scales)

    # --- Step 6: Gaussian Conditional /  Compression (CPU) ---
    # y is [1, 16, 16, 128*2]
    # We need 'means' dummy (usually 0)
    means = np.zeros_like(y)
    y_strings = gc.compress(y, scales, means)
    y_bytes = sum(len(s) for s in y_strings)

    y_hat = gc.decompress(y_strings, scales, means)

    if verbose:
        print_tensor_stats("   - Latent y_hat", y_hat)
        log(f"   - y_bytes: {y_bytes}")

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

    total_bytes = z_bytes + y_bytes
    if verbose:
        print_tensor_stats("   - Reconstruction", recon)
        log(f"   - Total Bytes: {total_bytes}")

    return recon, total_bytes
    #################################################


def run_hybrid_inference(
    xmodel_path: Path, dataset_path: Path, subset: int = 100, verbose: bool = False
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ----- Paths and Logging -----
    output_dir = xmodel_path.parent / "results"

    # Remove previous results to avoid contamination
    if output_dir.exists():
        print(f"[INFO] Cleaning previous results directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create fresh output directory and log file
    output_dir.mkdir(parents=True, exist_ok=True)
    global log_file
    log_file = output_dir / "inference.log"

    # Store inference metadata in a new manifest inside results
    manifest_path = xmodel_path.parent / "manifest.json"
    model_metadata = {"evaluated_at": timestamp}
    if manifest_path.exists():
        with open(manifest_path) as f:
            build_manifest = json.load(f)
            model_metadata["model_run_name"] = build_manifest.get("model_name", "Unknown")
            model_metadata["model_compiled_at"] = build_manifest.get("compiled_at", "Unknown")
    with open(output_dir / "inference_meta.json", "w") as f:
        json.dump(model_metadata, f, indent=4)

    # ----- Inference -----
    log(f"Starting Hybrid Inference at {timestamp}.")

    # 1. Load Model
    entropy_params_path = xmodel_path.parent / "entropy_params.npz"
    log(f"Loading graph from {xmodel_path}...")
    graph = xir.Graph.deserialize(str(xmodel_path))
    subgraph_map = identify_subgraphs(graph, xmodel_path.parent / "meta.json")
    log(f"Loading Entropy Models (Real/Interface) from {entropy_params_path}...")
    data = np.load(entropy_params_path)
    eb_channels = data["eb_cdf_length"].shape[0]
    eb = EntropyBottleneck(
        channels=eb_channels,
        quantized_cdf=data["eb_quantized_cdf"],
        cdf_length=data["eb_cdf_length"],
        offset=data["eb_offset"],
        medians=data["eb_medians"] if "eb_medians" in data else None,
    )
    gc = GaussianConditional(
        scale_table=data["gc_scale_table"],
        quantized_cdf=data["gc_quantized_cdf"],
        cdf_length=data["gc_cdf_length"],
        offset=data["gc_offset"],
    )

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
        recon_norm_logI, num_bytes = process_single_tile(noisy[i], runners, eb, gc, verbose=False)

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
        tracker_noisy.update(recon_linA, noisy_linA, num_bytes)
        tracker_adam.update(recon_linA, adam_linA, num_bytes)
        tracker_merlin.update(recon_linA, merlin_linA, num_bytes)

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
        noisy_file_path = tile_path / "sym_Noisy.npy"
        if not noisy_file_path.exists():
            log(f"Skipping {tile_path}, no sym_Noisy.npy found.")
            continue
        log(f"\n--- Running Large Tile Inference on {tile_path.name} ---")
        start_tile = time.time()
        # Load Data
        noisy_tile = np.load(noisy_file_path)  # [H, W, 2]
        if verbose:
            print_tensor_stats(f" - Loaded Noisy Tile ({tile_path.name})", noisy_tile)

        # Overlap-blended patch inference.
        # Replaces the previous pad_to_multiple → extract_patches → loop → reconstruct_from_patches
        # pipeline. patch_infer_fpga uses a sliding window with snap-to-border coverage,
        # so no explicit padding or post-crop is needed for non-multiple image sizes.
        def _infer_fn(patch_hwc: np.ndarray) -> Tuple[np.ndarray, int]:
            return process_single_tile(patch_hwc, runners, eb, gc, verbose=False)

        recon_norm_logI, total_bytes = patch_infer_fpga(
            noisy_tile,
            _infer_fn,
            patch_size=IMAGE_SIZE,
            overlap=16,
        )
        tile_bpp = MetricsTracker.compute_bitstream_bpp(noisy_tile[np.newaxis, ...], total_bytes)

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
        denoised_references = {
            "MERLIN": "linA_MERLIN.npy",
            "ADAM-NOC": "linA_ADAM_NOC.npy",
            "MERLIN_DDS": "linA_MERLIN_DDS.npy",
        }
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

    xmodel_path = Path(args.xmodel).resolve()
    if not xmodel_path.exists():
        log(f"Error: XModel file {xmodel_path} does not exist.")
        sys.exit(1)
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        log(f"Error: Data file {data_path} does not exist.")
        sys.exit(1)

    display_manifest(xmodel_path.parent)

    run_hybrid_inference(xmodel_path, data_path, args.subset, args.verbose)
