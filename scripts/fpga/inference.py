#!/usr/bin/env python3
"""This file is to be runs on target board, i.e., executed on the ARM CPU of the ZCU102.

It executes a compiled DPU model for inference using VART Python API, see `Vitis-AI_journey.md` for
details.
"""

import argparse
import json
from datetime import datetime
from typing import Callable, Dict, Iterable

import numpy as np

# These imports are from Vitis AI and are only present in the Docker container (or on the FPGA)
import vart  # type: ignore
import xir  # type: ignore

# ---------------------------------------------------------
# ---------- Code "duplicates" to avoid imports -----------
# ---------------------------------------------------------
# Normalization constants from src/utils/constants.py.
AMP_MIN = -4.605170249938965
AMP_MAX = 10.742239952087402
DATA_RANGE = 2 * AMP_MAX - 2 * AMP_MIN
EPS = 1e-2


def print_tensor_stats(name: str, tensor: np.ndarray):
    """Print statistics of a given tensor."""
    print(
        f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}, std={tensor.std():.4f}"
    )


class MetricsTracker:
    """Accumulates and averages a set of metrics over multiple batches.

    metrics_to_track: iterable of metric names, e.g. ["bpp", "mse", "psnr", "merlin"].
    For BPP we use the normalized x_hat + likelihoods.
    For all other metrics we assume both inputs are in log-intensity scale.
    """

    def __init__(self, metrics_to_track: Iterable[str]):
        self._metric_names = list(metrics_to_track)
        self._metric_fns: Dict[str, Callable[..., float]] = {
            "bpp": self.estimate_bpp,
            "mse": self.compute_mse,
            "psnr": self.compute_psnr,
            "ssim": self.compute_ssim,
            "ms_ssim": self.compute_ms_ssim,
            "merlin": self.compute_merlin_loss,
        }
        self._sums: Dict[str, float] = {name: 0.0 for name in self._metric_names}
        self._count: int = 0

    # ---------- static metric implementations ----------

    @staticmethod
    def denorm_from_unit(x: np.ndarray) -> np.ndarray:
        """Inverse of the (x - AMP_MIN) / (AMP_MAX - AMP_MIN) normalization."""
        return x * (AMP_MAX - AMP_MIN) + AMP_MIN

    @staticmethod
    def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Mean Squared Error (MSE) between two images."""
        return float(np.mean((a - b) ** 2))

    @staticmethod
    def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

        Inputs:
         - a: the predicted image, np.ndarray.
         - b: the reference image, np.ndarray.
        Returns:
         - PSNR value as float.
        """
        mse = MetricsTracker.compute_mse(a, b)
        peak = float(np.max(a))
        return 20 * np.log10(peak) - 10 * np.log10(mse)

    @staticmethod
    def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Placeholder: Compute Structural Similarity Index (SSIM) between two images."""
        return 0.0

    @staticmethod
    def compute_ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Placeholder: Compute Multi-Scale Structural Similarity Index (MS-SSIM) between two images."""
        return 0.0

    @staticmethod
    def compute_merlin_loss(r_log: np.ndarray, b_log: np.ndarray) -> float:
        """Compute MERLIN loss between reconstruction and target."""
        # In Log-Scale: (0.5 * r + exp(2*b - r))
        merlin_loss = 0.5 * r_log + np.exp(2 * b_log - r_log)
        return float(np.mean(merlin_loss))

    @staticmethod
    def estimate_bpp(
        x_hat_norm: np.ndarray,
        likelihoods: Dict[str, np.ndarray],
    ) -> float:
        """Compute BPP based on the estimated likelihoods (average bits per pixel).

        Uses normalized x_hat only to get N, H, W (tensors are [B, H, W, C]).
        """
        B, H, W, _ = x_hat_norm.shape  # CHANGED: assume [B, H, W, C]
        num_pixels = B * H * W
        bpp = sum((np.log(lh).sum() / (-np.log(2) * num_pixels)) for lh in likelihoods.values())
        return float(bpp)

    # ---------- public API ----------

    def update(
        self,
        recon_linA: np.ndarray,
        likelihoods: Dict[str, np.ndarray],
        target_linA: np.ndarray,
    ) -> Dict[str, float]:
        """
        recon_linA: reconstruction linear amplitude, shape [B, H, W, 1].
        likelihoods: dict of likelihood tensors (for BPP).
        target_linA: reference in linear amplitude.

        Returns per-batch metrics for the current reference.
        """
        batch_metrics: Dict[str, float] = {}

        for name in self._metric_names:
            fn = self._metric_fns.get(name)
            if fn is None:
                continue

            if name == "bpp":
                value = fn(recon_linA, likelihoods)
            elif name in ("mse", "psnr", "ssim", "ms_ssim", "merlin"):
                value = fn(recon_linA, target_linA)
            else:
                continue

            self._sums[name] += float(value)
            batch_metrics[name] = float(value)

        self._count += 1
        return batch_metrics

    def summary(self) -> Dict[str, float]:
        """Compute average metrics over all updates."""
        if self._count == 0:
            return {k: 0.0 for k in self._sums}
        return {k: v / self._count for k, v in self._sums.items()}

    @property
    def count(self) -> int:
        """Return the number of updates."""
        return self._count


# -----------------------------------------------------------
# -------------------- Inference code -----------------------
# -----------------------------------------------------------


def load_npy_data(
    dataset_path: str, subset_len: int = 100
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load data from NPY file.

    Inputs:
      - dataset_path: path to NPY file.
      - subset_len: number of samples to load, -1 for all.
    Returns:
      - noisy: np.ndarray of the SLC patchesof shape [B, H, W, 2] with real and imag channels.
      - ground_truths: dict with keys "adam_noc" and "merlin", each np.ndarray of shape [B, H, W, 1
    """
    data = np.load(dataset_path)
    if subset_len < 0 or subset_len > data.shape[0]:
        subset_len = data.shape[0]
    noisy = data[:subset_len, :, :, 0:2]  # [B, H, W, 2] real and imag channels only
    ground_truths = {  # [B, H, W, 1] already in log-intensity
        "adam_noc": data[:subset_len, :, :, 2:3],
        "merlin": data[:subset_len, :, :, 3:4],
    }
    return noisy, ground_truths


def float_to_DPU_int(data_float: np.ndarray, input_scale: float) -> np.ndarray:
    """Convert float data to DPU fixed-point INT8 using the given scale."""
    return (data_float * input_scale).astype(np.int8)


def DPU_int_to_float(data_int: np.ndarray, scale: float) -> np.ndarray:
    """Convert DPU fixed-point INT8 data back to float using the given scale."""
    return data_int.astype(np.float32) * scale


def denormalize_model_output(data_norm: np.ndarray) -> np.ndarray:
    """Denormalize model output.

    The model is fed with data in logarithm scale normalized with [AMP_MIN, AMP_MAX]. We revert
    this normalization here.
    """
    return np.exp(data_norm * (AMP_MAX - AMP_MIN) + AMP_MIN)


def get_child_subgraph_dpu(graph: xir.Graph) -> list[xir.Subgraph]:
    """From KP Labs Tutorial: https://docs.sml.kplabs.space/tutorials/ml_deployment/onboard_model_runner_python.html"""
    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def run_inference(xmodel_path: str, dataset_path: str, subset_len: int):
    """Main inference function using VART Python API."""
    # Load model and runner
    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = get_child_subgraph_dpu(graph)
    print(f"Found {len(subgraphs)} DPU subgraphs.")
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    # Get model tensors info
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_tensor = input_tensors[0]
    output_tensor = output_tensors[0]
    input_shape = tuple(input_tensor.dims)  # [B, H, W, C]
    output_shape = tuple(output_tensor.dims)  # [B, H, W, C]

    for input_tensor in input_tensors:
        print(f"Input tensor: {input_tensor.name=}, {input_tensor.dims=}, {input_tensor.dtype=}")
    for output_tensor in output_tensors:
        print(
            f"Output tensor: {output_tensor.name=}, {output_tensor.dims=}, {output_tensor.dtype=}"
        )

    # Get fixed-point scales for conversion @TODO check how that is computed
    input_fixpos = input_tensor.get_attr("fix_point")
    output_fixpos = output_tensor.get_attr("fix_point")

    input_scale = 2**input_fixpos  # Convert to scale factor
    output_scale = 1.0 / (2**output_fixpos)

    print(
        f"Expected input shape={input_shape} with fixpos={input_fixpos}, so scale={input_scale}."
    )
    print(
        f"Expected output shape={output_shape} with fixpos={output_fixpos}, so scale={output_scale}."
    )

    # VART needs input/output buffers with exact shape from DPU, order="C" ensures data is laid out in row-major order (unlike Fortran order)
    input_data = [np.empty(input_shape, dtype=np.int8, order="C")]
    output_data = [np.empty(output_shape, dtype=np.int8, order="C")]

    # Dataset
    noisy, ground_truths = load_npy_data(dataset_path, subset_len)
    print(
        f"Loaded Noisy SLC data with shape {noisy.shape} from {dataset_path}. References are {list(ground_truths.keys())}, each with shape {ground_truths['adam_noc'].shape}."
    )

    # Three trackers: vs original input, vs ADAM-NOC, vs MERLIN
    metric_list = ["bpp", "mse", "psnr", "ssim", "ms_ssim", "merlin"]
    tracker_noisy = MetricsTracker(metrics_to_track=metric_list)
    tracker_adam = MetricsTracker(metrics_to_track=metric_list)
    tracker_merlin = MetricsTracker(metrics_to_track=metric_list)
    n_batches = 0

    # Inference loop
    n_samples = len(noisy)
    batch_size = input_shape[0]
    print(f"Running inference on {n_samples} (batch size: {batch_size}) samples...")

    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        valid_len = batch_end - i
        noisy_lin = noisy[i:batch_end]  # [valid_len, H, W, 2]

        # Pad last batch if needed @TODO Delete if we enforce a batch size of 1
        if len(noisy_lin) < batch_size:
            pad_size = batch_size - len(noisy_lin)
            noisy_lin = np.pad(noisy_lin, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

        # Preprocess input (norm + quant) and store to DPU input buffer
        print(f"input_data info: {len(input_data)=}, {input_data[0].shape=}, {input_scale=}")

        noisy_lin_int = float_to_DPU_int(noisy_lin, input_scale)
        print(f"{noisy_lin_int.shape=}")
        input_data[0][:] = noisy_lin_int

        # Execute on DPU (synchronous)
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)

        # Read DPU output buffer and discard padded batches
        recon_norm_int = output_data[0][:valid_len]  # [valid_len, H, W, 2]
        # Bring back to float and denormalize
        recon_norm = DPU_int_to_float(recon_norm_int, output_scale)
        # Denormalize model output
        recon_lin = denormalize_model_output(recon_norm)
        # Convert to linear amplitude
        recon_linA = np.sqrt(
            0.5 * (np.square(recon_lin[..., 0:1]) + np.square(recon_lin[..., 1:2]))
        )  # [B, H, W, 1]

        # Build references in linear amplitude
        noisy_lin = noisy_lin[:valid_len]  # [valid_len, H, W, 2]
        noisy_linA = np.sqrt(
            np.square(noisy_lin[..., 0:1]) + np.square(noisy_lin[..., 1:2])
        )  # [valid_len, H, W, 1]

        # ADAM-NOC and MERLIN already linear amplitude [B, H, W]; expand channel dim to match [B, H, W, 1]
        adam_linA = ground_truths["adam_noc"][i:batch_end][:valid_len]  # [valid_len, H, W, 1]
        merlin_linA = ground_truths["merlin"][i:batch_end][:valid_len]  # [valid_len, H, W, 1]

        # Dummy likelihoods for BPP (same shape as output)
        likelihoods = {"y": np.ones_like(recon_linA, dtype=np.float32)}  # [valid_len, H, W, 1]
        # Update trackers
        batch_metrics_noisy = tracker_noisy.update(
            recon_linA,
            likelihoods,
            noisy_linA,
        )
        batch_metrics_adam = tracker_adam.update(
            recon_linA,
            likelihoods,
            adam_linA,
        )
        batch_metrics_merlin = tracker_merlin.update(
            recon_linA,
            likelihoods,
            merlin_linA,
        )
        n_batches += 1

        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i // batch_size}:")
            print(f"\t - {batch_metrics_noisy=}")
            print(f"\t - {batch_metrics_adam=}")
            print(f"\t - {batch_metrics_merlin=}")

    # 8. Summary
    avg_noisy = tracker_noisy.summary()
    avg_adam = tracker_adam.summary()
    avg_merlin = tracker_merlin.summary()

    print("\nFinal Results:")
    print("  Vs original input:")
    for name, value in sorted(avg_noisy.items()):
        print(f"    {name}: {value:.4f}")
    print("  Vs ADAM-NOC:")
    for name, value in sorted(avg_adam.items()):
        print(f"    {name}: {value:.4f}")
    print("  Vs MERLIN:")
    for name, value in sorted(avg_merlin.items()):
        print(f"    {name}: {value:.4f}")
    print(f"  Processed {n_samples} samples in {n_batches} batches")

    summary = {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "metrics_vs_noisy": avg_noisy,
        "metrics_vs_adam_noc": avg_adam,
        "metrics_vs_merlin": avg_merlin,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/inference_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmodel", required=True, help="Path to compiled .xmodel file")
    parser.add_argument("--data", required=True, help="Path to HDF5 test data")
    parser.add_argument("--subset", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()
    run_inference(args.xmodel, args.data, args.subset)
