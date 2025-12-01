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
import vart  # Vitis AI Runtime
import xir  # Xilinx Intermediate Representation

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
        """Compute Peak Signal-to-Noise Ratio (PSNR) between two images."""
        mse = MetricsTracker.compute_mse(a, b)
        if mse == 0:
            return float("inf")
        return 10 * np.log10((DATA_RANGE**2) / mse)

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
        x_hat_norm: np.ndarray,
        likelihoods: Dict[str, np.ndarray],
        target_log: np.ndarray,
    ) -> Dict[str, float]:
        """
        x_hat_norm: reconstruction in normalized [0,1] domain, shape [B, H, W, C].
        likelihoods: dict of likelihood tensors (for BPP).
        target_log: reference in log-intensity scale (already denormalized).

        Returns per-batch metrics for the current reference.
        """
        batch_metrics: Dict[str, float] = {}

        for name in self._metric_names:
            fn = self._metric_fns.get(name)
            if fn is None:
                continue

            if name == "bpp":
                value = fn(x_hat_norm, likelihoods)
            elif name in ("mse", "psnr", "ssim", "ms_ssim", "merlin"):
                value = fn(x_hat_norm, target_log)
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

    The dataset should always contain ADAM-NOC and MERLIN GT.
    """
    data = np.load(dataset_path)
    if subset_len < 0 or subset_len > data.shape[0]:
        subset_len = data.shape[0]
    slc_data = data[:subset_len, :, :, 0:2]  # [B, H, W, 2] real and imag channels only
    ground_truths = {  # [B, H, W, 1] already in log-intensity
        "adam_noc": data[:subset_len, :, :, 2:3],
        "merlin": data[:subset_len, :, :, 3:4],
    }
    return slc_data, ground_truths


def preprocess_input(data: np.ndarray, input_scale: float) -> np.ndarray:
    """Data normalization and quantization (fixed-point INT8)."""
    # Normalization is normally applied in TSXSSCDataset.__getitem__. Because we don't use it here, we must do it manually.
    data = np.square(data)
    data = np.log(data + 1e-2)
    data = (data - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)

    # The DPU processes INT8 data. We can either quantize it ourselves or let Vitis AI handle it.
    return (data * input_scale).astype(np.int8)


def postprocess_output(output_data_int: np.ndarray, output_scale: float) -> np.ndarray:
    """Convert DPU fixed-point output back to float, denormalize the reconstruction, and convert in
    Log-intensity scale."""
    output_data_float = output_data_int.astype(np.float32) * output_scale

    # Convert to log-intensity like in evaluation
    recon_lin = np.exp(output_data_float * (AMP_MAX - AMP_MIN) + AMP_MIN)
    recon_lin = 0.5 * (recon_lin[..., 0:1] + recon_lin[..., 1:2])
    recon_logI = np.log(recon_lin + EPS)
    return recon_logI


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

    # Get tensor info
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_tensor = input_tensors[0]
    output_tensor = output_tensors[0]
    input_shape = tuple(input_tensor.dims)  # [B, H, W, C]
    output_shape = tuple(output_tensor.dims)  # [B, H, W, C]

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
    slc_data, ref_data = load_npy_data(dataset_path, subset_len)
    print(
        f"Loaded SLC data with shape {slc_data.shape} from {dataset_path}. References are {list(ref_data.keys())}, each with shape {ref_data['adam_noc'].shape}."
    )

    # Three trackers: vs original input, vs ADAM-NOC, vs MERLIN
    metric_list = ["bpp", "mse", "psnr", "ssim", "ms_ssim", "merlin"]
    tracker_orig = MetricsTracker(metrics_to_track=metric_list)
    tracker_adam = MetricsTracker(metrics_to_track=metric_list)
    tracker_merlin = MetricsTracker(metrics_to_track=metric_list)
    n_batches = 0

    # Inference loop
    n_samples = len(slc_data)
    batch_size = input_shape[0]
    print(f"Running inference on {n_samples} (batch size: {batch_size}) samples...")

    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        valid_len = batch_end - i
        batch_data = slc_data[i:batch_end]  # [valid_len, H, W, 2]

        # Pad last batch if needed @TODO Delete if we enforce a batch size of 1
        if len(batch_data) < batch_size:
            pad_size = batch_size - len(batch_data)
            batch_data = np.pad(batch_data, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

        # Preprocess input (norm + quant) and store to DPU input buffer
        input_data[0][:] = preprocess_input(batch_data, input_scale)

        # Execute on DPU (synchronous)
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)

        # Read DPU output buffer and discard padded batches
        batch_output_int = output_data[0][:valid_len]  # [valid_len, H, W, 2]
        # Bring back to float and denormalize
        batch_output = postprocess_output(batch_output_int, output_scale)  # [valid_len, H, W, 1]

        # Build references in log-intensity
        # original: compute log-intensity from real/imag in channel-last form
        # batch_data (with padding) is [batch_size, H, W, 2]; we need only valid part
        batch_data_valid = batch_data[:valid_len]  # [valid_len, H, W, 2]
        orig_log = np.log(
            0.5 * (np.square(batch_data_valid[..., 0:1]) + np.square(batch_data_valid[..., 1:2]))
            + EPS
        )  # [valid_len, H, W, 1]

        # ADAM-NOC and MERLIN already log-intensity [B, H, W]; expand channel dim to match [B, H, W, 1]
        adam_slice = ref_data["adam_noc"][i:batch_end][:valid_len]  # [valid_len, H, W, 1]
        merlin_slice = ref_data["merlin"][i:batch_end][:valid_len]  # [valid_len, H, W, 1]

        # Dummy likelihoods for BPP (same shape as output)
        likelihoods = {"y": np.ones_like(batch_output, dtype=np.float32)}  # [valid_len, H, W, 1]

        # Update trackers
        batch_metrics_orig = tracker_orig.update(
            x_hat_norm=batch_output,
            likelihoods=likelihoods,
            target_log=orig_log,
        )
        batch_metrics_adam = tracker_adam.update(
            x_hat_norm=batch_output,
            likelihoods=likelihoods,
            target_log=adam_slice,
        )
        batch_metrics_merlin = tracker_merlin.update(
            x_hat_norm=batch_output,
            likelihoods=likelihoods,
            target_log=merlin_slice,
        )
        n_batches += 1

        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i // batch_size}:")
            print(f"\t - {batch_metrics_orig=}")
            print(f"\t - {batch_metrics_adam=}")
            print(f"\t - {batch_metrics_merlin=}")

    # 8. Summary
    avg_orig = tracker_orig.summary()
    avg_adam = tracker_adam.summary()
    avg_merlin = tracker_merlin.summary()

    print("\nFinal Results:")
    print("  Vs original input:")
    for name, value in sorted(avg_orig.items()):
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
        "metrics_vs_original": avg_orig,
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
