import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import vart  # type: ignore
import xir  # type: ignore

# Normalization constants
AMP_MIN = 4.605170249938965
AMP_MAX = 10.742239952087402
EPS = 1e-2
AMP_LIN_99 = 545.2018433569272


# -----------------------------------------------------------------------------
# DPU RUNNER HELPER
# -----------------------------------------------------------------------------


def float_to_DPU_int(data_float: np.ndarray, input_scale: float) -> np.ndarray:
    """Convert float data to DPU fixed-point INT8 using the given scale."""
    return (data_float * input_scale).astype(np.int8)


def DPU_int_to_float(data_int: np.ndarray, scale: float) -> np.ndarray:
    """Convert DPU fixed-point INT8 data back to float using the given scale."""
    return data_int.astype(np.float32) * scale


class DPUSubgraphRunner:
    """Helper to wrap a single DPU subgraph runner.

    Handles INT8 quantisation/dequantisation transparently so callers can pass float32 numpy arrays
    and receive float32 results.
    """

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
            f"[{name}] In: {self.input_shape} (scale={self.input_scale}), "
            f"Out: {self.output_shape} (scale={self.output_scale})"
        )

    def run(self, input_data: np.ndarray) -> np.ndarray:
        """Float-in, float-out inference (quantise -> DPU -> dequantise).

        input_data: Float numpy array matching input shape (NHWC).
        """
        # 1. Quantize Input (Float -> Int8)
        input_int8 = float_to_DPU_int(input_data, self.input_scale)

        # 2. Prepare Buffers
        # VART needs input/output buffers with exact shape from DPU,
        # order="C" ensures data is laid out in row-major order.
        input_buffer = np.ascontiguousarray(input_int8)
        output_buffer = np.empty(self.output_shape, dtype=np.int8, order="C")

        # 3. Execute (job_id is returned)
        job_id = self.runner.execute_async([input_buffer], [output_buffer])
        self.runner.wait(job_id)

        # 4. Dequantize Output (Int8 -> Float)
        output_float = DPU_int_to_float(output_buffer, self.output_scale)
        return output_float


# -----------------------------------------------------------------------------
# SUBGRAPH IDENTIFICATION
# -----------------------------------------------------------------------------


def identify_subgraphs(
    graph: xir.Graph, meta_path: Path, verbose: bool = True
) -> Dict[str, xir.Subgraph]:
    """Identify which xir.Subgraph corresponds to g_a, h_a, h_s, g_s using meta.json.

    Parameters
    ----------
    graph : xir.Graph
        The deserialized xmodel graph.
    meta_path : Path
        Path to the ``meta.json`` file produced by the Vitis-AI compiler.

    Returns
    -------
    dict
        Mapping ``{"g_a": subgraph, "h_a": ..., "h_s": ..., "g_s": ...}``.
    """
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta file not found at {meta_path}. Cannot identify subgraphs without it."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    kernels = meta.get("kernel", [])
    if verbose:
        print(f"[identify_subgraphs] meta.json lists {len(kernels)} kernels.")

    # Map kernel names to roles based on substrings
    name_to_role: Dict[str, str] = {}
    for k_name in kernels:
        for role in ("g_a", "g_s", "h_a", "h_s"):
            if role in k_name:
                name_to_role[k_name] = role
                break
        else:
            print(f"[identify_subgraphs] WARNING: Unrecognized kernel name: {k_name}")

    # Find the actual subgraphs in the graph object
    mapping: Dict[str, xir.Subgraph] = {}
    root = graph.get_root_subgraph()
    for sg in root.toposort_child_subgraph():
        if sg.get_name() in name_to_role:
            role = name_to_role[sg.get_name()]
            mapping[role] = sg
            if verbose:
                print(f"[identify_subgraphs] Mapped {role} -> {sg.get_name()}")

    # Validate
    required_keys = ["g_a", "h_a", "h_s", "g_s"]
    missing_keys = [k for k in required_keys if k not in mapping]
    if missing_keys:
        print(f"[identify_subgraphs] ERROR: Missing subgraphs: {missing_keys}")
        print(f"[identify_subgraphs] Found: {list(mapping.keys())}")

        print("\n--- Debug: All DPU Subgraphs ---")
        for sg in root.toposort_child_subgraph():
            if sg.has_attr("device") and sg.get_attr("device") == "DPU":
                inputs = list(sg.get_input_tensors())
                outputs = list(sg.get_output_tensors())
                in_shape = tuple(inputs[0].dims) if inputs else "None"
                out_shape = tuple(outputs[0].dims) if outputs else "None"
                print(f"  Subgraph: {sg.get_name()}")
                print(f"    Input:  {in_shape}")
                print(f"    Output: {out_shape}")
        print("--------------------------------\n")
        raise ValueError("Could not identify all required subgraphs in the model.")

    return mapping


# -----------------------------------------------------------------------------
# LOGGING UTILS
# -----------------------------------------------------------------------------


def display_manifest(script_path: Path) -> None:
    """Check for and display manifest.json contents."""
    script_dir = script_path.parent.resolve()
    manifest_path = script_dir / "manifest.json"

    if manifest_path.exists():
        print("\n" + "=" * 60)
        print("Build Manifest Found:")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Essential Keys
            model_name = manifest.get("model_name", "Unknown")
            compiled_at = manifest.get("compiled_at", "Unknown")
            print(f"  • Model: {model_name}")
            print(f"  • Compiled At: {compiled_at}")

            # Dynamic Keys (Print everything else)
            img = ["model_name", "compiled_at"]
            for k, v in manifest.items():
                if k not in img:
                    # Clean up key name for display
                    readable_key = k.replace("_", " ").title()
                    print(f"  • {readable_key}: {v}")

        except Exception as e:
            print(f"  [Error reading manifest: {e}]")
        print("=" * 60 + "\n")
    else:
        print("[INFO] No manifest.json found in current directory.")


def print_tensor_stats(name: str, tensor: np.ndarray):
    """Print statistics of a given tensor."""
    print(
        f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}, std={tensor.std():.4f}"
    )


# -----------------------------------------------------------------------------
# PROCESSING UTILS
# -----------------------------------------------------------------------------


def clip(
    img: np.ndarray,
    mean_std_norm: bool = True,
    clip_factor: float = 3.0,
    clip_percentiles: tuple = (5, 95),
) -> np.ndarray:
    """Clip image for better visualization."""
    if mean_std_norm:
        mean = img.mean()
        std = img.std()
        vmin = mean - clip_factor * std
        vmax = mean + clip_factor * std
    else:
        vmin = np.percentile(img, clip_percentiles[0])
        vmax = np.percentile(img, clip_percentiles[1])
    return np.clip(img, vmin, vmax)


class MetricsTracker:
    """Accumulates and averages a set of metrics over multiple batches.

    For BPP we use the normalized x_hat + num_bytes. For all other metrics we assume both inputs
    are in linear amplitude [0, +inf].
    """

    def __init__(self, metrics_to_track: Iterable[str]):
        self._metric_names = list(metrics_to_track)
        self._metric_fns: Dict[str, Callable[..., float]] = {
            "bpp": self.compute_bitstream_bpp,
            "mse": self.compute_mse,
            "psnr": self.compute_psnr,
            "ssim": self.compute_ssim,
            "ms_ssim": self.compute_ms_ssim,
            "enl": self.compute_enl,
            "ratio_mean": self.compute_ratio_mean,
            "ratio_enl": self.compute_ratio_enl,
            "epd": self.compute_epd,
        }
        self._sums: Dict[str, float] = {name: 0.0 for name in self._metric_names}
        self._count: int = 0

    @staticmethod
    def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
        """Compute MSE between 2 arrays expected in linear Amplitude scale as clipping between 0
        and AMP_LIN_99 is done."""
        # Clip target and predictions to 99% of distribution to avoid outliers dominating the PSNR computation.
        a = np.clip(a, 0, AMP_LIN_99)
        b = np.clip(b, 0, AMP_LIN_99)
        return float(np.mean((a - b) ** 2))

    @staticmethod
    def compute_psnr(a: np.ndarray, b: np.ndarray, mse_value: Optional[float] = None) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) between predicted_linA and target_linA
        tensors.

        Both tensors must be in linear Amplitude scale as peak=AMP_LIN_99 is used for PSNR
        computation.
        """
        mse_value = mse_value if mse_value is not None else MetricsTracker.compute_mse(a, b)
        psnr_value = 20 * np.log10(AMP_LIN_99) - 10 * np.log10(mse_value)
        return psnr_value

    @staticmethod
    def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute SSIM using cv2.quality.QualitySSIM_compute.

        Inputs are expected in linear amplitude [0, +inf], with arbitrary shape ([1, H, W, 1], [H,
        W, 1], or [H, W]). Both arrays are squeezed to 2-D float32 before the call. The returned
        value is the mean SSIM across channels (typically 1 channel).
        """
        a_2d = a.squeeze().astype(np.float32)
        b_2d = b.squeeze().astype(np.float32)
        # QualitySSIM_compute returns (scalar_per_channel, quality_map).
        # scalar_per_channel is a 4-element tuple; the first element holds channel 0.
        result, _ = cv2.quality.QualitySSIM_compute(a_2d, b_2d)  # type: ignore[attr-defined]
        return float(result[0])

    @staticmethod
    def compute_ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute MS-SSIM.

        Not available on this platform — returns 0.0.
        """
        return 0.0  # cv2 does not provide MS-SSIM

    @staticmethod
    def compute_enl(a: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> float:
        """ENL on linear intensity.

        roi=(r0, r1, c0, c1) optional crop for homogeneous regions.
        """
        arr = a.squeeze().astype(np.float32)
        if roi is not None:
            r0, r1, c0, c1 = roi
            arr = arr[r0:r1, c0:c1]
        lin_intensity = np.square(arr)
        mu = float(np.mean(lin_intensity))
        var = float(np.var(lin_intensity))
        return mu**2 / var if var > 0.0 else float("nan")

    @staticmethod
    def compute_ratio_mean(recon: np.ndarray, noisy: np.ndarray) -> float:
        """Mean of ratio image R = noisy_I / recon_I.

        >1 = residual speckle, <1 = over-smooth.
        """
        recon_I = np.square(recon.squeeze().astype(np.float32))
        noisy_I = np.square(noisy.squeeze().astype(np.float32))
        return float(np.mean(noisy_I / (recon_I + 1e-10)))

    @staticmethod
    def compute_ratio_enl(recon: np.ndarray, noisy: np.ndarray) -> float:
        """ENL of the ratio image R = noisy_I / recon_I."""
        recon_I = np.square(recon.squeeze().astype(np.float32))
        noisy_I = np.square(noisy.squeeze().astype(np.float32))
        ratio = noisy_I / (recon_I + 1e-10)
        mu = float(np.mean(ratio))
        var = float(np.var(ratio))
        return mu**2 / var if var > 0.0 else float("nan")

    @staticmethod
    def compute_epd(recon: np.ndarray, ref: np.ndarray) -> float:
        """Edge Preservation Degree in linA domain.

        EPD=1.0 means perfect edge preservation.
        """

        def _grad_mag(img: np.ndarray) -> np.ndarray:
            gx = np.zeros_like(img)
            gy = np.zeros_like(img)
            gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
            gy[1:-1, :] = img[2:, :] - img[:-2, :]
            return np.sqrt(gx**2 + gy**2)

        g_r = _grad_mag(recon.squeeze().astype(np.float32))
        g_f = _grad_mag(ref.squeeze().astype(np.float32))
        denom = float(np.sum(g_f**2))
        return float(np.sum(g_r * g_f) / denom) if denom > 0.0 else float("nan")

    @staticmethod
    def compute_bitstream_bpp(x_shape_holder: np.ndarray, num_bytes: int) -> float:
        """Compute BPP.

        Args:
            x_shape_holder: Tensor with shape [B, H, W, C] to get dimensions.
            num_bytes: Total number of bytes of the compressed representations.
        """
        B, H, W, _ = x_shape_holder.shape
        num_pixels = B * H * W
        return float((num_bytes * 8) / num_pixels)

    def update(
        self,
        recon_linA: np.ndarray,
        target_linA: np.ndarray,
        num_bytes: int,
    ) -> Dict[str, float]:
        """Update metrics with a new batch."""
        batch_metrics: Dict[str, float] = {}
        for name in self._metric_names:
            fn = self._metric_fns.get(name)
            if fn is None:
                continue

            if name == "bpp":
                value = self.compute_bitstream_bpp(recon_linA, num_bytes)
            elif name == "enl":
                value = self.compute_enl(recon_linA)
            else:
                value = fn(recon_linA, target_linA)

            self._sums[name] += float(value)
            batch_metrics[name] = float(value)
        self._count += 1
        return batch_metrics

    def summary(self) -> Dict[str, float]:
        if self._count == 0:
            return {k: 0.0 for k in self._sums}
        return {k: v / self._count for k, v in self._sums.items()}

    @property
    def count(self) -> int:
        """Return the number of updates."""
        return self._count


# -----------------------------------------------------------------------------
# VISUALIZATION UTILS
# -----------------------------------------------------------------------------


def visualize_patches(
    noisy_logI: np.ndarray,
    recon_logI: np.ndarray,
    adam_logI: np.ndarray,
    merlin_logI: np.ndarray,
    save_path: str,
    num_patches: int = 5,
):
    """Generate and save a 4-row comparison figure.

    If matplotlib is missing, skips visualization. Expects all inputs to be in log-Intensity
    format.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Matplotlib not found. Visualization skipped.")
        print(f"Would have saved to: {save_path}")
        return

    N = min(num_patches, len(noisy_logI))
    _, axes = plt.subplots(4, N, figsize=(4 * N, 16))
    if N == 1:
        axes = axes.reshape(4, 1)

    for i in range(N):
        # Clipping
        noisy_disp = clip(noisy_logI)
        recon_disp = clip(recon_logI)
        merlin_disp = clip(merlin_logI)
        adam_disp = clip(adam_logI)

        # Row 0: Original Noisy (LogI)
        axes[0, i].imshow(noisy_disp, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Noisy Input")

        # Row 1: Reconstruction (LogI)
        axes[1, i].imshow(recon_disp, cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstruction")

        # Row 2: ADAM NOC GT (LogI)
        axes[2, i].imshow(adam_disp, cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("ADAM NOC GT")

        # Row 3: MERLIN GT (LogI)
        axes[3, i].imshow(merlin_disp, cmap="gray")
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_title("MERLIN GT")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------------
# TILING UTILS
# -----------------------------------------------------------------------------


def pad_to_multiple(image: np.ndarray, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image using reflection so its dimensions are multiples of patch_size.

    Args:
        image: Input image [H, W, C]
        patch_size: Size of the patch

    Returns:
        padded_image: The padded image
        (h_pad, w_pad): The amount of padding added to height and width
    """
    h, w = image.shape[:2]
    h_pad = (patch_size - h % patch_size) % patch_size
    w_pad = (patch_size - w % patch_size) % patch_size

    if h_pad == 0 and w_pad == 0:
        return image, (0, 0)

    # Pad with reflection ((top, bottom), (left, right), (channels...))
    pad_width = ((0, h_pad), (0, w_pad)) + ((0, 0),) * (image.ndim - 2)
    padded_image = np.pad(image, pad_width, mode="reflect")
    return padded_image, (h_pad, w_pad)


def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract non-overlapping patches from the image.

    Args:
        image: Input image [H, W, C] (must be divisible by patch_size)

    Returns:
        patches: Array of shape [N_patches, patch_size, patch_size, C]
    """
    h, w = image.shape[:2]
    c = image.shape[2]

    # Reshape to (n_h, patch_h, n_w, patch_w, C)
    n_h = h // patch_size
    n_w = w // patch_size

    reshaped = image.reshape(n_h, patch_size, n_w, patch_size, c)
    # Transpose to (n_h, n_w, patch_h, patch_w, C)
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    # Reshape to (N, patch_h, patch_w, C)
    patches = transposed.reshape(-1, patch_size, patch_size, c)
    return patches


def reconstruct_from_patches(
    patches: np.ndarray, image_shape: Tuple[int, int], patch_size: int
) -> np.ndarray:
    """Reconstruct image from non-overlapping patches.

    Args:
        patches: [N, patch_size, patch_size, C]
        image_shape: (H, W) of the target image (must be divisible by patch_size)
        patch_size: size of patches

    Returns:
        Reconstructed image [H, W, C]
    """
    h, w = image_shape
    c = patches.shape[-1]
    n_h = h // patch_size
    n_w = w // patch_size

    # Reshape to (n_h, n_w, patch_h, patch_w, c)
    reshaped_patches = patches.reshape(n_h, n_w, patch_size, patch_size, c)
    # Transpose to (n_h, patch_h, n_w, patch_w, c)
    transposed = reshaped_patches.transpose(0, 2, 1, 3, 4)
    # Reshape to (H, W, C)
    image = transposed.reshape(h, w, c)
    return image


def patch_infer_fpga(
    image: np.ndarray,
    infer_fn: Callable[[np.ndarray], Tuple[np.ndarray, int]],
    patch_size: int = 256,
    overlap: int = 16,
    eliminate_border_px: int = 0,
    blend_profile: str = "sigmoid",
    blend_alpha: float = 6.0,
) -> Tuple[np.ndarray, int]:
    """Run overlap-blended patch inference on a large image using the FPGA pipeline.

    Inspired from simon-donike and opensr-utils (https://github.com/ESAOpenSR/opensr-utils/).

    The DPU input buffer has a fixed spatial size (typically 256×256), so large images
    must be split into patches, processed individually, and reassembled. A naive
    non-overlapping split introduces visible seams at patch boundaries because the model
    has no context outside each patch. This function solves that by using overlapping
    windows and blending the results in the overlap zones with smooth feathering ramps.

    **Tiling strategy.**  A sliding window of size ``patch_size × patch_size`` advances
    with stride ``patch_size - overlap``. An extra snap-to-border window is appended when
    the last regular window does not end exactly at the image edge, guaranteeing full
    coverage for any image dimension ≥ ``patch_size`` without explicit zero-padding
    or post-crop.

    **Blending.**  In overlap zones each patch contributes according to a 1-D feathering
    ramp (sigmoid, linear, or cosine) that rises from 0 at the leading edge to 1 toward
    the interior. The 2-D weight map is the outer product of the horizontal and vertical
    ramps. At the global image borders no ramp is applied, so the output has full weight
    at the image edges (no fading-to-zero frame). The final pixel value is the weighted
    average of all patches covering it: accumulated weighted sum ÷ accumulated weight.

    **FPGA I/O convention.**  Patches and outputs are kept in HWC layout throughout,
    matching the NHWC convention of VART and the DPU runners.

    **Canvas allocation.**  The output canvas is allocated lazily on the first inference
    result so that the output channel count (C_out) is inferred from the actual pipeline
    output, avoiding a redundant dummy DPU call just to probe the output shape.

    **Byte accounting.**  ``infer_fn`` returns both the reconstructed patch and the number
    of compressed bytes for that patch. The byte counts are summed across all patches and
    returned as ``total_bytes`` alongside the blended image. With overlap, some image
    regions are independently encoded more than once, so ``total_bytes`` exceeds what a
    non-overlapping tiling would produce — treat it as an upper-bound BPP estimate.

    Parameters
    ----------
    image : np.ndarray, shape [H, W, C_in]
        Input image in HWC layout (raw complex float, unnormalized). Normalization is
        handled inside infer_fn / process_single_tile.
    infer_fn : Callable
        FPGA inference function with signature::

            (patch_hwc: np.ndarray[patch_size, patch_size, C_in])
            -> (output_hwc: np.ndarray[patch_size, patch_size, C_out], num_bytes: int)

        Typically a lambda that closes over the DPU runners and entropy models, e.g.::

            lambda patch: process_single_tile(patch, runners, eb, gc)

    patch_size : int, default=256
        Square patch size in pixels. Must match the fixed DPU input buffer size.
    overlap : int, default=16
        Number of pixels of overlap between adjacent patches. Must be a positive even
        integer, strictly less than patch_size. Larger values give smoother transitions
        but increase compute and total_bytes.
    eliminate_border_px : int, default=0
        Outermost pixels at each patch edge forced to zero weight before the feathering
        ramp begins (hard-discards the strongest edge artefacts, replacing them entirely
        with data from the neighbouring patch). Must be a non-negative even integer,
        strictly less than overlap. Set to 0 to rely on feathering only.
    blend_profile : str, default="sigmoid"
        Shape of the 1-D feathering ramp in overlap zones:
          - "sigmoid" : S-curve, concentrates the transition in the middle of the
                        overlap zone. Recommended for most natural blends.
          - "linear"  : straight ramp from 0 to 1.
          - "cosine"  : half-cosine ease-in/ease-out, smooth at both ends.
    blend_alpha : float, default=6.0
        Steepness of the sigmoid curve. Only used when blend_profile="sigmoid".

    Returns
    -------
    output : np.ndarray, shape [H, W, C_out]
        Blended reconstruction in HWC layout (normalized log-intensity, matching the
        output convention of process_single_tile). Same spatial dimensions as the input.
    total_bytes : int
        Sum of compressed bytes across all processed patches (including overlapping ones).

    Raises
    ------
    ValueError
        If overlap / eliminate_border_px constraints are violated, or if the image is
        smaller than patch_size in either dimension.
    """
    image_np = image.astype(np.float32, copy=False)
    H, W = image_np.shape[:2]

    # ------------------------------------------------------------------
    # 1. Validate parameters
    # ------------------------------------------------------------------
    if overlap <= 0 or overlap % 2 != 0:
        raise ValueError(f"`overlap` must be a positive even integer, got {overlap}.")
    if overlap >= patch_size:
        raise ValueError(
            f"`overlap` ({overlap}) must be strictly less than `patch_size` ({patch_size})."
        )
    if eliminate_border_px < 0 or eliminate_border_px % 2 != 0:
        raise ValueError(
            f"`eliminate_border_px` must be a non-negative even integer, got {eliminate_border_px}."
        )
    if eliminate_border_px >= overlap:
        raise ValueError(
            f"`eliminate_border_px` ({eliminate_border_px}) must be strictly less than `overlap` ({overlap})."
        )
    if H < patch_size:
        raise ValueError(f"Image height ({H}) is smaller than `patch_size` ({patch_size}).")
    if W < patch_size:
        raise ValueError(f"Image width ({W}) is smaller than `patch_size` ({patch_size}).")

    # ------------------------------------------------------------------
    # 2. Build the list of overlapping patch windows
    #    Each window is (row_off, col_off) in image pixel coordinates.
    #    Strategy: sliding window with stride = patch_size - overlap,
    #    plus extra snap-to-border windows to guarantee full coverage.
    # ------------------------------------------------------------------
    stride = patch_size - overlap

    def _make_offsets(dim_size: int) -> List[int]:
        """Return starting offsets for one spatial dimension."""
        offsets = list(range(0, dim_size - patch_size + 1, stride))
        # Snap-to-border: ensure the last window ends exactly at the image edge
        last = dim_size - patch_size
        if not offsets or offsets[-1] != last:
            offsets.append(last)
        return offsets

    row_offsets = _make_offsets(H)
    col_offsets = _make_offsets(W)
    n_patches = len(row_offsets) * len(col_offsets)
    print(
        f"[patch_infer_fpga] {H}x{W} image => {n_patches} patches "
        f"({len(row_offsets)} rows x {len(col_offsets)} cols), "
        f"patch_size={patch_size}, overlap={overlap}, stride={stride}."
    )

    # ------------------------------------------------------------------
    # 3. Allocate accumulators.
    #    The output canvas is allocated lazily on the first inference call
    #    so that C_out is inferred from the actual output without a
    #    separate dummy DPU probe pass (which would waste a full round-trip
    #    through the encoder, entropy coder, and decoder).
    # ------------------------------------------------------------------
    canvas: Optional[np.ndarray] = None
    weight_canvas = np.zeros((H, W), dtype=np.float32)
    total_bytes: int = 0

    # ------------------------------------------------------------------
    # 4. Helper: build a 1-D feathering ramp of length n, going 0 → 1
    # ------------------------------------------------------------------
    def _make_ramp(n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if blend_profile == "linear":
            return t
        elif blend_profile == "sigmoid":
            a = float(blend_alpha)
            r = 1.0 / (1.0 + np.exp(-a * (t - 0.5)))
            r = (r - r[0]) / (r[-1] - r[0] + 1e-12)  # renormalize to [0, 1]
            return r.astype(np.float32)
        elif blend_profile == "cosine":
            return (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown blend_profile '{blend_profile}'. Choose 'sigmoid', 'linear', or 'cosine'."
            )

    # ------------------------------------------------------------------
    # 5. Helper: build the 2-D weight map for one patch
    #    Takes care of:
    #      - eliminate_border_px  (hard zero at outermost pixels)
    #      - feathering ramp      (smooth 0→1 across overlap zone)
    #      - global border guard  (no ramp at the image edge)
    # ------------------------------------------------------------------
    def _patch_weight_map(row_off: int, col_off: int) -> np.ndarray:
        """Return a [patch_size, patch_size] float32 weight map for this patch."""
        touch_top = row_off == 0
        touch_bottom = row_off + patch_size == H
        touch_left = col_off == 0
        touch_right = col_off + patch_size == W

        ramp_len = overlap - eliminate_border_px  # length of the actual ramp

        u = np.ones(patch_size, dtype=np.float32)  # horizontal (W) weights
        v = np.ones(patch_size, dtype=np.float32)  # vertical   (H) weights

        ramp = _make_ramp(ramp_len)  # goes 0 → 1

        for vec, touch_start, touch_end in [
            (u, touch_left, touch_right),
            (v, touch_top, touch_bottom),
        ]:
            if not touch_start:
                # Hard-zero the outermost eliminate_border_px pixels
                if eliminate_border_px > 0:
                    vec[:eliminate_border_px] = 0.0
                # Then apply the ramp (0 → 1) over the next ramp_len pixels
                if ramp_len > 0:
                    vec[eliminate_border_px : eliminate_border_px + ramp_len] = ramp

            if not touch_end:
                # Mirror of the above on the trailing edge (1 → 0)
                if eliminate_border_px > 0:
                    vec[-eliminate_border_px:] = 0.0
                if ramp_len > 0:
                    vec[
                        -(eliminate_border_px + ramp_len) : (
                            None if eliminate_border_px == 0 else -eliminate_border_px
                        )
                    ] = ramp[::-1]

        # Outer product: 2-D weight is the product of horizontal and vertical weights
        return (v[:, None] * u[None, :]).astype(np.float32)  # [patch_size, patch_size]

    # ------------------------------------------------------------------
    # 6. Main loop: extract patch → run FPGA pipeline → accumulate into canvas.
    #
    #    Each patch is extracted in HWC layout, the native convention for VART
    #    and DPU runners (NHWC). infer_fn executes the full pipeline — DPU
    #    encoder (g_a), hyper-encoder (h_a), entropy coder, entropy decoder,
    #    hyper-decoder (h_s), DPU decoder (g_s) — and returns the reconstructed
    #    HWC patch together with its compressed byte count.
    #
    #    The patch is multiplied by its 2-D weight map and added to the canvas.
    #    The weight map is simultaneously accumulated in weight_canvas. Dividing
    #    canvas by weight_canvas in step 7 yields the final weighted average,
    #    which smoothly blends contributions from all overlapping patches.
    # ------------------------------------------------------------------
    for row_off in row_offsets:
        for col_off in col_offsets:
            # Extract HWC patch — no transposition needed
            patch_hwc = image_np[
                row_off : row_off + patch_size,
                col_off : col_off + patch_size,
                :,
            ]  # [patch_size, patch_size, C_in]

            # Run the full FPGA pipeline (DPU encoder/decoder + entropy coding)
            out_hwc, patch_bytes = infer_fn(patch_hwc)  # [patch_size, patch_size, C_out], int

            # Lazy canvas allocation on the first result — avoids a dummy DPU probe
            if canvas is None:
                C_out = out_hwc.shape[-1]
                canvas = np.zeros((H, W, C_out), dtype=np.float32)
            total_bytes += patch_bytes

            # Accumulate weighted patch into canvas
            w2d = _patch_weight_map(row_off, col_off)  # [patch_size, patch_size]
            canvas[
                row_off : row_off + patch_size,
                col_off : col_off + patch_size,
                :,
            ] += (
                out_hwc * w2d[:, :, None]
            )
            weight_canvas[
                row_off : row_off + patch_size,
                col_off : col_off + patch_size,
            ] += w2d

    if canvas is None:
        raise ValueError("No patches were processed. Verify image dimensions and patch_size.")

    # ------------------------------------------------------------------
    # 7. Normalise: divide accumulated weighted sum by accumulated weights.
    #    Every pixel is covered by at least one patch, so weight_canvas
    #    should be > 0 everywhere. Guard against /0 with a small epsilon.
    # ------------------------------------------------------------------
    weight_canvas = np.maximum(weight_canvas, 1e-8)
    output = canvas / weight_canvas[:, :, None]  # [H, W, C_out]

    return output, total_bytes
