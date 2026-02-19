import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import vart  # type: ignore
import xir  # type: ignore

# Normalization constants
AMP_MIN = 4.605170249938965
AMP_MAX = 10.742239952087402
EPS = 1e-2
AMP_LIN_99 = 545.2018433569272

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
            "bpp": self.estimate_bpp,
            "mse": self.compute_mse,
            "psnr": self.compute_psnr,
            "ssim": self.compute_ssim,
            "ms_ssim": self.compute_ms_ssim,
        }
        self._sums: Dict[str, float] = {name: 0.0 for name in self._metric_names}
        self._count: int = 0

    @staticmethod
    def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
        """Compute MSE."""
        return float(np.mean((a - b) ** 2))

    @staticmethod
    def compute_psnr(a: np.ndarray, b: np.ndarray, mse_value: Optional[float] = None) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) between predicted_linA and target_linA
        tensors.

        Both tensors must be in linear Amplitude scale as peak=AMP_LIN_99 is used for PSNR
        computation.
        """
        # Clip target and predictions to 99% of distribution to avoid outliers dominating the PSNR computation.
        a = np.clip(a, 0, AMP_LIN_99)
        b = np.clip(b, 0, AMP_LIN_99)
        # Compute MSE and PSNR on 99% of the value
        mse_value = mse_value if mse_value is not None else MetricsTracker.compute_mse(a, b)
        psnr_value = 20 * np.log10(AMP_LIN_99) - 10 * np.log10(mse_value)
        return psnr_value

    @staticmethod
    def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute SSIM."""
        return 0.0  # Placeholder

    @staticmethod
    def compute_ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute MS-SSIM."""
        return 0.0  # Placeholder

    @staticmethod
    def estimate_bpp(x_shape_holder: np.ndarray, num_bytes: int) -> float:
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
        num_bytes: Optional[int] = None,
    ) -> Dict[str, float]:
        """Update metrics with a new batch."""
        batch_metrics: Dict[str, float] = {}
        for name in self._metric_names:
            fn = self._metric_fns.get(name)
            if fn is None:
                continue

            if name == "bpp":
                value = self.estimate_bpp(recon_linA, num_bytes)
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


def pad_to_multiple(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
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
    patches: np.ndarray, image_shape: tuple[int, int], patch_size: int
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
