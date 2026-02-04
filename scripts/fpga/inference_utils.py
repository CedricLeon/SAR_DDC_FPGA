from typing import Any, Callable, Dict, Iterable

import numpy as np

# Normalization constants
AMP_MIN = 4.605170249938965
AMP_MAX = 10.742239952087402
EPS = 1e-2


# -----------------------------------------------------------------------------
# LOGGING UTILS
# -----------------------------------------------------------------------------


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

    For BPP we use the normalized x_hat + likelihoods. For all other metrics we assume both inputs
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
    def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
        """Compute PSNR (peak assumed to be max of 'a').

        Ideally peak should be derived from range, but for SAR Amplitude it varies. We follow
        inference.py implementation using max(a).
        """
        mse = MetricsTracker.compute_mse(a, b)
        if mse == 0:
            return 100.0
        peak = float(np.max(a))
        if peak == 0:
            return 0.0
        return 20 * np.log10(peak) - 10 * np.log10(mse)

    @staticmethod
    def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute SSIM."""
        return 0.0  # Placeholder

    @staticmethod
    def compute_ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute MS-SSIM."""
        return 0.0  # Placeholder

    @staticmethod
    def estimate_bpp(x_shape_holder: np.ndarray, likelihoods: Dict[str, np.ndarray]) -> float:
        """Compute BPP."""
        B, H, W, _ = x_shape_holder.shape
        num_pixels = B * H * W
        bpp = sum((np.log(lh).sum() / (-np.log(2) * num_pixels)) for lh in likelihoods.values())
        return float(bpp)

    def update(
        self,
        recon_linA: np.ndarray,
        likelihoods: Dict[str, np.ndarray],
        target_linA: np.ndarray,
    ) -> Dict[str, float]:
        """Update metrics with a new batch."""
        batch_metrics: Dict[str, float] = {}
        for name in self._metric_names:
            fn = self._metric_fns.get(name)
            if fn is None:
                continue

            if name == "bpp":
                value = fn(recon_linA, likelihoods)
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
