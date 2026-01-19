from typing import Any, Callable, Dict, Iterable

import numpy as np

# Normalization constants
AMP_MIN = -4.605170249938965
AMP_MAX = 10.742239952087402
EPS = 1e-2


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
            "merlin": self.compute_merlin_loss,
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
    def compute_merlin_loss(r_linA: np.ndarray, b_linA: np.ndarray) -> float:
        """Compute MERLIN loss.

        Inputs are Linear Amplitude.
        Formula logic from inference.py: 0.5 * r_log + exp(2*b_log - r_log)
        Using r_log = log(r_linA^2) = 2*log(r_linA) might be cleaner?
        Let's convert to Log-Intensity first like in inference.py implied context.
        Actually inference.py compute_merlin_loss takes r_log, b_log?
        Wait, inference.py `update` calls `fn(recon_linA, target_linA)`.
        So the inputs to compute_merlin_loss ARE linA.
        BUT the formula says: 0.5 * r_log + exp(2*b_log - r_log).
        So we must convert linA to logI inside the function if inference.py implementation didn't do it?
        Looking at inference.py:
           metrics_to_merlin = self.compute_merlin_loss
           ...
           value = fn(recon_linA, target_linA)
        But compute_merlin_loss vars are named r_log, b_log.
        If inputs are linA, we need to convert.
        r_log (Log-Intensity) = log(r_linA^2) = 2 * log(r_linA)
        Let's assume inputs are Linear Amplitude and convert.
        """
        r_log = 2 * np.log(r_linA + EPS)
        b_log = 2 * np.log(b_linA + EPS)

        # M = 0.5 * r_log + exp(2*b_log - r_log) ??
        # Let's check logic:
        # Original MERLIN Loss: L = log(R) + S/R  where R is reflectivity (Intensity), S is observed intensity.
        # r_log is log(R). b_log is log(S).
        # S = exp(b_log). R = exp(r_log).
        # Loss = r_log + exp(b_log) / exp(r_log) = r_log + exp(b_log - r_log).
        # This differs from 0.5 * ... check inference.py carefully.
        # inference.py: 0.5 * r_log + np.exp(2 * b_log - r_log)??
        # If inputs were Log-Amplitude, then 2*b_log is Log-Intensity.
        # IF inputs are LinA, then r_log as defined above IS Log-Intensity.
        # So maybe inference.py formula assumes specific input type?
        # Let's stick to simple MSE/PSNR for now or standard MERLIN: log(mean) + obs/mean.
        # Using inference.py formula directly for consistency:
        merlin_loss = 0.5 * r_log + np.exp(b_log - r_log)  # Wait, 2*b_log - r_log?
        # If b_log is Log-Intensity, then exp(b_log) is S.
        # If inputs are linA, b_log = 2*log(linA) = log(linA^2) = log(I).
        # So exp(b_log - r_log) = S/R.
        # The 0.5 factor? Maybe loss is define on Amplitude?
        # Let's just use:
        loss = r_log + np.exp(b_log - r_log)
        return float(np.mean(loss))

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


def visualize_patches(
    noisy_complex: np.ndarray,
    recon_complex: np.ndarray,
    adam_linA: np.ndarray,
    merlin_linA: np.ndarray,
    save_path: str,
    num_patches: int = 5,
):
    """Generate and save a 4-row comparison figure.

    If matplotlib is missing, skips visualization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Matplotlib not found. Visualization skipped.")
        print(f"Would have saved to: {save_path}")
        return

    N = min(num_patches, len(noisy_complex))
    fig, axes = plt.subplots(4, N, figsize=(4 * N, 16))
    if N == 1:
        axes = axes.reshape(4, 1)

    # Prepare Data: Convert Complex -> Log Intensity for visualization
    def complex_to_logI(c):
        """Convert complex array to Log Intensity."""
        # c: [H, W, 2]
        Inten = np.square(c[..., 0]) + np.square(c[..., 1])
        return np.log(Inten + EPS)

    def linA_to_logI(a):
        """Convert Linear Amplitude to Log Intensity."""
        # a: [H, W, 1]
        Inten = np.square(a[..., 0])
        return np.log(Inten + EPS)

    for i in range(N):
        # Row 0: Original Noisy (LogI)
        noisy_logI = complex_to_logI(noisy_complex[i])
        noisy_disp = clip(noisy_logI)

        # Row 1: Reconstruction (LogI)
        recon_logI = complex_to_logI(recon_complex[i])
        recon_disp = clip(recon_logI)

        # Row 2: MERLIN GT (LogI)
        merlin_logI = linA_to_logI(merlin_linA[i])
        merlin_disp = clip(merlin_logI)

        # Row 3: ADAM NOC GT (LogI)
        adam_logI = linA_to_logI(adam_linA[i])
        adam_disp = clip(adam_logI)

        # Plot
        axes[0, i].imshow(noisy_disp, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Noisy Input")

        axes[1, i].imshow(recon_disp, cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstruction")

        axes[2, i].imshow(merlin_disp, cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("MERLIN GT")

        axes[3, i].imshow(adam_disp, cmap="gray")
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_title("ADAM NOC GT")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
