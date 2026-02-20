import warnings
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from matplotlib.ticker import FuncFormatter

from src.utils.constants import AMP_MAX, AMP_MIN, EPS
from src.utils.metrics import get_all_distortion_metrics, ms_ssim, mse, psnr, ssim
from src.utils.processing_utils import clip, process_large_patch
from src.utils.sar_utils import symmetrize


class CompareReconstructionToGT(Callback):
    """Callback to compare model reconstructions to MERLIN_DDS ground truth on a large validation
    patch."""

    def __init__(
        self,
        patch_dir: str,
        log_every_n_epochs: int,
        split_large_patch: bool = False,
        blend_method: str = "linear",
        stride: int = -1,
        verbose: bool = False,
    ):
        super().__init__()
        self.patch_dir = Path(patch_dir) / "visualization/Hamburg_[11000:12024-8500:9524]/"
        self.log_every_n_epochs = log_every_n_epochs
        # --- Details for clipping ---
        self.clip_for_visualization = True  # Enable or disable clipping
        self.mean_std_norm = True  # True: use mean/std, False use percentiles
        self.clip_factor = 3  # Clip to mean +/- self.clip_factor * std
        self.clip_percentiles = (5, 95)  # Clip to these percentiles
        # --- Processing large patch as small patches or not ---
        self.split_large_patch = split_large_patch
        self.blend_method = blend_method
        self.stride = stride

        self.with_compression = None
        self.verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Find the large patch and convert it to a torch tensor."""
        if self.verbose:
            print(
                f"\n[CompareReconstructionToGT] Setting up Callback. {self.clip_for_visualization=}, {self.clip_factor=}, {self.split_large_patch=} ({self.blend_method=}, {self.stride=})"
            )
            print(
                f"    Called with {pl_module.__class__.__name__}: net = {pl_module.net.__class__.__name__}, criterion = {pl_module.criterion.__class__.__name__}."
            )
        if pl_module.__class__.__name__ == "MerlinModule":
            self.with_compression = False
        elif pl_module.__class__.__name__ == "SARDDCModule":
            self.with_compression = True
        else:
            raise ValueError(f"Unsupported LightningModule class: {pl_module.__class__.__name__}")
        # ----- Load the noisy patch -----
        # For the files in patch_dir find the one that starts with sym_ and ends with .npy
        found_patch = False
        for file in self.patch_dir.glob("sym_Noisy.npy"):
            self.patch_path = file
            found_patch = True
            break
        if not found_patch:
            raise FileNotFoundError(
                f"No symmetrized patch found in {self.patch_dir}. "
                "Please ensure the directory contains a file starting with 'sym_' and ending with '.npy'."
            )

        # --- load and symmetrize ---
        patch_data = np.load(self.patch_path)  # [H, W, 2]
        if self.verbose:
            print(f"    Loaded Symmetrized PATCH from {self.patch_path}.")
            print(
                f"        Symmetrized PATCH (shape={patch_data.shape}) statistics: min={patch_data.min():.4f}, max={patch_data.max():.4f}, mean={patch_data.mean():.4f}, std={patch_data.std():.4f}. Is NaN={np.isnan(patch_data).any()}."
            )

        # --- Prepare noisy patch data as numpy arrays for visualization ---
        noisy_linI = np.square(patch_data[:, :, 0]) + np.square(patch_data[:, :, 1])
        self.noisy_linA = np.sqrt(noisy_linI)
        self.noisy_logI = np.log(noisy_linI + EPS)
        del noisy_linI
        if self.verbose:
            print(
                f"        NOISY LIN-A (shape={self.noisy_linA.shape}) statistics: min={self.noisy_linA.min():.4f}, max={self.noisy_linA.max():.4f}, mean={self.noisy_linA.mean():.4f}, std={self.noisy_linA.std():.4f}. Is NaN={np.isnan(self.noisy_linA).any()}."
            )
            print(
                f"        NOISY LOG-I (shape={self.noisy_logI.shape}) statistics: min={self.noisy_logI.min():.4f}, max={self.noisy_logI.max():.4f}, mean={self.noisy_logI.mean():.4f}, std={self.noisy_logI.std():.4f}. Is NaN={np.isnan(self.noisy_logI).any()}."
            )

        # --- Store as torch tensors on device for forward passes ---
        patch_tensor = torch.from_numpy(patch_data).to(pl_module.device).float()
        # NO NORMALIZATION, IT'S DONE IN model.forward()
        # Add batch and channel dimensions
        self.patch = patch_tensor.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # [1, 2, H, W]
        del patch_tensor, patch_data

        # ----- Load MERLIN_DDS Ground Truth -----
        found_merlin = False
        for file in self.patch_dir.glob("linA_MERLIN_DDS.npy"):
            self.merlin_gt_path = file
            self.merlin_linA = np.load(self.merlin_gt_path)

            # Denoised image from MERLIN_DDS comes in linear amplitude scale, see https://github.com/hi-paris/deepdespeckling
            self.merlin_logI = np.log(np.square(self.merlin_linA) + EPS)

            if self.verbose:
                print(f"    Loaded MERLIN_DDS GT from {self.merlin_gt_path}.")
                print(
                    f"        MERLIN_DDS LIN-A (shape={self.merlin_linA.shape}) statistics: min={self.merlin_linA.min():.4f}, max={self.merlin_linA.max():.4f}, mean={self.merlin_linA.mean():.4f}, std={self.merlin_linA.std():.4f}. Is NaN={np.isnan(self.merlin_linA).any()}."
                )

            # Quick print metrics between noisy and MERLIN_DDS GT
            metrics = get_all_distortion_metrics(self.noisy_linA, self.merlin_linA)
            if self.verbose:
                print("        Initial metrics between Noisy and MERLIN_DDS GT:", end="")
                for key, value in metrics.items():
                    print(f" {key}={value:.4f}", end=",")
                print()
            found_merlin = True
            break

        if not found_merlin:
            warnings.warn(
                f"No MERLIN_DDS Ground Truth found in {self.patch_dir}. Skipping GT logging."
            )
            self.merlin_linA = None
            self.merlin_logI = None

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the final reconstruction of the large patch at the end of testing."""
        if self.with_compression:
            recon = pl_module.forward(self.patch)
            criterion = pl_module.criterion(recon, self.patch)
            recon = recon["x_hat"]
        else:
            recon_real = pl_module.forward(self.patch[:, 0:1, :, :])
            recon_imag = pl_module.forward(self.patch[:, 1:2, :, :])
            recon = torch.cat([recon_real, recon_imag], dim=1)
            criterion = pl_module.criterion(recon, self.patch)

        # ----- Denorm the reconstructions  -----
        recon_denorm = recon * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = torch.exp(recon_denorm)
        recon_linI = 0.5 * (
            torch.square(recon_lin[:, 0, :, :]) + torch.square(recon_lin[:, 1, :, :])
        )
        recon_linA = torch.sqrt(recon_linI).squeeze().cpu().numpy()
        recon_logI = torch.log(recon_linI + EPS).squeeze().cpu().numpy()
        if self.clip_for_visualization:
            recon_logI = clip(
                recon_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
            )

        # compute metrics
        metrics_to_merlin = self._compute_metrics_to_merlin(criterion, recon_linA)

        # Save image locally
        log_dir = Path(trainer.log_dir) if trainer.log_dir else Path(trainer.default_root_dir)
        save_path = log_dir / "reconstruction_test.png"

        plt.imsave(save_path, recon_logI, cmap="gray")
        if self.verbose:
            print(f"[CompareReconstructionToGT] Saved test reconstruction to {save_path}")

        # Log to WandB
        if (
            pl_module.logger is not None
            and hasattr(pl_module.logger, "experiment")
            and hasattr(pl_module.logger.experiment, "log")
        ):
            # Create a caption from metrics
            # Filter out -1.0 metrics for cleaner caption
            valid_metrics = {k: v for k, v in metrics_to_merlin.items() if v != -1.0}
            caption = ", ".join([f"{k}={v:.4f}" for k, v in valid_metrics.items()])

            pl_module.logger.experiment.log(
                {"test/reconstruction_image": wandb.Image(str(save_path), caption=caption)}
            )

    def _compute_metrics_to_merlin(self, criterion: dict, recon_linA: np.ndarray) -> dict:
        """Compute distortion metrics between reconstruction and MERLIN_DDS GT in LINEAR-
        AMPLITUDE."""
        metrics_to_merlin = {"mse": -1.0, "psnr": -1.0, "bpp": -1.0, "ssim": -1.0, "ms_ssim": -1.0}
        metrics_to_merlin["loss"] = criterion["loss"].item()
        if self.merlin_linA is not None:
            for key, value in get_all_distortion_metrics(recon_linA, self.merlin_linA).items():
                metrics_to_merlin[key] = value
        if self.with_compression:
            metrics_to_merlin["bpp"] = criterion["bpp"].item()
        return metrics_to_merlin

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log reconstruction comparison with MERLIN_DDS GT."""
        # Only log on specified epochs and for the first batch
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return

        if self.verbose:
            print(f"\n[CompareReconstructionToGT] Epoch {trainer.current_epoch}.")

        # ----- Forward pass to get reconstruction and metrics -----
        with torch.no_grad():
            if self.split_large_patch:
                criterion, recon = process_large_patch(
                    model=pl_module,
                    input=self.patch,
                    target=self.patch,
                    stride=self.stride,
                    blend_method=self.blend_method,
                )
            else:
                if self.with_compression:
                    recon = pl_module.forward(self.patch)
                    criterion = pl_module.criterion(recon, self.patch)
                    recon = recon["x_hat"]
                else:
                    recon_real = pl_module.forward(self.patch[:, 0:1, :, :])
                    recon_imag = pl_module.forward(self.patch[:, 1:2, :, :])
                    recon = torch.cat([recon_real, recon_imag], dim=1)
                    criterion = pl_module.criterion(recon, self.patch)

        # ----- Denorm the reconstructions  -----
        recon_denorm = recon * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = torch.exp(recon_denorm)
        recon_linI = 0.5 * (
            torch.square(recon_lin[:, 0, :, :]) + torch.square(recon_lin[:, 1, :, :])
        )
        recon_linA = torch.sqrt(recon_linI).squeeze().cpu().numpy()
        recon_logI = torch.log(recon_linI + EPS).squeeze().cpu().numpy()
        if self.verbose:
            print(
                f"    RECON LIN-A: min={recon_linA.min():.4f}, max={recon_linA.max():.4f}, mean={recon_linA.mean():.4f}, std={recon_linA.std():.4f}. Is NaN={np.isnan(recon_linA).any()}."
            )
            print(
                f"    NOISY LIN-A: min={self.noisy_linA.min():.4f}, max={self.noisy_linA.max():.4f}, mean={self.noisy_linA.mean():.4f}, std={self.noisy_linA.std():.4f}."
            )

        fig_A, metrics_to_merlin = self._visualize_with_histograms(
            recon_linA,
            recon_logI,
            criterion,
            trainer,
        )

        # Log to WandB if available
        if (
            pl_module.logger is not None
            and hasattr(pl_module.logger, "experiment")
            and self.merlin_linA is not None
        ):
            dict_to_log = {
                f"val_large_patch/{key}_to_MERLIN": value if key not in ["loss", "bpp"] else None
                for key, value in get_all_distortion_metrics(recon_linA, self.merlin_linA).items()
            }
            pl_module.logger.experiment.log(  # type: ignore[attr-defined]
                {
                    "val_large_patch_comparison": fig_A,
                    "val_large_patch/loss": metrics_to_merlin["loss"],
                    "val_large_patch/bpp": metrics_to_merlin["bpp"],
                    **dict_to_log,
                }
            )

        plt.close(fig_A)

    def _visualize_with_histograms(
        self,
        recon_linA: np.ndarray,
        recon_logI: np.ndarray,
        criterion: dict,
        trainer: Trainer,
    ) -> tuple[Any, dict]:
        """Visualize the reconstruction, noisy input, MERLIN_DDS GT (if available) and their
        histograms."""
        metrics_to_merlin = self._compute_metrics_to_merlin(criterion, recon_linA)

        # ----- Prepare images for visualization in LOG-I-----
        if self.clip_for_visualization:
            noisy_logI = clip(
                self.noisy_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
            )
            recon_logI = clip(
                recon_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
            )
            if self.merlin_logI is not None:
                merlin_logI = clip(
                    self.merlin_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
                )

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        # ----- Row 1: Images -----
        # Original
        im0 = axes[0, 0].imshow(noisy_logI, cmap="gray")
        axes[0, 0].set_title("Noisy Log-I")
        axes[0, 0].axis("off")
        fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

        # Reconstruction
        im1 = axes[0, 1].imshow(recon_logI, cmap="gray")
        axes[0, 1].set_title("Recon Log-I")
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

        # MERLIN_DDS GT (if available)
        if self.merlin_logI is not None:
            im3 = axes[0, 2].imshow(merlin_logI, cmap="gray")
            axes[0, 2].set_title("MERLIN_DDS GT Log-I")
            axes[0, 2].axis("off")
            fig.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "MERLIN_DDS GT\nNot Available",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
            )
            axes[0, 2].axis("off")

        # ----- Row 2: Histograms -----
        def plot_histogram(ax, data, title):
            # Add a small check for NaN
            if np.isnan(data).any():
                ax.text(
                    0.5,
                    0.5,
                    "Data contains NaN\nHistogram cannot be displayed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                return

            ax.set_title(title)
            ax.hist(data.flatten(), bins=50, alpha=0.7, color="blue")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="y", labelsize=8)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, loc: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}")
            )
            mean = data.mean()
            std = data.std()
            ax.axvline(mean, color="red", linestyle="--", label="Mean")
            ax.axvline(
                mean + self.clip_factor * std,
                color="green",
                linestyle="--",
                label=f"Mean + {self.clip_factor}*Std",
            )
            ax.axvline(
                mean - self.clip_factor * std,
                color="green",
                linestyle="--",
                label=f"Mean - {self.clip_factor}*Std",
            )
            ax.legend(fontsize=8)

        # Noisy histogram
        plot_histogram(axes[1, 0], noisy_logI, "Noisy LOG-I Histogram")

        # Reconstruction histogram
        plot_histogram(axes[1, 1], recon_logI, "Recon LOG-I Histogram")

        # MERLIN_DDS GT histogram (if available)
        if self.merlin_logI is not None:
            plot_histogram(axes[1, 2], merlin_logI, "MERLIN_DDS GT LOG-I Histogram")
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "MERLIN_DDS GT\nHistogram\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
            axes[1, 2].axis("off")

        # ----- Add overall title with metrics -----
        fig.suptitle(
            f"Val Large patch ({'clipped and normalized' if self.clip_for_visualization else 'raw'}), epoch {trainer.current_epoch}: "
            f"Loss={metrics_to_merlin['loss']:.3f}, BPP={metrics_to_merlin['bpp']:.4f}."
            f"\n metrics to MERLIN_DDS GT (LIN-A): MSE={metrics_to_merlin['mse']:.4f}, PSNR={metrics_to_merlin['psnr']:.2f}dB, SSIM={metrics_to_merlin['ssim']:.4f}, MS-SSIM={metrics_to_merlin['ms_ssim']:.4f}",
            fontsize=14,
        )

        plt.tight_layout()

        return fig, metrics_to_merlin
