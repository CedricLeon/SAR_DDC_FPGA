import warnings
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.constants import amp_max, amp_min


class CompareReconstructionToGT(Callback):
    def __init__(
        self,
        patch_dir: str,
        log_every_n_epochs: int,
    ):
        super().__init__()
        self.patch_dir = Path(patch_dir) / "visualization"
        self.log_every_n_epochs = log_every_n_epochs
        self.eps = 1e-2
        self.clip_and_norm = True

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Find the large patch and convert it to a torch tensor."""
        # ----- Load the noisy patch -----
        # For the files in patch_dir find the one that starts with noisy_ and ends with .npy
        found_patch = False
        for file in self.patch_dir.glob("noisy_*.npy"):
            self.patch_path = file
            found_patch = True
            break
        if not found_patch:
            raise FileNotFoundError(
                f"No validation patch found in {self.patch_dir}. "
                "Please ensure the directory contains a file starting with 'val_' and ending with '.npy'."
            )

        # Read and prepare noisy patch data as numpy arrays for visualization
        patch_data = np.load(self.patch_path)  # [H, W, 2]
        I_noisy = patch_data[:, :, 0] ** 2 + patch_data[:, :, 1] ** 2
        self.A_noisy = np.sqrt(I_noisy)
        self.logI_noisy = np.log(I_noisy + self.eps)

        if self.clip_and_norm:
            self.A_noisy = self._clip_and_minmax_normalize(self.A_noisy)
            self.logI_noisy = self._clip_and_minmax_normalize(self.logI_noisy)

        # Store as torch tensors on device for forward passes
        patch_tensor = torch.from_numpy(patch_data).to(pl_module.device)
        # Normalize
        patch = torch.square(patch_tensor)
        patch = torch.log(patch + self.eps)
        patch = (patch - amp_max) / (amp_min - amp_max)
        # Add batch and channel dimensions
        self.real_tensor = patch[:, :, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        self.imag_tensor = patch[:, :, 1].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # ----- Load MERLIN Ground Truth -----
        found_merlin = False
        for file in self.patch_dir.glob("denoised_by_MERLIN_*.npy"):
            self.merlin_gt_path = file
            merlin_patch_dict = np.load(self.merlin_gt_path, allow_pickle=True).item()

            # Normalize the patch: log, clip +- 3 std, and scale (0,1)
            self.A_merlin = merlin_patch_dict["denoised"]["full"]
            self.logI_merlin = np.log(self.A_merlin**2 + self.eps)

            if self.clip_and_norm:
                self.A_merlin = self._clip_and_minmax_normalize(self.A_merlin)
                self.logI_merlin = self._clip_and_minmax_normalize(self.logI_merlin)

            print(f"Loaded MERLIN GT from {self.merlin_gt_path}.")
            print(
                f"MERLIN LINEAR-AMPLITUDE statistics: min={self.A_merlin.min():.4f}, max={self.A_merlin.max():.4f}, mean={self.A_merlin.mean():.4f}, std={self.A_merlin.std():.4f}. Is NaN={np.isnan(self.A_merlin).any()}."
            )
            print(
                f"MERLIN LOG-INTENSITY statistics: min={self.logI_merlin.min():.4f}, max={self.logI_merlin.max():.4f}, mean={self.logI_merlin.mean():.4f}, std={self.logI_merlin.std():.4f}. Is NaN={np.isnan(self.logI_merlin).any()}."
            )
            found_merlin = True
            break

        if not found_merlin:
            warnings.warn(
                f"No MERLIN Ground Truth found in {self.patch_dir}. Skipping GT logging."
            )
            self.A_merlin = None
            self.logI_merlin = None
        elif (
            self.merlin_gt_path.name.split("_")[3] != self.patch_path.name.split("_")[1]
        ):
            warnings.warn(
                f"Patch and MERLIN GT filenames do not match: {self.patch_path.name} vs {self.merlin_gt_path.name}. "
                "This may lead to incorrect logging."
            )

    def _clip_and_minmax_normalize(self, img: np.ndarray) -> np.ndarray:
        img = img.clip(img.mean() - 3 * img.std(), img.mean() + 3 * img.std())
        img = (img - img.min()) / (img.max() - img.min())
        return img

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log reconstruction comparison with MERLIN GT."""
        # Only log on specified epochs and for the first batch
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return
        print(
            f"Logging reconstruction comparison to MERLIN GT for epoch {trainer.current_epoch}, batch {batch_idx}."
        )

        # ----- Forward pass to get reconstruction and metrics -----
        with torch.no_grad():
            out_criterion_real, recon_real = pl_module._model_forward(
                self.real_tensor, self.imag_tensor
            )
            out_criterion_imag, recon_imag = pl_module._model_forward(
                self.imag_tensor, self.real_tensor
            )

        # ----- Denorm the reconstructions -----
        recon_real = torch.exp(recon_real.squeeze() * (amp_max - amp_min) + amp_min)
        recon_imag = torch.exp(recon_imag.squeeze() * (amp_max - amp_min) + amp_min)
        # Build full amplitude reconstruction
        I_recon = 0.5 * (torch.square(recon_real) + torch.square(recon_imag))
        A_recon = torch.sqrt(I_recon)
        logI_recon = torch.log(I_recon + self.eps)

        # Bring to numpy for visualization
        A_recon = A_recon.squeeze().cpu().numpy()
        logI_recon = logI_recon.squeeze().cpu().numpy()

        if self.clip_and_norm:
            A_recon = self._clip_and_minmax_normalize(A_recon)
            logI_recon = self._clip_and_minmax_normalize(logI_recon)

        # fig_A, _ = self._visualize_with_histograms(
        #     self.A_noisy,
        #     A_recon,
        #     self.A_merlin,
        #     out_criterion_real,
        #     out_criterion_imag,
        #     trainer,
        #     scale="A",
        # )
        fig_logI, metrics = self._visualize_with_histograms(
            self.logI_noisy,
            logI_recon,
            self.logI_merlin,
            out_criterion_real,
            out_criterion_imag,
            trainer,
            scale="logI",
        )

        pl_module.logger.experiment.log(
            {
                "val_large_patch_comparison": fig_logI,
                "val_large_patch/loss": metrics["loss"],
                "val_large_patch/bpp": metrics["bpp"],
                "val_large_patch/mse_to_MERLIN": metrics["mse"],
                "val_large_patch/psnr_to_MERLIN": metrics["psnr"],
            }
        )

        # plt.close(fig_A)
        plt.close(fig_logI)

    def _visualize_with_histograms(
        self,
        noisy: np.ndarray,
        recon: np.ndarray,
        merlin: np.ndarray | None,
        criterion_real: dict,
        criterion_imag: dict,
        trainer: Trainer,
        scale: str = "logI",
    ) -> tuple[plt.Figure, dict]:
        if scale == "logI":
            subtitles = ["Noisy Log-I", "Recon Log-I", "MERLIN GT Log-I"]
        elif scale == "A":
            subtitles = ["Noisy Amplitude", "Recon Amplitude", "MERLIN GT Amplitude"]
        else:
            raise ValueError(f"Unknown scale: {scale}")

        # ----- Create visualization -----
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Images
        # Original amplitude (sum of real + imag)
        im0 = axes[0, 0].imshow(noisy, cmap="gray")
        axes[0, 0].set_title(subtitles[0])
        axes[0, 0].axis("off")
        fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

        # Reconstruction
        im1 = axes[0, 1].imshow(recon, cmap="gray")
        axes[0, 1].set_title(subtitles[1])
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

        # MERLIN GT (if available)
        if merlin is not None:
            im2 = axes[0, 2].imshow(merlin, cmap="gray")
            axes[0, 2].set_title(subtitles[2])
            axes[0, 2].axis("off")
            fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "MERLIN GT\nNot Available",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
            )
            axes[0, 2].axis("off")

        # Row 2: Histograms
        # Original histogram
        axes[1, 0].hist(noisy.flatten(), bins=50, alpha=0.7, color="blue")
        axes[1, 0].set_title("Noisy Histogram")

        # Reconstruction histogram
        axes[1, 1].hist(recon.flatten(), bins=50, alpha=0.7, color="blue")
        axes[1, 1].set_title("Recon Histogram")

        # MERLIN GT histogram (if available)
        if merlin is not None:
            axes[1, 2].hist(merlin.flatten(), bins=50, alpha=0.7, color="blue")
            axes[1, 2].set_title("MERLIN GT Histogram")
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "MERLIN GT\nHistogram\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
            axes[1, 2].axis("off")

        # ----- Add overall title with metrics -----\
        metrics = {}
        metrics["loss"] = (
            criterion_real["loss"].item() + criterion_imag["loss"].item()
        ) / 2
        metrics["bpp"] = (
            criterion_real["bpp"].item() + criterion_imag["bpp"].item()
        ) / 2
        # Compute MSE, PSNR between reconstructions and MERLIN GT
        if merlin is not None:
            logI_diff = recon - merlin
            metrics["mse"] = np.mean((logI_diff) ** 2)
            peak = merlin.max()  # Should be amp_max - amp_min?
            metrics["psnr"] = (
                20 * np.log10(peak / np.sqrt(metrics["mse"]))
                if metrics["mse"] > 0
                else float("inf")
            )
            # ssim
        else:
            metrics["mse"] = metrics[
                "psnr"
            ] = -1  # Not available if MERLIN GT is not loaded

        fig.suptitle(
            f"Val Large patch, epoch {trainer.current_epoch}: "
            f"Loss={metrics['loss']:.3f}, BPP={metrics['bpp']:.4f}."
            f"\n Metrics to MERLIN GT: MSE={metrics['mse']:.4f}, PSNR={metrics['psnr']:.2f}dB, Diff mean={logI_diff.mean():.4f} and Diff std={logI_diff.std():.4f}",
            fontsize=14,
        )

        plt.tight_layout()

        return fig, metrics
