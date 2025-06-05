import os
import warnings
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer


class LogValidationPatch(Callback):
    def __init__(
        self,
        patch_dir: str,
        log_every_n_epochs: int,
        add_residuals: bool = False,
    ):
        """Initialize the callback.

        Args:
            log_every_n_epochs: Log frequency in epochs
            add_residuals: Whether to add difference between orginal and reconstruction as an additional column
        """
        super().__init__()
        self.patch_dir = Path(patch_dir)
        self.log_every_n_epochs = log_every_n_epochs
        self.add_residuals = add_residuals
        self.is_default_patch = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Find the large patch and convert it to a torch tensor."""
        # For the files in patch_dir find the one that starts with val_ and ends with .npy
        found_patch = False
        for file in os.listdir(self.patch_dir):
            if file.startswith("val_") and file.endswith(".npy"):
                self.patch_path = self.patch_dir / file
                found_patch = True
                break
        if not found_patch:
            warnings.warn(
                f"No patch found in{self.patch_dir}. Using default patch in parent."
            )
            self.patch_path = (
                self.patch_dir.parent / "default_large_validation_patch.npy"
            )
            self.is_default_patch = True

        # Read, convert to Tensor, and add batch dimension
        self.patch = (
            torch.from_numpy(np.load(self.patch_path))
            .to(pl_module.device)
            .unsqueeze(0)
            .float()
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log SAR images and despeckling results on validation batch end."""
        # Only log on specified epochs and for the first batch
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return

        # Forward pass and metrics calculation
        # [1, H, W, 2] -> [1, 1, H, W]
        real = self.patch[..., 0].unsqueeze(1)
        imag = self.patch[..., 1].unsqueeze(1)
        out_criterion, output_real = pl_module._model_forward(real, imag)

        # Always create 3 columns (reflectivity, reconstruction, residuals)
        ncols = 3 if self.add_residuals else 2
        fig, axs = plt.subplots(2, ncols, figsize=(5 * ncols, 10))

        real = torch.squeeze(real).cpu().numpy()
        imag = torch.squeeze(imag).cpu().numpy()
        output_real = torch.squeeze(output_real).cpu().numpy()
        reflectivity = np.add(real, imag) / 2
        residuals = abs(reflectivity - output_real)

        # Row 0: All plots with vmin=0, vmax=1
        # Column 0: Original reflectivity
        axs[0, 0].set_title(
            f"Reflectivity [0,1] (mean: {reflectivity.mean():.3f}, std: {reflectivity.std():.3f})"
        )
        im1 = axs[0, 0].imshow(reflectivity, cmap="gray", vmin=0, vmax=1)
        axs[0, 0].axis("off")
        fig.colorbar(im1, ax=axs[0, 0], shrink=0.8)

        # Column 1: Reconstructed real part
        axs[0, 1].set_title(
            f"Reconstruction [0,1] (mean: {output_real.mean():.3f}, std: {output_real.std():.3f})"
        )
        im2 = axs[0, 1].imshow(output_real, cmap="gray", vmin=0, vmax=1)
        axs[0, 1].axis("off")
        fig.colorbar(im2, ax=axs[0, 1], shrink=0.8)

        # Column 2: Residuals
        if self.add_residuals:
            axs[0, 2].set_title(
                f"Residuals [0,1] (mean: {residuals.mean():.3f}, std: {residuals.std():.3f})"
            )
            im3 = axs[0, 2].imshow(residuals, cmap="gray", vmin=0, vmax=1)
            axs[0, 2].axis("off")
            fig.colorbar(im3, ax=axs[0, 2], shrink=0.8)

        # Row 1: All plots with natural range
        # Column 0: Original reflectivity
        axs[1, 0].set_title(
            f"Reflectivity (mean: {reflectivity.mean():.3f}, std: {reflectivity.std():.3f})"
        )
        im4 = axs[1, 0].imshow(reflectivity, cmap="gray")
        axs[1, 0].axis("off")
        fig.colorbar(im4, ax=axs[1, 0], shrink=0.8)

        # Column 1: Reconstructed real part
        axs[1, 1].set_title(
            f"Reconstruction (mean: {output_real.mean():.3f}, std: {output_real.std():.3f})"
        )
        im5 = axs[1, 1].imshow(output_real, cmap="gray")
        axs[1, 1].axis("off")
        fig.colorbar(im5, ax=axs[1, 1], shrink=0.8)

        # Column 2: Residuals
        if self.add_residuals:
            axs[1, 2].set_title(
                f"Residuals (mean: {residuals.mean():.3f}, std: {residuals.std():.3f})"
            )
            im6 = axs[1, 2].imshow(residuals, cmap="gray")
            axs[1, 2].axis("off")
            fig.colorbar(im6, ax=axs[1, 2], shrink=0.8)

        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        default_patch_warning = (
            "\n /!\\ Default patch used  /!\\." if self.is_default_patch else ""
        )
        fig.suptitle(
            f"Val Large patch, epoch {trainer.current_epoch}: "
            f"Loss={out_criterion['loss']:.3f}, MSE={out_criterion['mse']:.3f}, "
            f"SSIM={out_criterion['ssim']:.4f}, MS-SSIM={out_criterion['ms_ssim']:.4f}, "
            f"BPP={out_criterion['bpp_loss']:.4f}{default_patch_warning}"
        )

        pl_module.logger.experiment.log(
            {
                "val_large_patch": fig,
                "val_patch/loss": out_criterion["loss"],
                "val_patch/bpp_loss": out_criterion["bpp_loss"],
                "val_patch/mse": out_criterion["mse"],
                "val_patch/psnr": out_criterion["psnr"],
                "val_patch/ssim": out_criterion["ssim"],
                "val_patch/ms_ssim": out_criterion["ms_ssim"],
            },
        )
        plt.close(fig)


class LogValidationBatchPlot(Callback):
    """Callback plotting and logging reconstructions results at the end of validation."""

    def __init__(
        self,
        log_every_n_epochs: int,
        num_images: int,
        cmap: str = "gray",
        add_residuals: bool = False,
    ):
        """Initialize the callback.

        Args:
            log_every_n_epochs: Log frequency in epochs
            num_images: Number of images to log
            cmap: Colormap for visualization
            add_residuals: Whether to add difference between orginal and reconstruction as an additional column
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_images = num_images
        self.cmap = cmap
        self.add_residuals = add_residuals
        self.image_indices = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Fix the image indices to log, to always log the same images."""
        batch_size = trainer.datamodule.hparams.batch_size
        if batch_size <= self.num_images:
            self.image_indices = torch.arange(batch_size)
        else:
            self.image_indices = torch.randperm(batch_size)[: self.num_images]

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log SAR images and despeckling results on validation batch end."""
        # Only log on specified epochs and for the first batch
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return

        # Forward pass and metrics calculation
        real = batch["real"][self.image_indices]
        imag = batch["imag"][self.image_indices]
        out_criterion, output_real = pl_module._model_forward(real, imag)

        ncols = 2 if not self.add_residuals else 3
        fig, axs = plt.subplots(
            self.num_images, ncols, figsize=(5 * ncols, 5 * self.num_images)
        )

        real = torch.squeeze(real).cpu().numpy()
        imag = torch.squeeze(imag).cpu().numpy()
        output_real = torch.squeeze(output_real).cpu().numpy()

        for i in range(self.num_images):
            reflectivity = np.add(real[i], imag[i]) / 2
            # Plot original reflectivity
            im1 = axs[i, 0].imshow(reflectivity, cmap=self.cmap, vmin=0, vmax=1)
            axs[i, 0].axis("off")
            # Add colorbar for the first image
            fig.colorbar(im1, ax=axs[i, 0])

            # Plot reconstructed real part
            im2 = axs[i, 1].imshow(output_real[i], cmap=self.cmap, vmin=0, vmax=1)
            axs[i, 1].axis("off")
            fig.colorbar(im2, ax=axs[i, 1])

            # Plot residuals if enabled
            if self.add_residuals:
                residuals = abs(reflectivity - output_real[i])
                im3 = axs[i, 2].imshow(residuals, cmap=self.cmap, vmin=0, vmax=1)
                axs[i, 2].axis("off")
                fig.colorbar(im3, ax=axs[i, 2])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        # Add a single colorbar after tight_layout or adjust_subplots
        # fig.colorbar(
        #     ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="gray"),
        #     ax=axs,
        #     location="right",
        #     shrink=0.8,
        #     aspect=30,
        # )
        column_titles = ["Input (Reflectivity)", "Reconstruction"]
        if self.add_residuals:
            column_titles.append("Residual (abs)")
        for ax, col in zip(axs[0], column_titles):
            ax.set_title(col)

        fig.suptitle(
            f"epoch {trainer.current_epoch}: Loss={out_criterion['loss']:.3f}, MSE={out_criterion['mse']:.3f}, SSIM={out_criterion['ssim']:.4f}, MS-SSIM={out_criterion['ms_ssim']:.4f} , BPP={out_criterion['bpp_loss']:.4f}\n input: {real.shape}, {real.min():.4f}, {real.max():.4f}, {real.mean():.4f}, {real.std():.4f}\n output: {output_real.shape}, {output_real.min():.4f}, {output_real.max():.4f}, {output_real.mean():.4f}, {output_real.std():.4f}",
        )
        # plt.tight_layout()
        pl_module.logger.experiment.log({"callback_reconstruction": fig})
        plt.close(fig)
