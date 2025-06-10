import os
import warnings
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
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
            raise FileNotFoundError(
                f"No validation patch found in {self.patch_dir}. "
                "Please ensure the directory contains a file starting with 'val_' and ending with '.npy'."
            )

        # Read, convert to Tensor, and add batch dimension
        self.patch = (
            torch.from_numpy(np.load(self.patch_path))
            .to(pl_module.device)
            .unsqueeze(0)
            .float()
        )

        # Load MERLIN Ground Truth
        merlin_gt_dir = self.patch_dir.parent.parent / "denoised"
        for file in os.listdir(merlin_gt_dir):
            if file.startswith("val_") and file.endswith(".npy"):
                self.merlin_gt_path = merlin_gt_dir / file
                merlin_patch_dict = np.load(
                    self.merlin_gt_path, allow_pickle=True
                ).item()
                merlin_patch = merlin_patch_dict["denoised"]["from_real"]
                # Access Lightning Datamodule hdf5_metadata to normalize the patch as needed
                self.log_base = trainer.datamodule.hdf5_metadata.get("log_base", None)
                self.min_val = trainer.datamodule.hdf5_metadata.get("norm_min", None)
                self.max_val = trainer.datamodule.hdf5_metadata.get("norm_max", None)

                if self.log_base == "nat":
                    merlin_patch = np.log(
                        (merlin_patch - self.min_val) / (self.max_val - self.min_val)
                    )
                elif self.log_base == "db":
                    merlin_patch = 10 * np.log10(
                        (merlin_patch - self.min_val) / (self.max_val - self.min_val)
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported log base: {self.log_base}. "
                        "Supported values are 'nat' and 'db'."
                    )

                self.merlin_gt = merlin_patch.astype(np.float32)
                found_patch = True
                break
        if not found_patch:
            warnings.warn(
                f"No MERLIN Ground Truth found in {merlin_gt_dir}. Skipping GT logging."
            )
            self.merlin_gt = None
        if self.merlin_gt_path.name.split("_")[1] != self.patch_path.name.split("_")[1]:
            warnings.warn(
                f"Patch and MERLIN GT filenames do not match: {self.patch_path.name} vs {self.merlin_gt_path.name}. "
                "This may lead to incorrect logging."
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
        # Now we have 3 rows: image with fixed range, histogram, image with natural range
        fig, axs = plt.subplots(3, ncols, figsize=(5 * ncols, 15))

        real = torch.squeeze(real).cpu().numpy()
        imag = torch.squeeze(imag).cpu().numpy()
        output_real = torch.squeeze(output_real).cpu().numpy()
        reflectivity = np.add(real, imag)
        residuals = abs(reflectivity - output_real)

        # Row 0: All plots with vmin=0, vmax=2
        # Column 0: Original reflectivity
        axs[0, 0].set_title(
            f"Reflectivity [0,1] (mean: {reflectivity.mean():.3f}, std: {reflectivity.std():.3f})"
        )
        im1 = axs[0, 0].imshow(reflectivity, cmap="gray", vmin=0, vmax=2)
        axs[0, 0].axis("off")
        fig.colorbar(im1, ax=axs[0, 0], shrink=0.8)

        # Column 1: Reconstructed real part
        axs[0, 1].set_title(
            f"Reconstruction [0,1] (mean: {output_real.mean():.3f}, std: {output_real.std():.3f})"
        )
        im2 = axs[0, 1].imshow(output_real, cmap="gray", vmin=0, vmax=2)
        axs[0, 1].axis("off")
        fig.colorbar(im2, ax=axs[0, 1], shrink=0.8)

        # Column 2: Residuals
        if self.add_residuals:
            axs[0, 2].set_title(
                f"Residuals [0,1] (mean: {residuals.mean():.3f}, std: {residuals.std():.3f})"
            )
            im3 = axs[0, 2].imshow(residuals, cmap="gray", vmin=0, vmax=2)
            axs[0, 2].axis("off")
            fig.colorbar(im3, ax=axs[0, 2], shrink=0.8)

        # Row 1: Histograms
        # Column 0: Original reflectivity histogram
        axs[1, 0].set_title("Reflectivity Histogram")
        axs[1, 0].hist(reflectivity.flatten(), bins=50, alpha=0.5)
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].tick_params(axis="y", labelsize=8)
        axs[1, 0].yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, loc: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}"
            )
        )

        # Column 1: Reconstructed real part histogram
        axs[1, 1].set_title("Reconstruction Histogram")
        axs[1, 1].hist(output_real.flatten(), bins=50, alpha=0.5)
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].tick_params(axis="y", labelsize=8)
        axs[1, 1].yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, loc: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}"
            )
        )

        # Column 2: Residuals histogram
        if self.add_residuals:
            axs[1, 2].set_title("Residuals Histogram")
            axs[1, 2].hist(residuals.flatten(), bins=50, alpha=0.5)
            axs[1, 2].grid(True, alpha=0.3)
            axs[1, 2].tick_params(axis="y", labelsize=8)
            axs[1, 2].yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, loc: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}"
                )
            )

        # Row 2: All plots with natural range (was Row 1 before)
        # Column 0: Original reflectivity
        axs[2, 0].set_title(
            f"Reflectivity (mean: {reflectivity.mean():.3f}, std: {reflectivity.std():.3f})"
        )
        im4 = axs[2, 0].imshow(reflectivity, cmap="gray")
        axs[2, 0].axis("off")
        fig.colorbar(im4, ax=axs[2, 0], shrink=0.8)

        # Column 1: Reconstructed real part
        axs[2, 1].set_title(
            f"Reconstruction (mean: {output_real.mean():.3f}, std: {output_real.std():.3f})"
        )
        im5 = axs[2, 1].imshow(output_real, cmap="gray")
        axs[2, 1].axis("off")
        fig.colorbar(im5, ax=axs[2, 1], shrink=0.8)

        # Column 2: Residuals
        if self.add_residuals:
            axs[2, 2].set_title(plt.subplots_adjust(wspace=0.15, hspace=0.3))
            im6 = axs[2, 2].imshow(residuals, cmap="gray")
            axs[2, 2].axis("off")
            fig.colorbar(im6, ax=axs[2, 2], shrink=0.8)

        plt.subplots_adjust(wspace=0.05, hspace=0.3)

        fig.suptitle(
            f"Val Large patch, epoch {trainer.current_epoch}: "
            f"Loss={out_criterion['loss']:.3f}, MSE={out_criterion['mse']:.3f}, "
            f"SSIM={out_criterion['ssim']:.4f}, MS-SSIM={out_criterion['ms_ssim']:.4f}, "
            f"BPP={out_criterion['bpp_loss']:.4f}"
        )

        # Let's also save comparison with MERLIN GT if available
        # @TODO: once happy with the metrics, move that to a standalone function that automatically takes care of numpoy to torch conversion
        if self.merlin_gt is not None:
            output_real_torch = (
                torch.from_numpy(output_real)
                .to(pl_module.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # Add batch and channel dims
            merlin_gt_torch = (
                torch.from_numpy(self.merlin_gt)
                .to(pl_module.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # Add batch and channel dims
            mse_merlin = torchmetrics.functional.mean_squared_error(
                output_real_torch, merlin_gt_torch
            )
            psnr_merlin = torchmetrics.functional.image.peak_signal_noise_ratio(
                output_real_torch, merlin_gt_torch
            )
            ssim_merlin = (
                torchmetrics.functional.image.structural_similarity_index_measure(
                    output_real_torch, merlin_gt_torch
                )
            )
            ms_ssim_merlin = torchmetrics.functional.image.multiscale_structural_similarity_index_measure(
                output_real_torch, merlin_gt_torch
            )

            fig2, axs = plt.subplots(1, 2)

            im0 = axs[0].imshow(output_real, cmap="gray")
            axs[0].set_title("Reconstruction")
            axs[0].axis("off")
            fig2.colorbar(im0, ax=axs[0], shrink=0.6)

            im1 = axs[1].imshow(self.merlin_gt, cmap="gray")
            axs[1].set_title("MERLIN GT")
            axs[1].axis("off")
            fig2.colorbar(im1, ax=axs[1], shrink=0.6)

            fig2.suptitle(
                f"epoch {trainer.current_epoch}, bpp={out_criterion['bpp_loss']:.4f}, mse={mse_merlin:.4f}, \n "
                f"psnr={psnr_merlin:.4f}, ssim={ssim_merlin:.4f}, ms_ssim={ms_ssim_merlin:.4f}"
            )
        else:
            fig2, ax = plt.subplots()
            ax.imshow(output_real, cmap="gray")
            ax.set_title("Reconstruction")
            ax.set_title(
                f"epoch {trainer.current_epoch}, bpp={out_criterion['bpp_loss']:.4f}"
            )

        pl_module.logger.experiment.log(
            {
                "val_large_patch": fig,
                "val_reconstruction": fig2,
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
        batch_size = trainer.datamodule.batch_size
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
