"""
SAR image logging callback for visualizing results.

This module implements a callback for logging SAR images, despeckling results,
and metrics to Weights & Biases during training.
"""

import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from lightning import Callback, LightningModule, Trainer


class LogReconstructionCallback(Callback):
    """Callback for logging SAR images and despeckling results.

    This callback visualizes:
    - Original SAR real and imaginary parts
    - Original intensity
    - Despeckled reflectivity
    - Rate metrics
    """

    def __init__(
        self,
        log_every_n_epochs: int,
        num_images: int,
        cmap: str = "gray",
        add_residuals: bool = False,
        verbose: bool = False,
    ):
        """Initialize the callback.

        Args:
            log_every_n_epochs: Log frequency in epochs
            num_images: Number of images to log
            cmap: Colormap for visualization
            add_residuals: Whether to add difference between orginal and reconstruction as an additional column
            verbose: Whether to log statistics on stdout
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_images = num_images
        self.cmap = cmap
        self.add_residuals = add_residuals
        self.verbose = verbose

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
        # 0. Only log on specified epochs and for the first batch
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return

        # 1. Check if the batch has at least num_images samples
        batch_size = batch["real"].shape[0]
        if batch_size < self.num_images:
            self.num_images = batch_size
            warnings.warn(
                f"Number of images to log ({self.num_images}) is larger than the batch size ({batch_size}).",
                UserWarning,
            )

        # 2. Forward pass and metrics calculation
        real = batch["real"][: self.num_images]
        imag = batch["imag"][: self.num_images]
        out_criterion, real_output = pl_module._model_forward(real, imag)

        if self.verbose:
            print("@TODO: Add verbose logging")
            print(
                f"Real shape: {real.shape}, Imag shape: {imag.shape}, Real output shape: {real_output.shape}."
            )
            print(f"Out criterion: {out_criterion}")

        # 3. Create a new figure with num_images rows and 2 columns (3 if add_residuals)
        ncols = 2 if not self.add_residuals else 3
        fig, axs = plt.subplots(
            self.num_images, ncols, figsize=(5 * ncols, 5 * self.num_images)
        )

        real = torch.squeeze(real).cpu().numpy()
        real_output = torch.squeeze(real_output).cpu().numpy()

        for i in range(self.num_images):
            # Plot original real part
            im1 = axs[i, 0].imshow(real[i], cmap=self.cmap)
            axs[i, 0].set_title("Original Real")
            axs[i, 0].axis("off")
            fig.colorbar(im1, ax=axs[i, 0], shrink=0.7)

            # Plot reconstructed real part
            im2 = axs[i, 1].imshow(real_output[i], cmap=self.cmap)
            axs[i, 1].set_title("Reconstructed Real")
            axs[i, 1].axis("off")
            fig.colorbar(im2, ax=axs[i, 1], shrink=0.7)

            # Plot residuals if enabled
            if self.add_residuals:
                im3 = axs[i, 2].imshow(abs(real[i] - real_output[i]), cmap=self.cmap)
                axs[i, 2].set_title("Residual")
                axs[i, 2].axis("off")
                fig.colorbar(im3, ax=axs[i, 2], shrink=0.7)

        fig.suptitle(f"Criterion: {out_criterion}")
        plt.tight_layout()
        pl_module.logger.experiment.log({"callback_reconstruction": fig})
        # Close the figure to avoid memory leaks
        plt.close(fig)
