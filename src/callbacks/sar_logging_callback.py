"""
SAR image logging callback for visualizing results.

This module implements a callback for logging SAR images, despeckling results,
and metrics to Weights & Biases during training.
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import wandb
from lightning import Callback, LightningModule, Trainer

from src.utils.sar_utils import calculate_equivalent_number_of_looks, visualize_sar


class SARVisualizationCallback(Callback):
    """Callback for logging SAR images and despeckling results.

    This callback visualizes:
    - Original SAR real and imaginary parts
    - Original intensity
    - Despeckled reflectivity
    - Rate metrics
    """

    def __init__(self, log_every_n_epochs: int = 5, max_samples: int = 4):
        """Initialize the callback.

        Args:
            log_every_n_epochs: Log frequency in epochs (default: 5)
            max_samples: Maximum number of samples to log (default: 4)
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples

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

        # Get real and imaginary parts and intensity from batch
        real_squared = batch["real"]
        imag_squared = batch["imag"]
        intensity = real_squared + imag_squared

        # Limit the number of samples to log
        n_samples = min(real_squared.size(0), self.max_samples)

        # Process real part
        real_output = pl_module.model(real_squared[:n_samples], training=False)
        real_x_hat = real_output["x_hat"]

        # Process imaginary part
        imag_output = pl_module.model(imag_squared[:n_samples], training=False)
        imag_x_hat = imag_output["x_hat"]

        # Average to get reflectivity estimate
        reflectivity = (real_x_hat + imag_x_hat) / 2

        # Calculate rate
        real_bpp = pl_module.calculate_bpp(
            real_output["likelihoods"], real_squared[:n_samples].shape
        )
        imag_bpp = pl_module.calculate_bpp(
            imag_output["likelihoods"], imag_squared[:n_samples].shape
        )
        total_bpp = real_bpp + imag_bpp

        # Log images to wandb
        if isinstance(trainer.logger, list):
            for logger in trainer.logger:
                if (
                    hasattr(logger, "experiment")
                    and logger.__class__.__name__ == "WandbLogger"
                ):
                    self._log_to_wandb(
                        logger.experiment,
                        real_squared[:n_samples],
                        imag_squared[:n_samples],
                        reflectivity,
                        intensity[:n_samples] if intensity is not None else None,
                        real_bpp,
                        imag_bpp,
                        total_bpp,
                        trainer.current_epoch,
                    )
        elif (
            hasattr(trainer.logger, "experiment")
            and trainer.logger.__class__.__name__ == "WandbLogger"
        ):
            self._log_to_wandb(
                trainer.logger.experiment,
                real_squared[:n_samples],
                imag_squared[:n_samples],
                reflectivity,
                intensity[:n_samples] if intensity is not None else None,
                real_bpp,
                imag_bpp,
                total_bpp,
                trainer.current_epoch,
            )

    def _log_to_wandb(
        self,
        experiment,
        real_squared,
        imag_squared,
        reflectivity,
        intensity,
        real_bpp,
        imag_bpp,
        total_bpp,
        epoch,
    ):
        """Log results to Weights & Biases."""
        images = []
        captions = []

        # Loop through each sample
        for i in range(real_squared.size(0)):
            # Extract single sample
            real_sample = real_squared[i]
            imag_sample = imag_squared[i]
            reflectivity_sample = reflectivity[i]
            intensity_sample = intensity[i] if intensity is not None else None

            # Create visualization figure
            fig = visualize_sar(
                real_sample, imag_sample, intensity_sample, reflectivity_sample
            )

            # Add to lists
            images.append(wandb.Image(fig))

            # Generate caption with rate information
            caption = f"Real bpp: {real_bpp:.4f}, Imag bpp: {imag_bpp:.4f}, Total: {total_bpp:.4f}"
            captions.append(caption)

            # Close figure to free memory
            plt.close(fig)

            # Calculate ENL if intensity is available
            if intensity_sample is not None:
                enl_metrics = calculate_equivalent_number_of_looks(
                    reflectivity_sample, intensity_sample
                )
                experiment.log(
                    {
                        f"ENL/sample_{i}/original": enl_metrics["enl_original"],
                        f"ENL/sample_{i}/despeckled": enl_metrics["enl_despeckled"],
                        f"ENL/sample_{i}/improvement": enl_metrics["improvement"],
                    }
                )

        # Log all images with captions
        experiment.log(
            {f"val_images/epoch_{epoch}": images, "captions": captions, "epoch": epoch}
        )
