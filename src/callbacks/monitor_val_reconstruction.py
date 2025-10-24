from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer


class MonitorValReconstruction(Callback):
    def __init__(
        self,
        log_every_n_epochs: int,
        num_images: int = 3,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_images = num_images

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        if pl_module.__class__.__name__ == "MerlinModule":
            self.with_compression = False
        elif pl_module.__class__.__name__ == "SARDDCModule":
            self.with_compression = True
        else:
            raise ValueError(f"Unsupported LightningModule class: {pl_module.__class__.__name__}")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log a few validation reconstruction examples."""
        # Only log on specified epochs and for the first batch
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return

        # Get the first few images from the batch
        num_images_to_show = min(self.num_images, batch["real"].shape[0])

        # Prepare input and target deterministically (always use real as input, imag as target)
        input = batch["real"]
        target = batch["imag"]

        # Forward pass to get reconstructions
        with torch.no_grad():
            reconstructions = pl_module(input)
            criterion = pl_module.criterion(reconstructions, target)
            if self.with_compression:
                reconstructions = reconstructions["x_hat"]

        # Create the visualization
        fig, axes = plt.subplots(3, num_images_to_show, figsize=(4 * num_images_to_show, 12))
        if num_images_to_show == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_images_to_show):
            # Get individual images and convert to numpy
            real_i = batch["real"][i, 0].cpu().numpy()  # Remove channel dim
            imag_i = batch["imag"][i, 0].cpu().numpy()  # Remove channel dim
            input_reflectivity_i = real_i + imag_i  # Sum for input reflectivity
            reconstruction_i = reconstructions[i, 0].cpu().numpy()  # Remove channel dim

            # Row 0: Input reflectivity
            im0 = axes[0, i].imshow(input_reflectivity_i, cmap="gray")
            axes[0, i].axis("off")
            fig.colorbar(im0, ax=axes[0, i], shrink=0.6)
            # Row 1: Reconstruction
            im1 = axes[1, i].imshow(reconstruction_i, cmap="gray")
            axes[1, i].axis("off")
            fig.colorbar(im1, ax=axes[1, i], shrink=0.6)
            # Row 2: Residuals (difference)
            residuals = np.abs(input_reflectivity_i - reconstruction_i)
            im2 = axes[2, i].imshow(residuals, cmap="gray")
            axes[2, i].axis("off")
            fig.colorbar(im2, ax=axes[2, i], shrink=0.6)

        # Add Row titles on the left side
        row_titles = [
            "Input Reflectivity\n(Real + Imag)",
            "Reconstruction",
            "Residuals",
        ]
        for i, title in enumerate(row_titles):
            axes[i, 0].text(
                -0.1,
                0.5,
                title,
                transform=axes[i, 0].transAxes,
                fontsize=12,
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                weight="bold",
            )
            # Add overall title with metrics
            title = (
                f"Validation Epoch {trainer.current_epoch} - "
                f"Loss: {criterion['loss']:.3f}, "
                f"MSE: {criterion['mse']:.4f}, "
                f"PSNR: {criterion['psnr']:.2f}dB"
            )

            if self.with_compression:
                title += f", BPP: {criterion['bpp']:.4f}"

            title += "\nMetrics computed between reconstructions (real) and target (imag)"
            fig.suptitle(title)

            plt.tight_layout()

        pl_module.logger.experiment.log(
            {
                "val_reconstructions": fig,
                "val_batch/loss": criterion["loss"],
                "val_batch/mse": criterion["mse"],
                "val_batch/psnr": criterion["psnr"],
            }
        )

        plt.close(fig)
