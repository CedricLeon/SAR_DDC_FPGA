from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.constants import EPS, amp_max, amp_min
from src.utils.processing_utils import clip


class MonitorValReconstruction(Callback):
    def __init__(
        self,
        log_every_n_epochs: int,
        num_images: int = 3,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_images = num_images
        # --- Details for clipping ---
        self.clip_for_visualization = True  # Enable or disable clipping
        self.mean_std_norm = False  # True: use mean/std, False use percentiles
        self.clip_factor = 3  # Clip to mean +/- self.clip_factor * std
        self.clip_percentiles = (5, 95)  # Clip to these percentiles

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Determine if the model uses compression based on its class name."""
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
        print(f"\n[MonitorValReconstruction] Epoch {trainer.current_epoch}.")

        # Get the first few images from the batch
        num_images_to_show = min(self.num_images, batch["real"].shape[0])

        # Prepare input and target deterministically (always use real as input, imag as target)
        input = torch.cat((batch["real"], batch["imag"]), dim=1).contiguous()

        # Forward pass to get reconstructions
        with torch.no_grad():
            reconstructions = pl_module(input)
            criterion = pl_module.criterion(reconstructions, target=input)
            if self.with_compression:
                reconstructions = reconstructions["x_hat"]
        # Denormalize reconstructions
        recon_denorm = reconstructions * (amp_max - amp_min) + amp_min
        recon = torch.exp(recon_denorm)
        recon_amp = torch.sqrt(
            0.5 * (torch.square(recon[:, 0, :, :]) + torch.square(recon[:, 1, :, :]))
        )  # [B, H, W]

        # Create the visualization
        fig, axes = plt.subplots(3, num_images_to_show, figsize=(4 * num_images_to_show, 12))
        if num_images_to_show == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_images_to_show):
            # Get individual images and convert to numpy
            real_i = batch["real"][i, 0].cpu().numpy()  # Remove channel dim
            imag_i = batch["imag"][i, 0].cpu().numpy()  # Remove channel dim
            noisy_amp_i = np.sqrt(
                np.square(real_i) + np.square(imag_i)
            )  # Sum for input reflectivity
            recon_amp_i = recon_amp[i].cpu().numpy()  # Remove channel dim
            if self.clip_for_visualization:
                noisy_amp_i = clip(
                    noisy_amp_i,
                    mean_std_norm=self.mean_std_norm,
                    clip_factor=self.clip_factor,
                    percentiles=self.clip_percentiles,
                )
                recon_amp_i = clip(
                    recon_amp_i,
                    mean_std_norm=self.mean_std_norm,
                    clip_factor=self.clip_factor,
                    percentiles=self.clip_percentiles,
                )
                clip_info = f" (clipped with {'mean/std' if self.mean_std_norm else f'percentiles {self.clip_percentiles}'})"
            else:
                clip_info = " (no clipping)"
            print(
                f"    RECON N°{i} AMPLITUDE{clip_info}: min={recon_amp_i.min():.4f}, max={recon_amp_i.max():.4f}, mean={recon_amp_i.mean():.4f}, std={recon_amp_i.std():.4f}. Is NaN={np.isnan(recon_amp_i).any()}."
            )
            print(
                f"    NOISY N°{i} AMPLITUDE{clip_info}: min={noisy_amp_i.min():.4f}, max={noisy_amp_i.max():.4f}, mean={noisy_amp_i.mean():.4f}, std={noisy_amp_i.std():.4f}, Is NaN={np.isnan(noisy_amp_i).any()}."
            )

            # Row 0: Input reflectivity
            im0 = axes[0, i].imshow(noisy_amp_i, cmap="gray")
            axes[0, i].axis("off")
            fig.colorbar(im0, ax=axes[0, i], shrink=0.6)
            # Row 1: Reconstruction
            im1 = axes[1, i].imshow(recon_amp_i, cmap="gray")
            axes[1, i].axis("off")
            fig.colorbar(im1, ax=axes[1, i], shrink=0.6)
            # Row 2: Residuals (difference)
            residuals = np.abs(noisy_amp_i - recon_amp_i)
            im2 = axes[2, i].imshow(residuals, cmap="gray")
            axes[2, i].axis("off")
            fig.colorbar(im2, ax=axes[2, i], shrink=0.6)

        # Add Row titles on the left side
        row_titles = [
            "Noisy amplitude\n(sqrt(Real^2 + Imag^2))",
            "Recon amplitude",
            "Residuals\n(no clipping)",
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
