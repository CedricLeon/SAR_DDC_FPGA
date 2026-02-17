from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.constants import AMP_MAX, AMP_MIN, EPS
from src.utils.processing_utils import clip


class MonitorValReconstruction(Callback):
    """Callback to monitor reconstructions by logging the first validation patches."""

    def __init__(
        self,
        log_every_n_epochs: int,
        num_images: int = 3,
        verbose: bool = False,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_images = num_images
        # --- Details for clipping ---
        self.clip_for_visualization = True  # Enable or disable clipping
        self.mean_std_norm = True  # True: use mean/std, False use percentiles
        self.clip_factor = 3  # Clip to mean +/- self.clip_factor * std
        self.clip_percentiles = (5, 95)  # Clip to these percentiles

        self.verbose = verbose

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
        if self.verbose:
            print(f"\n[MonitorValReconstruction] Epoch {trainer.current_epoch}.")

        # Get the first few images from the batch
        num_images_to_show = min(self.num_images, batch["real"].shape[0])

        # Prepare input and target deterministically (always use real as input, imag as target)
        input = torch.cat((batch["real"], batch["imag"]), dim=1).contiguous()

        if self.verbose:
            print(
                f"    Input shape: {input.shape}, dtype: {input.dtype}, min: {input.min().item():.4f}, max: {input.max().item():.4f}, mean: {input.mean().item():.4f}, std: {input.std().item():.4f}. Is NaN={torch.isnan(input).any().item()}."
            )
            input_norm = (torch.log(torch.square(input) + EPS) - 2 * AMP_MIN) / (
                2 * AMP_MAX - 2 * AMP_MIN
            )
            print(
                f"    Normalized input shape: {input_norm.shape}, dtype: {input_norm.dtype}, min: {input_norm.min().item():.4f}, max: {input_norm.max().item():.4f}, mean: {input_norm.mean().item():.4f}, std: {input_norm.std().item():.4f}. Is NaN={torch.isnan(input_norm).any().item()}."
            )

        # Forward pass to get reconstructions
        with torch.no_grad():
            if self.with_compression:
                reconstructions = pl_module(input)
                criterion = pl_module.criterion(reconstructions, target=input)
                reconstructions = reconstructions["x_hat"]
            else:
                recon_real = pl_module(input[:, 0:1, :, :])
                recon_imag = pl_module(input[:, 1:2, :, :])
                reconstructions = torch.cat([recon_real, recon_imag], dim=1)
                criterion = pl_module.criterion(reconstructions, target=input)
        # Denormalize reconstructions
        recon_denorm = reconstructions * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = torch.exp(recon_denorm)
        recon_linI = 0.5 * (
            torch.square(recon_lin[:, 0, :, :]) + torch.square(recon_lin[:, 1, :, :])
        )  # [B, H, W]
        recon_logI = torch.log(recon_linI + EPS)  # used for visualization
        # recon_linA = torch.sqrt(recon_linI)         # used for metrics computation

        # Create the visualization
        fig, axes = plt.subplots(3, num_images_to_show, figsize=(4 * num_images_to_show, 12))
        if num_images_to_show == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_images_to_show):
            # Get individual images and convert to numpy
            real_i = batch["real"][i, 0].cpu().numpy()  # Remove channel dim
            imag_i = batch["imag"][i, 0].cpu().numpy()  # Remove channel dim
            noisy_logI = np.log(
                np.square(real_i) + np.square(imag_i)
            )  # Sum for input reflectivity
            recon_logI_i = recon_logI[i].cpu().numpy()
            if self.clip_for_visualization:
                noisy_logI = clip(
                    noisy_logI,
                    self.mean_std_norm,
                    self.clip_factor,
                    self.clip_percentiles,
                )
                recon_logI_i = clip(
                    recon_logI_i,
                    self.mean_std_norm,
                    self.clip_factor,
                    self.clip_percentiles,
                )
                clip_info = f" (clipped with {'mean/std' if self.mean_std_norm else f'percentiles {self.clip_percentiles}'})"
            else:
                clip_info = " (no clipping)"

            if self.verbose:
                print(
                    f"    RECON N°{i} Log-Intensity{clip_info}: min={recon_logI_i.min():.4f}, max={recon_logI_i.max():.4f}, mean={recon_logI_i.mean():.4f}, std={recon_logI_i.std():.4f}. Is NaN={np.isnan(recon_logI_i).any()}."
                )
                print(
                    f"    NOISY N°{i} Log-Intensity{clip_info}: min={noisy_logI.min():.4f}, max={noisy_logI.max():.4f}, mean={noisy_logI.mean():.4f}, std={noisy_logI.std():.4f}, Is NaN={np.isnan(noisy_logI).any()}."
                )

            # Row 0: Input reflectivity
            im0 = axes[0, i].imshow(noisy_logI, cmap="gray")
            axes[0, i].axis("off")
            fig.colorbar(im0, ax=axes[0, i], shrink=0.6)
            # Row 1: Reconstruction
            im1 = axes[1, i].imshow(recon_logI_i, cmap="gray")
            axes[1, i].axis("off")
            fig.colorbar(im1, ax=axes[1, i], shrink=0.6)
            # Row 2: Residuals (difference)
            residuals = np.abs(noisy_logI - recon_logI_i)
            im2 = axes[2, i].imshow(residuals, cmap="gray")
            axes[2, i].axis("off")
            fig.colorbar(im2, ax=axes[2, i], shrink=0.6)

        # Add Row titles on the left side
        row_titles = [
            "Noisy log-intensity\n(log(Real^2 + Imag^2))",
            "Recon log-intensity\n(log(0.5 * (Real^2 + Imag^2)))",
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

        pl_module.logger.experiment.log(  # type: ignore[attr-defined]
            {
                "val_reconstructions": fig,
                "val_batch/loss": criterion["loss"],
                "val_batch/mse": criterion["mse"],
                "val_batch/psnr": criterion["psnr"],
            }
        )

        plt.close(fig)
