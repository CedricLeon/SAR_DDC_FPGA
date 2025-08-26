import warnings
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.constants import amp_max, amp_min
from src.utils.sar_utils import denormalize_tensor


class CompareReconstructionToGT(Callback):
    def __init__(
        self,
        patch_dir: str,
        log_every_n_epochs: int,
    ):
        super().__init__()
        self.patch_dir = Path(patch_dir)
        self.log_every_n_epochs = log_every_n_epochs

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Find the large patch and convert it to a torch tensor."""
        # ----- Load the validation patch -----
        # For the files in patch_dir find the one that starts with val_ and ends with .npy
        found_patch = False
        for file in self.patch_dir.glob("val_*.npy"):
            self.patch_path = file
            found_patch = True
            break
        if not found_patch:
            raise FileNotFoundError(
                f"No validation patch found in {self.patch_dir}. "
                "Please ensure the directory contains a file starting with 'val_' and ending with '.npy'."
            )

        # Read and prepare patch data as numpy arrays for visualization
        patch_data = np.load(self.patch_path)  # [H, W, 2]
        self.original_real = patch_data[:, :, 0]
        self.original_imag = patch_data[:, :, 1]
        self.original_reflectivity = np.sqrt(self.original_real + self.original_imag)

        # Store as torch tensors on device
        self.patch_tensor = (
            torch.from_numpy(patch_data).to(pl_module.device).unsqueeze(0).float()
        )  # [1, H, W, 2]
        self.real_tensor = self.patch_tensor[:, :, :, 0].unsqueeze(0)  # [1, 1, H, W]
        self.imag_tensor = self.patch_tensor[:, :, :, 1].unsqueeze(0)  # [1, 1, H, W]

        # ----- Load MERLIN Ground Truth -----
        merlin_gt_dir = self.patch_dir.parent.parent / "denoised"
        found_merlin = False
        for file in merlin_gt_dir.glob("val_*.npy"):
            self.merlin_gt_path = file
            merlin_patch_dict = np.load(self.merlin_gt_path, allow_pickle=True).item()

            # Normalize the patch: log, clip +- 3 std, and scale (0,1)
            merlin_patch = merlin_patch_dict["denoised"]["full"]
            merlin_patch = np.log(merlin_patch + 1e-2)
            merlin_patch = merlin_patch.clip(
                merlin_patch.mean() - 3 * merlin_patch.std(),
                merlin_patch.mean() + 3 * merlin_patch.std(),
            )
            self.merlin_gt = (merlin_patch - merlin_patch.min()) / (
                merlin_patch.max() - merlin_patch.min()
            )

            print(f"Loaded MERLIN GT from {self.merlin_gt_path}.")
            print(
                f"MERLIN statistics: min={self.merlin_gt.min():.4f}, max={self.merlin_gt.max():.4f}, mean={self.merlin_gt.mean():.4f}, std={self.merlin_gt.std():.4f}. Is NaN={np.isnan(self.merlin_gt).any()}."
            )
            found_merlin = True
            break

        if not found_merlin:
            warnings.warn(
                f"No MERLIN Ground Truth found in {merlin_gt_dir}. Skipping GT logging."
            )
            self.merlin_gt = None
        elif (
            self.merlin_gt_path.name.split("_")[1] != self.patch_path.name.split("_")[1]
        ):
            warnings.warn(
                f"Patch and MERLIN GT filenames do not match: {self.patch_path.name} vs {self.merlin_gt_path.name}. "
                "This may lead to incorrect logging."
            )

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
        recon_real = denormalize_tensor(recon_real.squeeze(), (amp_min, amp_max))
        recon_imag = denormalize_tensor(recon_imag.squeeze(), (amp_min, amp_max))
        # Build full amplitude reconstruction
        recon_full = torch.sqrt(
            0.5 * (torch.square(recon_real) + torch.square(recon_imag))
        )

        if torch.isnan(recon_full).any():
            raise ValueError("NaNs found in the denormed amplitude reconstruction.")

        # Log-scale, then compute statistics for clipping
        recon_log = torch.log(recon_full + 1e-2)
        # Clip using log-domain statistics and normalize
        recon = recon_log.clip(
            recon_log.mean() - 3 * recon_log.std(),
            recon_log.mean() + 3 * recon_log.std(),
        )
        recon = (recon - recon.min()) / (recon.max() - recon.min())

        # Check NaNs
        if torch.isnan(recon).any():
            raise ValueError("NaNs found in the normalized reconstruction.")

        # Bring to numpy for visualization
        reconstruction = recon.squeeze().cpu().numpy()

        # ----- Create visualization -----
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Images
        # Original amplitude (sum of real + imag)
        im0 = axes[0, 0].imshow(self.original_reflectivity, cmap="gray")
        axes[0, 0].set_title("Original Amplitude")
        axes[0, 0].axis("off")
        fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

        # Reconstruction
        im1 = axes[0, 1].imshow(reconstruction, cmap="gray")
        axes[0, 1].set_title("Recon amp, clipped and normalized")
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

        # MERLIN GT (if available)
        if self.merlin_gt is not None:
            im2 = axes[0, 2].imshow(self.merlin_gt, cmap="gray")
            axes[0, 2].set_title("MERLIN GT amp")
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
        axes[1, 0].hist(
            self.original_reflectivity.flatten(), bins=50, alpha=0.7, color="blue"
        )
        axes[1, 0].set_title("Original Histogram")

        # Reconstruction histogram
        axes[1, 1].hist(reconstruction.flatten(), bins=50, alpha=0.7, color="blue")
        axes[1, 1].set_title("Reconstruction Histogram")

        # MERLIN GT histogram (if available)
        if self.merlin_gt is not None:
            axes[1, 2].hist(self.merlin_gt.flatten(), bins=50, alpha=0.7, color="blue")
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

        # Add overall title with metrics
        loss = (
            out_criterion_real["loss"].item() + out_criterion_imag["loss"].item()
        ) / 2
        bpp_loss = (
            out_criterion_real["bpp_loss"].item()
            + out_criterion_imag["bpp_loss"].item()
        ) / 2
        # Compute MSE, PSNR between reconstructions and MERLIN GT
        if self.merlin_gt is not None:
            mse = np.mean((reconstruction - self.merlin_gt) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        else:
            mse = psnr = -1  # Not available if MERLIN GT is not loaded
        fig.suptitle(
            f"Val Large patch, epoch {trainer.current_epoch}: "
            f"Loss={loss:.3f}, BPP={bpp_loss:.4f}."
            f"\n Metrics to MERLIN GT: MSE={mse:.4f}, PSNR={psnr:.2f}dB",
            fontsize=14,
        )

        plt.tight_layout()

        pl_module.logger.experiment.log(
            {
                "val_large_patch_comparison": fig,
                "val_large_patch/loss": loss,
                "val_large_patch/bpp": bpp_loss,
                "val_large_patch/mse_to_MERLIN": mse,
                "val_large_patch/psnr_to_MERLIN": psnr,
            }
        )

        plt.close(fig)
