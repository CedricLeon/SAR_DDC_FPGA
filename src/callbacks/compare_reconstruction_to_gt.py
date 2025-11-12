import warnings
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib.ticker import FuncFormatter

from src.utils import process_large_patch
from src.utils.constants import amp_max, amp_min
from src.utils.sar_utils import symmetrize


class CompareReconstructionToGT(Callback):
    def __init__(
        self,
        patch_dir: str,
        log_every_n_epochs: int,
        split_large_patch: bool = False,
        blend_method: str = "linear",
        stride: int = -1,
    ):
        super().__init__()
        self.patch_dir = Path(patch_dir) / "visualization"
        self.log_every_n_epochs = log_every_n_epochs
        self.eps = 1e-2
        self.clip_and_norm = True
        self.clip_factor = 3  # Clip to mean +/- self.clip_factor * std
        self.split_large_patch = split_large_patch
        self.blend_method = blend_method
        self.stride = stride

        self.with_compression = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Find the large patch and convert it to a torch tensor."""
        print(
            f"\n[CompareReconstructionToGT] Setting up Callback. {self.clip_and_norm=}, {self.clip_factor=}, {self.split_large_patch=} ({self.blend_method=}, {self.stride=})"
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
        # For the files in patch_dir find the one that starts with raw_ and ends with .npy
        found_patch = False
        for file in self.patch_dir.glob("raw_*.npy"):
            self.patch_path = file
            found_patch = True
            break
        if not found_patch:
            raise FileNotFoundError(
                f"No validation patch found in {self.patch_dir}. "
                "Please ensure the directory contains a file starting with 'val_' and ending with '.npy'."
            )

        # --- load and symmetrize ---
        patch_data = np.load(self.patch_path)  # [H, W, 2]
        print(f"    Loaded RAW PATCH from {self.patch_path}.")
        print(
            f"        RAW PATCH (shape={patch_data.shape}) statistics: min={patch_data.min():.4f}, max={patch_data.max():.4f}, mean={patch_data.mean():.4f}, std={patch_data.std():.4f}. Is NaN={np.isnan(patch_data).any()}."
        )
        patch_data = symmetrize(patch_data)

        # --- Prepare noisy patch data as numpy arrays for visualization ---
        I_noisy = np.square(patch_data[:, :, 0]) + np.square(patch_data[:, :, 1])
        self.A_noisy = np.sqrt(I_noisy)
        self.logI_noisy = np.log(I_noisy + self.eps)
        print(
            f"        NOISY PATCH LOG-I (shape={self.logI_noisy.shape}) statistics: min={self.logI_noisy.min():.4f}, max={self.logI_noisy.max():.4f}, mean={self.logI_noisy.mean():.4f}, std={self.logI_noisy.std():.4f}. Is NaN={np.isnan(self.logI_noisy).any()}."
        )

        # --- Store as torch tensors on device for forward passes ---
        patch_tensor = torch.from_numpy(patch_data).to(pl_module.device)
        # Normalize
        patch = torch.square(patch_tensor)
        patch = torch.log(patch + self.eps)
        print(
            f"        NOISY TENSOR LOG (shape={patch.shape}) statistics: min={patch.min():.4f}, max={patch.max():.4f}, mean={patch.mean():.4f}, std={patch.std():.4f}. Is NaN={torch.isnan(patch).any()}."
        )
        patch = (patch - 2 * amp_min) / (2 * amp_max - 2 * amp_min)
        print(
            f"        NORMALIZED NOISY TENSOR LOG (shape={patch.shape}) statistics: min={patch.min():.4f}, max={patch.max():.4f}, mean={patch.mean():.4f}, std={patch.std():.4f}. Is NaN={torch.isnan(patch).any()}."
        )
        # Add batch and channel dimensions
        self.tensor = patch.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # [1, 2, H, W]
        self.real_tensor = patch[:, :, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        self.imag_tensor = patch[:, :, 1].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # ----- Load MERLIN Ground Truth -----
        found_merlin = False
        for file in self.patch_dir.glob("denoised_by_MERLIN_*.npy"):
            self.merlin_gt_path = file
            merlin_patch_dict = np.load(self.merlin_gt_path, allow_pickle=True).item()

            # Denoised image from MERLIN comes in linear amplitude scale, see https://github.com/hi-paris/deepdespeckling
            self.A_merlin = merlin_patch_dict["denoised"]["full"]
            self.logI_merlin = np.log(np.square(self.A_merlin) + self.eps)

            print(f"    Loaded MERLIN GT from {self.merlin_gt_path}.")
            print(
                f"        MERLIN LOG-INTENSITY  (shape={self.logI_merlin.shape}) statistics: min={self.logI_merlin.min():.4f}, max={self.logI_merlin.max():.4f}, mean={self.logI_merlin.mean():.4f}, std={self.logI_merlin.std():.4f}. Is NaN={np.isnan(self.logI_merlin).any()}."
            )
            found_merlin = True
            break

        if not found_merlin:
            warnings.warn(
                f"No MERLIN Ground Truth found in {self.patch_dir}. Skipping GT logging."
            )
            self.A_merlin = None
            self.logI_merlin = None
        elif self.merlin_gt_path.name.split("_")[3] != self.patch_path.name.split("_")[1]:
            warnings.warn(
                f"Patch and MERLIN GT filenames do not match: {self.patch_path.name} vs {self.merlin_gt_path.name}. "
                "This may lead to incorrect logging."
            )

    def _clip_and_minmax_normalize(self, img: np.ndarray) -> np.ndarray:
        """Clip to mean +/- self.clip_factor * std and min-max normalize to [0, 1]."""
        img = img.clip(
            img.mean() - self.clip_factor * img.std(),
            img.mean() + self.clip_factor * img.std(),
        )
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

        # ----- Forward pass to get reconstruction and metrics -----
        with torch.no_grad():
            if self.split_large_patch:
                criterion, recon = process_large_patch(
                    model=pl_module,
                    input=self.tensor,
                    target=self.tensor,
                    stride=self.stride,
                    blend_method=self.blend_method,
                )
                # criterion_real, recon_real = process_large_patch(
                #     model=pl_module,
                #     input=self.real_tensor,
                #     target=self.imag_tensor,
                #     stride=self.stride,
                #     blend_method=self.blend_method,
                # )
                # criterion_imag, recon_imag = process_large_patch(
                #     model=pl_module,
                #     input=self.imag_tensor,
                #     target=self.real_tensor,
                #     stride=self.stride,
                #     blend_method=self.blend_method,
                # )
            else:
                recon = pl_module(self.tensor)
                criterion = pl_module.criterion(recon, self.tensor)
                # recon_real = pl_module(self.real_tensor)
                # criterion_real = pl_module.criterion(recon_real, self.imag_tensor)
                # recon_imag = pl_module(self.imag_tensor)
                # criterion_imag = pl_module.criterion(recon_imag, self.real_tensor)

            if self.with_compression:
                assert isinstance(recon, dict)
                # , "SAR_DDC should return dict when with_compression=True"
                recon = recon["x_hat"]
                # recon_real = recon_real["x_hat"]
                # recon_imag = recon_imag["x_hat"]
            self.recon_as_output = 0.5 * (
                recon[:, :1, :, :] + recon[:, 1:, :, :]
            )  # 0.5 * (recon_real + recon_imag)
            print(
                f"    RECON: min={recon.min().item():.4f}, max={recon.max().item():.4f}, mean={recon.mean().item():.4f}, std={recon.std().item():.4f}. Is NaN={torch.isnan(recon).any().item()}."
            )
            print(
                f"    TARGET: min={self.imag_tensor.min().item():.4f}, max={self.imag_tensor.max().item():.4f}, mean={self.imag_tensor.mean().item():.4f}, std={self.imag_tensor.std().item():.4f}. Is NaN={torch.isnan(self.imag_tensor).any().item()}."
            )

        # ----- Denorm the reconstructions  -----
        # Either I denorm with the factor 2 or I don't square when building the input
        recon = torch.exp(recon * (amp_max - amp_min) + amp_min)
        # recon_real = torch.exp(recon_real.squeeze() * (amp_max - amp_min) + amp_min)
        # recon_imag = torch.exp(recon_imag.squeeze() * (amp_max - amp_min) + amp_min)
        print(
            f"    RECON DENORM LINEAR: min={recon.min().item():.4f}, max={recon.max().item():.4f}, mean={recon.mean().item():.4f}, std={recon.std().item():.4f}. Is NaN={torch.isnan(recon).any().item()}."
        )

        # Build full amplitude reconstruction
        # I_recon = 0.5 * (torch.square(recon_real) + torch.square(recon_imag))
        I_recon = 0.5 * (recon[:, :1, :, :] + recon[:, 1:, :, :])  # (recon_real + recon_imag)
        print(
            f"    RECON INTENSITY: min={I_recon.min().item():.4f}, max={I_recon.max().item():.4f}, mean={I_recon.mean().item():.4f}, std={I_recon.std().item():.4f}. Is NaN={torch.isnan(I_recon).any().item()}."
        )
        A_recon = torch.sqrt(I_recon)
        logI_recon = torch.log(I_recon + self.eps)

        # Bring to numpy for visualization
        A_recon = A_recon.squeeze().cpu().numpy()
        logI_recon = logI_recon.squeeze().cpu().numpy()

        # fig_A, _ = self._visualize_with_histograms(
        #     A_recon,
        #     criterion_real,
        #     criterion_imag,
        #     trainer,
        #     scale="A",
        # )
        fig_logI, metrics = self._visualize_with_histograms(
            logI_recon,
            criterion,
            # criterion_real,
            # criterion_imag,
            trainer,
            scale="logI",
        )

        # Log to WandB if available
        if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
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
        recon: np.ndarray,
        criterion: dict,
        # criterion_real: dict,
        # criterion_imag: dict,
        trainer: Trainer,
        scale: str = "logI",
    ) -> tuple[Any, dict]:
        """Visualize the reconstruction, noisy input, MERLIN GT (if available) and their
        histograms."""
        if scale == "logI":
            noisy = self.logI_noisy
            merlin = self.logI_merlin
            subtitles = ["Noisy Log-I", "Recon Log-I", "MERLIN GT Log-I"]
        elif scale == "A":
            noisy = self.A_noisy
            merlin = self.A_merlin
            subtitles = ["Noisy Lin-Amp", "Recon Lin-Amp", "MERLIN GT Lin-Amp"]
        else:
            raise ValueError(f"Unknown scale: {scale}")

        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        # ----- Row 1: Images -----
        # Original
        im0 = axes[0, 0].imshow(
            self._clip_and_minmax_normalize(noisy) if self.clip_and_norm else noisy,
            cmap="gray",
        )
        axes[0, 0].set_title(subtitles[0])
        axes[0, 0].axis("off")
        fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

        # Reconstruction
        im1 = axes[0, 1].imshow(
            self._clip_and_minmax_normalize(recon) if self.clip_and_norm else recon,
            cmap="gray",
        )
        axes[0, 1].set_title(subtitles[1])
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

        recon_as_output = self.recon_as_output.squeeze().cpu().numpy()
        im2 = axes[0, 2].imshow(
            (
                self._clip_and_minmax_normalize(recon_as_output)
                if self.clip_and_norm
                else recon_as_output
            ),
            cmap="gray",
        )
        axes[0, 2].set_title("Recon (exactly as output)")
        axes[0, 2].axis("off")
        fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

        # MERLIN GT (if available)
        if merlin is not None:
            im3 = axes[0, 3].imshow(
                self._clip_and_minmax_normalize(merlin) if self.clip_and_norm else merlin,
                cmap="gray",
            )
            axes[0, 3].set_title(subtitles[2])
            axes[0, 3].axis("off")
            fig.colorbar(im3, ax=axes[0, 3], shrink=0.8)
        else:
            axes[0, 3].text(
                0.5,
                0.5,
                "MERLIN GT\nNot Available",
                ha="center",
                va="center",
                transform=axes[0, 3].transAxes,
            )
            axes[0, 3].axis("off")

        # ----- Row 2: Histograms -----
        def plot_histogram(ax, data, title):
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
        plot_histogram(axes[1, 0], noisy, "Noisy Histogram")

        # Reconstruction histogram
        plot_histogram(axes[1, 1], recon, "Recon Histogram")
        plot_histogram(
            axes[1, 2],
            self.recon_as_output.squeeze().cpu().numpy(),
            "Recon output Histogram",
        )

        # MERLIN GT histogram (if available)
        if merlin is not None:
            plot_histogram(axes[1, 3], merlin, "MERLIN GT Histogram")
        else:
            axes[1, 3].text(
                0.5,
                0.5,
                "MERLIN GT\nHistogram\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 3].transAxes,
            )
            axes[1, 3].axis("off")

        # ----- Add overall title with metrics -----\
        metrics = {}
        metrics["loss"] = criterion[
            "loss"
        ].item()  # (criterion_real["loss"].item() + criterion_imag["loss"].item()) / 2
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
            metrics["mse"] = metrics["psnr"] = -1  # Not available if MERLIN GT is not loaded

        if self.with_compression:
            metrics["bpp"] = criterion[
                "bpp"
            ].item()  # (criterion_real["bpp"].item() + criterion_imag["bpp"].item()) / 2
        else:
            metrics["bpp"] = -1

        fig.suptitle(
            f"Val Large patch, epoch {trainer.current_epoch}: "
            f"Loss={metrics['loss']:.3f}, BPP={metrics['bpp']:.4f}."
            f"\n Metrics to MERLIN GT: MSE={metrics['mse']:.4f}, PSNR={metrics['psnr']:.2f}dB, Diff mean={logI_diff.mean():.4f} and Diff std={logI_diff.std():.4f}",
            fontsize=14,
        )

        plt.tight_layout()

        return fig, metrics
