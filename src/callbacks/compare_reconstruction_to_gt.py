import warnings
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib.ticker import FuncFormatter

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
            f"Setting CompareReconstructionToGT Callback. {self.clip_and_norm=}, {self.clip_factor=}, {self.split_large_patch=} ({self.blend_method=}, {self.stride=})"
        )
        print(
            f"Called with {pl_module.__class__.__name__}: net = {pl_module.net.__class__.__name__}, criterion = {pl_module.criterion.__class__.__name__}."
        )
        if pl_module.__class__.__name__ == "MerlinModule":
            self.with_compression = False
        elif pl_module.__class__.__name__ == "SARDDCModule":
            self.with_compression = True
        else:
            raise ValueError(
                f"Unsupported LightningModule class: {pl_module.__class__.__name__}"
            )
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

        # --- Read and prepare noisy patch data as numpy arrays for visualization ---
        patch_data = np.load(self.patch_path)  # [H, W, 2]
        patch_data = symmetrize(patch_data)
        I_noisy = np.square(patch_data[:, :, 0]) + np.square(patch_data[:, :, 1])
        self.A_noisy = np.sqrt(I_noisy)
        self.logI_noisy = np.log(I_noisy + self.eps)
        print(f"    Loaded NOISY PATCH from {self.patch_path}.")
        print(
            f"    NOISY PATCH LOG-I (shape={self.logI_noisy.shape}) statistics: min={self.logI_noisy.min():.4f}, max={self.logI_noisy.max():.4f}, mean={self.logI_noisy.mean():.4f}, std={self.logI_noisy.std():.4f}. Is NaN={np.isnan(self.logI_noisy).any()}."
        )

        # --- Store as torch tensors on device for forward passes ---
        patch_tensor = torch.from_numpy(patch_data).to(pl_module.device)
        # Normalize
        patch = torch.square(patch_tensor)
        patch = torch.log(patch + self.eps)
        print(
            f"    NOISY TENSOR LOG-I (shape={patch.shape}) statistics: min={patch.min():.4f}, max={patch.max():.4f}, mean={patch.mean():.4f}, std={patch.std():.4f}. Is NaN={torch.isnan(patch).any()}."
        )
        patch = (patch - 2 * amp_max) / (2 * amp_min - 2 * amp_max)
        print(
            f"    NORMALIZED NOISY TENSOR LOG-I (shape={patch.shape}) statistics: min={patch.min():.4f}, max={patch.max():.4f}, mean={patch.mean():.4f}, std={patch.std():.4f}. Is NaN={torch.isnan(patch).any()}."
        )
        # Add batch and channel dimensions
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
                f"    MERLIN LOG-INTENSITY  (shape={self.logI_merlin.shape}) statistics: min={self.logI_merlin.min():.4f}, max={self.logI_merlin.max():.4f}, mean={self.logI_merlin.mean():.4f}, std={self.logI_merlin.std():.4f}. Is NaN={np.isnan(self.logI_merlin).any()}."
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
                criterion_real, recon_real = self.process_large_patch(
                    pl_module,
                    self.real_tensor,
                    self.imag_tensor,
                    stride=self.stride,
                    blend_method=self.blend_method,
                )
                criterion_imag, recon_imag = self.process_large_patch(
                    pl_module,
                    self.imag_tensor,
                    self.real_tensor,
                    stride=self.stride,
                    blend_method=self.blend_method,
                )
            else:
                criterion_real, recon_real = pl_module._model_forward(
                    self.real_tensor, self.imag_tensor
                )
                criterion_imag, recon_imag = pl_module._model_forward(
                    self.imag_tensor, self.real_tensor
                )
            self.recon_real_as_output = recon_real
            print(
                f"   RECON: min={recon_real.min().item():.4f}, max={recon_real.max().item():.4f}, mean={recon_real.mean().item():.4f}, std={recon_real.std().item():.4f}. Is NaN={torch.isnan(recon_real).any().item()}."
            )
            print(
                f"    TARGET: min={self.imag_tensor.min().item():.4f}, max={self.imag_tensor.max().item():.4f}, mean={self.imag_tensor.mean().item():.4f}, std={self.imag_tensor.std().item():.4f}. Is NaN={torch.isnan(self.imag_tensor).any().item()}."
            )

        # ----- Denorm the reconstructions  -----
        recon_real = torch.exp(
            recon_real.squeeze() * (2 * amp_max - 2 * amp_min) + 2 * amp_min
        )
        recon_imag = torch.exp(
            recon_imag.squeeze() * (2 * amp_max - 2 * amp_min) + 2 * amp_min
        )
        # Build full amplitude reconstruction
        I_recon = 0.5 * (torch.square(recon_real) + torch.square(recon_imag))
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
            criterion_real,
            criterion_imag,
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
        criterion_real: dict,
        criterion_imag: dict,
        trainer: Trainer,
        scale: str = "logI",
    ) -> tuple[Any, dict]:
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

        im2 = axes[0, 2].imshow(
            self.recon_real_as_output.squeeze().cpu().numpy(), cmap="gray"
        )
        axes[0, 2].set_title("Recon Real part (exactly as output)")
        axes[0, 2].axis("off")
        fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

        # MERLIN GT (if available)
        if merlin is not None:
            im3 = axes[0, 3].imshow(
                self._clip_and_minmax_normalize(merlin)
                if self.clip_and_norm
                else merlin,
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
                FuncFormatter(
                    lambda x, loc: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}"
                )
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
            self.recon_real_as_output.squeeze().cpu().numpy(),
            "Recon Real part Histogram",
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
        metrics["loss"] = (
            criterion_real["loss"].item() + criterion_imag["loss"].item()
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

        if self.with_compression:
            metrics["bpp"] = (
                criterion_real["bpp"].item() + criterion_imag["bpp"].item()
            ) / 2
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

    def process_large_patch(
        self,
        pl_module: LightningModule,
        input: torch.Tensor,
        target: torch.Tensor,
        model_patch_size: int = 256,
        stride: int = -1,
        blend_method: str = "count",
    ) -> tuple[dict, torch.Tensor]:
        """Process a large patch by splitting into smaller patches, processing each, then recombining.

        Args:
            model: LightningModule with _model_forward method
            patch: Tensor of shape [B, 1, H, W]
            model_patch_size: Size of patches the model expects (e.g., 256)
            stride: Stride between patches (if -1, uses model_patch_size/2)
            blend_method: How to blend overlapping regions - "count" (average) or "linear" (weighted blend)

        Returns:
            Dictionary with results and metrics
        """
        # log.info(f"Processing large patch of shape {patch.shape} with model...")
        _, _, height, width = input.shape

        # Default stride is half the patch size (50% overlap)
        if stride == -1:
            stride = model_patch_size // 2

        # Create output tensors
        output_large = torch.zeros_like(input)
        counts = torch.zeros_like(input)  # Tracks number of contributions per pixel

        # Metrics storage
        output_large_criterion = {}
        patch_count = 0

        # log.info(
        #     f"Processing {height}x{width} image in {model_patch_size}x{model_patch_size} patches with stride {stride}"
        # )

        # Process each patch
        for y in range(0, height - model_patch_size + 1, stride):
            for x in range(0, width - model_patch_size + 1, stride):
                # Extract small patches (.contiguous() is necessary when doing that in torch)
                input_patch = input[
                    :, :, y : y + model_patch_size, x : x + model_patch_size
                ].contiguous()
                target_patch = target[
                    :, :, y : y + model_patch_size, x : x + model_patch_size
                ].contiguous()

                # Process patches + Accumulate metrics
                with torch.no_grad():
                    criterion, output = pl_module._model_forward(
                        input_patch, target_patch
                    )

                if patch_count == 0:
                    output_large_criterion = criterion
                else:
                    for key in output_large_criterion.keys():
                        output_large_criterion[key] += criterion[key]
                patch_count += 1

                # Prepare blend
                if blend_method == "linear":
                    # Create weight mask for smooth blending
                    weight = torch.ones_like(output)

                    if stride < model_patch_size:
                        # Calculate overlap size
                        overlap = model_patch_size - stride

                        # Create smooth transition weights using cosine taper on same device
                        taper = (
                            torch.cos(
                                torch.linspace(
                                    0, np.pi / 2, overlap, device=weight.device
                                )
                            )
                            ** 2
                        )

                        # Apply taper to overlapping regions
                        if y > 0:  # Top edge overlap
                            weight[:, :, :overlap, :] *= taper.view(-1, 1)
                        if x > 0:  # Left edge overlap
                            weight[:, :, :, :overlap] *= taper.view(1, -1)
                        if y + model_patch_size < height:  # Bottom edge overlap
                            weight[:, :, -overlap:, :] *= taper.flip(0).view(-1, 1)
                        if x + model_patch_size < width:  # Right edge overlap
                            weight[:, :, :, -overlap:] *= taper.flip(0).view(1, -1)

                    # Apply weighted update
                    output_large[
                        :, :, y : y + model_patch_size, x : x + model_patch_size
                    ] += output * weight
                    counts[
                        :, :, y : y + model_patch_size, x : x + model_patch_size
                    ] += weight
                elif (
                    blend_method == "count"
                ):  # "count" method - simple summation with counting
                    output_large[
                        :, :, y : y + model_patch_size, x : x + model_patch_size
                    ] += output
                    counts[
                        :, :, y : y + model_patch_size, x : x + model_patch_size
                    ] += 1
                else:
                    raise ValueError(
                        f"Unknown blend method: {blend_method}. Use 'linear' or 'count'."
                    )

        # Normalize by weights for overlapping regions
        if torch.any(counts == 0):
            warnings.warn(
                "Some pixels were not updated due to no contributions. This may indicate an issue with patch processing."
            )
        counts = counts + 1e-8  # Add small epsilon to avoid division by zero
        output_large = output_large / counts

        # Average the metrics
        for key in output_large_criterion.keys():
            output_large_criterion[key] /= patch_count

        return output_large_criterion, output_large
