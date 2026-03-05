import json
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from matplotlib.ticker import FuncFormatter

from src.utils.constants import AMP_MAX, AMP_MIN, EPS
from src.utils.debug import print_images_statistics
from src.utils.metrics import compute_bitstream_bpp, get_all_distortion_metrics
from src.utils.processing_utils import clip, patch_infer


class CompareReconstructionToGT(Callback):
    """Callback to compare model reconstructions to MERLIN_DDS ground truth on a large validation
    patch."""

    def __init__(
        self,
        patch_dir: str,
        log_every_n_epochs: int,
        blend_profile: Literal["sigmoid", "linear", "cosine"] = "sigmoid",
        overlap: int = 16,
        verbose: bool = False,
    ):
        super().__init__()
        self.patch_dir = Path(patch_dir) / "visualization/Hamburg_[11000:12024-8500:9524]/"
        self.log_every_n_epochs = log_every_n_epochs
        # --- Details for clipping ---
        self.clip_for_visualization = True  # Enable or disable clipping
        self.mean_std_norm = True  # True: use mean/std, False use percentiles
        self.clip_factor = 3  # Clip to mean +/- self.clip_factor * std
        self.clip_percentiles = (5, 95)  # Clip to these percentiles
        self.clip_info = (
            f" (clipped with {'mean/std' if self.mean_std_norm else f'percentiles {self.clip_percentiles}'})"
            if self.clip_for_visualization
            else " (no clipping)"
        )
        # --- Parameters to process large tile as patches during testing---
        self.overlap: int = overlap
        self.blend_profile: Literal["sigmoid", "linear", "cosine"] = blend_profile

        self.with_compression = None
        self.verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Find the large patch and convert it to a torch tensor."""
        if self.verbose:
            print(
                f"\n[CompareReconstructionToGT] Setting up Callback. {self.clip_for_visualization=}, {self.clip_factor=},({self.blend_profile=}, {self.overlap=})"
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
        # For the files in patch_dir find the one that starts with sym_ and ends with .npy
        found_patch = False
        for file in self.patch_dir.glob("sym_Noisy.npy"):
            self.patch_path = file
            found_patch = True
            break
        if not found_patch:
            raise FileNotFoundError(
                f"No symmetrized patch found in {self.patch_dir}. "
                "Please ensure the directory contains a file starting with 'sym_' and ending with '.npy'."
            )

        # --- load and symmetrize ---
        patch_data = np.load(self.patch_path)  # [H, W, 2]
        if self.verbose:
            print(f"    Loaded Symmetrized PATCH from {self.patch_path}.")

        # --- Prepare noisy patch data as numpy arrays for visualization ---
        noisy_linI = np.square(patch_data[:, :, 0]) + np.square(patch_data[:, :, 1])
        self.noisy_linA = np.sqrt(noisy_linI)
        self.noisy_logI = np.log(noisy_linI + EPS)
        del noisy_linI

        # --- Store as torch tensors on device for forward passes ---
        patch_tensor = torch.from_numpy(patch_data).to(pl_module.device).float()
        # NO NORMALIZATION, IT'S DONE IN model.forward()
        # Add batch and channel dimensions
        self.patch = patch_tensor.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # [1, 2, H, W]

        # ----- Load MERLIN_DDS Ground Truth -----
        found_merlin = False
        for file in self.patch_dir.glob("linA_MERLIN_DDS.npy"):
            self.merlin_gt_path = file
            self.merlin_linA = np.load(self.merlin_gt_path)
            if self.verbose:
                print(f"    Loaded MERLIN_DDS GT from {self.merlin_gt_path}.")

            # Denoised image from MERLIN_DDS comes in linear amplitude scale, see https://github.com/hi-paris/deepdespeckling
            self.merlin_logI = np.log(np.square(self.merlin_linA) + EPS)

            if self.verbose:
                # Quick print metrics between noisy and MERLIN_DDS GT
                metrics = get_all_distortion_metrics(self.noisy_linA, self.merlin_linA)
                print("    Initial metrics between Noisy and MERLIN_DDS GT:", end="")
                for key, value in metrics.items():
                    print(f" {key}={value:.4f}", end=",")
                print()

            found_merlin = True
            break

        if not found_merlin:
            warnings.warn(
                f"No MERLIN_DDS Ground Truth found in {self.patch_dir}. Skipping GT logging."
            )
            self.merlin_linA = None
            self.merlin_logI = None

        if self.verbose:
            # Print all images statistics for debugging
            print_images_statistics(
                {
                    "Noisy Symmetrized": patch_data,
                    "Noisy LinA": self.noisy_linA,
                    "MERLIN_DDS LinA": self.merlin_linA,
                    "Noisy LogI": self.noisy_logI,
                    "MERLIN_DDS LogI": self.merlin_logI,
                },
                title=f"Epoch {trainer.current_epoch} - Image Statistics{self.clip_info}",
            )
        del patch_tensor, patch_data

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the final reconstruction of the large patch at the end of testing."""
        # We only need to run this callback once, so we mute it if it's called on the "test_sub500.npy" set used for FPGA comparison
        prefix = getattr(pl_module, "test_prefix", "test")
        if "sub500" in prefix:
            print(
                f"\n[CompareReconstructionToGT] Skipping on {prefix} set to avoid redundant logging."
            )
            return

        # Move everything to CPU for final visualization and logging to avoid GPU memory issues, especially with large patches and compression outputs.
        pl_module.net.cpu()
        self.patch = self.patch.cpu()

        # Always split large patch during test; Build infer_fn for patch_infer based on module type.
        if self.with_compression:  # SARDDCModule

            def _infer_fn(patch: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
                out = pl_module.forward(patch)
                crit = pl_module.criterion(out, patch)
                patch_norm = (torch.log(torch.square(patch) + EPS) - 2 * AMP_MIN) / (
                    2 * AMP_MAX - 2 * AMP_MIN
                )
                out_enc = pl_module.net.compress(patch_norm)
                out_dec = pl_module.net.decompress(out_enc["strings"], out_enc["shape"])
                _N, _, _H, _W = patch.shape
                crit["bpp_bitstream"] = compute_bitstream_bpp(out_enc["strings"], _H, _W, _N)
                return out_dec, crit

        else:  # MerlinModule: real and imaginary channels are processed independently.

            def _infer_fn(patch: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
                recon_real = pl_module.forward(patch[:, 0:1])
                recon_imag = pl_module.forward(patch[:, 1:2])
                recon_patch = torch.cat([recon_real, recon_imag], dim=1)
                return recon_patch, pl_module.criterion(recon_patch, patch)

        recon, criterion = patch_infer(
            self.patch, _infer_fn, overlap=self.overlap, blend_profile=self.blend_profile
        )  # [1, 2, H, W], dict with keys like "loss", "bpp", "bpp_bitstream"

        # ----- Denorm the reconstructions  -----
        recon_denorm = recon * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = torch.exp(recon_denorm)
        recon_linI = 0.5 * (
            torch.square(recon_lin[:, 0, :, :]) + torch.square(recon_lin[:, 1, :, :])
        )
        recon_linA = torch.sqrt(recon_linI).squeeze().cpu().numpy()
        recon_logI = torch.log(recon_linI + EPS).squeeze().cpu().numpy()
        if self.clip_for_visualization:
            recon_logI = clip(
                recon_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
            )

        # compute metrics
        metrics_to_merlin = self._compute_metrics_to_merlin(criterion, recon_linA)

        # Save reconstructions locally: PNG for log-I and NPY for lin-A
        img_name = "recon_" + self.patch_dir.name
        log_dir = Path(trainer.log_dir) if trainer.log_dir else Path(trainer.default_root_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        png_path = log_dir / f"{img_name}_logI.png"
        plt.imsave(png_path, recon_logI, cmap="gray")
        npy_path = log_dir / f"{img_name}_linA.npy"
        np.save(npy_path, recon_linA)

        # Save per-tile metrics to JSON so the comparison notebook can read bpp / bpp_bitstream without re-running inference.
        metrics_json_path = log_dir / f"{img_name}_metrics.json"
        with open(metrics_json_path, "w") as _f:
            json.dump(metrics_to_merlin, _f, indent=4)

        if self.verbose:
            print(f"[CompareReconstructionToGT] Saved test reconstruction to {png_path}")
            print(f"[CompareReconstructionToGT] Saved test reconstruction (linA) to {npy_path}")
            print(f"[CompareReconstructionToGT] Saved tile metrics to {metrics_json_path}")

        print_images_statistics(
            {
                "Noisy LinA": self.noisy_linA,
                "recon_linA": recon_linA,
                "MERLIN_DDS LinA": self.merlin_linA,
                "recon_logI": recon_logI,
            },
            title=f"Stats [CompareReconstructionToGT] on_test_end() - PSNR to MERLIN_DDS = {metrics_to_merlin['psnr']:.2f}dB, bbp (criterion) = {metrics_to_merlin['bpp']:.4f}, bpp (bitstream) = {metrics_to_merlin['bpp_bitstream']:.4f}",
        )

        # Log to WandB
        if (
            pl_module.logger is not None
            and hasattr(pl_module.logger, "experiment")
            and hasattr(pl_module.logger.experiment, "log")
        ):
            # Create a caption from metrics
            # Filter out -1.0 metrics for cleaner caption
            valid_metrics = {k: v for k, v in metrics_to_merlin.items() if v != -1.0}
            caption = ", ".join([f"{k}={v:.4f}" for k, v in valid_metrics.items()])

            pl_module.logger.experiment.log(
                {"test/reconstruction_image_logI": wandb.Image(str(png_path), caption=caption)}
            )

    def _compute_metrics_to_merlin(self, criterion: dict, recon_linA: np.ndarray) -> dict:
        """Compute distortion metrics between reconstruction and MERLIN_DDS GT in LINEAR-
        AMPLITUDE."""
        metrics_to_merlin = {
            "mse": -1.0,
            "psnr": -1.0,
            "bpp": -1.0,
            "bpp_bitstream": -1.0,
            "ssim": -1.0,
            "ms_ssim": -1.0,
        }
        if "loss" in criterion:
            metrics_to_merlin["loss"] = criterion["loss"].item()

        if self.merlin_linA is not None:
            for key, value in get_all_distortion_metrics(recon_linA, self.merlin_linA).items():
                metrics_to_merlin[key] = value

        # Log BPP from criterion (likelihood)
        if "bpp" in criterion:
            val = criterion["bpp"]
            metrics_to_merlin["bpp"] = val.item() if isinstance(val, torch.Tensor) else val

        # Log BPP from bitstream if available
        if "bpp_bitstream" in criterion:
            metrics_to_merlin["bpp_bitstream"] = criterion["bpp_bitstream"]

        return metrics_to_merlin

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log reconstruction comparison with MERLIN_DDS GT."""
        # Only log on specified epochs and for the first batch
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return

        if self.verbose:
            print(f"\n[CompareReconstructionToGT] Epoch {trainer.current_epoch}.")

        # ----- Forward pass to get reconstruction and metrics -----
        with torch.no_grad():
            if self.with_compression:
                recon = pl_module.forward(self.patch)
                criterion = pl_module.criterion(recon, self.patch)
                recon = recon["x_hat"]
            else:
                recon_real = pl_module.forward(self.patch[:, 0:1, :, :])
                recon_imag = pl_module.forward(self.patch[:, 1:2, :, :])
                recon = torch.cat([recon_real, recon_imag], dim=1)
                criterion = pl_module.criterion(recon, self.patch)

        # ----- Denorm the reconstructions  -----
        recon_denorm = recon * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = torch.exp(recon_denorm)
        recon_linI = 0.5 * (
            torch.square(recon_lin[:, 0, :, :]) + torch.square(recon_lin[:, 1, :, :])
        )
        recon_linA = torch.sqrt(recon_linI).squeeze().cpu().numpy()
        recon_logI = torch.log(recon_linI + EPS).squeeze().cpu().numpy()
        if self.verbose:
            print_images_statistics(
                {
                    "Reconstruction LinA": recon_linA,
                    "Noisy LinA": self.noisy_linA,
                },
                title=f"Epoch {trainer.current_epoch} - Reconstruction Statistics{self.clip_info}",
            )

        fig_A, metrics_to_merlin = self._visualize_with_histograms(
            recon_linA,
            recon_logI,
            criterion,
            trainer,
        )

        # Log to WandB if available
        if (
            pl_module.logger is not None
            and hasattr(pl_module.logger, "experiment")
            and self.merlin_linA is not None
        ):
            dict_to_log = {
                f"val_large_patch/{key}_to_MERLIN": value if key not in ["loss", "bpp"] else None
                for key, value in get_all_distortion_metrics(recon_linA, self.merlin_linA).items()
            }
            pl_module.logger.experiment.log(  # type: ignore[attr-defined]
                {
                    "val_large_patch_comparison": fig_A,
                    "val_large_patch/loss": metrics_to_merlin["loss"],
                    "val_large_patch/bpp": metrics_to_merlin["bpp"],
                    **dict_to_log,
                }
            )

        plt.close(fig_A)

    def _visualize_with_histograms(
        self,
        recon_linA: np.ndarray,
        recon_logI: np.ndarray,
        criterion: dict,
        trainer: Trainer,
    ) -> tuple[Any, dict]:
        """Visualize the reconstruction, noisy input, MERLIN_DDS GT (if available) and their
        histograms."""
        metrics_to_merlin = self._compute_metrics_to_merlin(criterion, recon_linA)

        # ----- Prepare images for visualization in LOG-I-----
        if self.clip_for_visualization:
            noisy_logI = clip(
                self.noisy_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
            )
            recon_logI = clip(
                recon_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
            )
            if self.merlin_logI is not None:
                merlin_logI = clip(
                    self.merlin_logI, self.mean_std_norm, self.clip_factor, self.clip_percentiles
                )

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        # ----- Row 1: Images -----
        # Original
        im0 = axes[0, 0].imshow(noisy_logI, cmap="gray")
        axes[0, 0].set_title("Noisy Log-I")
        axes[0, 0].axis("off")
        fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

        # Reconstruction
        im1 = axes[0, 1].imshow(recon_logI, cmap="gray")
        axes[0, 1].set_title("Recon Log-I")
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

        # MERLIN_DDS GT (if available)
        if self.merlin_logI is not None:
            im3 = axes[0, 2].imshow(merlin_logI, cmap="gray")
            axes[0, 2].set_title("MERLIN_DDS GT Log-I")
            axes[0, 2].axis("off")
            fig.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "MERLIN_DDS GT\nNot Available",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
            )
            axes[0, 2].axis("off")

        # ----- Row 2: Histograms -----
        def plot_histogram(ax, data, title):
            # Add a small check for NaN
            if np.isnan(data).any():
                ax.text(
                    0.5,
                    0.5,
                    "Data contains NaN\nHistogram cannot be displayed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                return

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
        plot_histogram(axes[1, 0], noisy_logI, "Noisy LOG-I Histogram")

        # Reconstruction histogram
        plot_histogram(axes[1, 1], recon_logI, "Recon LOG-I Histogram")

        # MERLIN_DDS GT histogram (if available)
        if self.merlin_logI is not None:
            plot_histogram(axes[1, 2], merlin_logI, "MERLIN_DDS GT LOG-I Histogram")
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "MERLIN_DDS GT\nHistogram\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
            axes[1, 2].axis("off")

        # ----- Add overall title with metrics -----
        fig.suptitle(
            f"Val Large patch ({'clipped and normalized' if self.clip_for_visualization else 'raw'}), epoch {trainer.current_epoch}: "
            f"Loss={metrics_to_merlin['loss']:.3f}, BPP={metrics_to_merlin['bpp']:.4f}."
            f"\n metrics to MERLIN_DDS GT (LIN-A): MSE={metrics_to_merlin['mse']:.4f}, PSNR={metrics_to_merlin['psnr']:.2f}dB, SSIM={metrics_to_merlin['ssim']:.4f}, MS-SSIM={metrics_to_merlin['ms_ssim']:.4f}",
            fontsize=14,
        )

        plt.tight_layout()

        return fig, metrics_to_merlin
