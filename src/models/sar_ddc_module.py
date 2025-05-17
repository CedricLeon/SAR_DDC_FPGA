"""
SAR Despeckling and Data Compression (DDC) Lightning Module.

This module handles the training and testing logic for joint despeckling
and compression of SAR images using a Noise2Noise approach.
"""

import datetime
import os
from typing import Any, Dict, Tuple

import lightning
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.metrics import (
    calculate_psnr_1,
    calculate_psnr_max,
)


class SARDDCModule(lightning.LightningModule):
    """Lightning Module for SAR Despeckling and Data Compression.

    This module implements the training and testing logic for joint
    despeckling and compression of SAR images using a Noise2Noise approach.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        net_optimizer: torch.optim.Optimizer,
        aux_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gradient_clip_norm: float = 1.0,
        compile: bool = False,
        anomalies_log_dir: str = "anomalies",  # Directory for low PSNR logs
    ):
        """Initialize the Lightning Module.

        Args:
            net: Neural network module
            criterion: Loss criterion
            net_optimizer: Main optimizer for network parameters
            aux_optimizer: Auxiliary optimizer for quantiles
            scheduler: Learning rate scheduler
            gradient_clip_norm: Maximum gradient norm for clipping (default: 1.0)
            compile: Whether to compile the model (default: False)
            anomalies_log_dir: Directory to save anomalies (default: anomalies)
        """
        super().__init__()

        # Save hyperparameters to be accessible via self.hparams (ignore nn.Modules)
        self.save_hyperparameters(ignore=["criterion", "net"], logger=False)

        # Hydra recursive instantiation.
        self.net = net
        self.criterion = criterion

        # Activate manual optimization, because we have two optimizers.
        self.automatic_optimization = False

        # Set up directory for low PSNR logs
        self.low_psnr_count = 0
        self.psnr_ano_threshold = 5.0
        os.makedirs(self.hparams.anomalies_log_dir, exist_ok=True)

    def on_fit_start(self):
        """Called at the beginning of fit."""
        # # Initialize tracking dictionaries for monitoring
        # self.weight_norms_history = {}
        # self.gradient_norms_history = {}

        # Set up wandb watch to monitor parameters and gradients
        if isinstance(self.trainer.logger, lightning.pytorch.loggers.wandb.WandbLogger):
            self.trainer.logger.watch(
                self.net,
                log="all",  # Track both gradients and parameters
                log_freq=100,  # Log every 100 batches
                # log_graph=False,  # Disable logging model graph
            )

    def on_train_end(self):
        # Remove the hooks added by watch() to the model
        if isinstance(self.trainer.logger, lightning.pytorch.loggers.wandb.WandbLogger):
            self.trainer.logger.unwatch(self.net)

    def _random_switch_Re_Im(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get real and imaginary parts (already squared and normalized)
        real_squared, imag_squared = batch["real"], batch["imag"]

        # Deterministic random switching of inputs/targets using seeded generator
        if torch.rand(1).item() > 0.5:
            input_data, target_data = real_squared, imag_squared
        else:
            input_data, target_data = imag_squared, real_squared

        return input_data, target_data

    def forward(self, x: torch.Tensor):
        """Forward pass through the network."""
        return self.net(x)

    def _model_forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        output = self.forward(input)
        out_criterion = self.criterion(output, target)
        return out_criterion, output["x_hat"]

    def _log_anomalies(
        self,
        prefix: str,
        input: torch.Tensor,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        trigger: Tuple[str, float],
        additional_info: Dict = None,
    ) -> None:
        """Log and visualize current metrics and batch statistics.

        Args:
            prefix: Log prefix (train/valid/test)
            input: Input tensor
            target: Target tensor
            reconstruction: Reconstructed output
            trigger: Tuple with the metric and its value that triggered the logging
            additional_info: Additional information to include in the log
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.low_psnr_count += 1

        # Create log directory with timestamp
        log_dir = os.path.join(
            self.hparams.anomalies_log_dir,
            f"{prefix}_ano{self.low_psnr_count}_e{self.current_epoch}_step{self.global_step}_{trigger[0]}:{trigger[1]:.2f}dB",
        )
        os.makedirs(log_dir, exist_ok=True)

        # Create a general information file
        with open(os.path.join(log_dir, "info.txt"), "w") as f:
            f.write(
                f"===== LOW {trigger[0].upper()} DETECTED: {trigger[1]:.2f} dB (threshold: {self.psnr_ano_threshold:.2f} dB) =====\n\n"
            )
            f.write(f"Time: {timestamp}\n")
            f.write(f"Epoch: {self.current_epoch}\n")
            f.write(f"Global step: {self.global_step}\n\n")

            # Write additional information if provided
            if additional_info:
                f.write("Additional Metrics:\n")
                for key, value in additional_info.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Write batch statistics
            f.write("Batch Statistics:\n")
            batch_size = input.shape[0] if input.ndim > 3 else 1  # Use torch shape
            f.write(f"  Batch size: {batch_size}\n")

            # Write overall statistics for the entire batch
            f.write("\nOverall Statistics:\n")
            f.write("Input:\n")
            f.write(f"  Shape: {tuple(input.shape)}\n")  # Cast to tuple for printing
            f.write(
                f"  Mean: {torch.mean(input.float()).item():.6f}\n"
            )  # Convert to float before mean
            f.write(f"  Std: {torch.std(input.float()).item():.6f}\n")
            f.write(f"  Min: {torch.min(input).item():.6f}\n")
            f.write(f"  Max: {torch.max(input).item():.6f}\n")
            f.write(f"  NaN count: {torch.isnan(input.float()).sum().item()}\n")
            f.write(f"  Inf count: {torch.isinf(input.float()).sum().item()}\n\n")

            f.write("Target:\n")
            f.write(f"  Shape: {tuple(target.shape)}\n")  # Cast to tuple for printing
            f.write(f"  Mean: {torch.mean(target.float()).item():.6f}\n")
            f.write(f"  Std: {torch.std(target.float()).item():.6f}\n")
            f.write(f"  Min: {torch.min(target).item():.6f}\n")
            f.write(f"  Max: {torch.max(target).item():.6f}\n")
            f.write(f"  NaN count: {torch.isnan(target.float()).sum().item()}\n")
            f.write(f"  Inf count: {torch.isinf(target.float()).sum().item()}\n\n")

            f.write("Reconstruction:\n")
            f.write(
                f"  Shape: {tuple(reconstruction.shape)}\n"
            )  # Cast to tuple for printing
            f.write(f"  Mean: {torch.mean(reconstruction.float()).item():.6f}\n")
            f.write(f"  Std: {torch.std(reconstruction.float()).item():.6f}\n")
            f.write(f"  Min: {torch.min(reconstruction).item():.6f}\n")
            f.write(f"  Max: {torch.max(reconstruction).item():.6f}\n")
            f.write(
                f"  NaN count: {torch.isnan(reconstruction.float()).sum().item()}\n"
            )
            f.write(
                f"  Inf count: {torch.isinf(reconstruction.float()).sum().item()}\n\n"
            )

            # Write statistics for each sample in the batch
            num_samples = min(16, batch_size)
            num_vis_samples = min(4, batch_size)
            f.write("\nPer-Sample Statistics:\n")
            for i in range(num_samples):
                f.write(f"\nSample {i + 1}:\n")
                if input.ndim > 3:
                    sample_input = input[i].squeeze()
                    sample_target = target[i].squeeze()
                    sample_recon = reconstruction[i].squeeze()
                else:
                    sample_input = input.squeeze()
                    sample_target = target.squeeze()
                    sample_recon = reconstruction.squeeze()

                f.write("  Input:\n")
                f.write(f"    Mean: {torch.mean(sample_input.float()).item():.6f}\n")
                f.write(f"    Std: {torch.std(sample_input.float()).item():.6f}\n")
                f.write(f"    Min: {torch.min(sample_input).item():.6f}\n")
                f.write(f"    Max: {torch.max(sample_input).item():.6f}\n")
                f.write(
                    f"    NaN count: {torch.isnan(sample_input.float()).sum().item()}\n"
                )
                f.write(
                    f"    Inf count: {torch.isinf(sample_input.float()).sum().item()}\n"
                )

                f.write("  Target:\n")
                f.write(f"    Mean: {torch.mean(sample_target.float()).item():.6f}\n")
                f.write(f"    Std: {torch.std(sample_target.float()).item():.6f}\n")
                f.write(f"    Min: {torch.min(sample_target).item():.6f}\n")
                f.write(f"    Max: {torch.max(sample_target).item():.6f}\n")
                f.write(
                    f"    NaN count: {torch.isnan(sample_target.float()).sum().item()}\n"
                )
                f.write(
                    f"    Inf count: {torch.isinf(sample_target.float()).sum().item()}\n"
                )

                f.write("  Reconstruction:\n")
                f.write(f"    Mean: {torch.mean(sample_recon.float()).item():.6f}\n")
                f.write(f"    Std: {torch.std(sample_recon.float()).item():.6f}\n")
                f.write(f"    Min: {torch.min(sample_recon).item():.6f}\n")
                f.write(f"    Max: {torch.max(sample_recon).item():.6f}\n")
                f.write(
                    f"    NaN count: {torch.isnan(sample_recon.float()).sum().item()}\n"
                )
                f.write(
                    f"    Inf count: {torch.isinf(sample_recon.float()).sum().item()}\n"
                )

                # Calculate sample PSNR using torch.nn.functional.mse_loss and calculate_psnr_1
                sample_mse = F.mse_loss(
                    sample_recon.float(), sample_target.float()
                ).item()
                sample_psnr = calculate_psnr_1(sample_mse)
                f.write(f"  MSE: {sample_mse:.6f}\n")
                f.write(f"  PSNR: {sample_psnr:.2f} dB\n")

                # Visualize the first few samples
                if i < num_vis_samples:
                    # Convert back to numpy only for visualization
                    sample_input = sample_input.detach().cpu().numpy()
                    sample_target = sample_target.detach().cpu().numpy()
                    sample_recon = sample_recon.detach().cpu().numpy()

                    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle(
                        f"Low PSNR Sample {i + 1} - PSNR: {sample_psnr:.2f} dB (threshold: {self.psnr_ano_threshold:.2f} dB)",
                        fontsize=16,
                    )

                    # Plot original data
                    axs[0, 0].imshow(sample_input, cmap="gray")
                    axs[0, 0].set_title("Input")
                    axs[0, 0].axis("off")

                    axs[0, 1].imshow(sample_target, cmap="gray")
                    axs[0, 1].set_title("Target")
                    axs[0, 1].axis("off")

                    axs[0, 2].imshow(sample_recon, cmap="gray")
                    axs[0, 2].set_title("Reconstruction")
                    axs[0, 2].axis("off")

                    # Plot differences and error map
                    diff_input_target = np.abs(sample_input - sample_target)
                    axs[1, 0].imshow(diff_input_target, cmap="hot")
                    axs[1, 0].set_title("Input-Target Difference")
                    axs[1, 0].axis("off")

                    diff_recon_target = np.abs(sample_recon - sample_target)
                    axs[1, 1].imshow(diff_recon_target, cmap="hot")
                    axs[1, 1].set_title("Recon-Target Difference")
                    axs[1, 1].axis("off")

                    # Square error map (for better visualization of errors)
                    error_map = (sample_recon - sample_target) ** 2
                    axs[1, 2].imshow(error_map, cmap="hot")
                    axs[1, 2].set_title("Squared Error Map")
                    axs[1, 2].axis("off")

                    plt.tight_layout()
                    plt.savefig(os.path.join(log_dir, f"sample_{i + 1}.png"), dpi=150)
                    plt.close(fig)

        # Log to console
        print(
            f"WARNING: Low {trigger[0].upper()} ({trigger[1]:.2f} dB) detected at epoch {self.current_epoch}, step {self.global_step}"
        )
        print(f"Statistics saved to {log_dir}")

    def _log_metrics(
        self,
        prefix: str,
        out_criterion: Dict[str, Any],
        aux_loss: float,
        input: torch.Tensor,
        reconstructions: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Log training, validation, or test metrics."""
        mse_value = out_criterion["mse_loss"].item()
        psnr_value_1 = calculate_psnr_1(mse_value)
        psnr_value_max = calculate_psnr_max(mse_value, torch.max(input).item())
        bpp_value = out_criterion["bpp_loss"].item()

        # Enhanced metrics logging
        log_info = {
            f"{prefix}/aux": aux_loss,
            f"{prefix}/loss": out_criterion["loss"].item(),
            f"{prefix}/mse": mse_value,
            f"{prefix}/bpp": bpp_value,
            f"{prefix}/psnr_1": psnr_value_1,
            f"{prefix}/psnr_max": psnr_value_max,  # Both metrics are very similar
            # Track how much each loss contributes to total loss = R + lmbda * D
            f"{prefix}/mse_percent": (
                self.criterion.lmbda * mse_value / (out_criterion["loss"].item() + 1e-8)
            )
            * 100,
            f"{prefix}/bpp_percent": (bpp_value / (out_criterion["loss"].item() + 1e-8))
            * 100,
        }

        # # Manual tracking of weight and gradients
        # if prefix == "train" and self.global_step % 50 == 0:
        #     # Track weight norms
        #     for name, param in self.net.named_parameters():
        #         if param.requires_grad:
        #             norm = param.data.norm().item()
        #             log_info[f"{prefix}/weight_norm/{name}"] = norm

        #             # Track history for weight monitoring
        #             if name not in self.weight_norms_history:
        #                 self.weight_norms_history[name] = []
        #             self.weight_norms_history[name].append(norm)

        #             # If we have gradient, track it too
        #             if param.grad is not None:
        #                 grad_norm = param.grad.data.norm().item()
        #                 log_info[f"{prefix}/grad_norm/{name}"] = grad_norm

        #                 # Track gradient history
        #                 if name not in self.gradient_norms_history:
        #                     self.gradient_norms_history[name] = []
        #                 self.gradient_norms_history[name].append(grad_norm)

        # Configure per prefix (e.g. train/valid/test) logging **kwargs.
        on_step, on_epoch, prog_bar, sync_dist = None, None, False, True
        if prefix == "train":
            on_step, on_epoch, prog_bar, sync_dist = True, False, False, True
        elif prefix == "valid":
            on_step, on_epoch, prog_bar, sync_dist = False, True, True, True
        elif prefix == "test":
            on_step, on_epoch, prog_bar, sync_dist = False, True, False, True

        self.log_dict(
            log_info,
            sync_dist=sync_dist,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
        )

        # Log anomalies (low PSNR)
        if self.current_epoch > 0 and psnr_value_1 < self.psnr_ano_threshold:
            additional_info = {
                "bpp": bpp_value,
                "mse": mse_value,
                "loss": out_criterion["loss"].item(),
            }
            self._log_anomalies(
                prefix,
                input,
                target,
                reconstructions,
                ("PSNR", psnr_value_1),
                additional_info,
            )

    def training_step(self, batch, batch_idx):
        """Training step using Noise2Noise approach.
        Because we have two optimizers, we need to manually optimize.
        During training, we randomly switch between real and imaginary parts."""
        # Get the optimizers as a list to handle properly
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = list(optimizers)

        net_optimizer = optimizers[0]
        aux_optimizer = optimizers[1]

        net_optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # Forward pass
        input, target = self._random_switch_Re_Im(batch)

        # Verify input and target sanity before forward pass
        if torch.isnan(input).any() or torch.isinf(input).any():
            self.log("train/nan_inf_inputs", 1.0, on_step=True)
            print(f"WARNING: NaN or Inf detected in inputs at step {self.global_step}")

        if torch.isnan(target).any() or torch.isinf(target).any():
            self.log("train/nan_inf_targets", 1.0, on_step=True)
            print(f"WARNING: NaN or Inf detected in targets at step {self.global_step}")

        out_criterion, reconstructions = self._model_forward(input, target)

        # Monitor gradient norms for debugging
        if self.global_step % 100 == 0:  # Don't compute this every step to save time
            param_norm = 0
            for p in self.net.parameters():
                if p.grad is not None:
                    param_norm += p.grad.data.norm(2).item() ** 2
            param_norm = param_norm**0.5
            self.log("train/grad_norm", param_norm, on_step=True, on_epoch=False)

        # Check for NaN or Inf in loss or reconstructions
        if torch.isnan(out_criterion["loss"]) or torch.isinf(out_criterion["loss"]):
            self.log("train/nan_inf_loss", 1.0, on_step=True)
            print(f"WARNING: NaN or Inf detected in loss at step {self.global_step}")

        # Backward pass for the main loss
        self.manual_backward(out_criterion["loss"])
        if self.hparams.gradient_clip_norm > 0.0:  # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.hparams.gradient_clip_norm
            )
        net_optimizer.step()

        # Auxiliary loss
        aux_loss = self.net.aux_loss()
        self.manual_backward(aux_loss)
        aux_optimizer.step()

        # Log metrics
        self._log_metrics(
            "train", out_criterion, aux_loss.item(), input, reconstructions, target
        )

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        out_criterion, reconstructions = self._model_forward(input, target)
        aux_loss = self.net.aux_loss()
        self._log_metrics(
            "valid", out_criterion, aux_loss.item(), input, reconstructions, target
        )

    def test_step(self, batch, batch_idx):
        """Test step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        out_criterion, reconstructions = self._model_forward(input, target)
        aux_loss = self.net.aux_loss()
        self._log_metrics(
            "test", out_criterion, aux_loss.item(), input, reconstructions, target
        )

    def on_validation_epoch_end(self) -> None:
        """Update LR scheduler based on validation loss."""
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(self.trainer.callback_metrics["valid/loss"])

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar, you might have multiple.

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        main_params = [
            param
            for name, param in self.net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        ]
        aux_params = [
            param
            for name, param in self.net.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        ]

        # Validation: Ensure no parameter overlap and all parameters are accounted for
        all_params = set(
            param for _, param in self.net.named_parameters() if param.requires_grad
        )
        assert not set(main_params) & set(aux_params), (
            "Intersection found in main and auxiliary parameters"
        )
        assert set(main_params) | set(aux_params) == all_params, (
            "Union of main and auxiliary parameters does not match all model parameters"
        )

        # Instantiate optimizers from the configuration
        net_optimizer = self.hparams.net_optimizer(params=main_params)
        aux_optimizer = self.hparams.aux_optimizer(params=aux_params)

        # Configure the scheduler if provided
        if self.hparams.scheduler:
            lr_scheduler = self.hparams.scheduler(optimizer=net_optimizer)
            return [
                {
                    "optimizer": net_optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "name": "net_lr",  # "name" keywords are for the LearningRateMonitor callback
                        # "monitor": "valid/loss", # Unnecessary, because manual_optimization
                        # "interval": "epoch",
                        # "frequency": 1,
                    },
                },
                {"optimizer": aux_optimizer},
            ]

        return [{"optimizer": net_optimizer}, {"optimizer": aux_optimizer}]


if __name__ == "__main__":
    _ = SARDDCModule(None, None, None, None, None)
