"""
SAR Despeckling and Data Compression (DDC) Lightning Module.

This module handles the training and testing logic for joint despeckling
and compression of SAR images using a Noise2Noise approach.
"""

import datetime
import os
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule

from src.utils.metric import (
    calculate_psnr,
)
from src.utils.sar_utils import save_anomaly_visualization


class SARDDCModule(LightningModule):
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
        monitor_anomalies: bool = False,
        anomaly_log_dir: str = "anomaly_logs",
        anomaly_psnr_threshold: float = 0.0,  # PSNR below this value triggers visualization
        anomaly_bpp_threshold: float = 4.0,  # BPP above this value triggers visualization
        max_anomalies_per_epoch: int = 5,  # Maximum number of anomalies to log per epoch
        mse_spike_ratio: float = 10.0,  # Detect spikes in MSE (current vs rolling avg)
    ):
        """Initialize the Lightning Module.

        Args:
            lambda_: Rate-distortion tradeoff parameter (default: 0.01)
            lr: Learning rate for optimizer (default: 1e-4)
            net: Additional model parameters (default: None)
            monitor_anomalies: Whether to monitor and log anomalous batches
            anomaly_psnr_threshold: PSNR threshold below which to log anomalies
            anomaly_bpp_threshold: BPP threshold above which to log anomalies
            anomaly_log_dir: Directory to save anomaly visualizations
            max_anomalies_per_epoch: Maximum number of anomalies to log per epoch
            mse_spike_ratio: Factor to detect sudden MSE spikes compared to moving average
        """
        super().__init__()

        # Save hyperparameters to be accessible via self.hparams (ignore nn.Module)
        self.save_hyperparameters(ignore=["criterion", "net"], logger=False)

        # Hydra recursive instantiation.
        self.net = net
        self.criterion = criterion

        # Activate manual optimization, because we have two optimizers.
        self.automatic_optimization = False

        # # @TODO: Metrics: EuroSAT images (64x64) are too small for MS-SSIM => we use SSIM.
        # self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Anomaly detection setup
        self.monitor_anomalies = monitor_anomalies
        if self.monitor_anomalies:
            os.makedirs(anomaly_log_dir, exist_ok=True)
            self.anomaly_count = 0
            self.anomalies_this_epoch = 0
            self.current_epoch_tracked = 0
            self.mse_history = []  # Track MSE history for spike detection
            self.mse_window_size = 100  # Window size for rolling average

    def on_fit_start(self):
        """Called at the beginning of fit."""
        # Here _rng was set using thew seed but I suspect it's useless because done with lightning.seed_everything

    def on_train_epoch_start(self):
        """Reset anomaly counter at the start of each epoch."""
        if self.monitor_anomalies:
            self.anomalies_this_epoch = 0
            self.current_epoch_tracked = self.current_epoch

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
        # TMP MEMO (@TODO remove): compressai.losses.RateDistortionLoss computes "mse_loss", "bpp_loss", and loss:
        # out["mse_loss"] = nn.MSELoss(output["x_hat"], target)
        # out["bpp_loss"] = sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output["likelihoods"].values()
        # )
        # out["loss"] = self.lmbda * (255**2 * out["mse_loss"]) + out["bpp_loss"]
        # @TODO: That 255**2 most likely don't fit our SAR usage, might have to redefine my own loss function

        return out_criterion, output["x_hat"]

    def _check_for_anomalies(
        self, mse_value: float, psnr_value: float, bpp_value: float
    ) -> Tuple[bool, str]:
        """Check for various types of anomalies in the training process.

        Returns:
            Tuple[bool, str]: (is_anomaly, reason)
        """
        # Store MSE for tracking
        self.mse_history.append(mse_value)
        if len(self.mse_history) > self.mse_window_size:
            self.mse_history.pop(0)

        # Check for MSE spikes compared to recent history
        is_anomaly = False
        reason = ""

        # Only check for spikes if we have enough history
        if len(self.mse_history) >= 10:
            recent_avg = sum(self.mse_history[:-1]) / (len(self.mse_history) - 1)
            # Avoid division by zero
            if recent_avg > 1e-6:
                spike_ratio = mse_value / recent_avg
                if spike_ratio > self.hparams.mse_spike_ratio:
                    is_anomaly = True
                    reason += f"MSE spike: current={mse_value:.2f}, avg={recent_avg:.2f}, ratio={spike_ratio:.2f}. "

        # Check basic thresholds
        if psnr_value < self.hparams.anomaly_psnr_threshold:
            is_anomaly = True
            reason += f"PSNR={psnr_value:.2f} (threshold={self.hparams.anomaly_psnr_threshold}). "

        if bpp_value > self.hparams.anomaly_bpp_threshold:
            is_anomaly = True
            reason += f"BPP={bpp_value:.2f} (threshold={self.hparams.anomaly_bpp_threshold}). "

        # Check for NaN or extremely large values in the MSE
        if torch.isnan(torch.tensor(mse_value)) or mse_value > 1000:
            is_anomaly = True
            reason += f"Invalid MSE value: {mse_value}. "

        return is_anomaly, reason

    def _log_metrics(
        self,
        prefix: str,
        out_criterion: Dict[str, Any],
        aux_loss: float,  # Ensure aux_loss is a scalar (float or torch.Tensor.item()).
        input: torch.Tensor,
        reconstructions: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Log training, validation, or test metrics. From Tigran in RS_DC project."""
        # Calculate PSNR for anomaly detection
        mse_value = out_criterion["mse_loss"].item()
        psnr_value = calculate_psnr(mse_value)
        bpp_value = out_criterion["bpp_loss"].item()

        log_info = {
            f"{prefix}/aux": aux_loss,  # Use scalar aux_loss directly.
            f"{prefix}/loss": out_criterion["loss"].item(),
            f"{prefix}/mse": mse_value,  # * 255 ** 2 / 3
            f"{prefix}/bpp": bpp_value,
            f"{prefix}/psnr": psnr_value,
            # f"{prefix}/ssim": calculate_ssim(reconstructions, batch["what we aim to reconstruct"]),
        }
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

        # Check for anomalies
        if prefix == "train" and self.current_epoch > 1 and self.monitor_anomalies:
            # Reset counter if epoch changed
            if self.current_epoch_tracked != self.current_epoch:
                self.anomalies_this_epoch = 0
                self.current_epoch_tracked = self.current_epoch

            # Check if we've reached the maximum number of anomaly logs for this epoch
            if self.anomalies_this_epoch < self.hparams.max_anomalies_per_epoch:
                # Check for anomalies
                is_anomaly, anomaly_reason = self._check_for_anomalies(
                    mse_value, psnr_value, bpp_value
                )

                # Save visualization if anomaly detected
                if is_anomaly:
                    self.anomaly_count += 1
                    self.anomalies_this_epoch += 1
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.hparams.anomaly_log_dir}/anomaly_{prefix}_{timestamp}_{self.anomaly_count:03d}"

                    # Create a comprehensive info dict for logging
                    info = {
                        "prefix": prefix,
                        "timestamp": timestamp,
                        "reason": anomaly_reason,
                        "current_epoch": self.current_epoch,
                        "global_step": self.global_step,
                        "metrics": log_info,
                    }

                    save_anomaly_visualization(
                        input=input,
                        target=target,
                        reconstruction=reconstructions,
                        filename=filename,
                        info=info,
                    )

                    # Log anomaly to console for better visibility
                    self.log(
                        f"{prefix}/anomalies_detected", 1, on_step=True, on_epoch=False
                    )
                    print(
                        f"WARNING: Anomaly detected at epoch {self.current_epoch}, step {self.global_step}: {anomaly_reason}"
                    )

    def training_step(self, batch, batch_idx):
        """Training step using Noise2Noise approach.
        Because we have two optimizers, we need to manually optimize.
        During training, we randomly switch between real and imaginary parts."""
        # Get the optimizers and manually zero the gradients.
        net_optimizer, aux_optimizer = self.optimizers()
        net_optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # Forward pass.
        input, target = self._random_switch_Re_Im(batch)

        # Verify input and target sanity before forward pass
        if torch.isnan(input).any() or torch.isinf(input).any():
            self.log("train/nan_inf_inputs", 1.0, on_step=True)
            print(f"WARNING: NaN or Inf detected in inputs at step {self.global_step}")
            # Could return early here, but let's continue to log the error properly

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
            # We could skip the backward pass, but let's allow the training to continue and address this elsewhere

        # Backward pass for the main loss.
        self.manual_backward(out_criterion["loss"])
        if self.hparams.gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.hparams.gradient_clip_norm
            )
        net_optimizer.step()
        # The scheduler is updated in on_validation_epoch_end, because we monitor "valid/loss".

        # Auxiliary loss.
        aux_loss = self.net.aux_loss()
        self.manual_backward(aux_loss)
        aux_optimizer.step()

        # Log metrics, returning the loss is not required in manual optimization.
        self._log_metrics(
            "train", out_criterion, aux_loss.item(), input, reconstructions, target
        )

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        out_criterion, reconstructions = self._model_forward(input, target)
        # Ensure aux_loss is a scalar, original logging does not use self.net.aux_loss().item(), only self.net.aux_loss().
        aux_loss = self.net.aux_loss()
        self._log_metrics(
            "valid", out_criterion, aux_loss.item(), input, reconstructions, target
        )

    def test_step(self, batch, batch_idx):
        """Test step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        out_criterion, reconstructions = self._model_forward(input, target)
        # Ensure aux_loss is a scalar, original logging does not use self.net.aux_loss().item(), only self.net.aux_loss().
        aux_loss = self.net.aux_loss()
        self._log_metrics(
            "test", out_criterion, aux_loss.item(), input, reconstructions, target
        )

    def on_validation_epoch_end(self) -> None:
        """Update LR scheduler based on validation loss."""
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(self.trainer.callback_metrics["valid/loss"])

    def configure_optimizers(self):  # -> Can probably be optimized further.
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
        all_params = set(self.net.named_parameters())
        assert not set(main_params) & set(aux_params), (
            "Intersection found in main and auxiliary parameters"
        )
        assert set(main_params) | set(aux_params) == {
            param for _, param in all_params
        }, "Union of main and auxiliary parameters does not match all model parameters"

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
