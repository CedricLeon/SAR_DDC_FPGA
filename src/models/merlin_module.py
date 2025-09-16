"""
By nature the MERLIN method imposes no constraint on the type of neural network used.
However, by default we use the U-Net mentionned in the orginal paper:
    Dalsasso, E., Denis, L., & Tupin, F. (2022). As if by magic: Self-supervised training of deep despeckling networks with MERLIN. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–13. https://doi.org/10.1109/TGRS.2021.3128621

List of details from the experiments with TerraSAR-X stripmap (See page 4 and Table I):
- They use 3 images pachified 256x256 (total 50604 patches), a batch_size of 12
- 30 epochs, with gradient norm at 1.0, lr = 0.001, 0.0001 after 4 epochs, 0.00001 after 20 epochs
"""

from typing import Any, Dict, Optional, Tuple

import lightning
import torch
from torch import Tensor


class MerlinModule(lightning.LightningModule):
    """Lightning Module for SAR Despeckling using the MERLIN framework."""

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        gradient_clip_norm: float = 1.0,
    ):
        """Initialize the Lightning Module.

        Args:
            net: Neural network module
            criterion: Loss criterion
            net_optimizer: Main optimizer for network parameters
            scheduler: Learning rate scheduler
            gradient_clip_norm: Maximum gradient norm for clipping (default: 1.0)
        """
        super().__init__()

        # Disable automatic optimization
        self.automatic_optimization = False

        # Save hyperparameters to be accessible via self.hparams (ignore nn.Modules)
        self.save_hyperparameters(ignore=["criterion", "net"], logger=False)

        # Hydra recursive instantiation.
        self.net = net
        self.criterion = criterion

    def _random_switch_Re_Im(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        # Get real and imaginary parts (already squared and normalized)
        real_squared, imag_squared = batch["real"], batch["imag"]

        # Deterministic random switching of inputs/targets using seeded generator
        if torch.rand(1).item() > 0.5:
            input_data, target_data = real_squared, imag_squared
        else:
            input_data, target_data = imag_squared, real_squared

        return input_data, target_data

    def forward(self, x: Tensor):
        """Forward pass through the network."""
        return self.net(x)

    def _model_forward(
        self, input: Tensor, target: Tensor
    ) -> Tuple[Dict[str, Any], Tensor]:
        """Forward pass through the model and criterion computation.
        Args:
            input: Input tensor (squared real or imaginary part) [batch_size, 1, height, width]
            target: Target tensor (squared real or imaginary part) [batch_size, 1, height, width]

        Returns:
            (out_criterion, reconstruction): A tuple with a dictionary containing loss and metrics, and the reconstructed output tensor.
        """
        output = self.forward(input)
        criterion = self.criterion(output, target)
        return criterion, output

    def _log_metrics(self, prefix: str, criterion: Dict[str, Any]) -> None:
        """Log training, validation, or test metrics."""
        log_info = {
            f"{prefix}/loss": criterion["loss"].item(),
            f"{prefix}/loss_denorm": criterion["loss_denorm"].item(),
            f"{prefix}/loss_term1": criterion["loss_term1"].item(),
            f"{prefix}/loss_term2": criterion["loss_term2"].item(),
            f"{prefix}/mse": criterion["mse"].item(),
            f"{prefix}/ssim": criterion["ssim"].item(),
            f"{prefix}/ms_ssim": criterion["ms_ssim"].item(),
            f"{prefix}/psnr": criterion["psnr"].item(),
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

    def training_step(self, batch, batch_idx):
        """Training step using Noise2Noise approach.
        With manual optimization and gradient clipping."""
        optimizer = self.optimizers()

        input, target = self._random_switch_Re_Im(batch)
        criterion, _ = self._model_forward(input, target)

        # Manual backward pass
        self.manual_backward(criterion["loss"])

        # Apply gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.hparams.gradient_clip_norm,
            gradient_clip_algorithm="norm",
        )

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Step scheduler if available (for step-based schedulers)
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()

        # custom learning rate logging
        lr = optimizer.param_groups[0]["lr"]
        self.log(
            "train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        self._log_metrics("train", criterion)

        # Returning the loss not required in manual optimization
        return criterion["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        criterion, _ = self._model_forward(input, target)
        self._log_metrics("valid", criterion)

    def test_step(self, batch, batch_idx):
        """Test step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        criterion, _ = self._model_forward(input, target)
        self._log_metrics("test", criterion)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        print("Optimizer:", optimizer)
        self.epoch_based_scheduler = True  # Default to epoch-based
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            print("Scheduler:", scheduler)
            print("Scheduler params:", scheduler.state_dict())
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "interval" and "frequency" not necessary in manual optimization, see https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling
                    # "interval": "epoch",
                    # "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
