from typing import Any, Dict, Optional, Tuple

import lightning
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor


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
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        gradient_clip_norm: float = 1.0,
        compile: bool = False,
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
        """
        super().__init__()

        # Save hyperparameters to be accessible via self.hparams (ignore nn.Modules)
        self.save_hyperparameters(ignore=["criterion", "net"], logger=False)

        # Hydra recursive instantiation.
        self.net = net
        self.criterion = criterion

        # Activate manual optimization, because we have two optimizers.
        self.automatic_optimization = False

    def on_fit_start(self):
        """Called at the beginning of fit."""
        # Set up wandb watch to monitor parameters and gradients
        if isinstance(self.trainer.logger, WandbLogger):
            print("----------------------The WandB logger should watch gradient")
            self.trainer.logger.watch(
                self.net,
                log="all",  # Track both gradients and parameters
                log_freq=100,  # Log every 100 batches
                log_graph=False,  # Disable logging model graph
            )

    def on_train_end(self):
        # Remove the hooks added by watch() to the model
        if isinstance(self.trainer.logger, WandbLogger):
            wandb.unwatch(self.net)

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
        out_criterion = self.criterion(output, target)
        return out_criterion, output["x_hat"]

    def _log_metrics(
        self,
        prefix: str,
        out_criterion: Dict[str, Any],
        aux_loss: float,
    ) -> None:
        """Log training, validation, or test metrics."""
        log_info = {
            f"{prefix}/loss": out_criterion["loss"].item(),
            f"{prefix}/distortion": out_criterion["distortion"].item(),
            f"{prefix}/bpp": out_criterion["bpp"].item(),
            f"{prefix}/mse": out_criterion["mse"].item(),
            f"{prefix}/ssim": out_criterion["ssim"].item(),
            f"{prefix}/ms_ssim": out_criterion["ms_ssim"].item(),
            f"{prefix}/merlin": out_criterion["merlin"].item(),
            f"{prefix}/psnr": out_criterion["psnr"].item(),
            f"{prefix}/aux": aux_loss,
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
        Because we have two optimizers, we need to manually optimize.
        During training, we randomly switch between real and imaginary parts."""
        # Get the optimizers as a list to handle properly
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        net_optimizer = optimizers[0]
        aux_optimizer = optimizers[1]

        net_optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # Forward pass
        input, target = self._random_switch_Re_Im(batch)
        out_criterion, _ = self._model_forward(input, target)

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
        self._log_metrics("train", out_criterion, aux_loss.item())

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        out_criterion, _ = self._model_forward(input, target)
        aux_loss = self.net.aux_loss()
        self._log_metrics("valid", out_criterion, aux_loss.item())

    def test_step(self, batch, batch_idx):
        """Test step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        out_criterion, _ = self._model_forward(input, target)
        aux_loss = self.net.aux_loss()
        self._log_metrics("test", out_criterion, aux_loss.item())

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
