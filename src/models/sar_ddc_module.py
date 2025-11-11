from typing import Any, Dict, Optional, Tuple

import lightning
import torch
import torchmetrics.functional as TMF
import torchmetrics.functional.image as F
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

from src.utils.constants import amp_max, amp_min

EPS = 1e-2


class SARDDCModule(lightning.LightningModule):
    """Lightning Module for SAR Despeckling and Data Compression.

    This module implements the training and testing logic for joint despeckling and compression of
    SAR images using a Noise2Noise approach.
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

    # def on_fit_start(self):
    #     """Called at the beginning of fit."""
    #     # Set up wandb watch to monitor parameters and gradients
    #     if isinstance(self.trainer.logger, WandbLogger):
    #         print("----------------------The WandB logger should watch gradient")
    #         self.trainer.logger.watch(
    #             self.net,
    #             log="all",  # Track both gradients and parameters
    #             log_freq=100,  # Log every 100 batches
    #             log_graph=False,  # Disable logging model graph
    #         )

    # def on_train_end(self):
    #     # Remove the hooks added by watch() to the model
    #     if isinstance(self.trainer.logger, WandbLogger):
    #         wandb.unwatch(self.net)

    def _random_switch_Re_Im(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        # Get real and imaginary parts
        real, imag = batch["real"], batch["imag"]

        # Deterministic random switching of inputs/targets using seeded generator
        if torch.rand(1).item() > 0.5:
            input_data, target_data = real, imag
        else:
            input_data, target_data = imag, real

        return input_data, target_data

    def forward(self, x: Tensor):
        """Forward pass through the network."""
        return self.net(x)

    def _log_metrics(
        self,
        prefix: str,
        criterion: Dict[str, Any],
        aux_loss: float,
    ) -> None:
        """Log training, validation, or test metrics."""
        log_info = {
            f"{prefix}/loss": criterion["loss"].item(),
            f"{prefix}/bpp": criterion["bpp"].item(),
            f"{prefix}/mse": criterion["mse"].item(),
            f"{prefix}/ssim": criterion["ssim"].item(),
            f"{prefix}/ms_ssim": criterion["ms_ssim"].item(),
            f"{prefix}/merlin": criterion["merlin"].item(),
            f"{prefix}/psnr": criterion["psnr"].item(),
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

        Because we have two optimizers, we need to manually optimize. During training, we randomly
        switch between real and imaginary parts.
        """
        # Get the optimizers as a list to handle properly
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        net_optimizer = optimizers[0]
        aux_optimizer = optimizers[1]

        # Forward pass
        input, target = self._random_switch_Re_Im(batch)
        output = self.forward(input)
        criterion = self.criterion(output, target)

        # Backward pass for the main loss
        self.manual_backward(criterion["loss"])
        self.clip_gradients(
            net_optimizer,
            gradient_clip_val=self.hparams.gradient_clip_norm,
            gradient_clip_algorithm="norm",
        )
        net_optimizer.step()
        net_optimizer.zero_grad()

        # Auxiliary loss
        aux_loss = self.net.aux_loss()
        self.manual_backward(aux_loss)
        aux_optimizer.step()
        aux_optimizer.zero_grad()

        # Step scheduler if available (for epoch-based schedulers)
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()

        # custom learning rate logging
        lr = net_optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # Log metrics
        self._log_metrics("train", criterion, aux_loss.item())

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized processing of both real and imaginary parts."""
        input, target = self._random_switch_Re_Im(batch)
        output = self.forward(input)
        criterion = self.criterion(output, target)
        aux_loss = self.net.aux_loss()
        self._log_metrics("valid", criterion, aux_loss.item())

    def test_step(self, batch, batch_idx):
        """Test step: keep original metrics and, if references are available, compute GT metrics.

        - Logs standard criterion metrics (as before) using a random Re/Im switch.
        - Additionally, when batch contains 'adam_ref' and 'merlin_ref', computes reconstruction
          from both real and imag inputs, converts to log-intensity, and logs PSNR/SSIM/MS-SSIM/MSE
          against each reference.
        """
        real = batch["real"]
        imag = batch["imag"]
        adam_noc_ref = batch["adam_noc_ref"]
        merlin_ref = batch["merlin_ref"]

        out_r = self.forward(real)
        out_i = self.forward(imag)

        # SARDDC returns a dict with x_hat, while MERLIN returns just a tensor
        if isinstance(out_r, dict) and "x_hat" in out_r:
            out_r = out_r["x_hat"]
            out_i = out_i["x_hat"]

        # Convert to log-intensity like in evaluation
        recon_r_lin = torch.exp(out_r.squeeze(1) * (amp_max - amp_min) + amp_min)
        recon_i_lin = torch.exp(out_i.squeeze(1) * (amp_max - amp_min) + amp_min)
        I_recon = 0.5 * (recon_r_lin + recon_i_lin)
        recon_logI = torch.log(I_recon + EPS).unsqueeze(1)  # [B,1,H,W]

        all_metrics = {}
        data_range = float((recon_logI.max() - recon_logI.min()).detach().cpu())
        # MSE
        all_metrics["mse_adam_noc"] = TMF.mean_squared_error(recon_logI, adam_noc_ref)
        all_metrics["mse_merlin"] = TMF.mean_squared_error(recon_logI, merlin_ref)
        # PSNR
        all_metrics["psnr_adam_noc"] = F.peak_signal_noise_ratio(
            recon_logI, adam_noc_ref, data_range=data_range
        )
        all_metrics["psnr_merlin"] = F.peak_signal_noise_ratio(
            recon_logI, merlin_ref, data_range=data_range
        )
        # SSIM
        all_metrics["ssim_adam_noc"] = F.structural_similarity_index_measure(
            recon_logI, adam_noc_ref, data_range=data_range
        )
        all_metrics["ssim_merlin"] = F.structural_similarity_index_measure(
            recon_logI, merlin_ref, data_range=data_range
        )
        # MS-SSIM
        all_metrics["ms_ssim_adam_noc"] = F.multiscale_structural_similarity_index_measure(
            recon_logI, adam_noc_ref, data_range=data_range
        )
        all_metrics["ms_ssim_merlin"] = F.multiscale_structural_similarity_index_measure(
            recon_logI, merlin_ref, data_range=data_range
        )

        # Log extra metrics
        self.log_dict(
            all_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
        all_params = {param for _, param in self.net.named_parameters() if param.requires_grad}
        assert not set(main_params) & set(
            aux_params
        ), "Intersection found in main and auxiliary parameters"
        assert (
            set(main_params) | set(aux_params) == all_params
        ), "Union of main and auxiliary parameters does not match all model parameters"

        # Instantiate optimizers from the configuration
        net_optimizer = self.hparams.net_optimizer(params=main_params)
        aux_optimizer = self.hparams.aux_optimizer(params=aux_params)

        # Configure the scheduler if provided
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=net_optimizer)
            return [
                {
                    "optimizer": net_optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "name": "net_lr",  # "name" keywords are for the LearningRateMonitor callback
                        # "monitor": "valid/loss", # Unnecessary, because manual_optimization
                        # "interval": "epoch",
                        # "frequency": 1,
                    },
                },
                {"optimizer": aux_optimizer},
            ]

        return [{"optimizer": net_optimizer}, {"optimizer": aux_optimizer}]
