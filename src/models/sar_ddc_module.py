from typing import Any, Dict, Optional, Tuple

import lightning
import torch
import torchmetrics.functional as TMF
import torchmetrics.functional.image as F
from torch import Tensor

from src.utils.constants import AMP_MAX, AMP_MIN, EPS
from src.utils.metrics import mse, psnr


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

    def _random_switch_Re_Im(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Randomly switch between real and imaginary parts as input and target.

        Used only in training.
        """
        # Get real and imaginary parts
        real, imag = batch["real"], batch["imag"]

        # Deterministic random switching of inputs/targets using seeded generator
        if torch.rand(1).item() > 0.5:
            input_data, target_data = real, imag
        else:
            input_data, target_data = imag, real

        return input_data, target_data

    def forward(self, x: Tensor):
        """Normalize x and forward pass through the network."""
        x = (torch.log(torch.square(x) + EPS) - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)
        return self.net(x)

    def _log_metrics(
        self,
        prefix: str,
        criterion: Dict[str, Any],
        aux_loss: float,
    ) -> None:
        """Log training, validation, or test metrics."""
        log_info = {f"{prefix}/{key}": value for key, value in criterion.items()}
        log_info[f"{prefix}/aux"] = aux_loss

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
        # Get optimizers (simple AE may only provide one)
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        net_optimizer = optimizers[0]
        aux_optimizer = optimizers[1] if len(optimizers) > 1 else None

        # Forward pass
        input, target = self._random_switch_Re_Im(batch)
        output = self.forward(input)
        criterion = self.criterion(output, target)  # Noise2Noise: target is the other channel

        # Backward pass for the main loss
        self.manual_backward(criterion["loss"])
        self.clip_gradients(
            net_optimizer,  # type: ignore[attr-defined]
            gradient_clip_val=self.hparams.gradient_clip_norm,  # type: ignore[attr-defined]
            gradient_clip_algorithm="norm",
        )
        net_optimizer.step()
        net_optimizer.zero_grad()

        # Auxiliary loss (entropy bottleneck) if available
        aux_loss = (
            self.net.aux_loss()
            if hasattr(self.net, "aux_loss")
            else torch.zeros((), device=output["x_hat"].device)
        )
        if aux_optimizer is not None and aux_loss.requires_grad:
            self.manual_backward(aux_loss)
            aux_optimizer.step()
            aux_optimizer.zero_grad()

        # Step scheduler if available (for epoch-based schedulers)
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()  # type: ignore[attr-defined]

        # custom learning rate logging
        lr = net_optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # Log metrics
        self._log_metrics("train", criterion, aux_loss.item())

    def on_test_epoch_start(self) -> None:
        """Update the entropy bottleneck tables before testing."""
        self.net.update(force=True)

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized processing of both real and imaginary parts."""
        # input, target = self._random_switch_Re_Im(batch)
        input = torch.cat((batch["real"], batch["imag"]), dim=1).contiguous()
        output = self.forward(input)
        criterion = self.criterion(output, target=input)
        aux_loss = self.net.aux_loss()
        self._log_metrics("valid", criterion, aux_loss.item())

    def test_step(self, batch, batch_idx):
        """Test step: keep original metrics and, if references are available, compute GT metrics.

        - Logs standard criterion metrics (as before) using a random Re/Im switch.
        - Additionally, when batch contains 'adam_ref' and 'merlin_ref', computes reconstruction
          from both real and imag inputs, converts to log-intensity, and logs PSNR/SSIM/MS-SSIM/MSE
          against each reference.
        """
        input = torch.cat((batch["real"], batch["imag"]), dim=1).contiguous()
        output = self.forward(input)

        # Compute normal losses with Noise2Noise approach
        criterion = self.criterion(output, target=input)
        all_metrics = {f"test/{key}": value for key, value in criterion.items()}
        all_metrics["test/aux"] = self.net.aux_loss().item()

        # Convert to linear amplitude
        recon_denorm = output["x_hat"] * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = torch.exp(recon_denorm)
        clean_im_real = torch.square(recon_lin[:, 0, :, :])
        clean_im_imag = torch.square(recon_lin[:, 1, :, :])
        clean_im = torch.sqrt(0.5 * (clean_im_real + clean_im_imag)).unsqueeze(1)

        # Load GTs references
        adam_noc_ref = batch["adam_noc_ref"]
        merlin_ref = batch["merlin_ref"]
        peak = float(torch.max(clean_im))

        # MSE
        all_metrics["test/mse_adam_noc"] = mse(clean_im, adam_noc_ref)
        all_metrics["test/mse_merlin"] = mse(clean_im, merlin_ref)
        # PSNR
        all_metrics["test/psnr_adam_noc"] = psnr(
            clean_im, adam_noc_ref, mse_value=all_metrics["test/mse_adam_noc"]
        )
        all_metrics["test/psnr_merlin"] = psnr(
            clean_im, merlin_ref, mse_value=all_metrics["test/mse_merlin"]
        )
        # SSIM
        all_metrics["test/ssim_adam_noc"] = F.structural_similarity_index_measure(
            clean_im, adam_noc_ref, data_range=peak
        )
        all_metrics["test/ssim_merlin"] = F.structural_similarity_index_measure(
            clean_im, merlin_ref, data_range=peak
        )
        # MS-SSIM
        all_metrics["test/ms_ssim_adam_noc"] = F.multiscale_structural_similarity_index_measure(
            clean_im, adam_noc_ref, data_range=peak
        )
        all_metrics["test/ms_ssim_merlin"] = F.multiscale_structural_similarity_index_measure(
            clean_im, merlin_ref, data_range=peak
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
        assert not set(main_params) & set(aux_params)
        # , "Intersection found in main and auxiliary parameters"
        assert set(main_params) | set(aux_params) == all_params
        # , "Union of main and auxiliary parameters does not match all model parameters"

        # Instantiate optimizer(s)
        net_optimizer = self.hparams.net_optimizer(params=main_params)  # type: ignore[attr-defined]
        optimizers_cfg = []

        if getattr(self.hparams, "scheduler", None) is not None:
            scheduler = self.hparams.scheduler(optimizer=net_optimizer)  # type: ignore[attr-defined]
            optimizers_cfg.append(
                {
                    "optimizer": net_optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "name": "net_lr",
                    },
                }
            )
        else:
            optimizers_cfg.append({"optimizer": net_optimizer})

        # Only add auxiliary optimizer if there are auxiliary params (hyperprior model case)
        if len(aux_params) > 0:
            aux_optimizer = self.hparams.aux_optimizer(params=aux_params)  # type: ignore[attr-defined]
            optimizers_cfg.append({"optimizer": aux_optimizer})

        return optimizers_cfg
