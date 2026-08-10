from typing import Any, Dict, Optional, Tuple

import lightning
import torch
from torch import Tensor

from src.utils.constants import AMP_MAX, AMP_MIN, EPS
from src.utils.metrics import (
    compute_bitstream_bpp,
    enl,
    epd,
    ms_ssim,
    mse,
    psnr,
    ratio_enl,
    ratio_mean,
    ssim,
)


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

        # Each run is possibly evaluated on different test sets. We use a dynamic prefix to distinguish metrics for different test sets. Default is "test".
        self.test_prefix = "test"

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
        - Also computes actual bitstream BPP by running compress/decompress.
        """
        input = torch.cat((batch["real"], batch["imag"]), dim=1).contiguous()

        # Preprocess for network input (log-scale normalization)
        x = (torch.log(torch.square(input) + EPS) - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)

        # 1. Forward pass (for likelihood estimation metrics)
        output = self.net(x)

        # 2. Real compression (for actual bitstream BPP)
        out_enc = self.net.compress(x)
        out_dec = self.net.decompress(out_enc["strings"], out_enc["shape"])

        # Use decompressed output for reconstruction metrics to be as close to FPGA as possible.
        # Sanitize x_hat: high-lambda GDN models can produce NaN (from GDN(inf)=inf/inf when large
        # latents overflow inside g_s) as well as finite out-of-range values.
        # nan_to_num replaces NaN/±inf first, then clamp enforces [0,1].
        x_hat_raw = out_dec
        if False:
            has_nan = torch.isnan(x_hat_raw).any().item()
            has_inf = not torch.isfinite(x_hat_raw).all().item()
            is_out_of_range = bool(
                (x_hat_raw[torch.isfinite(x_hat_raw)].numel() > 0)
                and (
                    x_hat_raw[torch.isfinite(x_hat_raw)].max() > 1.0
                    or x_hat_raw[torch.isfinite(x_hat_raw)].min() < 0.0
                )
            )
            if has_nan or has_inf or is_out_of_range:
                print(
                    f"[test_step] batch_idx={batch_idx}: x_hat abnormal "
                    f"(nan={has_nan}, inf={has_inf}, out_of_range={is_out_of_range}). Sanitizing."
                )
        output["x_hat"] = torch.nan_to_num(x_hat_raw, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )

        # Compute normal losses with Noise2Noise approach
        criterion = self.criterion(output, target=input)
        prefix = getattr(self, "test_prefix", "test_unknown")
        all_metrics = {
            f"{prefix}/{key}": value
            for key, value in criterion.items()
            if key in ["bpp", "merlin", "loss"]
        }
        all_metrics[f"{prefix}/aux"] = self.net.aux_loss().item()

        # Calculate and log real bitstream BPP
        N, C, H, W = input.shape
        bpp_bits = compute_bitstream_bpp(out_enc["strings"], H, W, N)
        all_metrics[f"{prefix}/bpp_bitstream"] = bpp_bits

        # Convert to linear amplitude
        recon_denorm = output["x_hat"] * (AMP_MAX - AMP_MIN) + AMP_MIN
        recon_lin = torch.exp(recon_denorm)
        clean_im_real = torch.square(recon_lin[:, 0, :, :])
        clean_im_imag = torch.square(recon_lin[:, 1, :, :])
        clean_im = torch.sqrt(0.5 * (clean_im_real + clean_im_imag)).unsqueeze(1)

        # ── SAR quality metrics (reference-free / noisy-paired) ─────────────────────
        # noisy_linA: MERLIN convention — average Re²+Im² power, then sqrt
        noisy_linA = torch.sqrt(0.5 * (torch.square(batch["real"]) + torch.square(batch["imag"])))
        all_metrics[f"{prefix}/mse_noisy"] = mse(clean_im, noisy_linA)
        all_metrics[f"{prefix}/psnr_noisy"] = psnr(clean_im, noisy_linA)
        all_metrics[f"{prefix}/ssim_noisy"] = ssim(clean_im, noisy_linA)
        all_metrics[f"{prefix}/ms_ssim_noisy"] = ms_ssim(clean_im, noisy_linA)
        all_metrics[f"{prefix}/enl_recon"] = enl(clean_im)
        all_metrics[f"{prefix}/ratio_mean"] = ratio_mean(clean_im, noisy_linA)
        all_metrics[f"{prefix}/ratio_enl"] = ratio_enl(clean_im, noisy_linA)

        # Load GTs references
        if "adam_noc_ref" in batch and "merlin_ref" in batch:
            adam_noc_ref = batch["adam_noc_ref"]
            merlin_ref = batch["merlin_ref"]

            # MSE
            all_metrics[f"{prefix}/mse_adam_noc"] = mse(clean_im, adam_noc_ref)
            all_metrics[f"{prefix}/mse_merlin"] = mse(clean_im, merlin_ref)
            # PSNR
            all_metrics[f"{prefix}/psnr_adam_noc"] = psnr(
                clean_im, adam_noc_ref, mse_value=all_metrics[f"{prefix}/mse_adam_noc"]
            )
            all_metrics[f"{prefix}/psnr_merlin"] = psnr(
                clean_im, merlin_ref, mse_value=all_metrics[f"{prefix}/mse_merlin"]
            )
            # SSIM / MS-SSIM — clipped to AMP_LIN_99 with data_range=AMP_LIN_99, the same fixed
            # basis as MSE/PSNR above (src.utils.metrics._clip_to_amp99).
            all_metrics[f"{prefix}/ssim_adam_noc"] = ssim(clean_im, adam_noc_ref)
            all_metrics[f"{prefix}/ssim_merlin"] = ssim(clean_im, merlin_ref)
            all_metrics[f"{prefix}/ms_ssim_adam_noc"] = ms_ssim(clean_im, adam_noc_ref)
            all_metrics[f"{prefix}/ms_ssim_merlin"] = ms_ssim(clean_im, merlin_ref)
            # EPD (Edge Preservation Degree, linA)
            all_metrics[f"{prefix}/epd_adam_noc"] = epd(clean_im, adam_noc_ref)
            all_metrics[f"{prefix}/epd_merlin"] = epd(clean_im, merlin_ref)
        else:
            # Just skip metrics if references are not available (e.g. testing only on .npy patches)
            pass

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
