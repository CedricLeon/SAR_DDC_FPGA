"""DPU-friendly Residual Scale Hyperprior model.

This mirrors `src/models/components/res_scale_hyperprior.py::ResidualScaleHyperprior` but replaces
GDN with a DPU-safe variant (GDNPatched) during export / inference.

  - During normal training: use original CompressAI ops (EntropyBottleneck, GaussianConditional,
    GDN via make_activation).
  - During DPU export / inference: use GDNPatched + LowerBoundPatched; entropy models stay
    unpatched (we don't run them on DPU).
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from compressai.models import CompressionModel
from torch import Tensor

from context.compressai_original import (  # from compressai.entropy_models
    EntropyBottleneck,
    GaussianConditional,
)
from src.models.components.compressai_dpu import (
    EntropyBottleneckPatched,
    GaussianConditionalPatched,
)
from src.models.components.layers import ResidualBlock, make_activation
from src.utils.debug import log_tensor_shape

# -------------------------------------------------------------------------
# DPU-friendly ResidualScaleHyperprior
# -------------------------------------------------------------------------


class ResidualScaleHyperpriorPatched(CompressionModel):
    """Residual Scale Hyperprior model for SAR despeckling and compression (DPU-friendly).

    Architecture is identical to `ResidualScaleHyperprior`, but allows switching GDN to
    GDNPatched via `export_dpu=True` in order to make the model friendlier to DPU export.

    EntropyBottleneck and GaussianConditional are kept as the original CompressAI
    implementations (we don't run them on DPU).
    """

    def __init__(
        self,
        nb_channels_main: int = 128,
        activation: str = "gdn",
        no_output_padding: bool = True,
        export_dpu: bool = False,
    ):
        super().__init__()
        N = nb_channels_main
        M = 2 * N  # Number of channels for hyperprior
        self.activation = activation
        self.export_dpu = export_dpu

        # Same ConvTranspose2d config as original model
        convT_kernel = 4 if no_output_padding else 5
        convT_out_pad = 0 if no_output_padding else 1
        convT_padding = 1 if no_output_padding else 2

        # NOTE: keep entropy models unpatched for now
        if self.export_dpu:
            self.entropy_bottleneck = EntropyBottleneckPatched(M)
            self.gaussian_conditional = GaussianConditionalPatched(None)
        else:
            self.entropy_bottleneck = EntropyBottleneck(M)
            self.gaussian_conditional = GaussianConditional(None)

        # Main analysis transform (encoder g_a)
        self.g_a = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=self.export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=self.export_dpu),
            ),
            nn.Sequential(
                nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=self.export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=self.export_dpu),
            ),
            nn.Sequential(
                nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=self.export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=self.export_dpu),
            ),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )

        # Main synthesis transform (decoder g_s)
        self.g_s = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(
                    N,
                    N,
                    kernel_size=convT_kernel,
                    stride=2,
                    padding=convT_padding,
                    output_padding=convT_out_pad,
                ),
                make_activation(activation, N, inverse=True, use_patched_gdn=self.export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=self.export_dpu),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    N,
                    N,
                    kernel_size=convT_kernel,
                    stride=2,
                    padding=convT_padding,
                    output_padding=convT_out_pad,
                ),
                make_activation(activation, N, inverse=True, use_patched_gdn=self.export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=self.export_dpu),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    N,
                    N,
                    kernel_size=convT_kernel,
                    stride=2,
                    padding=convT_padding,
                    output_padding=convT_out_pad,
                ),
                make_activation(activation, N, inverse=True, use_patched_gdn=self.export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=self.export_dpu),
            ),
            nn.ConvTranspose2d(
                N,
                1,
                kernel_size=convT_kernel,
                stride=2,
                padding=convT_padding,
                output_padding=convT_out_pad,
            ),
        )

        # Hyperprior analysis transform (h_a)
        self.h_a = nn.Sequential(
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
        )

        # Hyperprior synthesis transform (h_s)
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(
                M,
                M,
                kernel_size=convT_kernel,
                stride=2,
                padding=convT_padding,
                output_padding=convT_out_pad,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                M,
                M,
                kernel_size=convT_kernel,
                stride=2,
                padding=convT_padding,
                output_padding=convT_out_pad,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                M,
                M,
                kernel_size=convT_kernel,
                stride=2,
                padding=convT_padding,
                output_padding=convT_out_pad,
            ),
        )

    # ---------------- Hyperprior path ----------------

    def scale_hyperprior(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Apply hyperprior to get scales and z-likelihoods."""
        z = torch.abs(y)
        log_tensor_shape("scale_hyperprior.z_abs", z)
        z = self.h_a(z)
        log_tensor_shape("scale_hyperprior.h_a(z)", z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        log_tensor_shape("scale_hyperprior.z_hat", z_hat)
        log_tensor_shape("scale_hyperprior.z_likelihoods", z_likelihoods)
        scales = self.h_s(z_hat)
        log_tensor_shape("scale_hyperprior.scales", scales)
        return scales, z_likelihoods

    # ---------------- Forward ----------------

    def forward(self, x: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        """Forward pass.

        Training:
            - x: [B, 1, H, W]
        Eval / inference:
            - x: [B, 2, H, W] (real, imag)

        When `export_dpu=True`, this module is intended to be used only in eval
        mode (no training), ensuring LowerBoundPatched is only used at inference.
        """
        if self.export_dpu and self.training:
            raise RuntimeError(
                "ResidualScaleHyperpriorPatched with export_dpu=True must not be used in training mode."
            )
        log_tensor_shape("forward.x", x)

        if self.training:
            assert x.shape[1] == 1, "Training: Input tensor must have 1 channel"

            # Analysis transform to get latent representation
            y = self.g_a(x)
            log_tensor_shape("forward.train.y", y)
            # Concatenate y with itself along channel dimension
            y = torch.cat((y, y), dim=1)

            # Apply hyperprior to get scales
            log_tensor_shape("forward.train.y_cat", y)
            scales, z_likelihoods = self.scale_hyperprior(y)

            # Apply entropy coding
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales)

            # Split: discard second half of y_hat
            log_tensor_shape("forward.train.y_hat_full", y_hat)
            y_hat = y_hat[:, : y_hat.shape[1] // 2, :, :]

            # Apply synthesis transform to reconstruct
            log_tensor_shape("forward.train.y_hat_half", y_hat)
            x_hat = self.g_s(y_hat)
            log_tensor_shape("forward.train.x_hat", x_hat)
        else:
            assert x.shape[1] == 2, "Inference: Input tensor must have 2 channels"
            x_real = x[:, :1, :, :]
            x_imag = x[:, 1:, :, :]

            # Analysis transform to get latent representation
            y_real = self.g_a(x_real)
            y_imag = self.g_a(x_imag)
            log_tensor_shape("forward.eval.y_real", y_real)
            log_tensor_shape("forward.eval.y_imag", y_imag)
            y = torch.cat((y_real, y_imag), dim=1)

            # Apply hyperprior to get scales
            log_tensor_shape("forward.eval.y_cat", y)
            scales, z_likelihoods = self.scale_hyperprior(y)

            # Apply entropy coding
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales)

            # Split real/imag paths
            log_tensor_shape("forward.eval.y_hat", y_hat)
            y_hat_real = y_hat[:, : y_hat.shape[1] // 2, :, :]
            y_hat_imag = y_hat[:, y_hat.shape[1] // 2 :, :, :]

            # Apply synthesis transform to reconstruct
            log_tensor_shape("forward.eval.y_hat_real", y_hat_real)
            log_tensor_shape("forward.eval.y_hat_imag", y_hat_imag)
            x_hat_real = self.g_s(y_hat_real)
            x_hat_imag = self.g_s(y_hat_imag)
            log_tensor_shape("forward.eval.x_hat_real", x_hat_real)
            log_tensor_shape("forward.eval.x_hat_imag", x_hat_imag)
            x_hat = torch.cat((x_hat_real, x_hat_imag), dim=1)
            log_tensor_shape("forward.eval.x_hat", x_hat)
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # ---------------- Aux loss ----------------

    def aux_loss(self) -> Tensor:
        """Return the EntropyBottleneck's auxiliary loss for training."""
        return self.entropy_bottleneck.loss()
