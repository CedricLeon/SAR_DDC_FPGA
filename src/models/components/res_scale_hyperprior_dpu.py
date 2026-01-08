"""DPU-friendly Residual Scale Hyperprior model.

This mirrors `src/models/components/res_scale_hyperprior.py::ResidualScaleHyperprior` but replaces
GDN with a DPU-safe variant (GDNPatched) during export / inference.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from compressai.models import CompressionModel
from torch import Size, Tensor

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

    The model architecture is identical to the original `ResidualScaleHyperprior` (called ADAM in the literature), but allows transforming the model in a DPU-exportable format via the `export_dpu` switch. The intended usage is:
      - During normal training: `export_dpu=False`. The model then uses original CompressAI ops (EntropyBottleneck, GaussianConditional, and GDN) that include a custom backward pass.
      - During DPU export/inference: `export_dpu=True`. The model then uses Patched variations of CompressAI ops, without a custom backward pass.
    """

    def __init__(
        self,
        nb_channels_main: int = 128,
        activation: str = "gdn",
        no_output_padding: bool = True,
        export_dpu: bool = False,
    ):
        """Initialize the ResidualScaleHyperpriorPatched model.

        Architectures information:
        - The Main encoder/decoder each have four stride=2 (de)convolution layers. Therefore, the latent space has a size 1/16 of the input resolution (e.g., 256 -> 16).
        - Similarly, the hyper encoder/decoder each have three stride=2 (de)convolution layers. Therefore, the hyperlatent space has a size 1/8 of the latent space resolution (e.g., 16 -> 2).
        With a default `nb_channels_main=128`, the hyper encoder/decoder have twice (concatenation of the latents for the real and imaginary parts), so 256 channels.
        For example, it gives the following dimensions (without batch dimension):
        - Input image: [2, 256, 256]
        - Latent space: [256, 16, 16]
        - Hyperlatent space: [256, 2, 2]

        Args:
            nb_channels_main (int): Number of channels for main path (default: 128)
            activation (str): Activation function to use. 'gdn', 'relu', etc. (default: 'gdn')
            no_output_padding (bool): If True, modifies ConvTranspose2d kernels size to not use output padding (default: True)
            export_dpu (bool): If True, use DPU-patched layers for inference/export (default: False)
        """
        super().__init__()
        N = nb_channels_main
        M = 2 * N  # Number of channels for hyperprior
        self.export_dpu: bool = export_dpu

        if export_dpu and not no_output_padding:
            raise ValueError(
                "DPU export with output padding, i.e., ConvTranspose2d use a non-zero `out_padding` argument, is not supported."
            )

        # Big ugly parameter for shape logging during inference
        self.DEBUG_MODE: bool = False

        # Same ConvTranspose2d config as original model
        convT_kernel: int = 4 if no_output_padding else 5
        convT_out_pad: int = 0 if no_output_padding else 1
        convT_padding: int = 1 if no_output_padding else 2

        if self.export_dpu:
            self.entropy_bottleneck: torch.nn.Module = EntropyBottleneckPatched(M)
            self.gaussian_conditional: torch.nn.Module = GaussianConditionalPatched(None)
        else:
            self.entropy_bottleneck: torch.nn.Module = EntropyBottleneck(M)
            self.gaussian_conditional: torch.nn.Module = GaussianConditional(None)

        # Main analysis transform (encoder g_a)
        self.g_a: nn.Sequential = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=export_dpu),
            ),
            nn.Sequential(
                nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=export_dpu),
            ),
            nn.Sequential(
                nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=export_dpu),
            ),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )

        # Main synthesis transform (decoder g_s)
        self.g_s: nn.Sequential = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(
                    N,
                    N,
                    kernel_size=convT_kernel,
                    stride=2,
                    padding=convT_padding,
                    output_padding=convT_out_pad,
                ),
                make_activation(activation, N, inverse=True, use_patched_gdn=export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=export_dpu),
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
                make_activation(activation, N, inverse=True, use_patched_gdn=export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=export_dpu),
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
                make_activation(activation, N, inverse=True, use_patched_gdn=export_dpu),
                ResidualBlock(N, activation, use_patched_gdn=export_dpu),
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

        # Hyperprior analysis transform
        self.h_a: nn.Sequential = nn.Sequential(
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
        )

        # Hyperprior synthesis transform
        self.h_s: nn.Sequential = nn.Sequential(
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

    def scale_hyperprior(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Apply hyperprior to predict the scales of each latent variable and compute the
        z-likelihoods.

        Args:
            y (Tensor): Latent representation from analysis transform. Shape: [B, 2*N, H', W']
        Returns:
            scales (Tensor): Predicted scales for each latent variable. Shape: [B, 2*N, H'', W'']
            z_likelihoods (Tensor): Likelihoods of the hyperprior latent variables. Shape: [B, 2*N, H'', W'']
        """
        z: Tensor = torch.abs(y)
        z = self.h_a(z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales: Tensor = self.h_s(z_hat)

        if self.DEBUG_MODE:
            log_tensor_shape("scale_hyperprior.z_abs", z)
            log_tensor_shape("scale_hyperprior.h_a(z)", z)
            log_tensor_shape("scale_hyperprior.z_hat", z_hat)
            log_tensor_shape("scale_hyperprior.z_likelihoods", z_likelihoods)
            log_tensor_shape("scale_hyperprior.scales", scales)

        return scales, z_likelihoods

    # ---------------- Forward ----------------
    def forward(self, x: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        """Forward pass.

        When `export_dpu=True`, this module is intended to be used only in eval
        mode (no training), ensuring LowerBoundPatched is only used at inference.
        Args:
            x (Tensor): Input tensor. Shape [B, 1, H, W] during training, [B, 2, H, W] (real, imag) during evaluation/inference.
        Returns:
            dict: Dictionary containing 'x_hat', 'y_hat', and the 'likelihoods'.
        """
        if self.export_dpu and self.training:
            raise RuntimeError(
                "ResidualScaleHyperpriorPatched with export_dpu=True must not be used in training mode."
            )
        if self.DEBUG_MODE:
            log_tensor_shape("forward.x", x)

        if self.training:
            assert x.shape[1] == 1, "Training: Input tensor must have 1 channel"

            # Analysis transform to get latent representation
            y: Tensor = self.g_a(x)
            if self.DEBUG_MODE:
                log_tensor_shape("forward.train.y", y)
            # Concatenate y with itself along channel dimension
            y = torch.cat((y, y), dim=1)

            # Apply hyperprior to get scales
            if self.DEBUG_MODE:
                log_tensor_shape("forward.train.y_cat", y)
            scales, z_likelihoods = self.scale_hyperprior(y)

            # Apply entropy coding
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales)

            # Split: discard second half of y_hat
            if self.DEBUG_MODE:
                log_tensor_shape("forward.train.y_hat_full", y_hat)
            y_hat = y_hat[:, : y_hat.shape[1] // 2, :, :]

            # Apply synthesis transform to reconstruct
            if self.DEBUG_MODE:
                log_tensor_shape("forward.train.y_hat_half", y_hat)
            x_hat: Tensor = self.g_s(y_hat)
            if self.DEBUG_MODE:
                log_tensor_shape("forward.train.x_hat", x_hat)
        else:
            assert x.shape[1] == 2, "Inference: Input tensor must have 2 channels"
            x_real: Tensor = x[:, :1, :, :]
            x_imag: Tensor = x[:, 1:, :, :]

            # Analysis transform to get latent representation
            y_real: Tensor = self.g_a(x_real)
            y_imag: Tensor = self.g_a(x_imag)
            if self.DEBUG_MODE:
                log_tensor_shape("forward.eval.y_real", y_real)
                log_tensor_shape("forward.eval.y_imag", y_imag)
            y: Tensor = torch.cat((y_real, y_imag), dim=1)

            # Apply hyperprior to get scales
            if self.DEBUG_MODE:
                log_tensor_shape("forward.eval.y_cat", y)
            scales, z_likelihoods = self.scale_hyperprior(y)

            # Apply entropy coding
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales)

            # Split real/imag paths
            if self.DEBUG_MODE:
                log_tensor_shape("forward.eval.y_hat", y_hat)
            y_hat_real: Tensor = y_hat[:, : y_hat.shape[1] // 2, :, :]
            y_hat_imag: Tensor = y_hat[:, y_hat.shape[1] // 2 :, :, :]

            # Apply synthesis transform to reconstruct
            if self.DEBUG_MODE:
                log_tensor_shape("forward.eval.y_hat_real", y_hat_real)
                log_tensor_shape("forward.eval.y_hat_imag", y_hat_imag)
            x_hat_real: Tensor = self.g_s(y_hat_real)
            x_hat_imag: Tensor = self.g_s(y_hat_imag)
            if self.DEBUG_MODE:
                log_tensor_shape("forward.eval.x_hat_real", x_hat_real)
                log_tensor_shape("forward.eval.x_hat_imag", x_hat_imag)
            x_hat: Tensor = torch.cat((x_hat_real, x_hat_imag), dim=1)
            if self.DEBUG_MODE:
                log_tensor_shape("forward.eval.x_hat", x_hat)
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # ---------------- Compress ----------------
    def compress(self, x: Tensor) -> dict[str, list[bytes] | Size]:
        """Compress an input tensor into strings.

        Args:
            x (Tensor): Input tensor. Shape [B, 2, H, W] (real, imag).
        Returns:
            dict: Dictionary containing 'strings' and 'shape'.
        """
        assert x.shape[1] == 2, "Compress: Input tensor must have 2 channels"
        x_real: Tensor = x[:, :1, :, :]
        x_imag: Tensor = x[:, 1:, :, :]

        # Analysis transform to get latent representation
        y_real: Tensor = self.g_a(x_real)
        y_imag: Tensor = self.g_a(x_imag)
        y: Tensor = torch.cat((y_real, y_imag), dim=1)

        # Apply hyperprior to get scales necessary for entropy coding of the main latent
        z: Tensor = self.h_a(torch.abs(y))
        z_strings, _ = self.entropy_bottleneck.compress(z)
        z_hat: Tensor = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales: Tensor = self.h_s(z_hat)

        # Apply entropy coding
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        print(f"Warning: I DID NOT CHECK WHY SHAPE IS {z.shape[-2:]=}.")

        return {"strings": [y_strings, z_strings], "shape": z.shape[-2:]}

    # ---------------- Decompress ----------------
    def decompress(self, strings: list[bytes], shape: Size) -> Tensor:
        """Decompress an input tensor from compressed strings.

        Args:
            strings (list[bytes]): List containing compressed strings for y and z.
            shape (Size): Shape of the hyperlatent tensor z.
        Returns:
            Tensor: Decompressed tensor. Shape [B, 2, H, W] (real, imag).
        """
        z_strings = strings[1]
        z_hat: Tensor = self.entropy_bottleneck.decompress(z_strings, shape)
        scales: Tensor = self.h_s(z_hat)

        y_strings = strings[0]
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat: Tensor = self.gaussian_conditional.decompress(y_strings, indexes)

        # Split real/imag paths
        y_hat_real: Tensor = y_hat[:, : y_hat.shape[1] // 2, :, :]
        y_hat_imag: Tensor = y_hat[:, y_hat.shape[1] // 2 :, :, :]

        # Apply synthesis transform to reconstruct
        x_hat_real: Tensor = self.g_s(y_hat_real)
        x_hat_imag: Tensor = self.g_s(y_hat_imag)
        x_hat: Tensor = torch.cat((x_hat_real, x_hat_imag), dim=1)

        return x_hat

    # ---------------- Aux loss ----------------
    def aux_loss(self) -> Tensor:
        """Return the EntropyBottleneck's auxiliary loss for training."""
        return self.entropy_bottleneck.loss()
