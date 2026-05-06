"""DPU-friendly Factorized Prior model."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from compressai.models import CompressionModel
from torch import Size, Tensor

from context.compressai_original import (  # from compressai.entropy_models
    EntropyBottleneck,
)
from src.models.components.compressai_dpu import EntropyBottleneckPatched
from src.models.components.layers import ResidualBlock, make_activation
from src.utils.debug import log_tensor_shape


# -------------------------------------------------------------------------
# DPU-friendly FactorizedPrior
# -------------------------------------------------------------------------
class ResidualFactorizedPriorPatched(CompressionModel):
    """Residual Factorized Prior model for SAR despeckling and compression (DPU-friendly version).

    Adaptation of "ADAM" model from the paper "Joint compression and despeckling by SAR
    representation learning" by Joel Amao-Oliva, Nils Foix-Colonier, Francescopaolo Sica. (2024).
    (ISPRS Journal of Photogrammetry and Remote Sensing) to the original FactorizedPrior
    architecture from Johannes Ballé.
    """

    def __init__(
        self,
        nb_channels_main: int = 128,
        activation: str = "gdn",
        no_output_padding: bool = True,
        no_residual_blocks: bool = False,
        export_dpu: bool = False,
    ):
        """Initialize the FactorizedPriorPatched model.

        Architectures information:
        - Unlike the ResidualScaleHyperprior, the FactorizedPrior does not have a separate hyperprior path for the scales.
        - The Main encoder/decoder each have four stride=2 (de)convolution layers. Therefore, the latent space has a size 1/16 of the input resolution (e.g., 256 -> 16).
        The entropy coder has 256 channels if `nb_channels_main=128`.
        Dimensions examples (without batch dimension):
        - Input image: [2, 256, 256]
        - Latent space: [256, 16, 16]

        Args:
            nb_channels_main (int): Number of channels for main path (default: 128)
            activation (str): Activation function to use. 'gdn', 'relu', 'gdn1', etc. (default: 'gdn')
            no_output_padding (bool): If True, modifies ConvTranspose2d kernels size to not use output padding as it is not supported by Vitis-AI DPU (default: True)
            no_residual_blocks (bool): If True, remove all ResidualBlocks from g_a and g_s (default: False)
            export_dpu (bool): If True, use DPU-patched layers for inference/export (default: False)
        """
        super().__init__()
        N = nb_channels_main
        M = 2 * N
        self.nb_channels_main: int = N
        self.export_dpu: bool = export_dpu
        self.no_residual_blocks: bool = no_residual_blocks
        self.activation: str = activation

        def _maybe_residual(channels: int) -> nn.Module:
            """Return a ResidualBlock or Identity depending on no_residual_blocks."""
            if no_residual_blocks:
                return nn.Identity()
            return ResidualBlock(channels, activation, use_patched_gdn=export_dpu)

        if export_dpu and not no_output_padding:
            raise ValueError(
                "DPU export with output padding, i.e., ConvTranspose2d use a non-zero `out_padding` argument, which is not supported by Vitis-AI DPU and will crash the compilation."
            )

        # Same ConvTranspose2d config as original model
        convT_kernel: int = 4 if no_output_padding else 5
        convT_out_pad: int = 0 if no_output_padding else 1
        convT_padding: int = 1 if no_output_padding else 2

        if self.export_dpu:
            self.entropy_bottleneck: torch.nn.Module = EntropyBottleneckPatched(M)
        else:
            self.entropy_bottleneck: torch.nn.Module = EntropyBottleneck(M)

        # Main analysis transform (encoder g_a)
        self.g_a: nn.Sequential = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=export_dpu),
                _maybe_residual(N),
            ),
            nn.Sequential(
                nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=export_dpu),
                _maybe_residual(N),
            ),
            nn.Sequential(
                nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
                make_activation(activation, N, inverse=False, use_patched_gdn=export_dpu),
                _maybe_residual(N),
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
                _maybe_residual(N),
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
                _maybe_residual(N),
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
                _maybe_residual(N),
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

    @property
    def main_downsampling_factor(self) -> int:
        """Return the overall downsampling factor of the main encoder."""
        return 16  # 2^4 from the 4 stride=2 layers in g_a

    def aux_loss(self) -> Tensor:
        """Return the EntropyBottleneck's auxiliary loss for training."""
        return self.entropy_bottleneck.loss()

    # ---------------- Forward ----------------
    def forward(self, x: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        """Complete forward pass. Expects 1-channel input during training and 2-channel input
        during evaluation/inference. When `export_dpu=True`, this module is intended to be used
        only in eval mode (no training), ensuring LowerBoundPatched is only used at inference.

        Args:
            x (Tensor): Input tensor. Shape [B, 1, H, W] during training, [B, 2, H, W] (real, imag) during evaluation/inference.
        Returns:
            dict: Dictionary containing 'x_hat' and the 'likelihoods'.
        """
        if self.export_dpu and self.training:
            raise RuntimeError(
                "FactorizedPriorPatched with export_dpu=True must not be used in training mode."
            )

        if self.training:
            assert x.shape[1] == 1, "Training: Input tensor must have 1 channel"

            # Analysis transform to get latent representation
            y: Tensor = self.g_a(x)
            # Concatenate y with itself along channel dimension
            y = torch.cat((y, y), dim=1)
            # Apply entropy coding
            y_hat, y_likelihoods = self.entropy_bottleneck(y)

            # Split: discard second half of y_hat
            y_hat = y_hat[:, : y_hat.shape[1] // 2, :, :]

            # Apply synthesis transform to reconstruct
            x_hat: Tensor = self.g_s(y_hat)
        else:
            assert x.shape[1] == 2, "Inference: Input tensor must have 2 channels"
            x_real: Tensor = x[:, :1, :, :]
            x_imag: Tensor = x[:, 1:, :, :]

            # Analysis transform to get latent representation
            y_real: Tensor = self.g_a(x_real)
            y_imag: Tensor = self.g_a(x_imag)
            y: Tensor = torch.cat((y_real, y_imag), dim=1)

            # Apply entropy coding
            y_hat, y_likelihoods = self.entropy_bottleneck(y)

            # Split real/imag paths
            y_hat_real: Tensor = y_hat[:, : y_hat.shape[1] // 2, :, :]
            y_hat_imag: Tensor = y_hat[:, y_hat.shape[1] // 2 :, :, :]

            # Apply synthesis transform to reconstruct
            x_hat_real: Tensor = self.g_s(y_hat_real)
            x_hat_imag: Tensor = self.g_s(y_hat_imag)
            x_hat: Tensor = torch.cat((x_hat_real, x_hat_imag), dim=1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    # ---------------- Update Entropy ----------------
    def update(self, force: bool = False) -> None:
        """Update entropy models (e.g., populate tables).

        We need to explicitly update components because we are using "Patched" components that are
        skipped by CompressionModel.update() (Patched components are no instances of
        compressai.entropy_models.EntropyModel).
        """
        print(f"  -> Updating {type(self.entropy_bottleneck).__name__}...")
        self.entropy_bottleneck.update(force=force)

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

        y_real: Tensor = self.g_a(x_real)
        y_imag: Tensor = self.g_a(x_imag)
        y: Tensor = torch.cat((y_real, y_imag), dim=1)

        y_strings = self.entropy_bottleneck.compress(y)

        return {"strings": [y_strings], "shape": y.shape[-2:]}

    # ---------------- Decompress ----------------
    def decompress(self, strings: list[bytes], shape: Size) -> Tensor:
        """Decompress an input tensor from compressed strings.

        Args:
            strings (list[bytes]): List containing compressed strings for y and z.
            shape (Size): Shape of the hyperlatent tensor z.
        Returns:
            Tensor: Decompressed tensor. Shape [B, 2, H, W] (real, imag).
        """
        y_hat: Tensor = self.entropy_bottleneck.decompress(strings[0], shape)

        y_hat_real: Tensor = y_hat[:, : y_hat.shape[1] // 2, :, :]
        y_hat_imag: Tensor = y_hat[:, y_hat.shape[1] // 2 :, :, :]

        x_hat_real: Tensor = self.g_s(y_hat_real)
        x_hat_imag: Tensor = self.g_s(y_hat_imag)
        x_hat: Tensor = torch.cat((x_hat_real, x_hat_imag), dim=1)

        return x_hat
