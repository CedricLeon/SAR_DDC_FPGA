"""DPU-friendly Residual Scale Hyperprior model.

This replaces CompressAi DPU-problematic operations by patched versions from `src.models.components.compressai_dpu`.
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
    get_scale_table,
)
from src.models.components.layers import ResidualBlock, make_activation
from src.utils.debug import log_tensor_shape


# -------------------------------------------------------------------------
# DPU-friendly ResidualScaleHyperprior
# -------------------------------------------------------------------------
class ResidualScaleHyperpriorPatched(CompressionModel):
    """Residual Scale Hyperprior model for SAR despeckling and compression (DPU-friendly version).

    Joel Amao-Oliva, Nils Foix-Colonier, Francescopaolo Sica. (2024). Joint compression and despeckling by SAR representation learning. (ISPRS Journal of Photogrammetry and Remote Sensing).

    The model architecture is identical to the original `ResidualScaleHyperprior` (called ADAM in the literature), but allows transforming the model in a DPU-exportable format via the `export_dpu` switch.
    The intended usage is:
      - During normal training: `export_dpu=False`. The model then uses original CompressAI ops (EntropyBottleneck, GaussianConditional, and GDN) that include a custom backward pass called `LowerBoundFunction(torch.autograd.Function)`.
      - During DPU export/inference: `export_dpu=True`. The model then uses Patched variations of CompressAI ops, without this custom backward pass which is not supported by Vitis-AI.
    """

    def __init__(
        self,
        nb_channels_main: int = 128,
        activation: str = "gdn",
        no_output_padding: bool = True,
        no_residual_blocks: bool = False,
        export_dpu: bool = False,
    ):
        """Initialize the ResidualScaleHyperpriorPatched model.

        Architectures information:
        - The Main encoder/decoder each have four stride=2 (de)convolution layers. Therefore, the latent space has a size 1/16 of the input resolution (e.g., 256 -> 16).
        - Similarly, the hyper encoder/decoder each have three stride=2 (de)convolution layers. Therefore, the hyperlatent space has a size 1/8 of the latent space resolution (e.g., 16 -> 2).
        The hyper encoder/decoder have a doubled channel size, because of the concatenation of the latents for the real and imaginary parts, so 256 channels if `nb_channels_main=128`.
        Dimensions examples (without batch dimension):
        - Input image: [2, 256, 256]
        - Latent space: [256, 16, 16]
        - Hyperlatent space: [256, 2, 2]

        Args:
            nb_channels_main (int): Number of channels for main path (default: 128)
            activation (str): Activation function to use. 'gdn', 'relu', 'gdn1', etc. (default: 'gdn')
            no_output_padding (bool): If True, modifies ConvTranspose2d kernels size to not use output padding as it is not supported by Vitis-AI DPU (default: True)
            no_residual_blocks (bool): If True, remove all ResidualBlocks from g_a and g_s (default: False)
            export_dpu (bool): If True, use DPU-patched layers for inference/export (default: False)
        """
        super().__init__()
        N = nb_channels_main
        M = 2 * N  # Number of channels for hyperprior
        self.export_dpu: bool = export_dpu
        self.no_residual_blocks: bool = no_residual_blocks
        self.activation: str = activation
        self.DEBUG_MODE: bool = False  # Big ugly parameter for shape logging during inference

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
            self.gaussian_conditional: torch.nn.Module = GaussianConditionalPatched(None)
        else:
            self.entropy_bottleneck: torch.nn.Module = EntropyBottleneck(M)
            self.gaussian_conditional: torch.nn.Module = GaussianConditional(None)

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

    @property
    def main_downsampling_factor(self) -> int:
        """Return the overall downsampling factor of the main encoder."""
        return 16  # 2^4 from the 4 stride=2 layers in g_a

    @property
    def hyper_downsampling_factor(self) -> int:
        """Return the overall downsampling factor of the hyperprior."""
        return 8  # 2^3 from the 3 stride=2 layers in h_a

    def aux_loss(self) -> Tensor:
        """Return the EntropyBottleneck's auxiliary loss for training."""
        return self.entropy_bottleneck.loss()

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
        """Complete forward pass. Expects 1-channel input during training and 2-channel input
        during evaluation/inference. When `export_dpu=True`, this module is intended to be used
        only in eval mode (no training), ensuring LowerBoundPatched is only used at inference.

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
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # ---------------- Update Entropy ----------------
    def update(self, force: bool = False) -> None:
        """Update entropy models (e.g., populate tables).

        We need to explicitly update components because we are using "Patched" components that are
        skipped by CompressionModel.update() (Patched components are no instances of
        compressai.entropy_models.EntropyModel).
        """
        print(
            f"  -> Updating {type(self.entropy_bottleneck).__name__} and {type(self.gaussian_conditional).__name__}..."
        )
        self.entropy_bottleneck.update(force=force)
        gc = self.gaussian_conditional

        # Check if scale_table is populated (from checkpoint) or needs initialization
        if gc.scale_table.numel() == 0:
            # Default Log-Scale table from CompressAI see https://interdigitalinc.github.io/CompressAI/models.html
            gc.update_scale_table(get_scale_table(), force=force)
        else:
            gc.update_scale_table(gc.scale_table, force=force)

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
        z: Tensor = self.h_a(torch.abs(y))  # [N, 2*N, H'', W''], e.g., [12, 256, 2, 2]
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat: Tensor = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales: Tensor = self.h_s(z_hat)

        # Apply entropy coding
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)

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
