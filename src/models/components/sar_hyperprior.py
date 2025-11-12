"""SAR Hyperprior Model for compression and despeckling.

This module implements a scale hyperprior architecture for SAR image compression and despeckling
based on CompressAI framework.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel
from torch import Tensor


# Helper functions for convolution and transposed convolution layers
def conv(in_channels, out_channels, kernel_size=5, stride=1):
    """Helper conv layer."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=1):
    """Helper deconv layer."""
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
    )


class ResidualBlock(nn.Module):
    """Residual block with skip connections."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv(channels, channels, kernel_size=5)
        self.gdn = GDN(channels)
        self.conv2 = conv(channels, channels, kernel_size=5)

    def forward(self, x):
        """Forward pass through the residual block."""
        residual = x
        out = self.gdn(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class ResidualScaleHyperprior(CompressionModel):
    """Residual Scale Hyperprior model for SAR image despeckling and compression. Similar to
    CompressAI's `bmshj2018-hyperprior` the model also incorporates residual blocks within the main
    transforms.

    The model is designed to take as input pre-processed and normalized SAR SLC images parts.
    Specifically, the model expects either the Real or Imaginary part of the SAR SLC image, squared
    and normalized to approximately [0, 1].
    """

    def __init__(self, nb_channels_main=128):
        """Initialize the SAR hyperprior model.

        Args:
            N: Number of channels for main transform (default: 128)
        """
        super().__init__()
        N = nb_channels_main
        M = 2 * N  # Number of channels for hyperprior

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)

        # Main analysis transform (encoder g_a)
        self.g_a = nn.Sequential(
            nn.Sequential(conv(1, N, kernel_size=5, stride=2), GDN(N), ResidualBlock(N)),
            nn.Sequential(conv(N, N, kernel_size=5, stride=2), GDN(N), ResidualBlock(N)),
            nn.Sequential(conv(N, N, kernel_size=5, stride=2), GDN(N), ResidualBlock(N)),
            conv(N, N, kernel_size=5, stride=2),
            # No GDN after final layer before bottleneck
        )

        # Main synthesis transform (decoder g_s)
        self.g_s = nn.Sequential(
            nn.Sequential(
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                ResidualBlock(N),
            ),
            nn.Sequential(
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                ResidualBlock(N),
            ),
            nn.Sequential(
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                ResidualBlock(N),
            ),
            deconv(N, 1, kernel_size=5, stride=2),
        )

        # Hyperprior analysis transform (h_a)
        self.h_a = nn.Sequential(
            conv(M, M, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )

        # Hyperprior synthesis transform (h_s)
        self.h_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, M, kernel_size=3, stride=2),
        )

    def scale_hyperprior(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply hyperprior to get scales and likelihoods."""
        z = torch.abs(y)
        z = self.h_a(z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales = self.h_s(z_hat)

        return scales, z_likelihoods

    def forward(self, x: Tensor) -> Dict[str, Tensor | Dict[str, Tensor]]:
        """Forward pass through the model.

        Args:
            x: Input tensor (squared real or imaginary part) [batch_size, 1, height, width]
            training: Whether the model is in training mode

        Returns:
            Dictionary with model outputs
        """
        if self.training:
            assert x.shape[1] == 1, "Training: Input tensor must have 1 channel"

            # Analysis transform to get latent representation
            y = self.g_a(x)
            # Concatenate y with itself along channel dimension
            y = torch.cat((y, y), dim=1)

            # Apply hyperprior to get scales
            scales, z_likelihoods = self.scale_hyperprior(y)

            # Apply entropy coding
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales)

            # Split: discard second half of y_hat
            y_hat = y_hat[:, : y_hat.shape[1] // 2, :, :]

            # Apply synthesis transform to reconstruct
            x_hat = self.g_s(y_hat)
        else:
            assert x.shape[1] == 2, "Inference: Input tensor must have 2 channels"
            x_real = x[:, :1, :, :]
            x_imag = x[:, 1:, :, :]

            # Analysis transform to get latent representation
            y_real = self.g_a(x_real)
            y_imag = self.g_a(x_imag)
            # Concatenate y_real and y_imag along channel dimension
            y = torch.cat((y_real, y_imag), dim=1)

            # Apply hyperprior to get scales
            scales, z_likelihoods = self.scale_hyperprior(y)

            # Apply entropy coding
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales)

            # Split: discard second half of y_hat
            y_hat_real = y_hat[:, : y_hat.shape[1] // 2, :, :]
            y_hat_imag = y_hat[:, y_hat.shape[1] // 2 :, :, :]

            # Apply synthesis transform to reconstruct
            x_hat_real = self.g_s(y_hat_real)
            x_hat_imag = self.g_s(y_hat_imag)
            x_hat = torch.cat((x_hat_real, x_hat_imag), dim=1)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # @TODO encode() and decode() methods should be used once training is over. Which means that they should deal with the doubles passes through the model of Real and Imaginary part and not concatenate and split the same way as in the forward pass.
    def compress(self, x):
        """Encode input to latent representation."""
        # Analysis transform
        y = self.g_a(x)

        # Hyperprior
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales = self.h_s(z_hat)

        # Compress y with obtained scales
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        """Decode latent representation to image space."""
        assert isinstance(strings, list) and len(strings) == 2
        # , "Invalid input format: strings must be a list containing y and z strings."
        y_strings, z_strings = strings

        # Get the scales from z_strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        scales = self.h_s(z_hat)

        # Decompress y
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, shape)
        y_hat = y_hat[:, : y_hat.shape[1] // 2, :, :]
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}

    def aux_loss(self) -> Tensor:
        """Return the EntropyBottleneck's auxiliary loss for training."""
        return self.entropy_bottleneck.loss()

    # def load_state_dict(self, state_dict, strict=True):
    #     # Custom load to handle compatibility with pre-trained models
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if param.shape == own_state[name].shape:
    #                 own_state[name].copy_(param)

    #     super().load_state_dict(own_state, strict=False)
