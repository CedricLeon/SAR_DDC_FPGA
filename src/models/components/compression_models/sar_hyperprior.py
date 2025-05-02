"""
SAR Hyperprior Model for compression and despeckling.

This module implements a scale hyperprior architecture for SAR image compression
and despeckling based on CompressAI framework.
"""

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel


# Helper functions for convolution and transposed convolution layers
def conv(in_channels, out_channels, kernel_size=5, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=1):
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
        residual = x
        out = self.gdn(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class SARHyperprior(CompressionModel):
    """Scale Hyperprior model for SAR image despeckling and compression.

    This model implements the architecture described in the specifications with:
    - Residual blocks integrated after GDN layers in the analysis transform
    - 128 channels for main transform, 256 for hyperprior
    - Architecture designed for squared real/imaginary components
    """

    def __init__(self, N=128, M=256):
        """Initialize the SAR hyperprior model.

        Args:
            N: Number of channels for main transform (default: 128)
            M: Number of channels for hyperprior (default: 256)
        """
        super().__init__()

        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)

        # Analysis transform (encoder g_a) with integrated residual blocks
        self.g_a_conv1 = conv(1, N, kernel_size=5, stride=2)
        self.g_a_gdn1 = GDN(N)
        self.g_a_res1 = ResidualBlock(N)

        self.g_a_conv2 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_gdn2 = GDN(N)
        self.g_a_res2 = ResidualBlock(N)

        self.g_a_conv3 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_gdn3 = GDN(N)
        self.g_a_res3 = ResidualBlock(N)

        self.g_a_conv4 = conv(N, N, kernel_size=5, stride=2)
        # No GDN after final layer before bottleneck

        # Synthesis transform (decoder g_s) with integrated residual blocks
        self.g_s_deconv1 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_res1 = ResidualBlock(N)

        self.g_s_deconv2 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.g_s_res2 = ResidualBlock(N)

        self.g_s_deconv3 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_res3 = ResidualBlock(N)

        self.g_s_deconv4 = deconv(N, 1, kernel_size=5, stride=2)

        # Hyperprior analysis transform (h_a)
        self.h_a = nn.Sequential(
            # nn.Identity(),  # abs operation happens in forward
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

    def g_a(self, x):
        """Analysis transform (encoder) with integrated residual blocks."""
        x = self.g_a_gdn1(self.g_a_conv1(x))
        x = self.g_a_res1(x)

        x = self.g_a_gdn2(self.g_a_conv2(x))
        x = self.g_a_res2(x)

        x = self.g_a_gdn3(self.g_a_conv3(x))
        x = self.g_a_res3(x)

        return self.g_a_conv4(x)

    def g_s(self, x):
        """Synthesis transform (decoder) with integrated residual blocks."""
        x = self.g_s_gdn1(self.g_s_deconv1(x))
        x = self.g_s_res1(x)

        x = self.g_s_gdn2(self.g_s_deconv2(x))
        x = self.g_s_res2(x)

        x = self.g_s_gdn3(self.g_s_deconv3(x))
        x = self.g_s_res3(x)

        return self.g_s_deconv4(x)

    def forward(self, x, training=True):
        """Forward pass through the model.

        Args:
            x: Input tensor (squared real or imaginary part)
            training: Whether the model is in training mode

        Returns:
            Dictionary with model outputs
        """
        # x.shape = [batch_size, 1, height, width]
        # Apply analysis transform to get latent representation
        y = self.g_a(x)
        # concatenate y with itself along channel dimension
        y = torch.cat((y, y), dim=1)  # 0 or last dimensions have to check y dimensions

        # Apply hyperprior to get scales
        z = torch.abs(y)
        z = self.h_a(z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales = self.h_s(z_hat)

        # Apply entropy coding
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales)

        # Split: discard second half of y_hat
        y_hat = y_hat[:, : y_hat.shape[1] // 2, :, :]

        # Apply synthesis transform to reconstruct
        x_hat = self.g_s(y_hat)

        # Return different outputs based on training vs. testing mode
        if training:
            # During training, return reconstructed image and likelihoods
            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }
        else:
            # During testing, also return latent representation for coding/joining
            return {
                "x_hat": x_hat,
                "y_hat": y_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

    def encode(self, x):
        """Encode input to latent representation."""
        # Analysis transform
        y = self.g_a(x)

        # Hyperprior
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Get scales from hyperprior
        scales = self.h_s(z_hat)

        # Compress y with obtained scales
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "y": y}

    def decode(self, y_hat):
        """Decode latent representation to image space."""
        return self.g_s(y_hat)

    def load_state_dict(self, state_dict, strict=True):
        # Custom load to handle compatibility with pre-trained models
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if param.shape == own_state[name].shape:
                    own_state[name].copy_(param)

        super().load_state_dict(own_state, strict=False)
