"""Simplified SAR Autoencoder (no entropy model, no hyperprior).

This model mirrors the encoder/decoder topology of `ResidualScaleHyperprior` but removes:
  - EntropyBottleneck / GaussianConditional modules
  - Hyperprior analysis/synthesis networks

It is intended as an FPGA-friendly stepping stone: operations are limited to conv, transposed
conv, GDN, residual blocks, simple pointwise arithmetic, and exponentials if later needed.

Forward interface tries to stay compatible with existing training code and loss functions by
returning a dict containing:
  - `x_hat`: reconstruction tensor
  - `y`: latent representation (before any entropy modeling)
  - `likelihoods`: a dict with a single key `y` whose tensor approximates per-element likelihoods
                   under a channel-wise Gaussian assumption. This enables reuse of
                   `estimate_bpp` from `src/utils/metrics.py` as a proxy for rate.

NOTE: The produced likelihoods are NOT true entropy-model likelihoods; they approximate the pdf of
      y assuming y_c ~ N(0, sigma_c^2) with sigma_c estimated from the current batch. They are meant
      only for *relative* rate comparisons (RD curves) during early FPGA prototyping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.layers import ResidualBlock, conv, deconv, make_activation


class ResidualSimpleAE(nn.Module):
    """Residual Autoencoder for SAR despeckling without latent compression.

    Training mode expects a single-channel (real OR imaginary) squared & normalized input.
    Inference mode expects a 2-channel tensor (real, imag). Both channels are processed
    independently through the encoder/decoder and re-concatenated.
    """

    def __init__(self, nb_channels_main: int = 128, activation: str = "gdn"):
        super().__init__()
        N = nb_channels_main
        self.activation = activation

        # Analysis transform (encoder)
        self.g_a = nn.Sequential(
            nn.Sequential(
                conv(1, N, kernel_size=5, stride=2),
                make_activation(activation, N, inverse=False),
                ResidualBlock(N, activation),
            ),
            nn.Sequential(
                conv(N, N, kernel_size=5, stride=2),
                make_activation(activation, N, inverse=False),
                ResidualBlock(N, activation),
            ),
            nn.Sequential(
                conv(N, N, kernel_size=5, stride=2),
                make_activation(activation, N, inverse=False),
                ResidualBlock(N, activation),
            ),
            conv(N, N, kernel_size=5, stride=2),
        )

        # Synthesis transform (decoder)
        self.g_s = nn.Sequential(
            nn.Sequential(
                deconv(N, N, kernel_size=5, stride=2),
                make_activation(activation, N, inverse=True),
                ResidualBlock(N, activation),
            ),
            nn.Sequential(
                deconv(N, N, kernel_size=5, stride=2),
                make_activation(activation, N, inverse=True),
                ResidualBlock(N, activation),
            ),
            nn.Sequential(
                deconv(N, N, kernel_size=5, stride=2),
                make_activation(activation, N, inverse=True),
                ResidualBlock(N, activation),
            ),
            deconv(N, 1, kernel_size=5, stride=2),
        )

    @staticmethod
    def _standardized_cumulative(inputs: Tensor) -> Tensor:
        """Standard normal CDF using erfc for numerical stability (like CompressAI)."""
        half = float(0.5)
        const = float(-(2**-0.5))
        return half * torch.erfc(const * inputs)

    @classmethod
    def _discrete_gaussian_likelihoods(cls, y: Tensor) -> Tensor:
        """Per-element probability mass of a unit-width quantization bin around y.

        We estimate per-channel mean/variance over batch+spatial dims:
            m_c = E[y_c],  s_c^2 = Var[y_c]
        Then approximate the probability mass that a unit-width quantizer would assign:
            P = Phi((0.5 - (y - m_c)) / s_c) - Phi((-0.5 - (y - m_c)) / s_c)

        This mirrors CompressAI's GaussianConditional likelihood computation and guarantees
        0 < P <= 1, so -log2(P) >= 0 and the resulting bpp proxy is non-negative.
        """
        eps = 1e-9
        # Per-channel mean and std
        mean = y.mean(dim=(0, 2, 3), keepdim=True)
        var = y.var(unbiased=False, dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
        std = torch.sqrt(var)

        # Lower bound on scale to avoid degenerate zero-variance
        s_min = 1e-2
        scales = torch.clamp(std, min=s_min)

        values = y - mean
        upper = cls._standardized_cumulative((0.5 - values) / scales)
        lower = cls._standardized_cumulative((-0.5 - values) / scales)
        likelihood = (upper - lower).clamp_min(eps)
        return likelihood

    def forward(self, x: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        if self.training:
            assert x.shape[1] == 1, "Training: Input tensor must have 1 channel"
            y = self.g_a(x)
            x_hat = self.g_s(y)
            likelihoods = {"y": self._discrete_gaussian_likelihoods(y)}
        else:
            assert x.shape[1] == 2, "Inference: Input tensor must have 2 channels (real, imag)"
            x_real = x[:, :1, :, :]
            x_imag = x[:, 1:, :, :]
            y_real = self.g_a(x_real)
            y_imag = self.g_a(x_imag)
            x_hat_real = self.g_s(y_real)
            x_hat_imag = self.g_s(y_imag)
            # Concatenate reconstructions (matches original hyperprior forward inference behavior)
            x_hat = torch.cat((x_hat_real, x_hat_imag), dim=1)
            # Merge latents for likelihood estimation (stack then treat as separate channels)
            y = torch.cat((y_real, y_imag), dim=1)
            likelihoods = {"y": self._discrete_gaussian_likelihoods(y)}

        return {"x_hat": x_hat, "y": y, "likelihoods": likelihoods}

    def aux_loss(self) -> Tensor:
        """Compatibility stub for LightningModule expecting an auxiliary loss.

        Returns zero (no entropy bottleneck present). Kept to avoid changing training loop logic.
        """
        return torch.zeros((), device=next(self.parameters()).device)
