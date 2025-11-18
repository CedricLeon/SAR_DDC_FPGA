import torch.nn as nn
from compressai.layers import GDN
from torch import Tensor


def conv(in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1) -> nn.Conv2d:
    """Helper conv layer."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(
    in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1
) -> nn.ConvTranspose2d:
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
    """Residual block with skip connections and custom activation."""

    def __init__(self, channels: int, act_type: str = "gdn", kernel_size: int = 5):
        super().__init__()
        self.conv1 = conv(channels, channels, kernel_size=kernel_size)
        self.act = make_activation(act_type, channels, inverse=False)
        self.conv2 = conv(channels, channels, kernel_size=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the residual block."""
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


def make_activation(act_type: str, channels: int, inverse: bool = False) -> nn.Module:
    t = (act_type or "gdn").lower()
    if t == "gdn":
        return GDN(channels, inverse=inverse)
    if t == "relu":
        return nn.ReLU(inplace=True)
    if t in ("lrelu", "leaky_relu"):
        return nn.LeakyReLU(0.1, inplace=True)
    if t == "silu":
        return nn.SiLU(inplace=True)
    if t == "gelu":
        return nn.GELU()
    if t in ("identity", "none"):
        return nn.Identity()
    if t in ("gn_relu", "groupnorm_relu"):
        return nn.Sequential(nn.GroupNorm(8, channels), nn.ReLU(inplace=True))
    raise ValueError(f"Unknown activation type: {act_type}")
