import torch.nn as nn
from compressai.layers import GDN, GDN1
from torch import Tensor

from src.models.components.compressai_dpu import GDN1Patched, GDNPatched


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

    def __init__(
        self,
        channels: int,
        act_name: str = "gdn",
        kernel_size: int = 5,
        use_patched_gdn: bool = False,
    ):
        super().__init__()
        self.conv1 = conv(channels, channels, kernel_size=kernel_size)
        self.act = make_activation(
            act_name, channels, inverse=False, use_patched_gdn=use_patched_gdn
        )
        self.conv2 = conv(channels, channels, kernel_size=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the residual block."""
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


def make_activation(
    act_name: str, channels: int, inverse: bool = False, use_patched_gdn: bool = False
) -> nn.Module:
    t = act_name.lower()
    if t == "gdn":
        if use_patched_gdn:
            return GDNPatched(channels, inverse=inverse)
        else:
            return GDN(channels, inverse=inverse)
    if t == "gdn1":
        if use_patched_gdn:
            return GDN1Patched(channels, inverse=inverse)
        else:
            return GDN1(channels, inverse=inverse)
    if t == "relu":
        return nn.ReLU(inplace=True)
    if t in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unknown activation type: {act_name}")
