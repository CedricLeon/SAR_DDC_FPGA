import math

import torch
from compressai.registry import register_criterion
from torch import nn
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)


@register_criterion("UnitaryRDLoss")
class UnitaryRDLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter.
    Exactly the same logic than the original compressai.RateDistortionLoss, but
    the distortion is not 255**2 * mse, but simply mse (input data is already normalized to [0, 1]).
    @TODO could have a parameter for the max value of the input data, e.g., 1 or 255."""

    def __init__(self, lmbda=0.01, metric="mse"):
        super().__init__()
        if metric not in ["mse", "ssim", "ms_ssim"]:
            raise NotImplementedError(f"{metric} is not supported!")

        self.lmbda = lmbda

        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse"] = self.mse(output["x_hat"], target)
        out["ssim"] = self.ssim(output["x_hat"], target)
        out["ms_ssim_loss"] = self.ms_ssim(output["x_hat"], target)

        if self.metric == "mse":
            distortion = out["mse"]
        elif self.metric == "ssim":
            distortion = 1 - out["ssim"]
        else:
            distortion = 1 - out["ms_ssim_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        return out


# @TODO: Is there a way to use torch.log10() and other math operation? And If yes, would it be faster?
def calculate_psnr_1(mse: float) -> float:
    """Calculate PSNR from MSE loss, assuming the max of the image is 1."""
    return -10 * math.log10(mse)


def calculate_psnr_max(mse: float, max_value: float) -> float:
    """Calculate PSNR from MSE loss."""
    return 10 * math.log10((max_value**2) / mse)
    # equivalent to: return 20 * math.log10(max_value) - 10 * math.log10(mse)
