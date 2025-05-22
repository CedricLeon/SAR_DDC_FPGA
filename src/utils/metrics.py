import math
from typing import Dict

import torch
from compressai.registry import register_criterion
from torch import Tensor, nn
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


@register_criterion("UnitaryRDLoss")
class UnitaryRDLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter.
    Exactly the same logic than the original compressai.RateDistortionLoss, but
    the distortion is not 255**2 * mse, but simply mse (input data is already normalized to [0, 1]).
    @TODO could have a parameter for the max value of the input data, e.g., 1 or 255."""

    def __init__(self, lmbda: float, metric: str = "mse"):
        super().__init__()
        if metric not in ["merlin", "mse", "ssim", "ms_ssim"]:
            raise NotImplementedError(f"{metric} is not supported!")
        self.metric = metric

        self.lmbda = lmbda

        self.mse = nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))

    def forward(self, output: Dict[str, Tensor], target: Tensor) -> Dict[str, Tensor]:
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # Average of the estimated number of bits needed to encode each pixel
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse"] = self.mse(output["x_hat"], target)
        out["psnr"] = self.psnr(output["x_hat"], target)
        out["ssim"] = self.ssim(output["x_hat"], target)
        out["ms_ssim"] = self.ms_ssim(output["x_hat"], target)

        if self.metric == "merlin":
            # sum over pixel k  0.5*output[k] + exp(input[k] − output[k])
            out["distortion"] = torch.mean(
                0.5 * output["x_hat"] + torch.exp(target - output["x_hat"])
            )
        elif self.metric == "mse":
            out["distortion"] = out["mse"]
        elif self.metric == "ssim":
            out["distortion"] = 1 - out["ssim"]
        elif self.metric == "ms-ssim":
            out["distortion"] = 1 - out["ms_ssim"]
        else:
            raise NotImplementedError(f"{self.metric} is not supported!")

        out["loss"] = self.lmbda * out["distortion"] + out["bpp_loss"]
        return out


def calculate_psnr_1(mse: Tensor) -> Tensor:
    """Calculate PSNR from MSE loss, assuming the max of the image is 1."""
    return -10 * torch.log10(mse)


def calculate_psnr_max(mse: Tensor, max_value: float) -> Tensor:
    """Calculate PSNR from MSE loss."""
    # equivalent to: 10 * torch.log10((max_value**2) / mse)
    return 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(mse)
