import math
from typing import Dict

import torch
from compressai.registry import register_criterion
from torch import Tensor, nn
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from src.utils.constants import amp_max, amp_min


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

        self.mse = MeanSquaredError()  # nn.MSELoss(reduction="sum")
        self.psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))

    def forward(self, output: Dict[str, Tensor], target: Tensor) -> Dict[str, Tensor]:
        out = {}
        # Compute BPP based on the estimated likelihoods (Average of the estimated number of bits needed to encode each pixel)
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out["bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        # Denorm the reconstructions and target before computing losses
        x_hat = output["x_hat"] * (2 * amp_max - 2 * amp_min) + 2 * amp_min
        target = target * (2 * amp_max - 2 * amp_min) + 2 * amp_min
        if torch.isnan(x_hat).any() or torch.isnan(target).any():
            raise ValueError("NaNs found in denormalized tensors.")

        out["mse"] = self.mse(x_hat, target)
        out["psnr"] = self.psnr(x_hat, target)
        out["ssim"] = self.ssim(x_hat, target)
        out["ms_ssim"] = self.ms_ssim(x_hat, target)

        # ----- MERLIN Loss -----
        # Classic:      (0.5 * log(output) + input^2 / output)
        # merlin = 0.5 * torch.log(x_hat + 1e-6) + torch.square(target) / (
        #     x_hat + 1e-6
        # )
        # In Log-Scale: (0.5 * output + exp(2*target - output))
        merlin = 0.5 * x_hat + torch.exp(2 * target - x_hat)
        out["merlin"] = (
            merlin.mean()
        )  # mean over pixels and batch (the loss should not depend on the patch_size)

        if self.metric == "merlin" or self.metric == "mse":
            out["distortion"] = out[self.metric]
        elif self.metric == "ssim" or self.metric == "ms_ssim":
            out["distortion"] = 1 - out[self.metric]

        out["loss"] = self.lmbda * out["distortion"] + out["bpp"]
        return out


def calculate_psnr_1(mse: Tensor) -> Tensor:
    """Calculate PSNR from MSE loss, assuming the max of the image is 1."""
    return -10 * torch.log10(mse)


def calculate_psnr_max(mse: Tensor, max_value: float) -> Tensor:
    """Calculate PSNR from MSE loss."""
    # equivalent to: 10 * torch.log10((max_value**2) / mse)
    return 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(mse)
