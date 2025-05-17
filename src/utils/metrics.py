import math
import warnings

import torch
from compressai.registry import register_criterion
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure


@register_criterion("UnitaryRDLoss")
class UnitaryRDLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter.
    Exactly the same logic than the original compressai.RateDistortionLoss, but
    the distortion is not 255**2 * mse_loss, but simply mse_loss (input data is already normalized to [0, 1]).
    @TODO could have a parameter for the max value of the input data, e.g., 1 or 255."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            raise NotImplementedError("ms-ssim is not implemented!")
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # if self.metric == ms_ssim:
        #     out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
        #     distortion = 1 - out["ms_ssim_loss"]
        # else:
        out["mse_loss"] = self.metric(output["x_hat"], target)
        # distortion = out["mse_loss"]  # * 255**2

        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]


# @TODO: Is there a way to use torch.log10() and other math operation? And If yes, would it be faster?
def calculate_psnr_1(mse_loss: float) -> float:
    """Calculate PSNR from MSE loss, assuming the max of the image is 1."""
    return -10 * math.log10(mse_loss)


def calculate_psnr_max(mse_loss: float, max_value: float) -> float:
    """Calculate PSNR from MSE loss."""
    return 10 * math.log10((max_value**2) / mse_loss)
    # equivalent to: return 20 * math.log10(max_value) - 10 * math.log10(mse_loss)


def calculate_bpp(likelihoods, input_shape):
    """Calculate bits per pixel.

    Args:
        likelihoods: Dictionary of likelihoods from model output
        input_shape: Shape of the input tensor

    Returns:
        Bits per pixel value as a tensor
    """
    warnings.warn(
        "calculate_bpp has not been verified and might do some weird things.",
        DeprecationWarning,
        stacklevel=2,
    )
    num_pixels = input_shape[0] * input_shape[2] * input_shape[3]
    bpp = 0

    for likelihood in likelihoods.values():
        bpp += torch.sum(torch.log2(likelihood)) / (-num_pixels)

    return bpp


def calculate_ssim(x, x_hat):
    """Calculate Structural Similarity Index Measure.

    Args:
        x: Original image
        x_hat: Reconstructed image

    Returns:
        SSIM value as a tensor
    """
    return structural_similarity_index_measure(x_hat, x)
