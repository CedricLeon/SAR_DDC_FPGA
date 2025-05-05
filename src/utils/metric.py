import math
import warnings

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def calculate_psnr(mse_loss: float) -> float:
    """Calculate PSNR from MSE loss."""
    warnings.warn(
        "calculate_psnr simply computes `-10 * math.log10(mse_loss)`. See if it should consider the max_value too.",
        DeprecationWarning,
        stacklevel=2,
    )
    return -10 * math.log10(mse_loss)


def calculate_bpp(self, likelihoods, input_shape):
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


def calculate_ssim(self, x, x_hat):
    """Calculate Structural Similarity Index Measure.

    Args:
        x: Original image
        x_hat: Reconstructed image

    Returns:
        SSIM value as a tensor
    """
    return structural_similarity_index_measure(x_hat, x)
