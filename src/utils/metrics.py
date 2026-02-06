import math
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from compressai.registry import register_criterion
from torch import Tensor, nn
from torchmetrics import MeanSquaredError
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure,
)
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from src.utils.constants import AMP_LIN_MAX, AMP_MAX, AMP_MIN, EPS
from src.utils.debug import print_statistics


def get_all_distortion_metrics(
    predicted: Union[Tensor, np.ndarray],
    target: Union[Tensor, np.ndarray],
) -> Dict[str, float]:
    """Compute all distortion metrics between predicted and target tensors.

    Args:
        predicted (Union[Tensor, np.ndarray]): Predicted tensor
        target (Union[Tensor, np.ndarray]): Target tensor
    Returns:
        Dict[str, float]: Dictionary containing MSE, PSNR, SSIM, and MS-SSIM values
    """
    if isinstance(predicted, np.ndarray):
        predicted = torch.from_numpy(predicted)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    mse_value = mse(predicted, target)
    psnr_value = psnr(predicted, target, mse_value)

    # Ensure tensors have shape [N, C, H, W]
    if predicted.ndim == 2:
        predicted = predicted.unsqueeze(0).unsqueeze(0)
    elif predicted.ndim == 3:
        predicted = predicted.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(0)
    ssim_value = ssim(predicted, target)
    ms_ssim_value = ms_ssim(predicted, target)
    return {
        "mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "ms_ssim": ms_ssim_value,
    }


def mse(predicted: Tensor, target: Tensor) -> float:
    """Compute Mean Squared Error (MSE) loss between predicted and target tensors."""
    return torch.mean((predicted - target) ** 2).item()


def psnr(predicted_linA: Tensor, target_linA: Tensor, mse_value: Optional[float] = None) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between predicted_linA and target_linA tensors.

    Both tensors must be in linear Amplitude scale as peak=AMP_LIN_MAX is used for PSNR
    computation.
    """
    mse_value = mse_value if mse_value is not None else mse(predicted_linA, target_linA)
    peak = AMP_LIN_MAX
    psnr_value = 20 * math.log10(peak) - 10 * math.log10(mse_value)
    return psnr_value


def ssim(predicted: Tensor, target: Tensor, data_range: Optional[float] = None) -> float:
    """Compute Structural Similarity Index Measure (SSIM)."""
    if data_range is None:
        data_range = float(torch.max(predicted))
    return structural_similarity_index_measure(predicted, target, data_range=data_range).item()


def ms_ssim(predicted: Tensor, target: Tensor, data_range: Optional[float] = None) -> float:
    """Compute Multi-Scale Structural Similarity Index Measure (MS-SSIM)."""
    if data_range is None:
        data_range = float(torch.max(predicted))
    return multiscale_structural_similarity_index_measure(
        predicted, target, data_range=data_range
    ).item()


def estimate_bpp(
    pred: Dict[str, Tensor],
) -> Union[Tensor, Literal[0]]:
    """Compute BPP based on the estimated likelihoods (Average of the estimated number of bits
    needed to encode each pixel)"""
    N, _, H, W = pred["x_hat"].size()
    num_pixels = N * H * W
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in pred["likelihoods"].values()
    )
    return bpp


@register_criterion("MerlinRDLoss")
class MerlinRDLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter.

    The distortion is replaced by the loss introduced by MERLIN to reconstruct the reflectivity of
    the input, i.e., despeckle the SAR image.
    """

    def __init__(self, lmbda: float, metric: str = "mse"):
        super().__init__()
        if metric not in ["merlin", "mse", "ssim", "ms_ssim"]:
            raise NotImplementedError(f"{metric} is not supported!")
        self.metric = metric

        self.lmbda = lmbda if lmbda >= 0 else None  # deactivate rate if lmbda < 0

        self.mse = MeanSquaredError()  # nn.MSELoss(reduction="sum")
        self.psnr = PeakSignalNoiseRatio(data_range=(2 * AMP_MIN, 2 * AMP_MAX))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(2 * AMP_MIN, 2 * AMP_MAX))
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=(2 * AMP_MIN, 2 * AMP_MAX)
        )

    def forward(self, output: Dict[str, Tensor], target: Tensor) -> Dict[str, Tensor]:
        """Compute rate-distortion loss.

        Args:
            output (Dict[str, Tensor]): Model output containing 'x_hat' and 'likelihoods'
            target (Tensor): Target tensor, in linear scale
        Returns:
            Dict[str, Tensor]: Dictionary containing loss, bpp, and distortion metrics
        """
        out = {}
        # Rate term (estimated bpp)
        out["bpp"] = estimate_bpp(output)

        # Denorm the reconstructions before computing losses
        log_hat_R = 2 * (output["x_hat"] * (AMP_MAX - AMP_MIN) + AMP_MIN)
        # print_statistics("      Predicted Reflectivity log_hat_R", log_hat_R)
        # ----- Classic MERLIN Loss (0.5 * log(r) + b^2 / r) -----
        hat_R = torch.exp(log_hat_R) + 1e-6  # must be non-zero
        # print_statistics("      Predicted Reflectivity hat_R", hat_R)
        b_square = torch.square(target)
        # print_statistics("      b_square", b_square)
        merlin_loss = 0.5 * log_hat_R + b_square / hat_R
        # ----- In Log-Scale MERLIN Loss (0.5 * log_r + exp(2*log_b - log_r)) -----
        # log_b = torch.log(torch.square(target) + EPS)
        # print_statistics("      Target (square + log )", log_b)
        # merlin_loss = 0.5 * log_hat_R + torch.exp(2 * log_b - log_hat_R)

        out["merlin"] = torch.mean(merlin_loss)

        # These metrics are just used for monitoring purposes, I compute them in log-scale
        clean = log_hat_R
        noisy = torch.log(b_square + EPS)

        out["mse"] = self.mse(clean, noisy)
        out["psnr"] = self.psnr(clean, noisy)
        out["ssim"] = self.ssim(clean, noisy)
        out["ms_ssim"] = self.ms_ssim(clean, noisy)

        # print(
        #     f"[DEBUG]: {out['merlin']=}, {out['mse']=}, {out['psnr']=}, {out['ssim']=}, {out['ms_ssim']=}"
        # )

        if self.metric == "merlin" or self.metric == "mse":
            out["distortion"] = out[self.metric]
        elif self.metric == "ssim" or self.metric == "ms_ssim":
            out["distortion"] = 1 - out[self.metric]

        if self.lmbda is None:
            out["loss"] = out["distortion"]
        else:
            out["loss"] = self.lmbda * out["distortion"] + out["bpp"]
        return out


class MerlinLoss(nn.Module):
    """MERLIN loss function for SAR image despeckling.

    Implements the loss function from:
        Dalsasso, E., Denis, L., & Tupin, F. (2022). As if by magic: Self-supervised training of deep despeckling networks with MERLIN. IEEE Transactions on Geoscience and Remote Sensing.

    The loss is computed in log-scale on DENORMALIZED images as: 0.5 * r + exp(2*b - r) where r is the predicted reflectivity and b is the observed (noisy) SAR image.
    """

    def __init__(self):
        super().__init__()

        self.mse = MeanSquaredError()
        self.psnr = PeakSignalNoiseRatio(data_range=(2 * AMP_MIN, 2 * AMP_MAX))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(2 * AMP_MIN, 2 * AMP_MAX))
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=(2 * AMP_MIN, 2 * AMP_MAX)
        )

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MERLIN loss.

        Args:
            predicted (torch.Tensor): Predicted reflectivity (r) in log-scale
            target (torch.Tensor): Observed SAR image (b) in log-scale

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss and metrics
        """
        out = {}

        # Denorm the reconstructions before computing losses
        log_hat_R = 2 * (predicted * (AMP_MAX - AMP_MIN) + AMP_MIN)
        # ----- Classic MERLIN Loss (0.5 * log(r) + b^2 / r) -----
        hat_R = torch.exp(log_hat_R) + 1e-6  # must be non-zero
        b_square = torch.square(target)
        merlin_loss = 0.5 * log_hat_R + b_square / hat_R
        # ----- In Log-Scale MERLIN Loss (0.5 * log_r + exp(2*log_b - log_r)) -----
        # log_b = torch.log(torch.square(target) + EPS)
        # print_statistics("      Target (square + log )", log_b)
        # merlin_loss = 0.5 * log_hat_R + torch.exp(2 * log_b - log_hat_R)

        clean = log_hat_R
        noisy = torch.log(b_square + EPS)

        out["loss"] = torch.mean(merlin_loss)

        out["mse"] = self.mse(clean, noisy)
        out["psnr"] = self.psnr(clean, noisy)
        out["ssim"] = self.ssim(clean, noisy)
        out["ms_ssim"] = self.ms_ssim(clean, noisy)

        return out
