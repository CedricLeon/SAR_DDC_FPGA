import math
from typing import Dict, Literal

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


def estimate_bpp(pred: Dict[str, Tensor]) -> Tensor | Literal[0]:
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

        self.lmbda = lmbda

        self.mse = MeanSquaredError()  # nn.MSELoss(reduction="sum")
        self.psnr = PeakSignalNoiseRatio(data_range=(2 * amp_min, 2 * amp_max))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(2 * amp_min, 2 * amp_max))
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=(2 * amp_min, 2 * amp_max)
        )

    def forward(self, output: Dict[str, Tensor], target: Tensor) -> Dict[str, Tensor]:
        out = {}
        # Rate term (estimated bpp)
        out["bpp"] = estimate_bpp(output)

        # Denorm the reconstructions and target before computing losses
        r_denorm = output["x_hat"] * (2 * amp_max - 2 * amp_min) + 2 * amp_min
        b_denorm = target * (2 * amp_max - 2 * amp_min) + 2 * amp_min

        out["mse"] = self.mse(r_denorm, b_denorm)
        out["psnr"] = self.psnr(r_denorm, b_denorm)
        out["ssim"] = self.ssim(r_denorm, b_denorm)
        out["ms_ssim"] = self.ms_ssim(r_denorm, b_denorm)

        # ----- MERLIN Loss -----
        # # Classic:      (0.5 * log(r) + b^2 / r)
        # r_denorm = torch.exp(r_denorm)
        # b_denorm = torch.exp(b_denorm)
        # merlin_loss = 0.5 * torch.log(r_denorm + 1e-2) + torch.square(b_denorm) / (
        #     r_denorm + 1e-6
        # )
        # In Log-Scale: (0.5 * r + exp(2*b - r))
        merlin_loss = 0.5 * r_denorm + torch.exp(2 * b_denorm - r_denorm)
        out["merlin"] = torch.mean(merlin_loss)

        if self.metric == "merlin" or self.metric == "mse":
            out["distortion"] = out[self.metric]
        elif self.metric == "ssim" or self.metric == "ms_ssim":
            out["distortion"] = 1 - out[self.metric]

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
        self.psnr = PeakSignalNoiseRatio(data_range=(amp_min, amp_max))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(amp_min, amp_max))
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=(amp_min, amp_max))

        self.count_calls: int | None = None  # 0 to enable printing
        self.print_every_n_call = 50

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MERLIN loss.

        Args:
            predicted (torch.Tensor): Predicted reflectivity (r) in log-scale
            target (torch.Tensor): Observed SAR image (b) in log-scale

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss and metrics
        """
        out = {}

        loss_denorm = 0.5 * predicted + torch.exp(2 * target - predicted)
        out["loss_denorm"] = torch.mean(loss_denorm)

        # Denorm the reconstructions and target before computing losses
        predicted_denorm = predicted * (2 * amp_max - 2 * amp_min) + 2 * amp_min
        target_denorm = target * (2 * amp_max - 2 * amp_min) + 2 * amp_min

        # MERLIN loss in log-scale: Sum over pixels 0.5 * r + exp(2*b - r)
        # Use mean instead of sum for numerical stability. therefore we scale lr by HxW
        term1 = 0.5 * predicted_denorm
        term2 = torch.exp(2 * target_denorm - predicted_denorm)

        # # TO CONSIDER: Clamp the exponential term to prevent overflow
        # term2 = torch.clamp(term2, max=1e6)
        if torch.isnan(term2).any():
            print(
                f"NaN detected in term2: pred_range=[{predicted_denorm.min()}, {predicted_denorm.max()}], target_range=[{target_denorm.min()}, {target_denorm.max()}]"
            )

        out["loss_term1"] = torch.mean(term1)
        out["loss_term2"] = torch.mean(term2)
        out["loss"] = torch.mean(term1 + term2)

        out["mse"] = self.mse(predicted_denorm, target_denorm)
        out["psnr"] = self.psnr(predicted_denorm, target_denorm)
        out["ssim"] = self.ssim(predicted_denorm, target_denorm)
        out["ms_ssim"] = self.ms_ssim(predicted_denorm, target_denorm)

        # Print predicted statistics every nth calls
        if self.count_calls is not None:
            if self.count_calls % self.print_every_n_call == 0:
                print(
                    f"[MerlinLoss: {out['loss']:.3f}] Call {self.count_calls}: Predicted min {predicted.min().item():.4f}, max {predicted.max().item():.4f}, mean {predicted.mean().item():.4f}, std {predicted.std().item():.4f}, isNaN {torch.isnan(predicted).any().item()}."
                    f" Target min {target.min().item():.4f}, max {target.max().item():.4f}, mean {target.mean().item():.4f}, std {target.std().item():.4f}, isNaN {torch.isnan(target).any().item()}."
                )
            self.count_calls += 1

        return out
