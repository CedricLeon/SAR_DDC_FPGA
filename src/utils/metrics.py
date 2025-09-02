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


@register_criterion("MerlinRDLoss")
class MerlinRDLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter.
    The distortion is replaced by the loss introduced by MERLIN
    to reconstruct the reflectivity of the input, i.e., despeckle the SAR image."""

    def __init__(self, lmbda: float, metric: str = "mse"):
        super().__init__()
        if metric not in ["merlin", "mse", "ssim", "ms_ssim"]:
            raise NotImplementedError(f"{metric} is not supported!")
        self.metric = metric

        self.lmbda = lmbda

        self.mse = MeanSquaredError()  # nn.MSELoss(reduction="sum")
        self.psnr = PeakSignalNoiseRatio(data_range=(amp_min, amp_max))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(amp_min, amp_max))
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=(amp_min, amp_max)
        )

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
        merlin_loss = 0.5 * r_denorm + torch.exp(b_denorm - r_denorm)
        out["merlin"] = torch.mean(merlin_loss)

        if self.metric == "merlin" or self.metric == "mse":
            out["distortion"] = out[self.metric]
        elif self.metric == "ssim" or self.metric == "ms_ssim":
            out["distortion"] = 1 - out[self.metric]

        out["loss"] = self.lmbda * out["distortion"] + out["bpp"]
        return out
