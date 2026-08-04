import math
from typing import Dict, Literal, Optional, Tuple, Union

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

from src.utils.constants import AMP_LIN_99, AMP_MAX, AMP_MIN, EPS


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


def _clip_to_amp99(*linA: Tensor) -> Tuple[Tensor, ...]:
    """Clip linear-amplitude tensors to ``AMP_LIN_99`` — the shared basis of every distortion
    metric.

    Every reference-based distortion metric in this module (MSE, PSNR, SSIM, MS-SSIM, EPD) scores on
    the 99th-percentile amplitude, so that no metric is driven by the handful of bright scatterers
    above it. This is "cheating" when comparing against methods that do not clip, but all our
    experiments use it, so they stay mutually comparable.

    Clipping is what makes a **fixed** ``data_range = AMP_LIN_99`` correct for SSIM/MS-SSIM. A
    per-image ``data_range`` (e.g. ``max(predicted)``) is not comparable across models: the SSIM
    stabilisers scale with ``data_range²``, so a recon with a bright pixel is scored on a larger
    ``C1/C2`` and saturates toward 1. That matters most for float32 vs INT8, where the DPU caps the
    recon at 2100 while float32 reaches ~1e5 (see docs/ssim_data_range_issue.md). It is also why EPD
    clips: gradients at bright scatterers otherwise dominate its sums.
    """
    return tuple(torch.clamp(t, max=AMP_LIN_99) for t in linA)


def mse(predicted_linA: Tensor, target_linA: Tensor) -> float:
    """Compute Mean Squared Error (MSE) loss between predicted and target tensors in linear
    Amplitude scale."""
    predicted_linA, target_linA = _clip_to_amp99(predicted_linA, target_linA)
    return torch.mean((predicted_linA - target_linA) ** 2).item()


def psnr(predicted_linA: Tensor, target_linA: Tensor, mse_value: Optional[float] = None) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between predicted_linA and target_linA tensors.

    Both tensors must be in linear Amplitude scale as peak=AMP_LIN_99 is used for PSNR computation.
    """
    # MSE (and therefore PSNR) is computed on 99% of the value
    mse_value = mse_value if mse_value is not None else mse(predicted_linA, target_linA)
    psnr_value = 20 * math.log10(AMP_LIN_99) - 10 * math.log10(mse_value)
    return psnr_value


def ssim(predicted: Tensor, target: Tensor) -> float:
    """Compute Structural Similarity Index Measure (SSIM) on the AMP_LIN_99 basis.

    Both inputs are clipped to ``AMP_LIN_99`` and ``data_range = AMP_LIN_99`` — the same basis as
    ``mse``/``psnr``, see ``_clip_to_amp99`` for why this is the only comparable choice.
    """
    predicted, target = _clip_to_amp99(predicted, target)
    return Tensor(
        structural_similarity_index_measure(predicted, target, data_range=AMP_LIN_99)
    ).item()


def ms_ssim(predicted: Tensor, target: Tensor) -> float:
    """Compute Multi-Scale SSIM on the AMP_LIN_99 basis (see ``ssim``)."""
    predicted, target = _clip_to_amp99(predicted, target)
    return multiscale_structural_similarity_index_measure(
        predicted, target, data_range=AMP_LIN_99
    ).item()


def enl(
    linA: Union[Tensor, np.ndarray],
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """Equivalent Number of Looks computed on linear intensity I = linA².

    ENL = E[I]² / Var[I].  Higher ENL → more speckle reduction.

    Args:
        linA: Reconstruction in linear amplitude.  Any shape — squeezed to 2-D before use.
        roi:  Optional (r0, r1, c0, c1) crop applied before computing.  Use to restrict
              the metric to a homogeneous region (avoids texture bias).
    """
    arr: np.ndarray = linA.detach().cpu().numpy() if isinstance(linA, Tensor) else np.asarray(linA)
    arr = arr.squeeze().astype(np.float32)
    if roi is not None:
        r0, r1, c0, c1 = roi
        arr = arr[r0:r1, c0:c1]
    lin_intensity = np.square(arr)
    mu = float(np.mean(lin_intensity))
    var = float(np.var(lin_intensity))
    return mu**2 / var if var > 0.0 else float("nan")


def ratio_mean(
    recon_linA: Union[Tensor, np.ndarray],
    noisy_linA: Union[Tensor, np.ndarray],
) -> float:
    """Mean of the ratio image R = I_noisy / I_recon (linear intensity).

    R ≈ 1.0 → ideal filter.  R > 1 → under-filtering (residual speckle). R < 1 → over-smoothing.
    """

    def _arr(x: Union[Tensor, np.ndarray]) -> np.ndarray:
        return (
            (x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x))
            .squeeze()
            .astype(np.float32)
        )

    recon_I = np.square(_arr(recon_linA))
    noisy_I = np.square(_arr(noisy_linA))
    return float(np.mean(noisy_I / (recon_I + 1e-10)))


def ratio_enl(
    recon_linA: Union[Tensor, np.ndarray],
    noisy_linA: Union[Tensor, np.ndarray],
) -> float:
    """ENL of the ratio image R = I_noisy / I_recon (linear intensity).

    For a perfect speckle filter R ~ Gamma(L, 1/L), so ENL(R) = L (number of looks). Deviations
    signal over-smoothing (ENL(R) < L) or residual speckle (ENL(R) > L).
    """

    def _arr(x: Union[Tensor, np.ndarray]) -> np.ndarray:
        return (
            (x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x))
            .squeeze()
            .astype(np.float32)
        )

    recon_I = np.square(_arr(recon_linA))
    noisy_I = np.square(_arr(noisy_linA))
    ratio = noisy_I / (recon_I + 1e-10)
    mu = float(np.mean(ratio))
    var = float(np.var(ratio))
    return mu**2 / var if var > 0.0 else float("nan")


def epd(
    recon_linA: Union[Tensor, np.ndarray],
    ref_linA: Union[Tensor, np.ndarray],
) -> float:
    r"""Edge Preservation Degree in linear amplitude.

    EPD = Σ(\|∇recon\| · \|∇ref\|) / Σ(\|∇ref\|²). EPD = 1.0 → perfect edge preservation.  EPD < 1
    → edge attenuation.

    Both inputs are clipped to ``AMP_LIN_99`` (see ``_clip_to_amp99``): unclipped, the sums are
    dominated by the gradients around bright scatterers, so EPD measures how well a backend
    represents point targets rather than how well it preserves edges.

    Uses a central-difference gradient for NumPy-only / Python-3.8 compatibility (consistent with
    the FPGA-side C++ implementation in ``inference_cpp/``).
    """

    def _arr(x: Union[Tensor, np.ndarray]) -> np.ndarray:
        arr = (
            (x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x))
            .squeeze()
            .astype(np.float32)
        )
        return np.minimum(arr, np.float32(AMP_LIN_99))

    def _grad_mag(img: np.ndarray) -> np.ndarray:
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        return np.sqrt(gx**2 + gy**2)

    grad_recon = _grad_mag(_arr(recon_linA))
    grad_ref = _grad_mag(_arr(ref_linA))
    denom = float(np.sum(grad_ref**2))
    return float(np.sum(grad_recon * grad_ref) / denom) if denom > 0.0 else float("nan")


def estimate_likelihoods_bpp(
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


def compute_bitstream_bpp(strings: list, height: int, width: int, batch_size: int) -> float:
    """Compute BPP based on the actual bitstream size.

    Args:
        strings (list): Nested list of byte strings from CompressAI (e.g. [[y_str, ...], [z_str, ...]])
        height (int): Image height
        width (int): Image width
        batch_size (int): Image batch size

    Returns:
        float: Calculated bits per pixel
    """
    nb_pixels = height * width * batch_size
    if nb_pixels == 0:
        return 0.0

    def _sum_bits(obj):
        if isinstance(obj, bytes):
            return len(obj) * 8
        elif isinstance(obj, list):
            return sum(_sum_bits(item) for item in obj)
        return 0

    return _sum_bits(strings) / nb_pixels


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
        out["bpp"] = estimate_likelihoods_bpp(output)

        # Denorm the reconstructions before computing losses
        log_hat_R = 2 * (output["x_hat"] * (AMP_MAX - AMP_MIN) + AMP_MIN)

        # ----- Classic MERLIN Loss (0.5 * log(r) + b^2 / r) -----
        # Early training has some heavy instability during the first step that can lead to the max value of x_hat exploding (for example max(x_hat) = 140 instead of x_hat ∈ [0, 1]).
        # In such a case, applying exp() creates a float overflow (exp(x) = Inf for x > ~88).
        # This further degenerates into NaNs everywhere as the Merlin loss computes Inf/Inf² = NaN.
        # Therefore, we clamp log_hat_R before exp to prevent float32 overflow, clamping to 83 still creates extremely large loss values, but it does not matter as gradient clippings ensure stable training and after a few steps the network stabilizes.
        # A valid x_hat ∈ [0, 1] maps log_hat_R to [9.2, 21.5], so the clamp never activates for a well-trained model.
        hat_R = torch.exp(log_hat_R.clamp(max=85.0)) + 1e-6  # must be non-zero

        b_square = torch.square(target)
        merlin_loss = 0.5 * log_hat_R + b_square / hat_R

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
        self.psnr = PeakSignalNoiseRatio(data_range=AMP_LIN_99)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=AMP_LIN_99)
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=AMP_LIN_99)

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
