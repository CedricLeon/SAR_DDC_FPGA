"""Guards for the shared AMP_LIN_99 basis of the distortion metrics.

Every reference-based distortion metric (MSE, PSNR, SSIM, MS-SSIM, EPD) clips to ``AMP_LIN_99``
and SSIM/MS-SSIM score at ``data_range = AMP_LIN_99``. Before this convention, SSIM used
``data_range = max(predicted)``, which made scores depend on the brightest pixel of each
reconstruction and therefore non-comparable across models — inflating float32 (peak ~1e5) against
INT8 (DPU-capped at 2100).
"""

import numpy as np
import pytest
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from src.utils.constants import AMP_LIN_99
from src.utils.metrics import epd, ms_ssim, mse, psnr, ssim


def _speckly_pair(seed: int = 0, size: int = 192):
    """A dark SAR-like scene (Rayleigh amplitude) and a smoothed 'reconstruction' of it.

    ``size >= 161``: torchmetrics MS-SSIM needs H,W > (kernel-1) * (len(betas)-1)**2.
    """
    rng = np.random.default_rng(seed)
    target = rng.rayleigh(scale=40.0, size=(size, size)).astype(np.float32)
    recon = target.copy()
    recon[1:-1, 1:-1] = 0.25 * (
        target[:-2, 1:-1] + target[2:, 1:-1] + target[1:-1, :-2] + target[1:-1, 2:]
    )
    to_t = lambda a: torch.from_numpy(a)[None, None]  # noqa: E731
    return to_t(recon), to_t(target)


def test_ssim_is_blind_to_peaks_above_the_clip():
    """The core property: two recons that differ only above AMP_LIN_99 must score identically.

    This is what makes float32 and INT8 comparable — the INT8 DPU caps the recon at 2100 while
    float32 reaches ~1e5, and neither peak may influence the score.
    """
    recon, target = _speckly_pair()
    dim_peak, bright_peak = recon.clone(), recon.clone()
    dim_peak[..., 10, 10] = 2_100.0  # INT8 board cap
    bright_peak[..., 10, 10] = 85_000.0  # float32 reach

    assert ssim(dim_peak, target) == pytest.approx(ssim(bright_peak, target), abs=1e-6)
    assert ms_ssim(dim_peak, target) == pytest.approx(ms_ssim(bright_peak, target), abs=1e-6)
    assert epd(dim_peak, target) == pytest.approx(epd(bright_peak, target), abs=1e-6)
    assert mse(dim_peak, target) == pytest.approx(mse(bright_peak, target), abs=1e-6)


def test_ssim_uses_the_fixed_amp99_data_range():
    """Known answer: ``ssim`` equals torchmetrics on clipped inputs at data_range=AMP_LIN_99,
    and differs from the old max(predicted) convention on a scene with a bright scatterer."""
    recon, target = _speckly_pair()
    recon[..., 10, 10] = 85_000.0

    clipped_r = torch.clamp(recon, max=AMP_LIN_99)
    clipped_t = torch.clamp(target, max=AMP_LIN_99)
    expected = float(
        structural_similarity_index_measure(clipped_r, clipped_t, data_range=AMP_LIN_99)
    )
    assert ssim(recon, target) == pytest.approx(expected, abs=1e-6)

    old = float(structural_similarity_index_measure(recon, target, data_range=float(recon.max())))
    assert old > ssim(recon, target)  # the old convention saturates toward 1


def test_ssim_scale_invariance_backs_the_cpp_workaround():
    """SSIM(x, y, L) == SSIM(s*x, s*y, s*L).

    OpenCV's ``QualitySSIM`` hard-codes its stabilisers to data_range=255 and takes no
    ``data_range``. ``inference_cpp/src/metrics.cpp::compute_ssim`` therefore clips to AMP_LIN_99
    and rescales both images by 255/AMP_LIN_99 to obtain SSIM at data_range=AMP_LIN_99. That is
    only valid because SSIM is homogeneous — assert it here, since the board path cannot be
    unit-tested on a host without opencv_quality.
    """
    recon, target = _speckly_pair(seed=1)
    recon, target = torch.clamp(recon, max=AMP_LIN_99), torch.clamp(target, max=AMP_LIN_99)
    scale = 255.0 / AMP_LIN_99

    at_amp99 = float(structural_similarity_index_measure(recon, target, data_range=AMP_LIN_99))
    at_255 = float(
        structural_similarity_index_measure(recon * scale, target * scale, data_range=255.0)
    )
    assert at_amp99 == pytest.approx(at_255, abs=1e-5)


def test_identical_inputs_score_perfectly():
    recon, target = _speckly_pair(seed=2)
    assert ssim(target, target) == pytest.approx(1.0, abs=1e-5)
    assert ms_ssim(target, target) == pytest.approx(1.0, abs=1e-5)
    assert epd(target, target) == pytest.approx(1.0, abs=1e-5)
    assert mse(target, target) == pytest.approx(0.0, abs=1e-6)
    del recon


def test_psnr_and_mse_unchanged_by_the_convention_move():
    """MSE/PSNR already clipped; the refactor into _clip_to_amp99 must not move them."""
    recon, target = _speckly_pair(seed=3)
    manual = float(
        torch.mean((torch.clamp(recon, max=AMP_LIN_99) - torch.clamp(target, max=AMP_LIN_99)) ** 2)
    )
    assert mse(recon, target) == pytest.approx(manual, abs=1e-6)
    assert psnr(recon, target) == pytest.approx(
        20 * np.log10(AMP_LIN_99) - 10 * np.log10(manual), abs=1e-6
    )
