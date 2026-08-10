"""Canonical reconstruction: model output (x_hat) -> linear-amplitude image.

Single source of truth shared by training-data generation
(`scripts/dataset/create_dataset.py`, which produced the MERLIN/ADAM ground truths), evaluation
(`src/evaluate.py`) and the symmetrization study. Reflectivity follows the MERLIN convention:
average the real and imaginary intensity predictions (factor 0.5). See docs/FPGA_inference.md §4.
"""

import torch

from src.utils.constants import AMP_MAX, AMP_MIN


def denorm_to_linA(recon_real: torch.Tensor, recon_imag: torch.Tensor) -> torch.Tensor:
    """Normalized-log-amplitude channels -> linear amplitude.

    Args:
        recon_real: model output channel 0 (real), normalized model domain, any shape.
        recon_imag: model output channel 1 (imag), same shape.
    Returns:
        Linear-amplitude reflectivity ``sqrt(0.5 * (real_lin**2 + imag_lin**2))``.
    """
    real_lin = torch.exp(recon_real * (AMP_MAX - AMP_MIN) + AMP_MIN)
    imag_lin = torch.exp(recon_imag * (AMP_MAX - AMP_MIN) + AMP_MIN)
    return torch.sqrt(0.5 * (torch.square(real_lin) + torch.square(imag_lin)))


@torch.no_grad()
def predict_linA(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Run a model on RAW ``[B,2,H,W]`` (real, imag) and return linear amplitude ``[B,H,W]``.

    ``SARDDCModule`` returns a 2-channel ``x_hat`` (channel 0 real, 1 imag); ``MerlinModule``
    despeckles one channel at a time. Both normalize internally, so ``batch`` is the raw
    (unnormalized) input, exactly as fed in ``create_dataset.py``.
    """
    # Lazy imports: models import from src.utils.*, so importing them at module top would cycle.
    from src.models.merlin_module import MerlinModule
    from src.models.sar_ddc_module import SARDDCModule

    if isinstance(model, SARDDCModule):
        x_hat = model(batch)["x_hat"]
        return denorm_to_linA(x_hat[:, 0], x_hat[:, 1])
    if isinstance(model, MerlinModule):
        return denorm_to_linA(model(batch[:, 0:1])[:, 0], model(batch[:, 1:2])[:, 0])
    raise TypeError(f"predict_linA: unsupported model type {type(model).__name__}")
