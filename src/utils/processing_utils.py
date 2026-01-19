import warnings
from pathlib import Path

import numpy as np
import torch
from lightning import LightningModule

from src.models.merlin_module import MerlinModule
from src.models.sar_ddc_module import SARDDCModule
from src.utils.metrics import estimate_bpp


def clip(
    img: np.ndarray,
    mean_std_norm: bool = True,
    clip_factor: int = 3,
    percentiles: tuple[int, int] = (5, 95),
) -> np.ndarray:
    """Clip to either mean +/- clip_factor * std or percentiles[0]th/percentiles[1]th
    percentile."""
    if mean_std_norm:
        img = img.clip(
            img.mean() - clip_factor * img.std(),
            img.mean() + clip_factor * img.std(),
        )
    else:
        p5 = np.percentile(img, percentiles[0])
        p95 = np.percentile(img, percentiles[1])
        img = img.clip(p5, p95)
    return img


def extract_short_name_from_TSX_filepath(filepath: Path):
    """Extract short name from the given filepath.

    Assumes the short name is the substring before the first underscore.
    """
    if "_" not in filepath.name:
        raise ValueError(
            f"File {filepath.name} does not contain an underscore '_' to extract the short name."
        )
    return filepath.name.split("_")[0]


def process_large_patch(
    model: LightningModule,
    input: torch.Tensor,
    target: torch.Tensor | None = None,
    model_patch_size: int = 256,
    stride: int | None = None,
    blend_method: str = "count",
) -> tuple[dict, torch.Tensor]:
    """Process a large patch by splitting into smaller patches, processing each of them, then
    recombining.

    Args:
        model: LightningModule (MerlinModule or SARDDCModule)
        patch: Tensor of shape [B, 1, H, W]
        target: Optional tensor of shape [B, 1, H, W]. If None, no metrics are computed for MerlinModule and only 'bpp' for SARDDCModule.
        model_patch_size: Size of patches the model expects (e.g., 256)
        stride: Optional stride between patches. If None, uses model_patch_size/2.
        blend_method: How to blend overlapping regions - "count" (average) or "linear" (weighted blend)

    Returns:
        Tuple[Dictionary with metrics, Reconstructed large patch tensor]
    """
    # log.info(f"Processing large patch of shape {patch.shape} with model...")
    _, _, height, width = input.shape

    # Default stride is half the patch size (50% overlap)
    if stride is None:
        stride = model_patch_size // 2
    elif stride <= 0 or stride > model_patch_size:
        raise ValueError(f"Invalid stride {stride}. Must be in range [1, {model_patch_size}].")

    # Create output tensors
    output_large = torch.zeros_like(input)
    counts = torch.zeros_like(input)  # Tracks number of contributions per pixel

    # Metrics storage
    large_criterion = {}
    patch_count = 0

    # log.info(
    #     f"Processing {height}x{width} image in {model_patch_size}x{model_patch_size} patches with stride {stride}"
    # )

    # Process each patch
    for y in range(0, height - model_patch_size + 1, stride):
        for x in range(0, width - model_patch_size + 1, stride):
            # Extract small patches (.contiguous() is necessary when doing that in torch)
            input_patch = input[
                :, :, y : y + model_patch_size, x : x + model_patch_size
            ].contiguous()

            # Process patches
            with torch.no_grad():
                output = model.forward(input_patch)

            # Compute criterion if target is provided
            if target is not None:
                target_patch = target[
                    :, :, y : y + model_patch_size, x : x + model_patch_size
                ].contiguous()
                criterion = model.criterion(output, target_patch)

                if patch_count == 0:
                    large_criterion = criterion
                else:
                    for key in large_criterion.keys():
                        large_criterion[key] += criterion[key]
            else:
                if isinstance(model, SARDDCModule):
                    if patch_count == 0:
                        large_criterion["bpp"] = estimate_bpp(output)
                    else:
                        large_criterion["bpp"] += estimate_bpp(output)
                # Rest is unnecessary logic but better safe than sorry
                elif isinstance(model, MerlinModule):
                    pass
                else:
                    raise NotImplementedError(
                        "Model must be either SARDDCModule or MerlinModule when target is None."
                    )

            if isinstance(model, SARDDCModule):
                output = output["x_hat"]

            # Prepare blend
            if blend_method == "linear":
                # Create weight mask for smooth blending
                weight = torch.ones_like(output)

                if stride < model_patch_size:
                    # Calculate overlap size
                    overlap = model_patch_size - stride

                    # Create smooth transition weights using cosine taper on same device
                    taper = (
                        torch.cos(torch.linspace(0, np.pi / 2, overlap, device=weight.device)) ** 2
                    )

                    # Apply taper to overlapping regions
                    if y > 0:  # Top edge overlap
                        weight[:, :, :overlap, :] *= taper.view(-1, 1)
                    if x > 0:  # Left edge overlap
                        weight[:, :, :, :overlap] *= taper.view(1, -1)
                    if y + model_patch_size < height:  # Bottom edge overlap
                        weight[:, :, -overlap:, :] *= taper.flip(0).view(-1, 1)
                    if x + model_patch_size < width:  # Right edge overlap
                        weight[:, :, :, -overlap:] *= taper.flip(0).view(1, -1)

                # Apply weighted update
                output_large[:, :, y : y + model_patch_size, x : x + model_patch_size] += (
                    output * weight
                )
                counts[:, :, y : y + model_patch_size, x : x + model_patch_size] += weight
            elif blend_method == "count":  # "count" method - simple summation with counting
                output_large[:, :, y : y + model_patch_size, x : x + model_patch_size] += output
                counts[:, :, y : y + model_patch_size, x : x + model_patch_size] += 1
            else:
                raise ValueError(f"Unknown blend method: {blend_method}. Use 'linear' or 'count'.")

            patch_count += 1

    # Normalize by weights for overlapping regions
    if torch.any(counts == 0):
        warnings.warn(
            "Some pixels were not updated due to no contributions. This may indicate an issue with patch processing."
        )
    counts = counts + 1e-8  # Add small epsilon to avoid division by zero
    output_large = output_large / counts

    # Average criterion over number of patches
    if large_criterion:
        for key in large_criterion.keys():
            large_criterion[key] /= patch_count

    return large_criterion, output_large
