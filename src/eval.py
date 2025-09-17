import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
)

log = RankedLogger(__name__, rank_zero_only=True)


def process_large_patch(
    model, patch, model_patch_size=256, stride=None, blend_method="count"
):
    """Process a large patch by splitting into smaller patches, processing each, then recombining.

    Args:
        model: LightningModule
        patch: Tensor of shape [B, H, W, 2] where last dim contains real/imag parts
        model_patch_size: Size of patches the model expects (e.g., 256)
        stride: Stride between patches (if None, uses model_patch_size/2)
        blend_method: How to blend overlapping regions - "count" (average) or "linear" (weighted blend)

    Returns:
        Dictionary with results and metrics
    """
    log.info(f"Processing large patch of shape {patch.shape} with model...")
    _, height, width, _ = patch.shape
    device = next(model.parameters()).device

    # Default stride is half the patch size (50% overlap)
    if stride is None:
        stride = model_patch_size // 2

    # Separate real and imaginary components
    real_large = patch[..., 0].unsqueeze(1)  # [B, 1, H, W]
    imag_large = patch[..., 1].unsqueeze(1)  # [B, 1, H, W]

    # Create output tensors
    real_output = torch.zeros_like(real_large)
    imag_output = torch.zeros_like(imag_large)
    counts = torch.zeros_like(real_large)  # Tracks number of contributions per pixel

    # Metrics storage
    total_bpp_real = 0
    total_bpp_imag = 0
    patch_count = 0

    log.info(
        f"Processing {height}x{width} image in {model_patch_size}x{model_patch_size} patches with stride {stride}"
    )

    # Process each patch
    for y in range(0, height - model_patch_size + 1, stride):
        for x in range(0, width - model_patch_size + 1, stride):
            # Extract small patches (.contiguous() is necessary when doing that in torch)
            real_patch = real_large[
                :, :, y : y + model_patch_size, x : x + model_patch_size
            ].contiguous()
            imag_patch = imag_large[
                :, :, y : y + model_patch_size, x : x + model_patch_size
            ].contiguous()

            # Process patches
            with torch.no_grad():
                output_real = model.forward(real_patch)
                criterion_real = model.criterion(output_real, imag_patch)

                output_imag = model.forward(imag_patch)
                criterion_imag = model.criterion(output_imag, real_patch)

            # Track metrics
            total_bpp_real += criterion_real["bpp"].item()
            total_bpp_imag += criterion_imag["bpp"].item()
            patch_count += 1

            if blend_method == "linear":
                # Create weight mask that fades near edges for smooth blending
                weight = torch.ones_like(real_patch)
                if stride < model_patch_size:
                    # Create linear falloff at edges (pixels at edges have lower weight)
                    falloff = torch.linspace(0, 1, stride, device=device)

                    # Apply falloff to edges where patches will overlap
                    if y > 0:  # Top edge
                        for i in range(stride):
                            weight[:, :, i, :] *= falloff[i]
                    if y + model_patch_size < height:  # Bottom edge
                        for i in range(stride):
                            weight[:, :, -(i + 1), :] *= falloff[i]
                    if x > 0:  # Left edge
                        for i in range(stride):
                            weight[:, :, :, i] *= falloff[i]
                    if x + model_patch_size < width:  # Right edge
                        for i in range(stride):
                            weight[:, :, :, -(i + 1)] *= falloff[i]

                # Apply weighted update
                real_output[
                    :, :, y : y + model_patch_size, x : x + model_patch_size
                ] += output_real * weight
                imag_output[
                    :, :, y : y + model_patch_size, x : x + model_patch_size
                ] += output_imag * weight
                counts[:, :, y : y + model_patch_size, x : x + model_patch_size] += (
                    weight
                )
            elif (
                blend_method == "count"
            ):  # "count" method - simple summation with counting
                real_output[
                    :, :, y : y + model_patch_size, x : x + model_patch_size
                ] += output_real
                imag_output[
                    :, :, y : y + model_patch_size, x : x + model_patch_size
                ] += output_imag
                counts[:, :, y : y + model_patch_size, x : x + model_patch_size] += 1
            else:
                raise ValueError(
                    f"Unknown blend method: {blend_method}. Use 'linear' or 'count'."
                )

    # Normalize by weights for overlapping regions
    if torch.any(counts == 0):
        log.warning(
            "Some pixels were not updated due to no contributions. This may indicate an issue with patch processing."
        )
    counts = counts + 1e-8  # Add small epsilon to avoid division by zero
    real_output = real_output / counts
    imag_output = imag_output / counts

    # Calculate average metrics
    avg_bpp_real = total_bpp_real / patch_count
    avg_bpp_imag = total_bpp_imag / patch_count
    avg_bpp = (avg_bpp_real + avg_bpp_imag) / 2

    log.info(f"Processed {patch_count} patches")

    return {
        "output_real": real_output,
        "output_imag": imag_output,
        "metrics": {
            "bpp_real": avg_bpp_real,
            "bpp_imag": avg_bpp_imag,
            "bpp_avg": avg_bpp,
        },
    }


# @task_wrapper (must return 2 objects: metric_dict and object_dict)
def custom_inference(cfg: DictConfig):
    """Run custom inference on specific inputs without using dataloaders."""
    checkpoint_path = cfg.ckpt_path
    # ------------ Load original training config ------------- #
    ckpt_dir = Path(checkpoint_path).parent.parent
    training_config_path = ckpt_dir / ".hydra" / "config.yaml"

    if training_config_path.exists():
        log.info(f"Loading original training config from {training_config_path}")
        cfg = OmegaConf.load(training_config_path)
    else:
        log.warning(f"⚠️ Original training config not found at {training_config_path} ⚠️")
        log.warning("Using evaluation config, which might cause compatibility issues.")

    # Apply extra utilities (e.g. ask for tags if none are provided in cfg, log.info cfg tree, etc.)
    extras(cfg)

    # ------------ Preparation ------------- #
    # Load checkpoint
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path)
    log.info(
        f"Loading checkpoint from {checkpoint_path}, at epoch {checkpoint['epoch']} and global step {checkpoint['global_step']} and Pytorch Lightning version {checkpoint['pytorch-lightning_version']}."
    )
    # log.info(f"Checkpoint keys: {checkpoint.keys()}")
    # log.info(f"Checkpoint state_dict keys: {checkpoint['state_dict'].keys()}")
    message = model.load_state_dict(checkpoint["state_dict"], strict=True)
    log.info(f"Model state_dict loaded with message: {message}")
    model.eval()

    # Load data
    data_dir = cfg.data.get("hdf5_dir", None)
    patch_path = Path(data_dir) / "val_Hamburg_1024x1024.npy"
    assert patch_path.exists(), (
        f"Patch file {patch_path} does not exist. Please check the path in the config."
    )

    # Hyperparameters logging
    cfg_lambda = cfg.model.criterion["lmbda"]
    normalization = patch_path.parent.name.split("_")[-1][4:]

    # ------------ Run inference ------------- #
    # Load, convert to Tensor, and add batch dimension
    patch = torch.from_numpy(np.load(patch_path)).to(model.device).unsqueeze(0).float()
    # Add channel dimension
    real = patch[..., 0].unsqueeze(1)
    imag = patch[..., 1].unsqueeze(1)

    with torch.no_grad():
        log.info("Running inference on the model...")
        log.info(
            f"The model uses lambda={cfg_lambda} on dataset {patch_path.parent.name}."
        )

        output_real = model.forward(real)
        criterion_real = model.criterion(output_real, imag)
        output_imag = model.forward(imag)
        criterion_imag = model.criterion(output_imag, real)
    # Log the bpp for each part, and its average
    log.info("Inference on whole image completed.")
    log.info(
        f"BPP: real={criterion_real['bpp']:.6f}, imag={criterion_imag['bpp']:.6f}, avg={(criterion_real['bpp'] + criterion_imag['bpp']) / 2:.6f}\n"
    )

    # Process using patch-based approach - no need for complex tensor reshaping
    patchified_results = process_large_patch(
        model=model,
        patch=patch,
        model_patch_size=256,
        stride=192,  # Sonnet default: 192
        blend_method="linear",  # "count" or "linear"
    )
    log.info("Inference on patchified image completed.")
    log.info(
        f"BPP: real={patchified_results['metrics']['bpp_real']:.6f}, imag={patchified_results['metrics']['bpp_imag']:.6f}, avg={patchified_results['metrics']['bpp_avg']:.6f}"
    )
    # Move back to numpy and remove all dimensions
    real = torch.squeeze(real).cpu().numpy()
    imag = torch.squeeze(imag).cpu().numpy()
    reflectivity = real + imag

    output_real = torch.squeeze(output_real).cpu().numpy()
    output_imag = torch.squeeze(output_imag).cpu().numpy()
    output_reflectivity = np.sqrt(0.5 * (output_real + output_imag))

    # If patchified results are available, use them
    output_real_patchified = (
        torch.squeeze(patchified_results["output_real"]).cpu().numpy()
    )
    output_imag_patchified = (
        torch.squeeze(patchified_results["output_imag"]).cpu().numpy()
    )
    output_reflectivity_patchified = np.sqrt(
        0.5 * (output_real_patchified + output_imag_patchified)
    )

    # ------------ Saving ------------- #
    # Create new dir called "inference_results"
    inference_dir = ckpt_dir / f"inference_results_{cfg_lambda}_{normalization}"
    inference_dir.mkdir(exist_ok=True)

    # Save images
    to_save = [
        real,
        imag,
        reflectivity,
        output_real,
        output_imag,
        output_reflectivity,
        output_real_patchified,
        output_imag_patchified,
        output_reflectivity_patchified,
    ]
    img_names = [
        "input_real",
        "input_imag",
        "input_reflectivity",
        "output_real",
        "output_imag",
        "output_reflectivity",
        "output_real_patchified",
        "output_imag_patchified",
        "output_reflectivity_patchified",
    ]
    log.info(f"Saving images {img_names} to {inference_dir}...")
    for img, name in zip(to_save, img_names):
        if name.startswith("input_"):
            img_clipped = img.clip(
                img.mean() - 3 * img.std(), img.mean() + 3 * img.std()
            )
        else:
            img_clipped = img.clip(0, img.mean() + 3 * img.std())
        plt.imsave(
            str(inference_dir / f"{name}_{cfg_lambda}_{normalization}.png"),
            img_clipped,
            cmap="gray",
        )

    # Write the criterion dict to file
    with open(inference_dir / "eval.logs", "w") as f:
        # Add a file header with current date and time, + the model checkpoint path, its epoch and global_step
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"Evaluation of the model checkpoint: {checkpoint_path}, the {current_datetime}\n"
        )
        f.write(
            f"Loaded at epoch: {checkpoint['epoch']} and global_step: {checkpoint['global_step']}\n"
        )
        f.write(f"Evaluated on data from: {data_dir}\n\n")

        for part, criterion in zip(
            ["Real", "Imaginary"], [criterion_real, criterion_imag]
        ):
            f.write(f"Criterion outputs for {part} input:\n")
            for key, value in criterion.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    assert cfg.ckpt_path, "Checkpoint path must be provided in the config."

    custom_inference(cfg)


if __name__ == "__main__":
    main()  # Hydra will automatically pass 'cfg' when called this way
