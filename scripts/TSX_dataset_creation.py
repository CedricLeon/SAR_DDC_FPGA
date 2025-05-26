#!/usr/bin/env python3
"""
TSX_dataset_creation_2.py - Streamlined SAR data preprocessing pipeline

This script processes TerraSAR-X CoSAR format (.cos) images and converts them to
HDF5 format for training deep learning models. Processing pipeline:
1. Load images from .cos files
2. Extract patches
3. Symmetrize patches
4. Square real and imaginary parts
5. Optionally preserve point-like scatterers
6. Optionally normalize data
7. Split into train/val/test sets
8. Save as HDF5 files with histograms

Basic usage:
    python TSX_dataset_creation_2.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR

Optional arguments:
    --preserve-threshold THRESHOLD  # Enable scatterer preservation with threshold in dB
    --norm-mode {db,nat}            # Enable normalization with specified log mode
    --norm-minmax PERCENT           # Percentile for min-max normalization (1, 5, 10, etc.)
    --patch-size SIZE               # Size of extracted patches (default: 256)
    --max-files N                   # Process only N files
    --verbose                       # Print detailed statistics at each processing stage
"""

# Imports
import argparse
import gc
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import SAR utilities
from src.utils.MERLIN_sar_utils import cos2mat, symetrisation_patch_test
from src.utils.pylogger import RankedLogger
from src.utils.sar_utils import (
    convert_to_db,
    extract_patches,
    preserve_point_like_scatterers,
)


# ANSI color codes for console output
class Colors:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def norm_minmax(x, min: float, max: float, clip: bool):
    x_norm = (x - min) / (max - min)
    return np.clip(x_norm, 0, 1) if clip else x_norm


def compute_statistics(data):
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "p1": np.percentile(data, 1),
        "p5": np.percentile(data, 5),
        "p10": np.percentile(data, 10),
        "p90": np.percentile(data, 90),
        "p95": np.percentile(data, 95),
        "p99": np.percentile(data, 99),
    }


def extract_filepath_short_name(file_path):
    """Extract a short name from file path, typically the city name.

    Args:
        file_path: Path to the file

    Returns:
        Short name extracted from file path
    """
    filename = os.path.basename(str(file_path))

    # Most filenames start with the city name
    parts = filename.split("_")
    if parts and len(parts) > 0:
        return parts[0]  # Usually the city name
    else:
        return filename[:10]  # Fall back to first 10 chars


def normalize_data(data, norm_mode="db", norm_minmax_val=0, clip=False, verbose=False):
    """
    Normalization using log transformation and min-max scaling.
    Processes data in batches to reduce memory usage.
    The logarithm base, min and max values, and clipping options can be specified.

    Args:
        data: Input data with shape [N, H, W, 2]
        norm_mode: "db" or "nat" for log mode
        norm_minmax_val: Percentile value for min-max normalization (0 for full min-max, other values for percentiles)
        clip: Whether to clip values to [0, 1] after normalization
        verbose: Whether to print statistics

    Returns:
        Normalized data
    """
    assert data.ndim == 4 or data.ndim == 3, (
        "Data must be 3D [H, W, 2] or 4D [N, H, W, 2] array"
    )
    assert data.shape[-1] == 2, "Data must have 2 channels."
    # Compute batch size based on data shape and available memory
    # Start with a batch size that's around 1/4 of the total data or max 1000 patches
    batch_size = min(1000, max(1, len(data) // 4))
    num_batches = (len(data) + batch_size - 1) // batch_size  # Ceiling division

    # Process real and imaginary parts independently
    normalized_data = np.zeros_like(data, dtype=np.float32)

    # First pass: calculate statistics on a smaller sample to save memory
    # Use either 20% of the data or 1000 patches, whichever is smaller
    sample_size = min(1000, max(1, int(len(data) * 0.2)))
    sample_indices = np.random.choice(len(data), sample_size, replace=False)
    sample_data = data[sample_indices]

    # Extract real and imaginary components from sample
    sample_real = sample_data[..., 0]
    sample_imag = sample_data[..., 1]

    # Compute log-transform on sample
    if norm_mode == "db":
        sample_real_log = convert_to_db(sample_real)
        sample_imag_log = convert_to_db(sample_imag)
        if verbose:
            log.info("      Applied dB (10*log10) transformation")
    elif norm_mode == "nat":
        sample_real_log = np.log(sample_real + np.spacing(1))
        sample_imag_log = np.log(sample_imag + np.spacing(1))
        if verbose:
            log.info("      Applied natural (nat) log transformation")
    else:
        raise ValueError(f"Invalid normalization mode: {norm_mode}")

    # Compute statistics from sample
    stats = {
        "real": compute_statistics(sample_real_log),
        "imag": compute_statistics(sample_imag_log),
    }

    # Free sample memory
    del sample_data, sample_real, sample_imag, sample_real_log, sample_imag_log
    gc.collect()

    # Determine min-max values based on percentile
    if norm_minmax_val == 0:
        # Use actual min and max
        min_real, max_real = stats["real"]["min"], stats["real"]["max"]
        min_imag, max_imag = stats["imag"]["min"], stats["imag"]["max"]

        if verbose:
            log.info("      Using full min-max range for normalization")
            log.info(f"         Real channel: min={min_real:.4f}, max={max_real:.4f}")
            log.info(f"         Imag channel: min={min_imag:.4f}, max={max_imag:.4f}")
    else:
        # Use percentiles
        min_real = stats["real"][f"p{norm_minmax_val}"]
        max_real = stats["real"][f"p{100 - norm_minmax_val}"]
        min_imag = stats["imag"][f"p{norm_minmax_val}"]
        max_imag = stats["imag"][f"p{100 - norm_minmax_val}"]

        if verbose:
            log.info(
                f"      Using {norm_minmax_val}-{100 - norm_minmax_val} percentile range for normalization"
            )
            log.info(
                f"          Real channel: p{norm_minmax_val}={min_real:.4f}, p{100 - norm_minmax_val}={max_real:.4f}"
            )
            log.info(
                f"          Imag channel: p{norm_minmax_val}={min_imag:.4f}, p{100 - norm_minmax_val}={max_imag:.4f}"
            )

    # Second pass: process in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))

        # Get batch
        batch = data[start_idx:end_idx]

        # Process real component
        real_batch = batch[..., 0]
        if norm_mode == "db":
            real_log = convert_to_db(real_batch)
        else:
            real_log = np.log(real_batch + np.spacing(1))

        # In-place normalization for real component
        normalized_data[start_idx:end_idx, ..., 0] = norm_minmax(
            real_log, min_real, max_real, clip
        )

        # Free memory
        del real_batch, real_log

        # Process imag component
        imag_batch = batch[..., 1]
        if norm_mode == "db":
            imag_log = convert_to_db(imag_batch)
        else:
            imag_log = np.log(imag_batch + np.spacing(1))

        # In-place normalization for imag component
        normalized_data[start_idx:end_idx, ..., 1] = norm_minmax(
            imag_log, min_imag, max_imag, clip
        )

        # Free memory
        del imag_batch, imag_log, batch
        gc.collect()

    return normalized_data


def preprocess_tsx_image(
    filepath,
    patch_size=256,
    preserve_threshold=None,
    norm_mode=None,
    norm_minmax_val=0,
    clip=False,
    verbose=False,
):
    """
    Full preprocessing pipeline for a single CoSAR image:
    1. Load images from .cos files
    2. Symmetrize image
    3. Square real and imaginary parts
    4. Optionally preserve point-like scatterers
    5. Optionally normalize data
    6. Extract patches

    Args:
        filepath: Path to the CoSAR image file
        patch_size: Size of patches to extract
        preserve_threshold: Threshold for preserving scatterers (None to disable)
        norm_mode: Normalization mode (None, "db", or "nat")
        norm_minmax_val: Percentile for min-max normalization
        verbose: Whether to print detailed statistics
        log: Logger object

    Returns:
        Dictionary with processed data at different stages
    """
    short_name = extract_filepath_short_name(filepath)

    # 1. Load TSX data
    log.info(f"  1. Loading TSX data from {short_name}...")
    tsx_data = cos2mat(str(filepath))
    if tsx_data is None:
        raise ValueError(f"Failed to load {short_name}")

    # 2. Apply symmetrization to whole image
    log.info("  2. Applying symmetrization to the whole image...")
    # Reshape to match MERLIN's expected format: [h, w, 2] -> real and imag_part [1, h, w, 1]
    real = tsx_data[:, :, 0]
    imag = tsx_data[:, :, 1]
    real_reshaped = real.reshape(1, *real.shape, 1)
    imag_reshaped = imag.reshape(1, *imag.shape, 1)
    real_sym, imag_sym = symetrisation_patch_test(real_reshaped, imag_reshaped)
    # Reshape back to [h, w, 2] format
    real_sym = real_sym[0, :, :, 0]
    imag_sym = imag_sym[0, :, :, 0]
    tsx_data_symmetrized = np.stack((real_sym, imag_sym), axis=2)

    # 3. Square all values
    log.info("  3. Squaring the image...")
    tsx_data_squared = np.square(tsx_data_symmetrized)

    # 4. Preserve scatterers (optional)
    tsx_data_preserved = tsx_data_squared
    scatterer_mask = None

    if preserve_threshold is not None:
        log.info(
            f"  4. Preserving point-like scatterers above {preserve_threshold} dB..."
        )
        tsx_data_preserved, scatterer_mask = preserve_point_like_scatterers(
            tsx_data_squared,
            threshold_db=preserve_threshold,
        )

        # Count preserved scatterers
        _, nb_scatterer_preserved = np.unique(scatterer_mask, return_counts=True)
        log.info(
            f"          {Colors.GREEN}Strong scatterers preserved = {nb_scatterer_preserved[1]} (or {nb_scatterer_preserved[1] / nb_scatterer_preserved[0] * 100:.6f}%).{Colors.RESET}"
        )
    else:
        log.info("  4. Scatterer preservation disabled, skipping...")

    # 5. Normalize data (optional)
    tsx_data_normalized = tsx_data_preserved
    norm_stats = None

    if norm_mode is not None:
        log.info(
            f"  5. Normalizing data using {norm_mode} log mode with {norm_minmax_val}% range..."
        )
        tsx_data_normalized = normalize_data(
            tsx_data_preserved,
            norm_mode=norm_mode,
            norm_minmax_val=norm_minmax_val,
            clip=clip,
            verbose=verbose,
        )
        norm_stats_real = compute_statistics(tsx_data_normalized[..., 0])
        norm_stats_imag = compute_statistics(tsx_data_normalized[..., 1])
        # Combine the statistics for real and imaginary parts
        norm_stats = {
            "real": norm_stats_real,
            "imag": norm_stats_imag,
        }
    else:
        log.info("  5. Normalization disabled, skipping...")

    # 6. Extract patches
    log.info(f"  6. Extracting patches of size {patch_size}x{patch_size}...")
    tsx_patches = extract_patches(tsx_data_normalized, patch_size, stride=patch_size)

    if len(tsx_patches) == 0:
        raise ValueError(f"No patches could be extracted from {short_name}")

    # Return the results at different stages of the pipeline
    return {
        "original_data": tsx_data,
        "symmetrized": tsx_data_symmetrized,
        "squared": tsx_data_squared,
        "preserved": tsx_data_preserved,
        "scatterer_mask": scatterer_mask,
        "normalized": tsx_data_normalized,
        "norm_stats": norm_stats,
        "final_patches": tsx_patches,
    }


def preprocess_tsx_patches(
    filepath,
    patch_size=256,
    preserve_threshold=None,
    norm_mode=None,
    norm_minmax_val=0,
    clip=False,
    verbose=False,
):
    """
    Full preprocessing pipeline for a single CoSAR image:
    1. Load images from .cos files
    2. Extract patches
    3. Symmetrize patches
    4. Square real and imaginary parts
    5. Optionally preserve point-like scatterers
    6. Optionally normalize data

    Args:
        filepath: Path to the CoSAR image file
        patch_size: Size of patches to extract
        preserve_threshold: Threshold for preserving scatterers (None to disable)
        norm_mode: Normalization mode (None, "db", or "nat")
        norm_minmax_val: Percentile for min-max normalization
        verbose: Whether to print detailed statistics
        log: Logger object

    Returns:
        Dictionary with processed data at different stages
    """
    short_name = extract_filepath_short_name(filepath)

    # 1. Load TSX data
    log.info(f"  1. Loading TSX data from {short_name}...")
    tsx_data = cos2mat(str(filepath))
    if tsx_data is None:
        raise ValueError(f"Failed to load {short_name}")

    if verbose:
        log.info(f"    Original data shape: {tsx_data.shape}")
        log.info(
            f"    Real part - "
            f"min: {np.min(tsx_data[:, :, 0]):.2f}, mean: {np.mean(tsx_data[:, :, 0]):.2f}, max: {np.max(tsx_data[:, :, 0]):.2f}"
        )
        log.info(
            f"    Imag part - "
            f"min: {np.min(tsx_data[:, :, 1]):.2f}, mean: {np.mean(tsx_data[:, :, 1]):.2f}, max: {np.max(tsx_data[:, :, 1]):.2f}"
        )

    # 2. Extract patches
    log.info(f"  2. Extracting patches of size {patch_size}x{patch_size}...")
    original_patches = extract_patches(tsx_data, patch_size, stride=patch_size)
    if verbose:
        log.info(
            f"    Extracted {len(original_patches)} patches of size {patch_size}x{patch_size}. Patches shape: {original_patches.shape}."
        )

    if len(original_patches) == 0:
        raise ValueError(f"No patches could be extracted from {short_name}")

    # 3. Apply symmetrization to each patch
    log.info("  3. Applying symmetrization to each patch...")
    symmetrized_patches = []
    for patch in original_patches:
        # Reshape to match MERLIN's expected format: [h, w, 2] -> real and imag_part [1, h, w, 1]
        real_part = patch[:, :, 0]
        imag_part = patch[:, :, 1]
        real_part_reshaped = real_part.reshape(1, *real_part.shape, 1)
        imag_part_reshaped = imag_part.reshape(1, *imag_part.shape, 1)
        real_sym, imag_sym = symetrisation_patch_test(
            real_part_reshaped, imag_part_reshaped
        )
        # Reshape back to [h, w, 2] format
        real_sym = real_sym[0, :, :, 0]
        imag_sym = imag_sym[0, :, :, 0]
        symmetrized_patch = np.stack((real_sym, imag_sym), axis=2)
        symmetrized_patches.append(symmetrized_patch)

    symmetrized_patches = np.array(symmetrized_patches)

    if verbose:
        log.info(f"    Symmetrized patches shape: {symmetrized_patches.shape}")
        log.info(
            f"    Real part - "
            f"min: {np.min(symmetrized_patches[:, :, :, 0]):.2f}, "
            f"mean: {np.mean(symmetrized_patches[:, :, :, 0]):.2f}, "
            f"max: {np.max(symmetrized_patches[:, :, :, 0]):.2f}"
        )
        log.info(
            f"    Imag part - "
            f"min: {np.min(symmetrized_patches[:, :, :, 1]):.2f}, "
            f"mean: {np.mean(symmetrized_patches[:, :, :, 1]):.2f}, "
            f"max: {np.max(symmetrized_patches[:, :, :, 1]):.2f}"
        )

    # 4. Square the real and imaginary parts
    log.info("  4. Squaring the real and imaginary parts...")
    squared_patches = np.square(symmetrized_patches)

    if verbose:
        log.info(f"    Squared patches shape: {squared_patches.shape}")
        log.info(
            f"    Real part² - "
            f"min: {np.min(squared_patches[:, :, :, 0]):.2f}, "
            f"mean: {np.mean(squared_patches[:, :, :, 0]):.2f}, "
            f"max: {np.max(squared_patches[:, :, :, 0]):.2f}"
        )
        log.info(
            f"    Imag part² - "
            f"min: {np.min(squared_patches[:, :, :, 1]):.2f}, "
            f"mean: {np.mean(squared_patches[:, :, :, 1]):.2f}, "
            f"max: {np.max(squared_patches[:, :, :, 1]):.2f}"
        )

    # 5. Preserve scatterers (optional)
    preserved_patches = squared_patches
    scatterer_masks = None

    if preserve_threshold is not None:
        log.info(
            f"  5. Preserving point-like scatterers above {preserve_threshold} dB..."
        )
        preserved_patches_list = []
        scatterer_masks_list = []

        for patch in squared_patches:
            preserved_patch, scatterer_mask = preserve_point_like_scatterers(
                patch[:, :, 0], patch[:, :, 1], threshold_db=preserve_threshold
            )
            preserved_patches_list.append(preserved_patch)
            scatterer_masks_list.append(scatterer_mask)

        preserved_patches = np.array(preserved_patches_list)
        scatterer_masks = np.array(scatterer_masks_list)

        # Count preserved scatterers
        _, nb_scatterer_preserved = np.unique(scatterer_masks, return_counts=True)
        log.info(
            f"          {Colors.GREEN}Strong scatterers preserved = {nb_scatterer_preserved[1]} (or {nb_scatterer_preserved[1] / nb_scatterer_preserved[0] * 100:.6f}%).{Colors.RESET}"
        )

        if verbose:
            log.info(f"    Preserved patches shape: {preserved_patches.shape}")
            log.info(
                f"    Real part - "
                f"min: {np.min(preserved_patches[:, :, :, 0]):.2f}, "
                f"mean: {np.mean(preserved_patches[:, :, :, 0]):.2f}, "
                f"max: {np.max(preserved_patches[:, :, :, 0]):.2f}"
            )
            log.info(
                f"    Imag part - "
                f"min: {np.min(preserved_patches[:, :, :, 1]):.2f}, "
                f"mean: {np.mean(preserved_patches[:, :, :, 1]):.2f}, "
                f"max: {np.max(preserved_patches[:, :, :, 1]):.2f}"
            )
    else:
        log.info("  5. Scatterer preservation disabled, skipping...")

    # 6. Normalize data (optional)
    normalized_patches = None
    norm_stats = None

    if norm_mode is not None:
        log.info(
            f"  6. Normalizing data using {norm_mode} log mode with {norm_minmax_val}% range..."
        )
        normalized_patches = normalize_data(
            preserved_patches,
            norm_mode=norm_mode,
            norm_minmax_val=norm_minmax_val,
            clip=clip,
            verbose=verbose,
        )
        norm_stats_real = compute_statistics(normalized_patches[..., 0])
        norm_stats_imag = compute_statistics(normalized_patches[..., 1])
        # Combine the statistics for real and imaginary parts
        norm_stats = {
            "real": norm_stats_real,
            "imag": norm_stats_imag,
        }

        if verbose:
            log.info(f"    Normalized patches shape: {normalized_patches.shape}")
            log.info(
                f"    Real part - "
                f"min: {norm_stats_real['min']:.4f}, "
                f"mean: {norm_stats_real['mean']:.4f}, "
                f"max: {norm_stats_real['max']:.4f}"
            )
            log.info(
                f"    Imag part - "
                f"min: {norm_stats_imag['min']:.4f}, "
                f"mean: {norm_stats_imag['mean']:.4f}, "
                f"max: {norm_stats_imag['max']:.4f}"
            )
    else:
        log.info("  6. Normalization disabled, skipping...")

    # Return the results at different stages of the pipeline
    result = {
        "original_data": tsx_data,
        "original": original_patches,
        "symmetrized": symmetrized_patches,
        "squared": squared_patches,
        "preserved": preserved_patches,
        "scatterer_mask": scatterer_masks,
        "normalized": normalized_patches,
        "norm_stats": norm_stats,
    }

    # Determine which set of patches should be the final output
    if normalized_patches is not None:
        result["final_patches"] = normalized_patches
        log.info("  Final patches: normalized_patches")
    else:
        result["final_patches"] = preserved_patches
        log.info("  Final patches: preserved_patches")

    return result


def plot_processing_histograms(results, output_path, title_prefix=""):
    """
    Plot histograms of the TSX data at different processing stages.

    Args:
        results: Dictionary with processed data at different stages
        output_path: Path to save the histogram plots
        title_prefix: Prefix for plot titles

    Returns:
        None
    """
    # Determine which stages are available in the results
    has_preserved = (
        results["preserved"] is not None and results["scatterer_mask"] is not None
    )
    has_normalized = results["normalized"] is not None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{title_prefix} TSX Data Processing Pipeline", fontsize=16)

    # Plot all histograms
    plot_hist(
        results["original_data"],
        axes[0, 0],
        "Original Patches (Intensity [dB])",
        intensity=True,
        is_squared=False,
    )
    plot_hist(
        results["symmetrized"],
        axes[0, 1],
        "Symmetrized Patches (Intensity [dB])",
        intensity=True,
        is_squared=False,
    )
    plot_hist(
        results["squared"],
        axes[0, 2],
        "Squared Patches (Intensity [dB])",
        intensity=True,
    )
    if has_preserved:
        plot_hist(
            results["preserved"],
            axes[1, 0],
            f"Preserved Patches (>{results.get('preserve_threshold', 0)} dB, Intensity [dB])",
            intensity=True,
        )
    else:
        axes[1, 0].set_visible(False)

    # Plot normalized patches if available
    if has_normalized:
        plot_hist(
            results["normalized"][..., 0].flatten(),
            axes[1, 1],
            f"Real Norm ({results['norm_mode']} at {results['norm_minmax_val']}%)",
        )
        plot_hist(
            results["normalized"][..., 1].flatten(),
            axes[1, 2],
            f"Imag Norm ({results['norm_mode']} at {results['norm_minmax_val']}%)",
        )
    else:
        axes[1, 1].set_visible(False)
        axes[1, 2].set_visible(False)

    # Add tight layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# Helper function to plot histogram
def plot_hist(data, ax, title, intensity=False, is_squared=True, data_percent=10):
    # Sample only a percentage of patches to handle large datasets
    n_sample = max(1, int(len(data) * data_percent / 100))
    indices = np.random.choice(len(data), n_sample, replace=False)
    data = data[indices]

    if intensity:
        assert (len(data.shape) == 4) or (len(data.shape) == 3), (
            f"Data must be [N,h,w,2] or [h,w,2] for intensity histogram, but {data.shape}"
        )
        assert data.shape[-1] == 2, (
            f"Data must have 2 channels (real and imag), but {data.shape}"
        )
        # Convert to intensity (dB)
        if is_squared:
            intensity_patches = convert_to_db(data[..., 0] + data[..., 1])
        else:
            intensity_patches = convert_to_db(data[..., 0] ** 2 + data[..., 1] ** 2)
        data = intensity_patches.flatten()
    else:
        assert len(data.shape) == 1, (
            f"Data must be flattened for non-intensity histogram, but {data.shape}"
        )
    # Compute statistics
    mean = np.mean(data)
    median = np.median(data)
    p5 = np.percentile(data, 5)
    p95 = np.percentile(data, 95)
    min_val = np.min(data)
    max_val = np.max(data)

    # Plot histogram
    ax.hist(data, bins=100, color="blue", alpha=0.7)

    # Add vertical lines for key statistics
    ax.axvline(mean, color="r", linestyle="--", label=f"Mean: {mean:.4f}")
    ax.axvline(median, color="g", linestyle="--", label=f"Median: {median:.4f}")
    ax.axvline(p5, color="orange", linestyle=":", label=f"5%: {p5:.4f}")
    ax.axvline(p95, color="orange", linestyle=":", label=f"95%: {p95:.4f}")

    # Set labels and legend
    ax.set_title(f"{title} - min: {min_val:.4f}, max: {max_val:.4f}")
    ax.set_xlabel("Intensity (dB)" if intensity else "Value")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def create_combined_histogram(
    train_path,
    val_path,
    test_path,
    output_path,
    data_percent=10,
    is_squared=True,
    intensity=False,
):
    """
    Create a combined histogram figure with train, val, and test subplots.

    Args:
        train_path: Path to train HDF5 file
        val_path: Path to validation HDF5 file
        test_path: Path to test HDF5 file
        output_path: Path to save the histogram plot
        data_percent: Percentage of data to use for statistics
        is_squared: Whether data is already squared (default: True)
        intensity: Whether to plot intensity instead of log-intensity (default: False)

    Returns:
        None
    """
    log.info(f"Creating combined histogram (using {data_percent}% of data)...")

    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Load and plot training data
    log.info(f"  Loading training data from {train_path}")
    with h5py.File(train_path, "r") as f:
        train_patches = f["patches"][:]
        plot_hist(
            train_patches[..., 0].flatten(),
            axs[0],
            "Training Set (Real)",
            intensity=intensity,
            is_squared=is_squared,
            data_percent=data_percent,
        )

    # Load and plot validation data
    log.info(f"  Loading validation data from {val_path}")
    with h5py.File(val_path, "r") as f:
        val_patches = f["patches"][:]
        plot_hist(
            val_patches[..., 0].flatten(),
            axs[1],
            "Validation Set (Real)",
            intensity=intensity,
            is_squared=is_squared,
            data_percent=data_percent,
        )

    # Load and plot test data
    log.info(f"  Loading test data from {test_path}")
    with h5py.File(test_path, "r") as f:
        test_patches = f["patches"][:]
        plot_hist(
            test_patches[..., 0].flatten(),
            axs[2],
            "Test Set (Real)",
            intensity=intensity,
            is_squared=is_squared,
            data_percent=data_percent,
        )

    # Add title and adjust layout
    fig.suptitle("Dataset Distributions of Real parts", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure
    plt.savefig(output_path, dpi=300)
    log.info(f"  Saved combined histogram to {output_path}")
    plt.close(fig)


def write_hdf5(patches, metadata, path):
    """
    Write dataset patches to HDF5 at path.

    Args:
        patches: Array of patches to write
        metadata: Dictionary with metadata and statistics
        path: Path to save HDF5 file

    Returns:
        bool: Success status
    """
    n_patches = len(patches)

    with h5py.File(path, "w") as f:
        # Store patches
        f.create_dataset("patches", data=patches, dtype="float32")

        # Add all metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)):
                f.attrs[key] = value
            elif isinstance(value, dict):
                group = f.create_group(key)
                for k, v in value.items():
                    if isinstance(v, dict):
                        subgroup = group.create_group(k)
                        for sk, sv in v.items():
                            subgroup.attrs[sk] = sv
                    else:
                        group.attrs[k] = v

    return True


def split_patches(patches, metadata, train_frac=0.8, val_frac=0.1):
    """
    Split patches into train, validation, and test sets.
    Memory-optimized implementation that processes data in batches.

    Args:
        patches: Array of patches to split
        metadata: Dictionary with metadata and statistics
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation

    Returns:
        Dictionary with train, val, and test patch arrays and metadata
    """
    # Calculate number of patches for each split
    n_samples = len(patches)
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    n_test = n_samples - n_train - n_val

    log.info(
        f"Splitting {n_samples} patches: {n_train} train, {n_val} val, {n_test} test"
    )

    # Create random permutation of indices
    indices = np.random.permutation(n_samples)

    # Split the indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # Process in batches to reduce memory usage
    # First, create empty arrays to hold the data
    if n_samples > 0:
        patch_shape = patches[0].shape

        # Create arrays with proper shapes but use efficient memory allocation
        train_patches = np.zeros((n_train,) + patch_shape, dtype=patches.dtype)
        val_patches = np.zeros((n_val,) + patch_shape, dtype=patches.dtype)
        test_patches = np.zeros((n_test,) + patch_shape, dtype=patches.dtype)

        # Determine batch size based on available memory (adjust as needed)
        batch_size = 1000  # Start with a reasonable batch size

        # Copy training data in batches
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            batch_indices = train_indices[i:end_idx]
            train_patches[i:end_idx] = patches[batch_indices]
            # Force garbage collection periodically
            if i % (batch_size * 10) == 0:
                gc.collect()

        # Copy validation data in batches
        for i in range(0, n_val, batch_size):
            end_idx = min(i + batch_size, n_val)
            batch_indices = val_indices[i:end_idx]
            val_patches[i:end_idx] = patches[batch_indices]
            # Force garbage collection periodically
            if i % (batch_size * 10) == 0:
                gc.collect()

        # Copy test data in batches
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            batch_indices = test_indices[i:end_idx]
            test_patches[i:end_idx] = patches[batch_indices]
            # Force garbage collection periodically
            if i % (batch_size * 10) == 0:
                gc.collect()
    else:
        # Handle edge case of empty array
        train_patches = patches[train_indices]
        val_patches = patches[val_indices]
        test_patches = patches[test_indices]

    # Add split information to metadata
    split_metadata = metadata.copy()
    split_metadata.update(
        {
            "train_size": n_train,
            "val_size": n_val,
            "test_size": n_test,
            "train_fraction": train_frac,
            "val_fraction": val_frac,
            "test_fraction": 1 - train_frac - val_frac,
        }
    )

    return {
        "train": {"patches": train_patches, "metadata": split_metadata},
        "val": {"patches": val_patches, "metadata": split_metadata},
        "test": {"patches": test_patches, "metadata": split_metadata},
        "metadata": split_metadata,
    }


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    max_files=None,
    patch_size=256,
    train_frac=0.8,
    val_frac=0.1,
    preserve_threshold=None,
    norm_mode=None,
    norm_minmax_val=0,
    clip=False,
    verbose=False,
):
    """
    Process all TSX CoSAR files, extract patches, and create train/val/test datasets.

    Args:
        input_dir: Directory containing .cos files
        output_dir: Directory to save results
        max_files: Maximum number of files to process
        patch_size: Size of patches to extract
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        preserve_threshold: Threshold for preserving scatterers (None to disable)
        norm_mode: Normalization mode (None, "db", or "nat")
        norm_minmax_val: Percentile for min-max normalization
        verbose: Whether to print detailed statistics
        log: Logger object

    Returns:
        bool: Success status
    """
    # Record start time for tracking processing duration
    start_time = datetime.now()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find .cos files
    input_path = input_dir
    cos_files = list(input_path.glob("*.cos"))

    if len(cos_files) == 0:
        log.error(f"No .cos files found in {input_dir}")
        return False

    log.info(f"Found {len(cos_files)} .cos files in {input_dir}")
    for file in cos_files:
        short_name = extract_filepath_short_name(file)
        log.info(f"  - {short_name}")

    # Limit the number of files if specified
    if max_files and max_files < len(cos_files):
        cos_files = cos_files[:max_files]
        log.info(f"Processing only the first {max_files} files")

    # Create temporary directory for patches
    output_path = output_dir
    patches_dir = output_path / "tmp_patches"
    os.makedirs(patches_dir, exist_ok=True)

    # Initialize counters
    total_patches = 0
    file_info = []

    # Process each file and save patches
    log.info(f"{Colors.BLUE}Processing TSX CoSAR files...{Colors.RESET}")
    for i, file_path in enumerate(cos_files):
        short_name = extract_filepath_short_name(file_path)
        log.info(
            f"{Colors.BLUE}Processing {i + 1}/{len(cos_files)}: {short_name}{Colors.RESET}"
        )

        # Process the file
        results = preprocess_tsx_image(
            file_path,
            patch_size=patch_size,
            preserve_threshold=preserve_threshold,
            norm_mode=norm_mode,
            norm_minmax_val=norm_minmax_val,
            clip=clip,
            verbose=verbose,
        )
        # Add additional metadata to results
        results["preserve_threshold"] = preserve_threshold
        results["norm_mode"] = norm_mode
        results["norm_minmax_val"] = norm_minmax_val

        # Get final patches
        patches = results["final_patches"]
        num_patches = len(patches)

        # Create histograms for this file
        hist_path = output_path / f"{short_name}_histograms.png"
        plot_processing_histograms(results, hist_path, title_prefix=short_name)
        log.info(f"  Saved processing histograms to {hist_path}")

        # Determine split sizes for this file using random split
        num_train = int(num_patches * train_frac)
        num_val = int(num_patches * val_frac)
        num_test = num_patches - num_train - num_val

        # Update total count
        total_patches += num_patches

        # Create random permutation for this file
        indices = np.random.permutation(num_patches)
        train_indices = indices[:num_train]
        val_indices = indices[num_train : num_train + num_val]
        test_indices = indices[num_train + num_val :]

        # Prepare metadata
        metadata = {
            "source_file": str(file_path),
            "short_name": short_name,
            "num_patches": num_patches,
            "patch_size": patch_size,
            "preserve_threshold": preserve_threshold,
            "norm_mode": norm_mode,
            "norm_minmax_val": norm_minmax_val,
        }
        if results["norm_stats"] is not None:
            metadata.update({"norm_stats": results["norm_stats"]})

        # Save patches as temporary HDF5 files
        if num_train > 0:
            train_file = patches_dir / f"{short_name}_train.h5"
            train_patches = patches[train_indices]
            write_hdf5(train_patches, metadata, train_file)
            log.info(f"  Saved {num_train} training patches to {train_file}")
            del train_patches
        if num_val > 0:
            val_file = patches_dir / f"{short_name}_val.h5"
            val_patches = patches[val_indices]
            write_hdf5(val_patches, metadata, val_file)
            log.info(f"  Saved {num_val} validation patches to {val_file}")
            del val_patches
        if num_test > 0:
            test_file = patches_dir / f"{short_name}_test.h5"
            test_patches = patches[test_indices]
            write_hdf5(test_patches, metadata, test_file)
            log.info(f"  Saved {num_test} test patches to {test_file}")
            del test_patches

        # Record file info for merging
        file_info.append(
            {
                "name": short_name,
                "train_file": train_file if num_train > 0 else None,
                "val_file": val_file if num_val > 0 else None,
                "test_file": test_file if num_test > 0 else None,
                "num_train": num_train,
                "num_val": num_val,
                "num_test": num_test,
                "metadata": metadata,
            }
        )

        # Force garbage collection
        del patches
        del results
        gc.collect()

    # Merge all patches into final datasets
    log.info(f"{Colors.BLUE}Merging patches from all files...{Colors.RESET}")

    # Calculate split sizes
    total_train = sum(info["num_train"] for info in file_info)
    total_val = sum(info["num_val"] for info in file_info)
    total_test = sum(info["num_test"] for info in file_info)

    log.info(f"Total patches: {total_patches}")
    log.info(
        f"  - Training: {total_train} patches ({total_train / total_patches * 100:.2f}%)"
    )
    log.info(
        f"  - Validation: {total_val} patches ({total_val / total_patches * 100:.2f}%)"
    )
    log.info(
        f"  - Testing: {total_test} patches ({total_test / total_patches * 100:.2f}%)"
    )

    # Function to merge patches from multiple files
    def merge_patches(file_list, output_file, set_name, start_time=None):
        if not file_list:
            log.warning(f"No files for {set_name} set. Skipping.")
            return None

        # Determine total size and shape
        total_count = sum(h5py.File(f, "r")["patches"].shape[0] for f in file_list)
        with h5py.File(file_list[0], "r") as f:
            patch_shape = f["patches"].shape[1:]

        # Chunk size for reading and writing
        chunk_size = 1000  # Adjust based on available memory

        # Create dataset with chunks
        with h5py.File(output_file, "w") as out_f:
            # Create extensible dataset
            patches_dset = out_f.create_dataset(
                "patches",
                shape=(0,) + patch_shape,
                maxshape=(total_count,) + patch_shape,
                dtype="float32",
                chunks=(chunk_size,) + patch_shape,
            )

            # Copy metadata from first file
            with h5py.File(file_list[0], "r") as first_f:
                for attr_name, attr_value in first_f.attrs.items():
                    if attr_name != "num_patches":  # We'll update this
                        out_f.attrs[attr_name] = attr_value

                # Copy groups like norm_stats if they exist
                for group_name in first_f.keys():
                    if group_name != "patches" and isinstance(
                        first_f[group_name], h5py.Group
                    ):
                        group = out_f.create_group(group_name)
                        for attr_name, attr_value in first_f[group_name].attrs.items():
                            group.attrs[attr_name] = attr_value

            # Add patches from each file in chunks
            start_idx = 0
            for i, file_path in enumerate(
                tqdm(file_list, desc=f"Merging {set_name} files")
            ):
                with h5py.File(file_path, "r") as in_f:
                    num_file_patches = in_f["patches"].shape[0]
                    for chunk_start in range(0, num_file_patches, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, num_file_patches)
                        file_patches_chunk = in_f["patches"][chunk_start:chunk_end]

                        # Resize dataset and copy patches
                        num_chunk_patches = len(file_patches_chunk)
                        patches_dset.resize(start_idx + num_chunk_patches, axis=0)
                        patches_dset[start_idx : start_idx + num_chunk_patches] = (
                            file_patches_chunk
                        )

                        # Update start index
                        start_idx += num_chunk_patches

                        del file_patches_chunk
                        gc.collect()  # Important to free memory after each chunk

                # Collect short names of all images
                image_short_names = []
                for file_path in file_list:
                    with h5py.File(file_path, "r") as in_f:
                        if "short_name" in in_f.attrs:
                            short_name = in_f.attrs["short_name"]
                            if short_name not in image_short_names:
                                image_short_names.append(short_name)

                # Calculate statistics for a subset of data (10%)
                stats_data_percent = 10
                # log.info(
                #     f"Calculating statistics using {stats_data_percent}% of the data..."
                # )
                # Make sure we don't exceed the dataset size (actual range is 0 to total_count-1)
                sample_size = max(
                    1, min(total_count, int(total_count * stats_data_percent / 100))
                )

                # Use sequential indices instead of random sampling to avoid potential h5py fancy indexing issues
                # This is more memory efficient and avoids out of range errors
                max_idx = total_count - 1
                step = max(1, max_idx // sample_size)

                # Generate evenly spaced indices
                indices = np.arange(0, max_idx, step)[:sample_size]

                # Read sample data in chunks
                sample_data = []
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(indices))
                    sample_indices = indices[chunk_start:chunk_end]
                    # Read one index at a time to avoid fancy indexing issues with h5py
                    chunk_data = []
                    for idx in sample_indices:
                        chunk_data.append(patches_dset[idx : idx + 1])
                    if chunk_data:
                        sample_data.append(np.concatenate(chunk_data, axis=0))

                if sample_data:
                    sample_data = np.concatenate(sample_data, axis=0)
                else:
                    # Fallback if we couldn't get any samples
                    log.warning(
                        "Could not sample data for statistics. Using first 10 elements."
                    )
                    sample_size = min(10, total_count)
                    sample_data = patches_dset[:sample_size]

                # Calculate statistics for real and imaginary parts
                real_data = sample_data[..., 0]
                imag_data = sample_data[..., 1]

                # Update final metadata
                out_f.attrs["creation_date"] = str(datetime.now())
                out_f.attrs["processing_duration"] = str(datetime.now() - start_time)
                out_f.attrs["preserve_threshold"] = (
                    "None" if preserve_threshold is None else preserve_threshold
                )
                out_f.attrs["norm_mode"] = "None" if norm_mode is None else norm_mode
                out_f.attrs["norm_minmax_val"] = norm_minmax_val
                out_f.attrs["total_patches"] = total_count
                out_f.attrs["patch_size"] = patch_size
                out_f.attrs["images_short_names"] = "/".join(image_short_names)
                out_f.attrs["train_size"] = total_train
                out_f.attrs["val_size"] = total_val
                out_f.attrs["test_size"] = total_test
                out_f.attrs["stats_data_percent"] = stats_data_percent

                # Add statistics for real and imaginary parts
                out_f.attrs["min_real"] = float(np.min(real_data))
                out_f.attrs["max_real"] = float(np.max(real_data))
                out_f.attrs["mean_real"] = float(np.mean(real_data))
                out_f.attrs["p5_real"] = float(np.percentile(real_data, 5))
                out_f.attrs["p95_real"] = float(np.percentile(real_data, 95))

                out_f.attrs["min_imag"] = float(np.min(imag_data))
                out_f.attrs["max_imag"] = float(np.max(imag_data))
                out_f.attrs["mean_imag"] = float(np.mean(imag_data))
                out_f.attrs["p5_imag"] = float(np.percentile(imag_data, 5))
                out_f.attrs["p95_imag"] = float(np.percentile(imag_data, 95))

        return output_file

    # Merge sets
    train_files = [
        info["train_file"] for info in file_info if info["train_file"] is not None
    ]
    val_files = [info["val_file"] for info in file_info if info["val_file"] is not None]
    test_files = [
        info["test_file"] for info in file_info if info["test_file"] is not None
    ]

    # Merge each split
    log.info(f"Merging {len(train_files)} training files...")
    train_result = merge_patches(
        train_files, output_path / "train.h5", "training", start_time
    )

    log.info(f"Merging {len(val_files)} validation files...")
    val_result = merge_patches(
        val_files, output_path / "val.h5", "validation", start_time
    )

    log.info(f"Merging {len(test_files)} test files...")
    test_result = merge_patches(test_files, output_path / "test.h5", "test", start_time)

    # Create combined histogram
    combined_histogram_path = output_path / "combined_histograms.png"
    create_combined_histogram(
        train_result, val_result, test_result, combined_histogram_path
    )

    # Clean up temporary files
    log.info("Cleaning up temporary files...")
    for info in file_info:
        if info["train_file"] and os.path.exists(info["train_file"]):
            os.remove(info["train_file"])
        if info["val_file"] and os.path.exists(info["val_file"]):
            os.remove(info["val_file"])
        if info["test_file"] and os.path.exists(info["test_file"]):
            os.remove(info["test_file"])

    # Remove tmp directory
    os.rmdir(patches_dir)

    # Success if at least one split was created
    return train_result is not None or val_result is not None or test_result is not None


def main():
    """Main function for TSX dataset creation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process TSX .cos files and create HDF5 datasets."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing .cos files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Parent directory to save the different datasets",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Size of patches to extract (default: 256)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (optional)",
    )
    parser.add_argument(
        "--preserve-threshold",
        type=float,
        default=None,
        help="Threshold above which point-like scatterers are preserved (default: None)",
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        choices=["db", "nat"],
        default=None,
        help="Normalization mode: 'db' or 'nat' (natural) (default: None -> no normalization is done)",
    )
    parser.add_argument(
        "--norm-minmax",
        type=int,
        default=0,
        help="Percentiles used in place of min and max in the normalization: 0, 1, 5, or 10 (default: 0 -> normal min and max values used)",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Whether to clip the data to the range [0, 1] after normalization. Disabled is --norm-minmax is 0 (data already between 0 and 1). (default: False)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics during processing",
    )

    args = parser.parse_args()

    # Build dataset name
    pres_name = (
        "nopres"
        if args.preserve_threshold is None
        else f"pres{int(args.preserve_threshold)}"
    )
    norm_name = (
        "nonorm"
        if args.norm_mode is None
        else f"norm{args.norm_minmax}{args.norm_mode}"
    )
    norm_name += "clip" if args.clip else ""
    dataset_name = f"randomsplit{args.max_files}im_{pres_name}_{norm_name}"

    # Setup logging
    # 1. Configure rank_zero_only
    from lightning_utilities.core.rank_zero import rank_zero_only

    rank_zero_only.rank = 0  # Set rank for single-process script

    # 2. Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Setup root logger first
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 4. Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 5. File handler
    log_file_path = os.path.join(args.output_dir, dataset_name, "dataset_creation.log")
    file_handler = logging.FileHandler(log_file_path)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # 6. Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 7. Create RankedLogger AFTER setting up root logger
    global log
    log = RankedLogger(__name__, rank_zero_only=True)

    # 8. Log that we've setup logging
    log.info(f"Logging configured. Log file: {log_file_path}")

    # Validate arguments
    if args.train_frac + args.val_frac > 1.0:
        log.warning("Train + validation fractions exceed 1.0. Adjusting values...")
        total = args.train_frac + args.val_frac
        args.train_frac /= total
        args.val_frac /= total
        log.info(
            f"Adjusted fractions: train={args.train_frac:.2f}, "
            f"val={args.val_frac:.2f}, test={1 - args.train_frac - args.val_frac:.2f}"
        )
    if args.clip and args.norm_minmax == 0:
        log.warning(
            "Clipping is enabled, but norm_minmax is set to 0. "
            "Clipping will be disabled."
        )
        args.clip = False

    # Print configuration
    log.info(f"{Colors.YELLOW}TSX Dataset Creation - Configuration:{Colors.RESET}")
    log.info(f"  Input directory: {args.input_dir}")
    log.info(f"  Output directory: {args.output_dir}")
    log.info(f"  Patch size: {args.patch_size}x{args.patch_size}")
    log.info(f"  Train fraction: {args.train_frac}")
    log.info(f"  Validation fraction: {args.val_frac}")
    log.info(f"  Test fraction: {1 - args.train_frac - args.val_frac}")
    log.info(f"  Seed: {args.seed}")
    if args.max_files:
        log.info(f"  Max files: {args.max_files}")
    if args.preserve_threshold is not None:
        log.info(f"  Preserve scatterers threshold: {args.preserve_threshold} dB")
    else:
        log.info("  Preserve scatterers: Disabled")
    if args.norm_mode is not None:
        log.info(f"  Normalization mode: {args.norm_mode}")
        log.info(f"  Min-max normalization percentile: {args.norm_minmax}%")
        log.info(f"  Clipping [0,1]: {'Enabled' if args.clip else 'Disabled'}")
    else:
        log.info("  Normalization: Disabled")
    log.info(f"  Verbose mode: {'Enabled' if args.verbose else 'Disabled'}")
    log.info("")

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Process the dataset
    start_time = datetime.now()
    log.info(
        f"Starting the creation of the dataset {dataset_name} at {start_time}",
        Colors.YELLOW,
    )

    success = process_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) / dataset_name,
        max_files=args.max_files,
        patch_size=args.patch_size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        preserve_threshold=args.preserve_threshold,
        norm_mode=args.norm_mode,
        norm_minmax_val=args.norm_minmax,
        clip=args.clip,
        verbose=args.verbose,
    )

    end_time = datetime.now()
    duration = end_time - start_time

    if success:
        log.info(f"{Colors.GREEN}Processing completed successfully!{Colors.RESET}")
        log.info(f"Started: {start_time}")
        log.info(f"Finished: {end_time}")
        log.info(f"Total duration: {duration}")
        return 0
    else:
        log.error(f"{Colors.RED}Processing completed with errors.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
