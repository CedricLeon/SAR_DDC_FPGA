#!/usr/bin/env python3
"""
TSX_dataset_creation.py - Process SAR .cos files to HDF5 datasets

This script processes TerraSAR-X CoSAR format (.cos) images and converts them to
HDF5 format for training deep learning models. It performs preprocessing steps:
- Loading images from .cos files
- Patch extraction
- SAR preprocessing (symmetrization)
- Scatterer preservation
- Creating train/val/test splits
- Saving as HDF5 files and log-intensity histograms
All steps posses parameters for customization, see --help for details.

Usage:
    python TSX_dataset_creation.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR --mode {test|random_split|spatial_split}
"""

# Imports
import argparse
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

# Import original SAR utilities
from src.utils.MERLIN_sar_utils import (
    cos2mat,
    symetrisation_patch_test,
)
from src.utils.sar_utils import (
    extract_filepath_short_name,
    extract_patches,
    print_sar_statistics,
)


# Configure logging - simplified version
def setup_logging(output_dir, filename="dataset_creation.log", level=logging.INFO):
    """Set up logging to both console and file.

    Args:
        output_dir: Directory to save log file
        filename: Log filename
        level: Logging level

    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create file handler
    log_path = os.path.join(output_dir, filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatters
    # Simple formatter that just passes through the message
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)

    # File formatter without colors
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ANSI color codes for console output
class Colors:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


# Simple log function
def log(message, level="info", color=None):
    """Log a message with optional color.
    The color will appear in the console but not in the log file.

    Args:
        message: Message to log
        level: Logging level (info, warning, error, debug)
        color: Color to use for console output (from Colors class)
    """
    # Add color to the message for console display
    colored_message = (
        f"{color if color else ''}{message}{Colors.RESET if color else ''}"
    )

    # Log at the appropriate level
    if level == "info":
        logging.info(colored_message)
    elif level == "warning":
        logging.warning(colored_message)
    elif level == "error":
        logging.error(colored_message)
    elif level == "debug":
        logging.debug(colored_message)


def preserve_point_like_scatterers(real2, imag2, threshold_db=9):
    """Preserve point-like scatterers in SAR image above a certain threshold.

    Args:
        real2: Squared real part of SAR image
        imag2: Squared imaginary part of SAR image
        threshold_db: Threshold in dB for scatterer preservation (default: 9dB)

    Returns:
        Tuple of (restacked_data, scatterer_mask)
    """
    # Convert intensity to dB (10*log10(intensity)), add a small epsilon to avoid log(0)
    intensity = real2 + imag2
    intensity_db = 10 * np.log10(intensity + 1e-10)

    # Create mask for scatterers above threshold
    scatterer_mask = intensity_db > threshold_db

    # Create copies of the input arrays to avoid modifying originals
    real2_proc = real2.copy()
    imag2_proc = imag2.copy()

    # For pixels above threshold, assign the same value to both real and imaginary parts
    # Value = sqrt(intensity/2), which gives half the power to each component
    scatterer_value = np.sqrt(intensity[scatterer_mask] / 2)
    real2_proc[scatterer_mask] = scatterer_value
    imag2_proc[scatterer_mask] = scatterer_value

    return np.stack((real2_proc, imag2_proc), axis=2), scatterer_mask


def preprocess_sar_image(filepath, patch_size=256, preserve_scatterers_threshold=None):
    """Full preprocessing pipeline for a single SAR image using original functions.

    Args:
        filepath: Path to the SAR image file
        patch_size: Size of patches to extract

    Returns:
        Dictionary with processed data at different stages
    """
    short_name = extract_filepath_short_name(filepath)

    # 1. Load SAR data
    log("    1. Loading SAR data...")
    sar_data = cos2mat(str(filepath), verbose=False)
    if sar_data is None:
        raise ValueError(f"Failed to load {short_name}")

    # Store original data
    original_data = sar_data.copy()
    # print_sar_statistics("Original data", original_data, indent="        ")

    # 2. Extract patches with NO OVERLAP (stride=patch_size//2 for 50% overlap)
    log(
        f"    2. Extracting patches of size {patch_size}x{patch_size} with no overlap..."
    )
    original_patches = extract_patches(original_data, patch_size, stride=patch_size)
    log(
        f"          Extracted {len(original_patches)} patches of size {patch_size}x{patch_size}"
    )

    if len(original_patches) == 0:
        raise ValueError(f"No patches could be extracted from {short_name}")

    # 3. Apply symmetrization to each patch
    log("    3. Applying symmetrization to each patch...")
    symmetrized_patches = []
    for patch in original_patches:
        # Reshape to match MERLIN's expected format: [h, w, 2] -> real_ and imag_part [1, h, w, 1]
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
    # log(f"          Symmetrized {len(symmetrized_patches)} patches")
    print_sar_statistics("Symmetrized patches", symmetrized_patches, indent="        ")

    # 4. Square the real and imaginary parts
    log("    4. Squaring the real and imaginary parts of the patches...")
    symmetrized_patches_squared = np.square(symmetrized_patches)
    # print_sar_statistics(
    #     "Squared patches", symmetrized_patches_squared, with_intensity=False, indent="        "
    # )

    # 5. Preserve scatterers in each patch
    if preserve_scatterers_threshold:
        log("    5. Preserving point-like scatterers in each patch...")
        preserved_patches_squared = []
        scatterer_masks = []
        for patch in symmetrized_patches_squared:
            preserved_patch_squared, scatterer_mask = preserve_point_like_scatterers(
                patch[:, :, 0],
                patch[:, :, 1],
                threshold_db=preserve_scatterers_threshold,
            )
            preserved_patches_squared.append(preserved_patch_squared)
            scatterer_masks.append(scatterer_mask)

        preserved_patches_squared = np.array(preserved_patches_squared)
        scatterer_masks = np.array(scatterer_masks)
        _, nb_scatterer_preserved = np.unique(scatterer_masks, return_counts=True)
        log(
            f"          Strong scatterers preserved = {nb_scatterer_preserved[1]} (or {nb_scatterer_preserved[1] / nb_scatterer_preserved[0] * 100:.6f}%).",
            color=Colors.GREEN,
        )
        # print_sar_statistics(
        #     "Preserved patches", preserved_patches_squared, with_intensity=False, indent="        "
        # )

    # Return the results at different stages of the pipeline
    return {
        "original_data": original_data,
        "original_patches": original_patches,
        "symmetrized_patches": symmetrized_patches,
        "scatterer_masks": scatterer_masks if preserve_scatterers_threshold else None,
        "patches": preserved_patches_squared
        if preserve_scatterers_threshold
        else symmetrized_patches_squared,  # Final processed patches
    }


def write_hdf5(patches, statistics, path):
    """Write dataset patches to HDF5 at path.

    Args:
        patches: Array of patches to write
        statistics: Dictionary with statistics (mean, median, and various percentiles)
        path: Path to save HDF5 file

    Returns:
        bool: Success status
    """
    n_patches = len(patches)
    log(f"      Writing {n_patches} patches to {path}...")

    with h5py.File(path, "w") as f:
        f.create_dataset("patches", data=patches, dtype="float32")

        # Add metadata
        f.attrs["num_patches"] = n_patches
        f.attrs["mean"] = statistics["mean"]
        f.attrs["median"] = statistics["median"]
        f.attrs["p1"] = statistics["p1"]
        f.attrs["p5"] = statistics["p5"]
        f.attrs["p10"] = statistics["p10"]
        f.attrs["p90"] = statistics["p90"]
        f.attrs["p95"] = statistics["p95"]
        f.attrs["p99"] = statistics["p99"]
        f.attrs["patch_size"] = patches.shape[1]  # Assuming square patches
        f.attrs["creation_date"] = str(datetime.now())
    return True


def plot_intensity_histogram(
    data, output_path, title="Log-Intensity Histogram", is_squared=True
):
    """Plot histogram of log-intensity values and print key statistics.

    Args:
        data: SAR data in [h, w, 2] format (single image) or [n, h, w, 2] format (batch of patches)
        output_path: Path to save the histogram plot
        title: Title for the plot
        is_squared: If True, assumes the data is already squared (e.g., preserved_patches)

    Returns:
        Dictionary with statistics
    """
    # Check if we're dealing with a batch of patches or a single image
    # log(f"Data shape: {data.shape}, is_squared: {is_squared}")
    is_batch = len(data.shape) == 4  # [n, h, w, 2] format

    if is_batch:
        # log(f"Processing batch of {len(data)} patches for histogram...")

        intensity_values = []
        for patch in data:
            if is_squared:
                intensity = patch[:, :, 0] + patch[:, :, 1]
            else:
                intensity = patch[:, :, 0] ** 2 + patch[:, :, 1] ** 2
            # Flatten and append to our list
            intensity_values.append(intensity.flatten())
        # Concatenate all flattened arrays into a single 1D array
        intensity_all = np.concatenate(intensity_values)
    else:
        if is_squared:
            intensity = data[:, :, 0] + data[:, :, 1]
        else:
            intensity = data[:, :, 0] ** 2 + data[:, :, 1] ** 2
        intensity_all = intensity.flatten()

    # Convert to dB (log scale)
    epsilon = 1e-10  # Small value to avoid log(0)
    log_intensity = 10 * np.log10(intensity_all + epsilon)

    # Calculate statistics
    statistics = {
        "mean": np.mean(log_intensity),
        "median": np.median(log_intensity),
        "min": np.min(log_intensity),
        "p1": np.percentile(log_intensity, 1),
        "p5": np.percentile(log_intensity, 5),
        "p10": np.percentile(log_intensity, 10),
        "p90": np.percentile(log_intensity, 90),
        "p95": np.percentile(log_intensity, 95),
        "p99": np.percentile(log_intensity, 99),
        "max": np.max(log_intensity),
    }

    # Print statistics
    log(f"      {title} [dB] statistics:")
    for key, value in statistics.items():
        log(f"        - {key.capitalize()}: {value:.2f} dB")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    hist, bins, _ = plt.hist(log_intensity, bins=100, color="blue", alpha=0.7)

    # Add vertical lines for key statistics
    plt.axvline(
        float(statistics["mean"]),
        color="r",
        linestyle="--",
        label=f"Mean ({statistics['mean']:.2f} dB)",
    )
    plt.axvline(
        float(statistics["median"]),
        color="g",
        linestyle="--",
        label=f"Median ({statistics['median']:.2f} dB)",
    )
    plt.axvline(
        float(statistics["p5"]),
        color="orange",
        linestyle=":",
        label=f"5th percentile ({statistics['p5']:.2f} dB)",
    )
    plt.axvline(
        float(statistics["p95"]),
        color="orange",
        linestyle=":",
        label=f"95th percentile ({statistics['p95']:.2f} dB)",
    )

    # Add labels and title
    plt.xlabel("Log-Intensity (dB)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    log(f"      Saved histogram to {output_path}")
    plt.close()

    # Return statistics dictionary for potential further use
    return statistics


def process_all_files(file_paths, patch_size=256, max_files=None):
    """Process all SAR files and extract patches.

    Args:
        file_paths: List of file paths to process
        patch_size: Size of patches to extract
        max_files: Maximum number of files to process (optional)

    Returns:
        List of all extracted patches
    """
    all_patches = []
    files_to_process = file_paths[:max_files] if max_files else file_paths
    log(f"Processing {len(files_to_process)} files...", color=Colors.BLUE)

    for file_path in tqdm(files_to_process, desc="Extracting patches from files"):
        short_name = extract_filepath_short_name(file_path)
        results = preprocess_sar_image(file_path, patch_size=patch_size)
        file_patches = results["patches"]  # The fully pre-processed patches

        log(
            f"  Extracted {len(file_patches)} patches from {short_name}",
            color=Colors.BLUE,
        )
        all_patches.append(file_patches)

    # Combine all patches
    if all_patches:
        combined_patches = np.vstack(all_patches)
        log(f"Total patches collected: {len(combined_patches)}")
        return combined_patches
    else:
        log("No valid patches processed.", level="error")
        return None


def split_patches(patches, train_frac=0.8, val_frac=0.1, seed=42):
    """Split patches into train, validation, and test sets.

    Args:
        patches: Array of patches to split
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        Dictionary with train, val, and test patch arrays
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Calculate number of patches for each split
    n_samples = len(patches)
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    n_test = n_samples - n_train - n_val

    log(f"Splitting {n_samples} patches into:")
    log(f"  - Training: {n_train} patches ({train_frac * 100:.1f}%)")
    log(f"  - Validation: {n_val} patches ({val_frac * 100:.1f}%)")
    log(f"  - Testing: {n_test} patches ({(1 - train_frac - val_frac) * 100:.1f}%)")

    # Create random permutation of indices
    indices = np.random.permutation(n_samples)

    # Split the data
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # Create the splits
    train_patches = patches[train_indices]
    val_patches = patches[val_indices]
    test_patches = patches[test_indices]

    return {"train": train_patches, "val": val_patches, "test": test_patches}


def verify_hdf5(file_path):
    """Verify HDF5 file can be read and return summary.

    Args:
        file_path: Path to HDF5 file

    Returns:
        bool: Success status
    """
    if not os.path.exists(file_path):
        log(f"File {file_path} does not exist!", level="error")
        return False

    with h5py.File(file_path, "r") as f:
        log(f"HDF5 file: {file_path}")
        log(f"Keys: {list(f.keys())}")
        log(f"Attributes: {dict(f.attrs)}")
        log("Dataset shapes:")
        for key in f.keys():
            log(f"  {key}: {f[key].shape}")
        return True


def split_dataset_by_file(file_paths, test_ratio=0.2, val_ratio=0.1, seed=42):
    """Split dataset into train, val, and test sets based on files.

    Args:
        file_paths: List of file paths
        test_ratio: Ratio of files to use for testing
        val_ratio: Ratio of files to use for validation
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', and 'test' lists of files
    """
    # Set seed for reproducibility
    random.seed(seed)

    # Shuffle the file paths deterministically
    shuffled_files = file_paths.copy()
    random.shuffle(shuffled_files)

    # Calculate the number of files for each split
    n_files = len(shuffled_files)
    n_test = max(1, int(n_files * test_ratio))
    n_val = max(1, int(n_files * val_ratio))
    n_train = n_files - n_test - n_val

    # Split the files
    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train : n_train + n_val]
    test_files = shuffled_files[n_train + n_val :]

    log(f"Split {n_files} files into:")
    log(f"  - Train: {len(train_files)} files")
    log(f"  - Validation: {len(val_files)} files")
    log(f"  - Test: {len(test_files)} files")

    return {"train": train_files, "val": val_files, "test": test_files}


def run_test_mode(
    input_dir,
    output_dir,
    patch_size=256,
    file_number=0,
    preserve_scatterers_threshold=None,
):
    """Run the test mode: process one image and verify the results.

    Args:
        input_dir: Directory containing .cos files
        output_dir: Directory to save results
        patch_size: Size of patches to extract
        file_number: Index of the file to process (default: 0 = first file)

    Returns:
        bool: Success status
    """
    log("Running in TEST mode", color=Colors.YELLOW)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find .cos files
    input_path = Path(input_dir)
    cos_files = list(input_path.glob("*.cos"))

    assert len(cos_files) > 0, f"No .cos files found in {input_dir}"

    # Make sure file_number is within range
    if file_number >= len(cos_files):
        log(
            f"Warning: file_number {file_number} exceeds the number of files ({len(cos_files)}). Using the first file instead.",
            level="warning",
        )
        file_number = 0

    # Process the specified file
    test_file = cos_files[file_number]
    short_name = extract_filepath_short_name(test_file)
    log(f"Testing with file {file_number}: {test_file}")

    # Process the test file
    results = preprocess_sar_image(test_file, patch_size, preserve_scatterers_threshold)

    # Create histogram
    output_path = Path(output_dir)
    hist_path = output_path / f"test_histogram_{short_name}.png"

    # Create and save histogram
    patches_statistics = plot_intensity_histogram(
        results["patches"],
        hist_path,
        title=f"Log-Intensity Histogram - {short_name}",
        is_squared=True,
    )

    # Save a sample to HDF5
    test_output = output_path / "test_sample.h5"
    write_hdf5(results["patches"], patches_statistics, test_output)

    # Verify the HDF5 file
    log("Verifying the created HDF5 file:")
    verify_hdf5(test_output)

    log("Test mode completed successfully!")
    return True


def run_random_split_mode(
    input_dir,
    output_dir,
    patch_size=256,
    train_frac=0.8,
    val_frac=0.1,
    seed=42,
    preserve_scatterers_threshold=None,
):
    """Run random split mode: process all images, merge and shuffle patches from all images together, then split. Not optimal for the generalization of the model + may spatially overfit, but fast implementation (and most likely better results)
    Memory-optimized version that processes files individually to reduce memory usage.

    Args:
        input_dir: Directory containing .cos files
        output_dir: Directory to save results
        patch_size: Size of patches to extract
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        bool: Success status
    """
    log("Running in RANDOM SPLIT mode", color=Colors.YELLOW)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find .cos files
    input_path = Path(input_dir)
    cos_files = list(input_path.glob("*.cos"))

    assert len(cos_files) > 0, f"No .cos files found in {input_dir}"

    log(f"\nFound {len(cos_files)} .cos files in {input_dir}")
    for file in cos_files:
        short_name = extract_filepath_short_name(file)
        log(f"  - {short_name}")

    # Create output directory paths
    output_path = Path(output_dir)
    patches_dir = output_path / "patches"
    os.makedirs(patches_dir, exist_ok=True)

    # Process each file individually and save split patches
    total_patches_train = 0
    total_patches_val = 0
    total_patches_test = 0
    file_info = []

    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # First pass: process each file separately and save patches
    for i, file_path in enumerate(cos_files):
        short_name = extract_filepath_short_name(file_path)
        log(f"\nProcessing {i}/{len(cos_files)}: {short_name}...", color=Colors.BLUE)

        # Process the file
        results = preprocess_sar_image(
            file_path, patch_size, preserve_scatterers_threshold
        )

        # Get the patches
        patches = results["patches"]
        num_patches = len(patches)

        if num_patches == 0:
            log(
                f"Warning: No patches extracted from {short_name}. Skipping.",
                level="warning",
            )
            continue

        # Calculate split sizes for this file
        num_train = int(num_patches * train_frac)
        num_val = int(num_patches * val_frac)
        num_test = num_patches - num_train - num_val

        # Update totals
        total_patches_train += num_train
        total_patches_val += num_val
        total_patches_test += num_test

        # Create random permutation for this file
        indices = np.random.permutation(num_patches)
        train_indices = indices[:num_train]
        val_indices = indices[num_train : num_train + num_val]
        test_indices = indices[num_train + num_val :]

        # Save patches to separate files to avoid keeping everything in memory
        train_file = patches_dir / f"{short_name}_train.h5"
        val_file = patches_dir / f"{short_name}_val.h5"
        test_file = patches_dir / f"{short_name}_test.h5"

        # Create and save histograms for this file's patches
        hist_path = output_path / f"{short_name}_histogram.png"
        file_stats = plot_intensity_histogram(
            patches, hist_path, title=f"{short_name} Log-Intensity", is_squared=True
        )

        # Save train patches
        if num_train > 0:
            train_patches = patches[train_indices]
            write_hdf5(train_patches, file_stats, train_file)
            del train_patches  # Explicitly free memory

        # Save validation patches
        if num_val > 0:
            val_patches = patches[val_indices]
            write_hdf5(val_patches, file_stats, val_file)
            del val_patches  # Explicitly free memory

        # Save test patches
        if num_test > 0:
            test_patches = patches[test_indices]
            write_hdf5(test_patches, file_stats, test_file)
            del test_patches  # Explicitly free memory

        # Record file info for second pass
        file_info.append(
            {
                "name": short_name,
                "train_file": train_file if num_train > 0 else None,
                "val_file": val_file if num_val > 0 else None,
                "test_file": test_file if num_test > 0 else None,
                "num_train": num_train,
                "num_val": num_val,
                "num_test": num_test,
            }
        )

        # Force garbage collection to free memory
        del patches
        del results
        import gc

        gc.collect()

    # Second pass: combine patches from all files for each split
    total_patches = total_patches_train + total_patches_val + total_patches_test
    log(
        f"\nProcessed all files. Total patches: {total_patches}. Combining patches from all files..."
    )

    # Function to merge patches from multiple files
    def merge_and_save_split(split_name, file_list, output_file):
        if not file_list:
            log(f"No files for {split_name} split. Skipping.")
            return None
        # log(f"Combining {len(file_list)} files for {split_name} split...")

        # Initialize combined statistics
        combined_stats = None

        # Create the output dataset using chunks
        with h5py.File(output_file, "w") as out_f:
            # First determine total size
            total_patches = sum(
                h5py.File(f, "r")["patches"].shape[0] for f in file_list
            )

            # Get shape of a single patch for dataset creation
            with h5py.File(file_list[0], "r") as sample_f:
                sample_shape = sample_f["patches"].shape[1:]

            # Create extensible dataset with chunks
            chunk_size = min(
                1000, total_patches
            )  # Adjust chunk size based on your data
            patches_dset = out_f.create_dataset(
                "patches",
                shape=(0,) + sample_shape,
                maxshape=(total_patches,) + sample_shape,
                dtype="float32",
                chunks=(chunk_size,) + sample_shape,
            )

            # Add patches from each file
            start_idx = 0
            intensity_values = []

            for i, file_path in enumerate(file_list):
                with h5py.File(file_path, "r") as in_f:
                    file_patches = in_f["patches"][:]
                    num_file_patches = len(file_patches)

                    # Resize dataset to accommodate new patches
                    patches_dset.resize(start_idx + num_file_patches, axis=0)

                    # Copy patches to output file
                    patches_dset[start_idx : start_idx + num_file_patches] = (
                        file_patches
                    )

                    # Calculate intensity for statistics
                    for j in range(
                        0, num_file_patches, max(1, num_file_patches // 10)
                    ):  # Sample ~10% for stats
                        patch = file_patches[j]
                        intensity = (
                            patch[:, :, 0] + patch[:, :, 1]
                        )  # Assuming squared values
                        intensity_values.append(intensity.flatten())

                    # Update start index
                    start_idx += num_file_patches

                    # Copy attributes if this is the first file
                    if i == 0:
                        for attr_name, attr_value in in_f.attrs.items():
                            if (
                                attr_name != "num_patches"
                            ):  # We'll update this at the end
                                out_f.attrs[attr_name] = attr_value

                # Free memory
                del file_patches
                gc.collect()

            # Calculate statistics on the combined dataset
            if intensity_values:
                intensity_all = np.concatenate(
                    [sample for sample in intensity_values if len(sample) > 0]
                )
                log_intensity = 10 * np.log10(intensity_all + 1e-10)

                combined_stats = {
                    "mean": float(np.mean(log_intensity)),
                    "median": float(np.median(log_intensity)),
                    "p1": float(np.percentile(log_intensity, 1)),
                    "p5": float(np.percentile(log_intensity, 5)),
                    "p10": float(np.percentile(log_intensity, 10)),
                    "p90": float(np.percentile(log_intensity, 90)),
                    "p95": float(np.percentile(log_intensity, 95)),
                    "p99": float(np.percentile(log_intensity, 99)),
                }

                # Update dataset attributes with combined statistics
                for key, value in combined_stats.items():
                    out_f.attrs[key] = value

                # Update num_patches attribute
                out_f.attrs["num_patches"] = total_patches
                out_f.attrs["creation_date"] = str(datetime.now())

                # log(f"  {split_name} split has {total_patches} patches")
                log(f"  Mean intensity: {combined_stats['mean']:.2f} dB")

        return combined_stats

    # Merge train, val, and test splits
    train_output = output_path / "train.h5"
    val_output = output_path / "val.h5"
    test_output = output_path / "test.h5"

    # Get lists of files for each split
    train_files = [
        info["train_file"] for info in file_info if info["train_file"] is not None
    ]
    val_files = [info["val_file"] for info in file_info if info["val_file"] is not None]
    test_files = [
        info["test_file"] for info in file_info if info["test_file"] is not None
    ]

    # Merge and save each split
    log(
        f"  - Training: {total_patches_train} patches ({(total_patches_train / total_patches) * 100:.4f}%)"
    )
    train_stats = merge_and_save_split("Training", train_files, train_output)
    log(
        f"  - Validation: {total_patches_val} patches ({(total_patches_val / total_patches) * 100:.4f}%)"
    )
    val_stats = merge_and_save_split("Validation", val_files, val_output)
    log(
        f"  - Testing: {total_patches_test} patches ({(total_patches_test / total_patches) * 100:.4f}%)"
    )
    test_stats = merge_and_save_split("Testing", test_files, test_output)

    # Create combined histograms
    train_hist = output_path / "train_histogram.png"
    val_hist = output_path / "val_histogram.png"
    test_hist = output_path / "test_histogram.png"

    # Function to create histogram from statistics
    def create_histogram_from_stats(stats, output_path, title):
        if not stats:
            log(f"No statistics available for {title}. Skipping histogram.")
            return

        plt.figure(figsize=(10, 6))

        # Draw vertical lines for key statistics
        plt.axvline(
            stats["mean"],
            color="r",
            linestyle="--",
            label=f"Mean ({stats['mean']:.2f} dB)",
        )
        plt.axvline(
            stats["median"],
            color="g",
            linestyle="--",
            label=f"Median ({stats['median']:.2f} dB)",
        )
        plt.axvline(
            stats["p5"],
            color="orange",
            linestyle=":",
            label=f"5th percentile ({stats['p5']:.2f} dB)",
        )
        plt.axvline(
            stats["p95"],
            color="orange",
            linestyle=":",
            label=f"95th percentile ({stats['p95']:.2f} dB)",
        )

        # Add labels and title
        plt.xlabel("Log-Intensity (dB)")
        plt.ylabel("Frequency (approximate)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Set x-axis limits based on the 1st and 99th percentiles
        plt.xlim([stats["p1"] - 5, stats["p99"] + 5])

        plt.tight_layout()
        plt.savefig(output_path)
        log(f"  Saved histogram to {output_path}")
        plt.close()

    # Generate histogram plots if statistics are available
    log("\nCreating histograms for merged datasets...")
    create_histogram_from_stats(train_stats, train_hist, "Training Set Log-Intensity")
    create_histogram_from_stats(val_stats, val_hist, "Validation Set Log-Intensity")
    create_histogram_from_stats(test_stats, test_hist, "Test Set Log-Intensity")

    # # Verify the HDF5 files
    # log("\nVerifying the created HDF5 files:")
    # log("\nTrain set:")
    # verify_hdf5(train_output)
    # log("\nValidation set:")
    # verify_hdf5(val_output)
    # log("\nTest set:")
    # verify_hdf5(test_output)

    # Optionally, clean up temporary files
    log("\nCleaning up temporary files...", color=Colors.BLUE)
    for info in file_info:
        if info["train_file"] and os.path.exists(info["train_file"]):
            os.remove(info["train_file"])
        if info["val_file"] and os.path.exists(info["val_file"]):
            os.remove(info["val_file"])
        if info["test_file"] and os.path.exists(info["test_file"]):
            os.remove(info["test_file"])

    # Remove temp directory if it's empty
    try:
        os.rmdir(patches_dir)
        log(f"Removed temporary directory: {patches_dir}")
    except:
        log(f"Could not remove temporary directory: {patches_dir}")

    log("\nRandom split mode completed successfully!", color=Colors.GREEN)
    return True


def run_spatial_split_mode(
    input_dir,
    output_dir,
    patch_size=256,
    test_ratio=0.2,
    val_ratio=0.1,
    seed=42,
    preserve_scatterers_threshold=None,
):
    """Run spatial split mode: first split by files, then process each group.
    For this dataset the idea is to download many more tiles (like 20), first split each tile into their respective dataset, for example 15 train/valid and 5 test, and then build the datasets from maybe 15% of the patches of each tile.

    Args:
        input_dir: Directory containing .cos files
        output_dir: Directory to save results
        patch_size: Size of patches to extract
        test_ratio: Ratio of files to use for testing
        val_ratio: Ratio of files to use for validation
        seed: Random seed for reproducibility

    Returns:
        bool: Success status
    """
    log("\nSPATIAL SPLIT mode is not yet implemented.", level="error")
    log(
        "This mode will split the dataset spatially to avoid data leakage between train and test sets.",
        level="warning",
    )
    log("The implementation will be added in a future update.", level="warning")
    return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process SAR .cos files and create HDF5 datasets for training."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing .cos files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save HDF5 files"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "random_split", "spatial_split"],
        required=True,
        help="Processing mode",
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
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (optional)",
    )
    parser.add_argument(
        "--file-number",
        type=int,
        default=0,
        help="File index to process in test mode (default: 0)",
    )
    parser.add_argument(
        "--preserve-scatterers-threshold",
        type=float,
        default=None,
        help="Threshold above which point-like scatterers are preserved (default: None = no preservation)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.train_frac + args.val_frac > 1.0:
        log(
            "Warning: Train + validation fractions exceed 1.0. Adjusting values...",
            level="warning",
        )
        total = args.train_frac + args.val_frac
        args.train_frac /= total
        args.val_frac /= total
        log(
            f"Adjusted fractions: train={args.train_frac:.2f}, val={args.val_frac:.2f}, test={1 - args.train_frac - args.val_frac:.2f}"
        )

    # Print configuration
    log("\nSAR Dataset Creation - Configuration:")
    log(f"  Input directory: {args.input_dir}")
    log(f"  Output directory: {args.output_dir}")
    log(f"  Mode: {args.mode}")
    log(f"  Preserve point-like scatterers: {args.preserve_scatterers_threshold}")
    log(f"  Patch size: {args.patch_size}x{args.patch_size}")
    log(f"  Train fraction: {args.train_frac}")
    log(f"  Validation fraction: {args.val_frac}")
    log(f"  Test fraction: {1 - args.train_frac - args.val_frac}")
    log(f"  Random seed: {args.seed}")
    if args.file_number > 0:
        log(f"  File number for test mode: {args.file_number}")
    if args.max_files:
        log(f"  Max files: {args.max_files}")
    log("")

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize logging
    global logger
    logger = setup_logging(args.output_dir)

    # Run the selected mode
    if args.mode == "test":
        success = run_test_mode(
            args.input_dir,
            args.output_dir,
            args.patch_size,
            args.file_number,
            args.preserve_scatterers_threshold,
        )
    elif args.mode == "random_split":
        success = run_random_split_mode(
            args.input_dir,
            args.output_dir,
            args.patch_size,
            args.train_frac,
            args.val_frac,
            args.seed,
            args.preserve_scatterers_threshold,
        )
    elif args.mode == "spatial_split":
        success = run_spatial_split_mode(
            args.input_dir,
            args.output_dir,
            args.patch_size,
            1 - args.train_frac - args.val_frac,
            args.val_frac,
            args.seed,
            args.preserve_scatterers_threshold,
        )
    else:
        log(f"Invalid mode selected: {args.mode}", level="error")
        return 1

    if success:
        log("\nProcessing completed successfully!", color=Colors.GREEN)
        return 0
    else:
        log("\nProcessing completed with errors.", level="error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
