"""
SAR Utilities for loading and preprocessing SAR data.

This module provides utility functions for SAR data handling:
- Loading CoSAR format files
- Symmetrization for zero Doppler centering
- Visualization functions for SAR images
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Quick ANSI color code shortcuts
r = "\033[31m"
y = "\033[33m"
g = "\033[32m"
b = "\033[34m"
e = "\033[0m"

# Constants for normalization from the log intensity of "random_split" (Genoa/Roma/Warsaw/Cologne/Hamburg), 30/04/2025
M = 50.32925033569336  # 95th percentile or 54.32472229003906 (99th)
m = 28.17565727233887  # 5th percentile or 20.96910095214844 (1st)


def normalize_sar(im):
    # np.spacing(1) is even smaller than 1e-12
    return ((np.log(im + np.spacing(1)).clip(min=0) - m) / (M - m)).astype("float32")


def denormalize_sar(im):
    return np.exp((M - m) * (np.squeeze(im)).astype("float32") + m)


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


def extract_patches(data, patch_size, stride=None):
    """Extract patches of size patch_size from the input data.

    Args:
        data: Input data array (2D or 3D)
        patch_size: Size of patches to extract (patch_size x patch_size)
        stride: Stride for extraction (default: patch_size for no overlap)

    Returns:
        List of extracted patches
    """
    if stride is None:
        stride = patch_size  # Default: no overlap

    patches = []

    # Check if we're dealing with 3D data (real+imag channels)
    if len(data.shape) == 3:
        h, w, _ = data.shape
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = data[i : i + patch_size, j : j + patch_size, :]
                if patch.shape[:2] == (patch_size, patch_size):
                    patches.append(patch)
        return np.array(patches)
    else:  # 2D data (intensity only)
        h, w = data.shape
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = data[i : i + patch_size, j : j + patch_size]
                if patch.shape == (patch_size, patch_size):
                    patches.append(patch)
        return np.array(patches)


def print_sar_statistics(name, data, with_intensity=True, indent=""):
    """Print statistics of some SAR data, 1D to 4D. Statististics are computed over all elements in the array, regardless of dimensionality.

    Args:
        name: Name of the data
        data: SAR data, 1D, 2D (assumed to be Intensity), 3D [h, w, 2] (assumed Real + Imaginary), or 4D [N, h, w, 2]
    """

    def print_statistics_obj(obj, general_indent, obj_name=None):
        indent = general_indent + "    " + obj_name + ": " if obj_name else "    "
        print(
            f"{indent}mean = {r}{np.mean(obj):.3f}{e}, std = {r}{np.std(obj):.3f}{e}, min = {r}{np.min(obj):.3f}{e}, max = {r}{np.max(obj):.3f}{e}"
        )

    print(f"{indent}Statistics for {b}{name}{e} {data.shape}:")
    if len(data.shape) == 1:
        print_statistics_obj(data, indent)
    elif len(data.shape) == 2:
        print_statistics_obj(data, indent, "Intensity")
    elif len(data.shape) == 3:
        print_statistics_obj(data[:, :, 0], indent, "Real")
        print_statistics_obj(data[:, :, 1], indent, "Imaginary")
        if with_intensity:
            print_statistics_obj(
                data[:, :, 0] ** 2 + data[:, :, 1] ** 2, indent, "Intensity"
            )
    elif len(data.shape) == 4:
        # Compute the statistics across the batch dimension
        print_statistics_obj(data[:, :, :, 0], indent, "Real")
        print_statistics_obj(data[:, :, :, 1], indent, "Imaginary")
        if with_intensity:
            print_statistics_obj(
                data[:, :, :, 0] ** 2 + data[:, :, :, 1] ** 2, indent, "Intensity"
            )
    else:
        print(
            f"{indent}{y}Warning unsupported format: {name} has above 4 dimensions, {r}{data.shape}{e}"
        )


def visualize_sar(
    real_part,
    imag_part,
    intensity=None,
    reflectivity=None,
    figsize=(12, 10),
    log_scale=True,
):
    """Visualize SAR data components.

    Args:
        real_part: Real part of SAR image
        imag_part: Imaginary part of SAR image
        intensity: Original intensity (optional)
        reflectivity: Reconstructed reflectivity (optional)
        figsize: Figure size (default: (12, 10))
        log_scale: Whether to use log scale (default: True)

    Returns:
        matplotlib figure
    """
    # Convert to numpy arrays
    if isinstance(real_part, torch.Tensor):
        real_part = real_part.detach().cpu().numpy()
    if isinstance(imag_part, torch.Tensor):
        imag_part = imag_part.detach().cpu().numpy()
    if isinstance(intensity, torch.Tensor) and intensity is not None:
        intensity = intensity.detach().cpu().numpy()
    if isinstance(reflectivity, torch.Tensor) and reflectivity is not None:
        reflectivity = reflectivity.detach().cpu().numpy()

    # Remove singleton dimensions
    if real_part.ndim > 2 and real_part.shape[0] == 1:
        real_part = real_part[0]
    if imag_part.ndim > 2 and imag_part.shape[0] == 1:
        imag_part = imag_part[0]
    if intensity is not None and intensity.ndim > 2 and intensity.shape[0] == 1:
        intensity = intensity[0]
    if (
        reflectivity is not None
        and reflectivity.ndim > 2
        and reflectivity.shape[0] == 1
    ):
        reflectivity = reflectivity[0]

    # Calculate power if not provided
    if intensity is None:
        intensity = real_part**2 + imag_part**2

    # Count number of subplots needed
    n_plots = 3 if reflectivity is None else 4

    # Create figure with subplots
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)

    # Plot real part
    axs[0].imshow(real_part, cmap="gray")
    axs[0].set_title("Real Part")
    axs[0].axis("off")

    # Plot imaginary part
    axs[1].imshow(imag_part, cmap="gray")
    axs[1].set_title("Imaginary Part")
    axs[1].axis("off")

    # Plot original intensity
    if log_scale:
        # Apply log transformation for better visualization
        intensity_log = np.log(intensity + np.spacing(1))
        im = axs[2].imshow(intensity_log, cmap="gray")
    else:
        im = axs[2].imshow(intensity, cmap="gray")
    axs[2].set_title("Intensity (Original)")
    axs[2].axis("off")

    # Plot reflectivity if available
    if reflectivity is not None:
        if log_scale:
            # Apply log transformation for better visualization
            reflectivity_log = np.log(reflectivity + 1e-10)
            im = axs[3].imshow(reflectivity_log, cmap="gray")
        else:
            im = axs[3].imshow(reflectivity, cmap="gray")
        axs[3].set_title("Reflectivity (Despeckled)")
        axs[3].axis("off")

    # Add colorbar
    fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def calculate_equivalent_number_of_looks(reflectivity, intensity):
    """Calculate equivalent number of looks (ENL) for despeckling quality evaluation.

    Args:
        reflectivity: Despeckled reflectivity estimate
        intensity: Original intensity

    Returns:
        ENL value
    """
    # Convert to numpy arrays
    if isinstance(reflectivity, torch.Tensor):
        reflectivity = reflectivity.detach().cpu().numpy()
    if isinstance(intensity, torch.Tensor):
        intensity = intensity.detach().cpu().numpy()

    # Remove singleton dimensions
    if reflectivity.ndim > 2 and reflectivity.shape[0] == 1:
        reflectivity = reflectivity[0]
    if intensity.ndim > 2 and intensity.shape[0] == 1:
        intensity = intensity[0]

    # Calculate statistics in a homogeneous region (center patch)
    h, w = reflectivity.shape
    center_h, center_w = h // 2, w // 2
    patch_size = min(h, w) // 4

    h_start, h_end = center_h - patch_size, center_h + patch_size
    w_start, w_end = center_w - patch_size, center_w + patch_size

    # Extract patches
    reflectivity_patch = reflectivity[h_start:h_end, w_start:w_end]
    intensity_patch = intensity[h_start:h_end, w_start:w_end]

    # Calculate ENL
    enl_orig = np.mean(intensity_patch) ** 2 / np.var(intensity_patch)
    enl_desp = np.mean(reflectivity_patch) ** 2 / np.var(reflectivity_patch)

    return {
        "enl_original": enl_orig,
        "enl_despeckled": enl_desp,
        "improvement": enl_desp / enl_orig,
    }


def save_anomaly_visualization(input, target, reconstruction, filename, info=None):
    """Save visualization and information for anomalous batches.

    Args:
        input: Input tensor [B, H, W] or [B, H, W, C]
        target: Target tensor [B, H, W] or [B, H, W, C]
        reconstruction: Reconstructed output [B, H, W] or [B, H, W, C]
        filename: Base filename to save outputs
        info: Dictionary containing additional information about the anomaly
    """

    # Convert tensors to numpy if needed
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    input_np = to_numpy(input)
    target_np = to_numpy(target)
    reconstruction_np = to_numpy(reconstruction)

    # Save detailed information to a text file
    with open(f"{filename}_info.txt", "w") as f:
        f.write("===== ANOMALY DETECTED =====\n\n")

        if info:
            f.write(f"Reason: {info.get('reason', 'Unknown')}\n")
            f.write(f"Epoch: {info.get('current_epoch', 'Unknown')}\n")
            f.write(f"Global step: {info.get('global_step', 'Unknown')}\n\n")

            # Print metrics
            f.write("Metrics:\n")
            metrics = info.get("metrics", {})
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        # Save data statistics
        f.write("Data Statistics:\n")
        f.write("Input:\n")
        _write_statistics(f, input_np, indent="  ")

        f.write("Target:\n")
        _write_statistics(f, target_np, indent="  ")

        f.write("Reconstruction:\n")
        _write_statistics(f, reconstruction_np, indent="  ")

    # Create visualizations - select up to 4 samples if batched
    num_samples = min(4, input_np.shape[0]) if input_np.ndim > 3 else 1

    for i in range(num_samples):
        # Extract the sample
        if input_np.ndim > 3:
            sample_input = np.squeeze(input_np[i])
            sample_target = np.squeeze(target_np[i])
            sample_recon = np.squeeze(reconstruction_np[i])
        else:
            sample_input = np.squeeze(input_np)
            sample_target = np.squeeze(target_np)
            sample_recon = np.squeeze(reconstruction_np)

        # Create figure with multiple plots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Anomaly Detection - {info.get('reason', '')}", fontsize=16)

        # Row 1: Original data
        _plot_sample(axs[0, 0], sample_input, "Input")
        _plot_sample(axs[0, 1], sample_target, "Target")
        _plot_sample(axs[0, 2], sample_recon, "Reconstruction")

        # Row 2: Difference and error maps
        _plot_difference(
            axs[1, 0], sample_input, sample_target, "Input-Target Difference"
        )
        _plot_difference(
            axs[1, 1], sample_recon, sample_target, "Recon-Target Difference"
        )
        _plot_error_map(
            axs[1, 2], sample_recon, sample_target, "Error Map (Recon vs Target)"
        )

        plt.tight_layout()
        sample_suffix = f"_sample{i}" if num_samples > 1 else ""
        plt.savefig(f"{filename}{sample_suffix}.png", dpi=150)
        plt.close(fig)


def _write_statistics(file, data, indent=""):
    """Write statistical information about the data to a file."""
    file.write(f"{indent}Shape: {data.shape}\n")
    file.write(f"{indent}Mean: {np.mean(data):.6f}\n")
    file.write(f"{indent}Std: {np.std(data):.6f}\n")
    file.write(f"{indent}Min: {np.min(data):.6f}\n")
    file.write(f"{indent}Max: {np.max(data):.6f}\n")
    file.write(f"{indent}NaN count: {np.isnan(data).sum()}\n")
    file.write(f"{indent}Inf count: {np.isinf(data).sum()}\n\n")


def _plot_sample(ax, data, title):
    """Plot a single sample on the given axis."""
    if data.ndim > 2 and data.shape[-1] == 2:
        # Complex data (real + imaginary)
        intensity = data[..., 0] + data[..., 1]
        ax.imshow(intensity, cmap="gray")
    else:
        # Real data
        ax.imshow(data, cmap="gray")

    ax.set_title(title)
    ax.axis("off")


def _plot_difference(ax, data1, data2, title):
    """Plot difference between two samples."""
    # Handle complex data
    if data1.ndim > 2 and data1.shape[-1] == 2:
        # For complex data, compute difference of intensities
        intensity1 = data1[..., 0] + data1[..., 1]
        intensity2 = data2[..., 0] + data2[..., 1]
        diff = intensity1 - intensity2
    else:
        diff = data1 - data2

    im = ax.imshow(diff, cmap="gray", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")


def _plot_error_map(ax, data1, data2, title):
    """Plot error map between reconstruction and target."""
    # Handle complex data
    if data1.ndim > 2 and data1.shape[-1] == 2:
        # For complex data, compute MSE between intensities
        intensity1 = data1[..., 0] + data1[..., 1]
        intensity2 = data2[..., 0] + data2[..., 1]
        error_map = (intensity1 - intensity2) ** 2
    else:
        error_map = (data1 - data2) ** 2

    im = ax.imshow(error_map, cmap="jet", norm=plt.Normalize(0, error_map.max()))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")
