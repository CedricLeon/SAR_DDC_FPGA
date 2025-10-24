"""SAR Utilities for loading and preprocessing SAR data.

This module provides utility functions for SAR data handling:
- Loading CoSAR format files
- Symmetrization for zero Doppler centering
- Visualization functions for SAR images
"""

import struct
import warnings
from logging import Logger
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import signal


def convert_to_db(x: np.ndarray):
    """Convert input to decibels (dB)."""
    return 10 * np.log10(x + 1e-2)


def convert_from_db(x: np.ndarray):
    """Convert input from decibels (dB) to linear scale."""
    return pow(10, x / 10)


def load_cosar(path: Path, logger: Logger | None = None) -> np.ndarray | None:
    """Convert a CoSAR image to a numpy array. Function from MERLIN (originally named `cos2mat`)
    'improved' with Copilot.

    Args:
        path (Path): Path to the .cos file.
        logger (Logger | None): Logger instance for logging. Default is None.

    Returns:
        The image as a numpy array with dimensions [nlines, ncolumns, 2], where [:,:,0] is real part and [:,:,1] is imaginary part. None if the file could not be open.
    """
    try:
        fin = open(path, "rb")
    except OSError:
        if logger:
            logger.error(f"{path}: it is a not openable file")
            logger.error("Failed to call cos2mat")
        return None

    # Read header information
    ibib = struct.unpack(">i", fin.read(4))[0]
    irsri = struct.unpack(">i", fin.read(4))[0]
    irs = struct.unpack(">i", fin.read(4))[0]
    ias = struct.unpack(">i", fin.read(4))[0]
    ibi = struct.unpack(">i", fin.read(4))[0]
    irtnb = struct.unpack(">i", fin.read(4))[0]
    itnl = struct.unpack(">i", fin.read(4))[0]

    nlig = struct.unpack(">i", fin.read(4))[0]
    ncoltot = int(irtnb / 4)
    ncol = ncoltot - 2
    nlig = ias

    if logger:
        logger.info(f"      Reading image in CoSAR format. ncolumns={ncol} nlines={nlig}")

    # Reset file position and skip headers
    fin.seek(0)
    for _ in range(4):  # Skip 4 header lines
        firm = fin.read(4 * ncoltot)

    # Read image data
    imgcxs = np.empty([nlig, ncol], dtype=np.complex64)

    for iut in range(nlig):
        firm = fin.read(4 * ncoltot)
        if len(firm) < 4 * ncoltot:  # Check if we've reached EOF
            if logger:
                logger.warning(f"Warning: Reached EOF at line {iut}/{nlig}")
            break

        imgligne = np.ndarray(2 * ncoltot, ">h", firm)
        imgcxs[iut, :] = imgligne[4 : 2 * ncoltot : 2] + 1j * imgligne[5 : 2 * ncoltot : 2]

    fin.close()

    # Extract real and imaginary parts
    real_part = np.real(imgcxs)
    imag_part = np.imag(imgcxs)

    if logger:
        logger.info(
            f"      Successfully loaded image with shape: {real_part.shape} ([:,:,0] real and [:,:,1] imaginary)."
        )
    return np.stack((real_part, imag_part), axis=2)


def symmetrize(image: np.ndarray) -> np.ndarray:
    """Symmetrize the real and imaginary parts of the image and assure it's zero Doppler centered.

    Original function from MERLIN (called `symetrisation_patch_test`. Yes, with one 'm').
    Added logic to support my data format.
    Args:
        image: Input image with shape [H, W, 2] (real and imaginary parts)
    Returns:
        The symmetrized image with shape [H, W, 2]
    """
    # Reshape to match MERLIN's expected format: [h, w, 2] -> real and imag_part [1, h, w, 1]
    real = image[:, :, 0]
    imag = image[:, :, 1]
    real_part = real.reshape(1, *real.shape, 1)
    imag_part = imag.reshape(1, *imag.shape, 1)

    # -------------------------- MERLIN SYMETRIZATION --------------------------
    S = np.fft.fftshift(np.fft.fft2(real_part[0, :, :, 0] + 1j * imag_part[0, :, :, 0]))
    p = np.zeros(S.shape[0])  # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i, :]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = np.roll(p, shift_az_2)
    window = signal.windows.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symmetrized and zero-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)

    q = np.zeros(S.shape[1])  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:, j]))
    sq = q[::-1]
    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    shift_range_1 = int(round(-(d2 - 1) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_2 = np.roll(q, shift_range_2)
    window_r = signal.windows.gaussian(q.shape[0], std=0.2 * q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]

    Sf = np.roll(S2, int(shift_range * q.shape[0]), axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    # ima2 = ima2.reshape(1, np.size(ima2, 0), np.size(ima2, 1), 1)
    ####################################################################
    # Reshape back to [h, w, 2] format
    return np.stack((np.real(ima2), np.imag(ima2)), axis=2)


def preserve_point_like_scatterers(
    image2: np.ndarray, threshold_db: float = 60.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Preserve point-like scatterers in a TSX image. For pixels whose intensity is above a certain
    threshold, equally distribute their intensity between real and imaginary parts.

    Args:
        image2: Squared SAR image
        threshold_db: Threshold in dB for scatterer preservation

    Returns:
        Tuple of (preserved_patch, scatterer_mask) where preserved_patch is the processed image
    """
    warnings.warn(
        "preserve_point_like_scatterers() is not necessary. See https://ieeexplore.ieee.org/abstract/document/10021242."
    )
    warnings.warn(
        "This function is deprecated, MERLIN already conserves point-like scatterers. See See https://arxiv.org/abs/2207.11095."
    )

    real2_proc = image2[..., 0].copy()
    imag2_proc = image2[..., 1].copy()

    intensity = image2[..., 0] + image2[..., 1]
    intensity_db = convert_to_db(intensity)
    scatterer_mask = intensity_db > threshold_db

    # Value = sqrt(intensity/2), which gives half the power to each component
    scatterer_value = np.sqrt(intensity[scatterer_mask] / 2)
    real2_proc[scatterer_mask] = scatterer_value
    imag2_proc[scatterer_mask] = scatterer_value

    return np.stack((real2_proc, imag2_proc), axis=2), scatterer_mask


def extract_patches(image: np.ndarray, patch_size: int, stride: int | None = None) -> np.ndarray:
    """Extract patches of size patch_size from the input image.

    Args:
        image: Input image [H, W, 2]
        patch_size: Size of patches to extract (patch_size x patch_size)
        stride: Stride for extraction. Default: patch_size for no overlap

    Returns:
        An ndarray of patches [num_patches, patch_size, patch_size, 2]
    """
    assert image.ndim == 3, "Image must be 3D [H, W, 2]."
    if stride is None:
        stride = patch_size  # Default: no overlap

    patches = []
    h, w, _ = image.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i : i + patch_size, j : j + patch_size, :]
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch)
    return np.array(patches)
