"""
SAR Utilities for TerraSAR-X Image Processing

This module contains utilities for processing Synthetic Aperture Radar (SAR) images,
particularly TerraSAR-X data in CoSAR format.

Some functions are re-implemented from MERLIN (https://github.com/EarthObservation/MERLIN)
"""

import os
import numpy as np
import struct
from PIL import Image
from scipy import special, signal


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
# These constants are from MERLIN's utils.py
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle


def normalize_sar(im):
    """
    Normalize SAR image using logarithmic transformation.
    
    Re-implemented from MERLIN-TSX-stripmap-test/utils.py
    
    Args:
        im: Input SAR image (intensity)
        
    Returns:
        Normalized image in the range [0, 255]
    """
    return ((np.log(im + np.spacing(1)) - m) * 255 / (M - m)).astype('float32')


def denormalize_sar(im):
    """
    Denormalize SAR image (inverse of normalize_sar).
    
    Re-implemented from MERLIN-TSX-stripmap-test/utils.py
    
    Args:
        im: Normalized SAR image
        
    Returns:
        Denormalized image (intensity)
    """
    return np.exp((M - m) * (np.squeeze(im)).astype('float32') + m)


def symetrisation_patch(real_part, imag_part):
    """
    Apply symmetrization to center the Doppler spectrum (Zero Doppler Centering).
    
    Re-implemented from MERLIN-TSX-stripmap-test/utils.py: symetrisation_patch_test
    
    Args:
        real_part: Real part of the complex SAR image, shape [1, height, width, 1]
        imag_part: Imaginary part of the complex SAR image, shape [1, height, width, 1]
        
    Returns:
        Tuple (real_part, imag_part) of the symmetrized image
    """
    # Create complex image from real and imaginary parts
    S = np.fft.fftshift(np.fft.fft2(real_part[0,:,:,0] + 1j * imag_part[0,:,:,0]))
    
    # Azimuth symmetrization
    p = np.zeros((S.shape[0]))  # azimuth (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i,:]))
    
    # Reversed sequence for correlation
    sp = p[::-1]
    
    # Compute correlation
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    
    # Calculate shift options
    shift_az_1 = int(round(-(d1-1)/2)) % p.shape[0] + int(p.shape[0]/2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(round(-(d1-1-p.shape[0])/2)) % p.shape[0] + int(p.shape[0]/2)
    p2_2 = np.roll(p, shift_az_2)
    
    # Select best shift using Gaussian window
    window = signal.windows.gaussian(p.shape[0], std=0.2*p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    
    # Choose the shift that maximizes correlation with Gaussian window
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    
    # Apply azimuth shift
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)
    
    # Range symmetrization
    q = np.zeros((S.shape[1]))  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:,j]))
    
    # Reversed sequence for correlation
    sq = q[::-1]
    
    # Compute correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    
    # Calculate shift options
    shift_range_1 = int(round(-(d2-1)/2)) % q.shape[0] + int(q.shape[0]/2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(round(-(d2-1-q.shape[0])/2)) % q.shape[0] + int(q.shape[0]/2)
    q2_2 = np.roll(q, shift_range_2)
    
    # Select best shift using Gaussian window
    window_r = signal.windows.gaussian(q.shape[0], std=0.2*q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    
    # Choose the shift that maximizes correlation with Gaussian window
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]
    
    # Apply range shift
    Sf = np.roll(S2, int(shift_range * q.shape[0]), axis=1)
    
    # Convert back to spatial domain
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    ima2 = ima2.reshape(1, np.size(ima2, 0), np.size(ima2, 1), 1)
    
    return np.real(ima2), np.imag(ima2)


def cos2mat(image_file):
    """
    Load a CoSAR format SAR image.
    
    Re-implemented from MERLIN/load_cosar.py
    
    Args:
        image_file: Path to the CoSAR file
        
    Returns:
        SAR image data with shape [nlines, ncolumns, 2]
        where [:,:,0] contains the real part and [:,:,1] contains the imaginary part
    """
    print('Converting CoSAR to numpy array of size [ncolumns,nlines,2]')

    try:
        fin = open(image_file, 'rb')
    except IOError:
        legx = image_file + ': it is a not openable file'
        print(legx)
        print('failed to call cos2mat')
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
    
    # Calculate dimensions
    ncoltot = int(irtnb / 4)
    ncol = ncoltot - 2
    nlig = ias  # Use ias as number of lines

    print(f'Reading image in CoSAR format. ncolumns={ncol} nlines={nlig}')

    # Initialize arrays
    firm = np.zeros(4 * ncoltot, dtype=np.byte)
    imgcxs = np.empty([nlig, ncol], dtype=np.complex64)

    # Skip header
    fin.seek(0)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    
    # Read data
    for iut in range(nlig):
        firm = fin.read(4 * ncoltot)
        imgligne = np.ndarray(2 * ncoltot, '>h', firm)
        imgcxs[iut, :] = imgligne[4:2 * ncoltot:2] + 1j * imgligne[5:2 * ncoltot:2]

    fin.close()
    
    print('[:,:,0] contains the real part of the SLC image data')
    print('[:,:,1] contains the imaginary part of the SLC image data')
    return np.stack((np.real(imgcxs), np.imag(imgcxs)), axis=2)


def store_data_and_plot(im, threshold, filename):
    """
    Store data and generate visualization.
    
    Re-implemented from MERLIN-TSX-stripmap-test/utils.py
    
    Args:
        im: Image data
        threshold: Clipping threshold
        filename: Output filename
    """
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))


def multilook(image, n_looks):
    """
    Apply multi-look processing to reduce speckle.
    
    Args:
        image: Input SAR image
        n_looks: Number of looks (int or tuple of (row_looks, col_looks))
        
    Returns:
        Multi-looked image
    """
    # Ensure n_looks is a tuple with 2 elements (row_looks, col_looks)
    if isinstance(n_looks, int):
        n_looks = (n_looks, n_looks)
    
    # Calculate new dimensions
    rows, cols = image.shape
    new_rows = rows // n_looks[0]
    new_cols = cols // n_looks[1]
    
    # Reshape and average
    reshaped = image[:new_rows*n_looks[0], :new_cols*n_looks[1]]
    reshaped = reshaped.reshape(new_rows, n_looks[0], new_cols, n_looks[1])
    return reshaped.mean(axis=(1, 3))


def extract_patch(image, start_row, start_col, patch_size=256):
    """
    Extract a patch from an image.
    
    Args:
        image: Input image
        start_row: Starting row index
        start_col: Starting column index
        patch_size: Size of the patch
        
    Returns:
        Extracted patch
    """
    end_row = min(start_row + patch_size, image.shape[0])
    end_col = min(start_col + patch_size, image.shape[1])
    return image[start_row:end_row, start_col:end_col]


