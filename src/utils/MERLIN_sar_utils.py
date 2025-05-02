"""
Original functions from MERLIN  (https://gitlab.telecom-paris.fr/ring/MERLIN): do not MODIFY
SAR Utilities for TerraSAR-X Image Processing
"""

import struct

import numpy as np
from scipy import signal, special

# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645  # Probably empirical 90% percentile of MERLIN's training set
m = -1.429329123112601  # Probably empirical 10% percentile of MERLIN's training set
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle


# From MERLIN: utils.py (removed / 255)
def normalize_sar(im):
    print(
        f"Warning: deprecated normalization function using MERLIN's constants: {m} and {M}. Prefer to use sar_utils.py version."
    )
    return ((np.log(im + np.spacing(1)) - m) / (M - m)).astype("float32")


# From MERLIN:  self.Y_input in `__init__()` in model.py # The 2 times comes from the power in the log
def normalize_sar_as_network_input(im):
    return ((np.log(im**2 + 1e-3) - 2 * m) / 2 * (M - m)).astype("float32")


def denormalize_sar(im):
    return np.exp((M - m) * (np.squeeze(im)).astype("float32") + m)


def symetrisation_patch_test(real_part, imag_part):
    S = np.fft.fftshift(np.fft.fft2(real_part[0, :, :, 0] + 1j * imag_part[0, :, :, 0]))
    p = np.zeros((S.shape[0]))  # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i, :]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(
        p.shape[0] / 2
    )
    p2_2 = np.roll(p, shift_az_2)
    window = signal.windows.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symetrized and zero-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)

    q = np.zeros((S.shape[1]))  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:, j]))
    sq = q[::-1]
    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    shift_range_1 = int(round(-(d2 - 1) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(
        q.shape[0] / 2
    )
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
    ima2 = ima2.reshape(1, np.size(ima2, 0), np.size(ima2, 1), 1)
    return np.real(ima2), np.imag(ima2)


def cos2mat(path_to_cosar_image: str, verbose: bool = True):
    """Convert a CoSAR image to a numpy array. Function from MERLIN 'improved' with Copilot.

    Args:
        path_to_cosar_image (str): path to the image which is a cos file

    Returns:
        numpy array: The image as a numpy array with dimensions [nlines, ncolumns, 2],
                    where [:,:,0] is real part and [:,:,1] is imaginary part
    """
    try:
        fin = open(path_to_cosar_image, "rb")
    except IOError:
        print(f"{path_to_cosar_image}: it is a not openable file")
        print("Failed to call cos2mat")
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

    if verbose:
        print(f"        Reading image in CoSAR format. ncolumns={ncol} nlines={nlig}")

    # Reset file position and skip headers
    fin.seek(0)
    for _ in range(4):  # Skip 4 header lines
        firm = fin.read(4 * ncoltot)

    # Read image data
    imgcxs = np.empty([nlig, ncol], dtype=np.complex64)

    for iut in range(nlig):
        firm = fin.read(4 * ncoltot)
        if len(firm) < 4 * ncoltot:  # Check if we've reached EOF
            print(f"Warning: Reached EOF at line {iut}/{nlig}")
            break

        imgligne = np.ndarray(2 * ncoltot, ">h", firm)
        imgcxs[iut, :] = (
            imgligne[4 : 2 * ncoltot : 2] + 1j * imgligne[5 : 2 * ncoltot : 2]
        )

    fin.close()

    # Extract real and imaginary parts
    real_part = np.real(imgcxs)
    imag_part = np.imag(imgcxs)

    if verbose:
        print(
            f"      Successfully loaded image with shape: {real_part.shape} ([:,:,0] real and [:,:,1] imaginary)."
        )
    return np.stack((real_part, imag_part), axis=2)
