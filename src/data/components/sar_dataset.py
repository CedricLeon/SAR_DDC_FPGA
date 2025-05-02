"""
SAR Dataset for loading and preprocessing SAR images.

This module contains dataset classes for handling SAR images in both CoSAR format
and pre-processed HDF5 format with proper deterministic behavior.
"""

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class TSXSSCDataset(Dataset):
    """Dataset for pre-processed SAR patches in HDF5 format.

    This dataset loads pre-processed SAR patches from HDF5 files created
    by the TSX_dataset_creation.py script.
    """

    def __init__(self, hdf5_path: Path, transform=None):
        """Initialize the dataset.

        Args:
            hdf5_file: Path to the HDF5 file containing pre-processed patches
            transform: Optional transform to apply to samples (default: None)
        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform

        # Open the HDF5 file
        # We don't keep it open to avoid issues with multiprocessing
        # Just check that it exists and get the number of patches
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            self.num_patches = f["patches"].shape[0]

            # Store dataset attributes for normalization
            self.attrs = dict(f.attrs)

    def __len__(self):
        """Return the number of patches in the dataset."""
        return self.num_patches

    def __getitem__(self, idx):
        """Get a patch from the dataset.

        The HDF5 file is opened and closed for each access to avoid
        multiprocessing issues.

        Args:
            idx: Index of the patch to retrieve

        Returns:
            Dictionary containing the patch data
        """
        # Get the patch from HDF5 file
        with h5py.File(self.hdf5_path, "r") as f:
            # HDF5 patches are stored as [N, H, W, 2] with real and imaginary parts
            patch = f["patches"][idx]

        # Extract real and imaginary components (squared from processing)
        real_part = patch[:, :, 0]
        imag_part = patch[:, :, 1]

        # Calculate intensity (sum of real and imaginary parts)
        # intensity = real_part + imag_part

        # Apply normalization
        from src.utils.sar_utils import normalize_sar

        real_norm = normalize_sar(real_part)
        imag_norm = normalize_sar(imag_part)
        # intensity_norm = normalize_sar(intensity)

        # Convert to tensors and add channel dimension
        real_tensor = torch.from_numpy(real_norm).float().unsqueeze(0)
        imag_tensor = torch.from_numpy(imag_norm).float().unsqueeze(0)
        # intensity_tensor = torch.from_numpy(intensity_norm).float().unsqueeze(0)

        # Create sample dictionary
        sample = {
            "real": real_tensor,
            "imag": imag_tensor,
            # "intensity": intensity_tensor,  # @TODO to remove, it should not be used
            # "hdf5_path": self.hdf5_path,
            # "patch_idx": idx,
        }

        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)

        return sample
