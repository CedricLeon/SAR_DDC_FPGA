"""SAR Dataset for loading and preprocessing SAR images.

This module contains dataset classes for handling SAR images in both CoSAR format and pre-processed
HDF5 format with proper deterministic behavior.
"""

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from src.utils.constants import amp_max, amp_min


class TSXSSCDataset(Dataset):
    """Dataset for pre-processed SAR patches in HDF5 format.

    This dataset loads pre-processed SAR patches from HDF5 files created by the
    TSX_dataset_creation.py script.
    """

    def __init__(
        self,
        hdf5_path: Path,
    ):
        """Initialize the dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing pre-processed patches
        """
        super().__init__()
        self.hdf5_path = hdf5_path

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            self.num_patches = f["patches"].shape[0]
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
        with h5py.File(self.hdf5_path, "r") as f:
            patch = torch.from_numpy(f["patches"][idx]).float()

            patch = torch.square(patch)
            patch = torch.log(patch + 1e-2)
            patch = (patch - 2 * amp_min) / (2 * amp_max - 2 * amp_min)

            return {  # Add channel dimension
                "real": patch[:, :, 0].unsqueeze(0),
                "imag": patch[:, :, 1].unsqueeze(0),
            }
