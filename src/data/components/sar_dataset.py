"""
SAR Dataset for loading and preprocessing SAR images.

This module contains dataset classes for handling SAR images in both CoSAR format
and pre-processed HDF5 format with proper deterministic behavior.
"""

import warnings
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class TSXSSCDataset(Dataset):
    """Dataset for pre-processed SAR patches in HDF5 format.

    This dataset loads pre-processed SAR patches from HDF5 files created
    by the TSX_dataset_creation.py script.
    """

    def __init__(
        self,
        hdf5_path: Path,
        log_mode: str = "natural",
        must_normalize: float | None = None,
        transform=None,
    ):
        """Initialize the dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing pre-processed patches
            log_mode: logarithmic base of the loaded data, either "db": Transformed with 10*log10() or "natural": Transformed with log(). Default: "natural".
            must_normalize: Min-max normalization used for the data. None means the data is already normalized, while any other percent indicates the percentiles that should be used in place of min and max values, e.g., 1 <=> norm_x = (x - p1) / (p99 - p1). In particular, 0 implies the traditionnal min-max normalization. Default: None.
            transform: Optional transform to apply to samples (default: None)
        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.log_mode = log_mode
        self.must_normalize = must_normalize
        self.transform = transform

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
            patch = f["patches"][idx]
            real = patch[:, :, 0]
            imag = patch[:, :, 1]

            if self.must_normalize is not None:
                # @TODO: implement support for unnormalized data
                warnings.warn(
                    f"Unnormalized data is not supported. TSXSSCDataset was instantiated with {self.must_normalize=} and {self.log_mode=}."
                )
                # # Apply normalization
                # real = normalize_sar(real)
                # imag = normalize_sar(imag)

            if self.transform:
                warnings.warn(
                    "Transforms are not supported yet. TSXSSCDataset was instantiated with {self.transform}."
                )
                # sample = self.transform(sample)

            # Convert to tensors and add channel dimension
            real_tensor = torch.from_numpy(real).float().unsqueeze(0)
            imag_tensor = torch.from_numpy(imag).float().unsqueeze(0)

            return {
                "real": real_tensor,
                "imag": imag_tensor,
            }
