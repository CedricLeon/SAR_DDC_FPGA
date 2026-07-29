from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class TSXSSCDataset(Dataset):
    """Dataset for pre-processed SAR patches in HDF5 format.

    This dataset loads pre-processed SAR patches from HDF5 files created by the
    `scripts/dataset/create_dataset.py` script. Each patch is unnormalized, has 2 channels (real
    and imaginary), and can come with references ("Ground-Truths") as 2 additional channels.
    """

    def __init__(
        self,
        hdf5_path: Path,
        with_refs: bool = False,
    ):
        """Initialize the dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing pre-processed patches
            with_refs: Whether to include reference maps in the dataset
        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.with_refs = with_refs

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

            item = {}

            if self.with_refs:
                item["adam_noc_ref"] = patch[:, :, 2].unsqueeze(0)
                item["merlin_ref"] = patch[:, :, 3].unsqueeze(0)
                patch = patch[:, :, :2]

            item["real"] = patch[:, :, 0].unsqueeze(0)
            item["imag"] = patch[:, :, 1].unsqueeze(0)

            return item
