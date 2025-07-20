"""
SAR DataModule for handling TerraSAR-X data.

This module handles loading, preprocessing, and splitting of SAR data
with deterministic behavior.
"""

import logging
from pathlib import Path

import h5py
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.sar_dataset import TSXSSCDataset

log = logging.getLogger(__name__)


class TSXSSCDataModule(LightningDataModule):
    """Lightning DataModule for pre-processed SAR images in HDF5 format.

    This module handles loading pre-processed SAR patches from HDF5 files
    created by TSX_dataset_creation.py.
    """

    def __init__(
        self,
        hdf5_dir: str,
        must_normalize: float | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform=None,
        **kwargs,
    ):
        """Initialize the DataModule.

        Args:
            hdf5_dir: Directory containing HDF5 dataset files
            must_normalize: Min-max normalization used for the data. None means the data is already normalized, while any other percent indicates the percentiles that should be used in place of min and max values, e.g., 1 <=> norm_x = (x - p1) / (p99 - p1). In particular, 0 implies the traditionnal min-max normalization. Default: None.
            batch_size: Batch size (default: 16)
            num_workers: Number of workers for DataLoader (default: 4)
            pin_memory: Whether to pin memory (default: True)
            transform: Optional transform to apply (default: None)
        """
        super().__init__()

        # Save hyperparameters
        self.hdf5_root_dir = Path(hdf5_dir)
        self.must_normalize = must_normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Set file paths as Path objects
        self.train_path = self.hdf5_root_dir / "train.h5"
        self.val_path = self.hdf5_root_dir / "val.h5"
        self.test_path = self.hdf5_root_dir / "test.h5"

        # Data transformations
        self.transform = transform

    def log_patches_per_split(
        self, split: str, total_patches: int, patches_per_image_str: str
    ):
        patches_per_image = {
            name: int(count)
            for name, count in (
                entry.split("-") for entry in patches_per_image_str.split("_")
            )
        }
        patches_summary = ", ".join(
            f"{count} from {name}" for name, count in patches_per_image.items()
        )
        log.info(
            f"{split} set contains a total of {total_patches} patches with {patches_summary}."
        )

    def prepare_data(self):
        """Data preparation (download, etc.) - runs once on the node."""
        # Check if the HDF5 files exist
        for path in [self.train_path, self.val_path, self.test_path]:
            if not path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {path}")

        # Read hdf5 attributes to set pre-processing information
        with h5py.File(self.train_path, "r") as f:
            attrs = f["patches"].attrs
            self.hdf5_metadata = dict(attrs)
            log.info(
                f"Loaded datasets created at {attrs.get('creation_date', 'Unknown')} with seed {attrs.get('seed', 'Unknown')} and "
                f"{'normalized' if attrs.get('normalize', False) else 'NOT normalized'} data."
            )
            self.log_patches_per_split(
                "Train", attrs.get("nb_patches", -1), attrs.get("patches_per_image", "")
            )
        with h5py.File(self.val_path, "r") as f:
            self.log_patches_per_split(
                "Validation",
                f["patches"].attrs.get("nb_patches", -1),
                f["patches"].attrs.get("patches_per_image", ""),
            )
        with h5py.File(self.test_path, "r") as f:
            self.log_patches_per_split(
                "Test",
                f["patches"].attrs.get("nb_patches", -1),
                f["patches"].attrs.get("patches_per_image", ""),
            )

    def setup(self, stage=None):
        """Data setup per stage - runs on every process."""
        if stage == "fit" or stage is None:
            # Create training dataset
            self.data_train = TSXSSCDataset(
                self.train_path,
                must_normalize=self.must_normalize,
                transform=self.transform,
            )

            # Create validation dataset
            self.data_val = TSXSSCDataset(
                self.val_path,
                must_normalize=self.must_normalize,
                transform=self.transform,
            )

        if stage == "test" or stage is None:
            # Create test dataset
            self.data_test = TSXSSCDataset(
                self.test_path,
                must_normalize=self.must_normalize,
                transform=self.transform,
            )

    def train_dataloader(self):
        """Create train dataloader."""
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
