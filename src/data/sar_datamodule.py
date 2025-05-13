"""
SAR DataModule for handling TerraSAR-X data.

This module handles loading, preprocessing, and splitting of SAR data
with deterministic behavior.
"""

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.sar_dataset import TSXSSCDataset


class TSXSSCDataModule(LightningDataModule):
    """Lightning DataModule for pre-processed SAR images in HDF5 format.

    This module handles loading pre-processed SAR patches from HDF5 files
    created by TSX_dataset_creation.py.
    """

    def __init__(
        self,
        hdf5_dir: str,
        log_mode: str = "natural",
        normalize: float | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform=None,
        **kwargs,
    ):
        """Initialize the DataModule.

        Args:
            hdf5_dir: Directory containing HDF5 dataset files
            log_mode: logarithmic base of the loaded data, either "db": Transformed with 10*log10() or "natural": Transformed with log(). Default: "natural".
            normalize: Min-max normalization used for the data. None means not normalized, while any other percent indicates the percentiles used in place of min and max values, e.g., 1 <=> norm_x = (x - p1) / (p99 - p1). In particular, 0 implies the traditionnal min-max normalization. Default: None.
            batch_size: Batch size (default: 16)
            num_workers: Number of workers for DataLoader (default: 4)
            pin_memory: Whether to pin memory (default: True)
            transform: Optional transform to apply (default: None)
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        # Set file paths as Path objects
        hdf5_root_dir = Path(self.hparams.hdf5_dir)
        self.train_path = hdf5_root_dir / "train.h5"
        self.val_path = hdf5_root_dir / "val.h5"
        self.test_path = hdf5_root_dir / "test.h5"

        # Data transformations
        self.transform = transform

        # Data split information
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Data preparation (download, etc.) - runs once on the node."""
        # Check if the HDF5 files exist
        for path in [self.train_path, self.val_path, self.test_path]:
            if not path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {path}")

    def setup(self, stage=None):
        """Data setup per stage - runs on every process."""
        if stage == "fit" or stage is None:
            # Create training dataset
            self.data_train = TSXSSCDataset(
                self.train_path,
                log_mode=self.hparams.log_mode,
                normalize=self.hparams.normalize,
                transform=self.transform,
            )

            # Create validation dataset
            self.data_val = TSXSSCDataset(
                self.val_path,
                log_mode=self.hparams.log_mode,
                normalize=self.hparams.normalize,
                transform=self.transform,
            )

        if stage == "test" or stage is None:
            # Create test dataset
            self.data_test = TSXSSCDataset(
                self.test_path,
                log_mode=self.hparams.log_mode,
                normalize=self.hparams.normalize,
                transform=self.transform,
            )

    def train_dataloader(self):
        """Create train dataloader."""
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
