from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    """Dataset for loading image patches from a .npy file.

    Expects data shape [N, H, W, 2].
    """

    def __init__(self, data_path: Path):
        super().__init__()
        self.data_path = data_path
        self.data = np.load(str(data_path))  # [N, H, W, 2]
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = self.data[idx]  # [H, W, 2]
        real = patch[..., 0]  # [H, W]
        imag = patch[..., 1]  # [H, W]

        return {
            "real": torch.from_numpy(real).unsqueeze(0),  # [1, H, W]
            "imag": torch.from_numpy(imag).unsqueeze(0),  # [1, H, W]
        }


def run_dual_evaluation(
    trainer: Trainer,
    model: LightningModule,
    datamodule: Any = None,
    dataloaders: Any = None,
    ckpt_path: Optional[str] = None,
    hdf5_dir: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Dict[str, float]:
    """
    Runs evaluation on two sets:
    1. Full Test Set (from datamodule/dataloaders) -> Prefix: "test_full"
    2. Subset 300 (from .npy file in hdf5_dir) -> Prefix: "test_sub300"

    Returns combined metrics dictionary.
    """
    final_metrics = {}

    # --- 1. Standard Test (Full Dataset) ---
    print("    Running test on FULL dataset...")
    # Temporarily set prefix on model (Requires model to support test_prefix attribute)
    original_prefix = getattr(model, "test_prefix", "test")
    model.test_prefix = "test_full"

    if datamodule:
        results_full = trainer.test(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=False
        )
    else:
        results_full = trainer.test(
            model=model, dataloaders=dataloaders, ckpt_path=ckpt_path, verbose=False
        )

    if results_full:
        final_metrics.update(results_full[0])
        print(f"    Full Test Metrics: {results_full[0]}")

    # --- 2. Subset Test (from .npy) ---
    print("    Running test on SUBSET 300 dataset (from .npy)...")

    if hdf5_dir:
        npy_files = list(Path(hdf5_dir).glob("test_sub300*.npy"))
        if npy_files:
            npy_path = npy_files[0]
            print(f"    Found subset file: {npy_path}.\n    Running test on this subset...")
            dataset = NumpyDataset(npy_path)
            subset_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=True,
            )

            model.test_prefix = "test_sub300"
            # Important: Do not reload the checkpoint here (ckpt_path=None).
            # The model is already loaded and initialized (buffers resized via update()) from the first test run.
            # Reloading the original checkpoint (which has empty buffers) would cause a size mismatch error.
            results_sub = trainer.test(
                model=model, dataloaders=subset_loader, ckpt_path=None, verbose=False
            )
            if results_sub:
                final_metrics.update(results_sub[0])
                print(f"    Subset Test Metrics: {results_sub[0]}")
        else:
            print(
                "    WARNING: Subset file matching 'test_sub300*.npy' not found. Skipping test_sub300."
            )
    else:
        print("    WARNING: hdf5_dir not provided. Skipping test_sub300.")

    # Reset prefix
    # model.test_prefix = original_prefix
    setattr(model, "test_prefix", original_prefix)

    return final_metrics
