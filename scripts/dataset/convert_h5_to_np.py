import argparse
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Path to input H5 file")
parser.add_argument(
    "--subset",
    type=int,
    default=None,
    help="Optional number of images to extract from the dataset",
)


def main():
    """Convert H5 dataset of image patches to NumPy binary format (.npy)."""
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    subset = args.subset
    with h5py.File(dataset_path, "r") as f:
        total = f["patches"].shape[0]
        if subset is not None:
            if subset <= 0:
                raise ValueError(f"--subset {subset} must be a positive integer")
            elif subset > total:
                print(
                    f"Requested subset ({subset}) exceeds dataset size ({total}). Using full dataset instead."
                )
                patches = f["patches"][:]
            else:
                patches = f["patches"][:subset]
        else:
            patches = f["patches"][:]
        print(f"Saving patches ({patches.shape=}) to NumPy format (.npy).")

    # Save as NumPy binary format (.npy)
    out_path = dataset_path.parent / f"{dataset_path.stem}_{subset}.npy"
    np.save(out_path, patches)


if __name__ == "__main__":
    main()
