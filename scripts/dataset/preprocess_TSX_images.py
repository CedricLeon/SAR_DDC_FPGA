"""
This script creates a pre-processed dataset from the orgiginal TSX COS files.

The dataset is created using spatial splits, a file stating which image belongs to which split is required (see `--split-file` argument). The size of the patches can be specified, default is 256x256. The seed can be set for reproducibility, default is 42.

Regarding normalization, none is made by default, i.e., the only processing done is the symmetrization of the images (Zero-Doppler centering) and the patchification. If the `--normalize` flag is set, the images are squared, moved to a natural log-scale, and normalized using the global minimum and maximum of the amplitude of the images (computed in a different script, these values should remain FIXED).

Example usage (from repo root):
$ python scripts/dataset/preprocess_TSX_images.py --input-dir data/TSX_cos_files --output-dir data/processed_hdf5/ --split-file data/TSX_cos_files/spatial_splits_1.json --normalize
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.constants import amp_max, amp_min

# from src.utils.pylogger import RankedLogger
from src.utils.sar_utils import (
    extract_patches,
    load_and_symmetrize_TSX_image,
    normalize_ndarray,
    preprocess_TSX_patch,
)


# ANSI color codes for console output
class Colors:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Process SAR .cos files and create HDF5 datasets."
)
parser.add_argument(
    "--input-dir", type=str, required=True, help="Directory containing .cos files"
)
parser.add_argument(
    "--output-dir", type=str, required=True, help="Directory to save HDF5 files"
)
parser.add_argument(
    "--split-file",
    type=str,
    required=True,
    help="Path of the split file (JSON format).",
)
parser.add_argument(
    "--normalize",
    action="store_true",
    help="Apply normalization (natural log-scale + min/max using amplitude global minimum/maximum) to the images",
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=256,
    help="Size of patches to extract. Size under 128 not recommended because of MS-SSIM. (default: 256)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
args = parser.parse_args()


def setup_logging(output_dir: Path, dataset_name: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # File handler
    log_dir = output_dir / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dataset_creation.log"
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    # Create RankedLogger AFTER setting up root logger
    global log
    log = logger
    log.info(f"Logging configured. Log file: {log_file}")


def extract_short_name_from_filepath(filepath: Path):
    if "_" not in filepath.name:
        raise ValueError(
            f"File {filepath.name} does not contain an underscore '_' to extract the short name."
        )
    return filepath.name.split("_")[0]


def add_patch_for_persistent_visualization(filenames, save_dir):
    """Preprocess and save a unique, large patch used for visualizing improvments during training."""
    # Cheery picked the towncenter of Hamburg (ncolumns=14686 nlines=32901)
    CROP_COORDINATES = (11000, 8500)
    PATCH_SIZE = (1024, 1024)
    image_path = filenames["val"][0]
    short_name = extract_short_name_from_filepath(image_path)

    patch_np = preprocess_TSX_patch(
        image_path,
        CROP_COORDINATES,
        PATCH_SIZE,
        preserve_threshold=False,
        log_base="nat",
        min_max=(2 * amp_min, 2 * amp_max),
        clip=False,
        logger=None,
    )

    patch_path = save_dir / f"val_{short_name}_{PATCH_SIZE[0]}x{PATCH_SIZE[1]}.npy"
    np.save(patch_path, patch_np)
    log.info(
        f"Successfully saved persistent visualization patch of {short_name} at {patch_path}."
    )


def add_metadata_to_dataset(
    dset: h5py.Dataset,
    nb_total_patches: int,
    patches_per_image: dict[str, int],
):
    """Add metadata to the shuffled dataset."""
    dset.attrs["creation_date"] = datetime.now().isoformat()
    dset.attrs["seed"] = args.seed
    dset.attrs["split_file"] = args.split_file
    dset.attrs["normalize"] = args.normalize
    dset.attrs["patch_size"] = args.patch_size
    dset.attrs["nb_patches"] = nb_total_patches
    dset.attrs["patches_per_image"] = "_".join(
        f"{name}-{count}" for name, count in patches_per_image.items()
    )  # e.g., "Roma-123_Geneva-456"

    log.info(f"Metadata added to dataset: {dset.attrs}.")


def process_dataset(
    filenames: dict[str, list[Path]],
    output_dir: Path,
):
    # ---- Manage training files (chunked processing) ----
    log.info(
        f"{Colors.YELLOW}Processing training split with {len(filenames['train'])} files.{Colors.RESET}"
    )
    with h5py.File(output_dir / "unshuffled.h5", "w") as f:
        hdf5_dataset = f.create_dataset(
            "patches",
            shape=(0, args.patch_size, args.patch_size, 2),
            maxshape=(None, args.patch_size, args.patch_size, 2),
            dtype="float32",
            chunks=True,
        )

        patches_per_image = {}
        for i, file_path in enumerate(filenames["train"]):
            short_name = extract_short_name_from_filepath(file_path)
            log.info(
                f" - {Colors.BLUE}Processing {i + 1}/{len(filenames['train'])}: {short_name}{Colors.RESET}"
            )
            image = load_and_symmetrize_TSX_image(
                image_path=file_path,
                logger=log,
            )
            if args.normalize:
                image = np.square(image)
                image = normalize_ndarray(
                    image,
                    log_base="nat",
                    min_max=(2 * amp_min, 2 * amp_max),
                    clip=False,
                )
            patches = extract_patches(image, args.patch_size, stride=args.patch_size)

            nb_patches = patches.shape[0]
            patches_per_image[short_name] = nb_patches
            log.info(f"      Extracted {nb_patches} patches from {short_name}.")
            hdf5_dataset.resize(hdf5_dataset.shape[0] + nb_patches, axis=0)
            hdf5_dataset[-nb_patches:] = patches

            del patches  # Free memory
    gc.collect()

    # Out-of-core shuffle
    log.info("Shuffling training patches...")
    with h5py.File(output_dir / "unshuffled.h5", "r") as f:
        patches_dataset = f["patches"]
        nb_total_patches = patches_dataset.shape[0]
        # Create a random permutation of indices
        indices = np.random.permutation(nb_total_patches)

        with h5py.File(output_dir / "train.h5", "w") as out_f:
            shuffled_dataset = out_f.create_dataset(
                "patches",
                shape=(nb_total_patches, args.patch_size, args.patch_size, 2),
                dtype="float32",
            )
            chunk_size = 1024
            for i in range(0, nb_total_patches, chunk_size):
                batch_indices = indices[i : i + chunk_size]
                # Read patches in original order
                tmp_data = patches_dataset[
                    batch_indices.min() : batch_indices.max() + 1
                ]
                # Map to the shuffled order
                remapped_indices = batch_indices - batch_indices.min()
                # Write to output in shuffled order
                shuffled_dataset[i : i + len(batch_indices)] = tmp_data[
                    remapped_indices
                ]

            add_metadata_to_dataset(
                shuffled_dataset,
                nb_total_patches,
                patches_per_image,
            )
        log.info(
            f"Saved {nb_total_patches} shuffled training patches to {output_dir / 'train.h5'}"
        )

    (output_dir / "unshuffled.h5").unlink(missing_ok=True)

    # For validation and training no need to chunk the processing, as they should only contain one image each.
    for split in ["val", "test"]:
        log.info(
            f"{Colors.YELLOW}Processing {split} split with {len(filenames[split])} files.{Colors.RESET}"
        )
        patches_per_image = {}
        all_patches = []
        for i, file_path in enumerate(filenames[split]):
            short_name = extract_short_name_from_filepath(file_path)
            log.info(f"  - {Colors.BLUE}Processing {short_name}{Colors.RESET}")

            image = load_and_symmetrize_TSX_image(
                image_path=file_path,
                logger=log,
            )
            if args.normalize:
                image = np.square(image)
                image = normalize_ndarray(
                    image,
                    log_base="nat",
                    min_max=(2 * amp_min, 2 * amp_max),
                    clip=False,
                )
            patches = extract_patches(image, args.patch_size, stride=args.patch_size)

            nb_patches = patches.shape[0]
            patches_per_image[short_name] = nb_patches
            log.info(f"      Extracted {nb_patches} patches from {short_name}.")
            all_patches.append(patches)
            del patches

        patches = np.concatenate(all_patches, axis=0)
        del all_patches

        patches = patches[np.random.permutation(patches.shape[0])]
        with h5py.File(output_dir / f"{split}.h5", "w") as f:
            dset = f.create_dataset(
                "patches",
                data=patches,
                maxshape=(None, args.patch_size, args.patch_size, 2),
                dtype="float32",
            )
            add_metadata_to_dataset(
                dset,
                patches.shape[0],
                patches_per_image,
            )
        del patches
        log.info(
            f"Saved {split} split with {nb_total_patches} patches to {output_dir / f'{split}.h5'}"
        )


def main():
    # ----- Setup logging -----
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    split_file = Path(args.split_file)
    dataset_name = f"TSX_preprocessed_{split_file.stem}_{args.patch_size}x{args.patch_size}{'_normalized' if args.normalize else ''}"
    setup_logging(output_dir, dataset_name)

    # ----- Validate script arguments -----
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory does not exist: {output_dir}")

    # ----- Print configuration -----
    log.info(f"{Colors.YELLOW}TSX Dataset Creation - Configuration:{Colors.RESET}")
    log.info(f"  Input directory: {args.input_dir}")
    log.info(f"  Output directory: {args.output_dir}")
    log.info(f"  Split file: {args.split_file}")
    log.info(f"  Patch size: {args.patch_size}x{args.patch_size}")
    log.info(f"  Normalize: {args.normalize}")
    log.info(f"  Seed: {args.seed}")
    log.info("")

    # ----- Determine splits -----
    with open(args.split_file, "r") as f:
        splits = json.load(f)
    filenames = {
        "train": splits.get("train", []),
        "val": splits.get("val", []),
        "test": splits.get("test", []),
    }
    for split, files in filenames.items():
        log.info(f"Files for {split} split:")
        filepaths = []
        for file in files:
            file_path = input_dir / file
            log.info(f"  - {extract_short_name_from_filepath(file_path)}")
            if not file_path.is_file():
                raise RuntimeError(
                    f"Expected file for {split} split does not exist: {file_path}"
                )
            filepaths.append(file_path)
        filenames[split] = filepaths

    # ----- Set seeds -----
    np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # ----- Process -----
    start_time = datetime.now()
    log.info(
        f"{Colors.GREEN}Starting the creation of the dataset {dataset_name} at {start_time}{Colors.RESET}",
    )
    add_patch_for_persistent_visualization(filenames, output_dir / dataset_name)
    process_dataset(
        filenames,
        output_dir / dataset_name,
    )
    end_time = datetime.now()
    log.info(f"{Colors.GREEN}Processing completed successfully!{Colors.RESET}")
    log.info(
        f"Started: {start_time}. Finished: {end_time}. Total duration: {end_time - start_time}."
    )


if __name__ == "__main__":
    main()
