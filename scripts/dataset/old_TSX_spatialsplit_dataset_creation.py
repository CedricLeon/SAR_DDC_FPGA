"""
For the moment I will only list a few design decisions.
- Doing spatial split means no need for validation and test fraction arguments.
- --max-files only applies to training folder, validation and test folders will always be fully processed (but should contain only one image anyway).

"""

# Imports
import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from lightning_utilities.core.rank_zero import rank_zero_only

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.constants import PERCENTILES, M, amp_max, amp_min, m
from src.utils.pylogger import RankedLogger
from src.utils.sar_utils import preprocess_TSX_image, preprocess_TSX_patch


# ANSI color codes for console output
class Colors:
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Process SAR .cos files and create HDF5 datasets for training."
)
parser.add_argument(
    "--input-dir", type=str, required=True, help="Directory containing .cos files"
)
parser.add_argument(
    "--output-dir", type=str, required=True, help="Directory to save HDF5 files"
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
parser.add_argument(
    "--max-files-train",
    type=int,
    default=5,
    help="Maximum number of files to process for training split (default: 5)",
)
parser.add_argument(
    "--preserve-threshold",
    type=float,
    default=None,
    help="Threshold above which point-like scatterers are preserved (default: None = no preservation)",
)
parser.add_argument(
    "--log-base",
    type=str,
    choices=["db", "nat"],
    default=None,
    help="Logarithm base: 'db' or 'nat' (natural) (default: None -> no normalization is done)",
)
parser.add_argument(
    "--norm-minmax",
    type=int,
    default=-1,
    help="Percentiles used in place of min and max in the normalization: 0, 1, 5, or 10 (default: 0 -> min and max are used). you can also use -1 to use MERLIN's empirical values (not recommended). See `src/utils/constants` for more info.",
)
parser.add_argument(
    "--clip",
    action="store_true",
    help="Whether to clip the data to the range [0, 1] after normalization. Not recommended. Disabled is --norm-minmax is 0 (data already between 0 and 1). (default: False)",
)
args = parser.parse_args()


def extract_filepath_short_name(file_path: Path):
    # Will crash if file is not named "Roma_ ..."
    return file_path.name.split("_")[0]


def build_dataset_name():
    pres_name = (
        "nopres"
        if args.preserve_threshold is None
        else f"pres{int(args.preserve_threshold)}"
    )
    norm_name = (
        "nonorm"
        if args.log_base is None
        else f"norm{'MERLIN' if args.norm_minmax == -1 else args.norm_minmax}{args.log_base}"
    )
    norm_name += "clip" if args.clip else ""
    return f"spatialsplit{args.max_files_train}_{pres_name}_{norm_name}"


def main():
    dataset_name = build_dataset_name()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # ----- Setup logging -----
    rank_zero_only.rank = 0  # Set rank for single-process script
    logger = logging.getLogger()
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
    log = RankedLogger(__name__, rank_zero_only=True)
    log.info(f"Logging configured. Log file: {log_file}")

    # ----- Validate script arguments -----
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory does not exist: {output_dir}")
    if args.clip and args.norm_minmax == 0:
        log.warning(
            "Clipping is enabled, but norm_minmax is set to 0. "
            "Clipping will be disabled."
        )
        args.clip = False

    # ----- Print configuration -----
    log.info(f"{Colors.YELLOW}TSX Dataset Creation - Configuration:{Colors.RESET}")
    log.info(f"  Input directory: {args.input_dir}")
    log.info(f"  Output directory: {args.output_dir}")
    log.info(f"  Patch size: {args.patch_size}x{args.patch_size}")
    log.info(f"  Seed: {args.seed}")
    if args.max_files_train:
        log.info(f"  Max files: {args.max_files_train}")
    if args.preserve_threshold is not None:
        log.info(f"  Preserve scatterers threshold: {args.preserve_threshold} dB")
    else:
        log.info("  Preserve scatterers: Disabled")
    if args.log_base is not None:
        log.info(f"  Logarithm base: {args.log_base}")
        log.info(f"  Min-max normalization percentile: {args.norm_minmax}%")
        log.info(f"  Clipping [0,1]: {'Enabled' if args.clip else 'Disabled'}")
    else:
        log.info("  Normalization: Disabled")
    log.info("")

    # ----- Determine splits -----
    with open(input_dir / "spatial_splits.json", "r") as f:
        splits = json.load(f)
    filenames = {
        "train": splits.get("train", []),
        "val": splits.get("val", []),
        "test": splits.get("test", []),
    }
    for split, files in filenames.items():
        for file in files:
            file_path = input_dir / file
            if not file_path.is_file():
                raise RuntimeError(
                    f"Expected file for {split} split does not exist: {file_path}"
                )
    if (
        args.max_files_train is not None
        and len(filenames["train"]) > args.max_files_train
    ):
        log.info(
            f"Limiting training files to {args.max_files_train} out of {len(filenames['train'])} available."
        )
        filenames["train"] = filenames["train"][: args.max_files_train]

    # ----- Solve normalization settings -----
    if args.norm_minmax == -1:  # Use MERLIN's empirical values
        min_max = (m, M)
    elif args.norm_minmax == 0:  # Use min and max
        min_max = (amp_min, amp_max)
    else:
        min_max = (
            PERCENTILES[f"p{args.norm_minmax}"],
            PERCENTILES[f"p{100 - args.norm_minmax}"],
        )

    # ----- Set seeds -----
    np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # ----- Process -----
    start_time = datetime.now()
    log.info(
        f"Starting the creation of the dataset {dataset_name} at {start_time}",
        Colors.YELLOW,
    )
    add_patch_for_persistent_visualization(
        filenames, min_max, output_dir / dataset_name
    )
    process_dataset(
        input_dir,
        filenames,
        output_dir / dataset_name,
        min_max,
    )
    end_time = datetime.now()
    log.info(f"{Colors.GREEN}Processing completed successfully!{Colors.RESET}")
    log.info(
        f"Started: {start_time}. Finished: {end_time}. Total duration: {end_time - start_time}."
    )


def process_dataset(
    input_dir: Path,
    filenames: dict[str, list[str]],
    output_dir: Path,
    min_max: tuple[float, float],
):
    # ---- Manage training files (chunked processing) ----
    with h5py.File(output_dir / "unshuffled.h5", "w") as f:
        hdf5_dataset = f.create_dataset(
            "patches",
            shape=(0, args.patch_size, args.patch_size, 2),
            maxshape=(None, args.patch_size, args.patch_size, 2),
            dtype="float32",
            chunks=True,
        )

        patches_per_image = {}
        for i, filename in enumerate(filenames["train"]):
            file_path = input_dir / filename
            short_name = extract_filepath_short_name(file_path)
            log.info(
                f"{Colors.BLUE}Processing {i + 1}/{len(filenames['train'])}: {short_name}{Colors.RESET}"
            )

            patches = preprocess_TSX_image(
                image_path=file_path,
                preserve_threshold=args.preserve_threshold,
                log_base=args.log_base,
                min_max=min_max,
                clip=args.clip,
                patch_size=args.patch_size,
                logger=log,
            )

            nb_patches = patches.shape[0]
            patches_per_image[short_name] = nb_patches
            log.info(f"  - Extracted {nb_patches} patches from {short_name}.")
            hdf5_dataset.resize(hdf5_dataset.shape[0] + nb_patches, axis=0)
            hdf5_dataset[-nb_patches:] = patches

            del patches  # Free memory
    gc.collect()

    # Out-of-core shuffle
    with h5py.File(output_dir / "unshuffled.h5", "r") as f:
        patches_dataset = f["patches"]
        nb_total_patches = patches_dataset.shape[0]
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
                temp_data = patches_dataset[
                    batch_indices.min() : batch_indices.max() + 1
                ]
                # Map to the shuffled order
                remapped_indices = batch_indices - batch_indices.min()
                # Write to output in shuffled order
                shuffled_dataset[i : i + len(batch_indices)] = temp_data[
                    remapped_indices
                ]

            add_metadata_to_dataset(
                shuffled_dataset,
                nb_total_patches,
                patches_per_image,
                min_max,
            )

    (output_dir / "unshuffled.h5").unlink(missing_ok=True)

    # For validation and training no need to chunk the processing, as they should only contain one image each.
    for split in ["val", "test"]:
        log.info(f"Processing {split} split with {len(filenames[split])} files.")
        patches_per_image = {}
        all_patches = []
        for i, filename in enumerate(filenames[split]):
            file_path = input_dir / filename
            short_name = extract_filepath_short_name(file_path)
            log.info(f"  - Processing {short_name}")

            patches = preprocess_TSX_image(
                image_path=file_path,
                preserve_threshold=args.preserve_threshold,
                log_base=args.log_base,
                min_max=min_max,
                clip=args.clip,
                patch_size=args.patch_size,
                logger=log,
            )
            nb_patches = patches.shape[0]
            patches_per_image[short_name] = nb_patches
            log.info(f"  - Extracted {nb_patches} patches from {short_name}.")
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
                nb_total_patches=patches.shape[0],
                patches_per_image=patches_per_image,
                min_max=min_max,
            )
        del patches
        log.info(
            f"Saved {split} split with {nb_total_patches} patches to {output_dir / f'{split}.h5'}"
        )


def add_metadata_to_dataset(
    dset: h5py.Dataset,
    nb_total_patches: int,
    patches_per_image: dict[str, int],
    min_max: tuple[float, float],
):
    """Add metadata to the shuffled dataset."""
    dset.attrs["creation_date"] = datetime.now().isoformat()
    dset.attrs["name"] = build_dataset_name()
    dset.attrs["seed"] = args.seed

    # HDF5 does not have a "null value" representation -> handle as a string
    dset.attrs["max_files_train"] = (
        "None" if args.max_files_train is None else args.max_files_train
    )
    dset.attrs["preserve_threshold"] = (
        "None" if args.preserve_threshold is None else args.preserve_threshold
    )
    dset.attrs["log_base"] = "None" if args.log_base is None else args.log_base
    # Store boolean as integer (0 or 1)
    dset.attrs["clip"] = int(args.clip)

    dset.attrs["norm_min"] = min_max[0]
    dset.attrs["norm_max"] = min_max[1]

    dset.attrs["patch_size"] = args.patch_size
    dset.attrs["nb_patches"] = nb_total_patches
    dset.attrs["patches_per_image"] = "_".join(
        f"{name}-{count}" for name, count in patches_per_image.items()
    )  # e.g., "Roma-123_Geneva-456"

    log.info(f"Metadata added to dataset: {dset.attrs}.")


def add_patch_for_persistent_visualization(filenames, min_max, save_dir):
    """Preprocess and save a unique, large patch used for visualizing improvments during training."""
    # Cheery picked the towncenter of Hamburg (ncolumns=14686 nlines=32901)
    CROP_COORDINATES = (11000, 8500)
    PATCH_SIZE = (1024, 1024)
    image_path = Path(args.input_dir) / filenames["val"][0]
    short_name = extract_filepath_short_name(image_path)

    patch_np = preprocess_TSX_patch(
        image_path,
        CROP_COORDINATES,
        PATCH_SIZE,
        args.preserve_threshold,
        args.log_base,
        min_max,
        args.clip,
        logger=log,
    )
    patch_path = save_dir / f"val_{short_name}_{PATCH_SIZE[0]}x{PATCH_SIZE[1]}.npy"
    np.save(patch_path, patch_np)
    log.info(
        f"Successfully saved persistent visualization patch of {short_name} at {patch_path}."
    )


if __name__ == "__main__":
    main()
