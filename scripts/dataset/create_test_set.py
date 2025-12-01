import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# from src.utils.pylogger import RankedLogger
from src.models.merlin_module import MerlinModule
from src.models.sar_ddc_module import SARDDCModule
from src.utils.constants import amp_max, amp_min
from src.utils.processing_utils import extract_short_name_from_TSX_filepath
from src.utils.sar_utils import (
    extract_patches,
    load_cosar,
    symmetrize,
)


class Colors:
    """ANSI color codes for console output."""

    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process SAR .cos files and create HDF5 datasets.")
parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .cos files")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save HDF5 files")
parser.add_argument(
    "--split-file",
    type=str,
    required=True,
    help="Path of the split file (JSON format).",
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

ADAM_NOC_CKPT_PATH = Path("data/method_ground_truths/ADAM_NOC/checkpoints/last.ckpt")
MERLIN_CKPT_PATH = Path("data/method_ground_truths/MERLIN/checkpoints/last.ckpt")

EPS = 1e-2


def setup_logging(output_dir: Path, dataset_name: str):
    """Setup logging to file and console."""
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


def _checkpoint_run_dir(ckpt_path: Path) -> Path:
    """Get the run directory from the checkpoint path: .../runs/<date>/checkpoints/<file>.ckpt -> parent.parent"""
    return ckpt_path.parent.parent


def _load_training_cfg_from_ckpt(ckpt_path: Path):
    """Load the original training config from the checkpoint's run directory."""
    run_dir = _checkpoint_run_dir(ckpt_path)
    training_config_path = run_dir / ".hydra" / "config.yaml"
    if not training_config_path.exists():
        raise FileNotFoundError(f"Training config not found at {training_config_path}.")
    log.info(f"Loading original training config from {training_config_path}")
    return OmegaConf.load(training_config_path)


def _instantiate_model_and_load_weights(train_cfg, ckpt_path: Path) -> torch.nn.Module:
    """Instantiate model from Hydra cfg and load weights."""
    log.info(f"Instantiating model <{train_cfg.model._target_}>")
    model = hydra.utils.instantiate(train_cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
    log.info(f"Loaded checkpoint state_dict with message: {msg}")
    model.eval()
    return model


@torch.no_grad()
def _predict_logI_from_real_imag(
    model: torch.nn.Module, real_b: torch.Tensor, imag_b: torch.Tensor
) -> torch.Tensor:
    """Run model on batches of normalized real/imag and return log-intensity reconstructions.

    Inputs:
      - real_b, imag_b: tensors [B,1,H,W] normalized in model domain.
    Returns:
      - recon_logI: tensor [B,1,H,W] in log-intensity domain.
    """
    # ADAM forward passes in evaluation takes input with 2 channels
    if isinstance(model, SARDDCModule):
        input = torch.cat((real_b, imag_b), dim=1).contiguous()
        output = model(input)
        out_r = output["x_hat"][:, 0:1, :, :]
        out_i = output["x_hat"][:, 1:2, :, :]
    elif isinstance(model, MerlinModule):
        out_r = model(real_b)
        out_i = model(imag_b)
    # Type guard for linters
    assert isinstance(out_r, torch.Tensor), "Model output must be a Tensor or dict with 'x_hat'"
    assert isinstance(out_i, torch.Tensor), "Model output must be a Tensor or dict with 'x_hat'"

    # Denormalize to linear amplitude per channel, average, then log-intensity
    recon_r_lin = torch.exp(out_r.squeeze(1) * (amp_max - amp_min) + amp_min)
    recon_i_lin = torch.exp(out_i.squeeze(1) * (amp_max - amp_min) + amp_min)
    recon_amp = 0.5 * (recon_r_lin + recon_i_lin)  # [B,H,W]
    recon_logI = torch.log(recon_amp + EPS).unsqueeze(1)  # [B,1,H,W]
    return recon_logI


def add_metadata_to_dataset(
    dset: h5py.Dataset,
    nb_patches: int,
    patches_per_image: dict[str, int],
):
    """Add metadata to the shuffled dataset."""
    dset.attrs["creation_date"] = datetime.now().isoformat()
    dset.attrs["seed"] = args.seed
    dset.attrs["split_file"] = args.split_file
    dset.attrs["patch_size"] = args.patch_size
    dset.attrs["nb_patches"] = nb_patches
    dset.attrs["patches_per_image"] = "_".join(
        f"{name}-{count}" for name, count in patches_per_image.items()
    )  # e.g., "Roma-123_Geneva-456"

    log.info(f"Metadata added to dataset: {dset.attrs}.")


def process_test_dataset(
    filenames: list[Path],
    output_dir: Path,
):
    """Process the dataset: load images, symmetrize, patchify, run models, and save HDF5.

    Output dataset layout: patches -> [N, patch, patch, 4] = [real, imag, ADAM-NOC, MERLIN]
    where the last two channels are per-patch denoised log-intensity maps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load both models once
    if not ADAM_NOC_CKPT_PATH.exists():
        raise FileNotFoundError(f"ADAM-NOC checkpoint not found at {ADAM_NOC_CKPT_PATH}")
    if not MERLIN_CKPT_PATH.exists():
        raise FileNotFoundError(f"MERLIN checkpoint not found at {MERLIN_CKPT_PATH}")

    adam_cfg = _load_training_cfg_from_ckpt(ADAM_NOC_CKPT_PATH)
    merlin_cfg = _load_training_cfg_from_ckpt(MERLIN_CKPT_PATH)

    adam_model = _instantiate_model_and_load_weights(adam_cfg, ADAM_NOC_CKPT_PATH).to(device)
    merlin_model = _instantiate_model_and_load_weights(merlin_cfg, MERLIN_CKPT_PATH).to(device)

    # ---- Process test files ----
    log.info(f"{Colors.YELLOW}Processing test split with {len(filenames)} files.{Colors.RESET}")

    patches_per_image: dict[str, int] = {}
    per_file_patches: list[np.ndarray] = []
    total_patches = 0

    for i, file_path in enumerate(filenames):
        short_name = extract_short_name_from_TSX_filepath(file_path)
        log.info(f"  - {Colors.BLUE}Processing {short_name}{Colors.RESET}")

        # Load, symmetrize, patchify
        image = load_cosar(file_path, logger=log)
        if image is None:
            raise ValueError(f"Failed to load {file_path}")
        image = symmetrize(image)
        patches = extract_patches(image, args.patch_size, stride=args.patch_size)  # [N,H,W,2]

        nb_patches = int(patches.shape[0])
        patches_per_image[short_name] = nb_patches
        total_patches += nb_patches
        log.info(f"      Extracted {nb_patches} patches from {short_name}.")

        # Prepare outputs for this file
        out_file = np.empty((nb_patches, args.patch_size, args.patch_size, 4), dtype=np.float32)

        # Fill real/imag directly (raw, no normalization)
        out_file[..., 0] = patches[..., 0]
        out_file[..., 1] = patches[..., 1]

        # Batch inference for predictions
        batch_size = 16
        for start in tqdm(
            range(0, nb_patches, batch_size),
            desc=f"Infer {short_name}",
            unit="patch",
            leave=False,
        ):
            end = min(start + batch_size, nb_patches)
            batch = patches[start:end]  # [B,H,W,2]

            # Normalize to model domain: log of squared channel, then min-max using (2*amp_min, 2*amp_max)
            batch_sq = np.square(batch).astype(np.float32)
            batch_log = np.log(batch_sq + EPS)
            batch_norm = (batch_log - 2 * amp_min) / (2 * amp_max - 2 * amp_min)

            real_b = torch.from_numpy(batch_norm[:, :, :, 0]).to(device).unsqueeze(1).float()
            imag_b = torch.from_numpy(batch_norm[:, :, :, 1]).to(device).unsqueeze(1).float()

            # Predictions (log-intensity)
            adam_logI = _predict_logI_from_real_imag(adam_model, real_b, imag_b)
            merlin_logI = _predict_logI_from_real_imag(merlin_model, real_b, imag_b)

            out_file[start:end, :, :, 2] = adam_logI.squeeze(1).cpu().numpy()
            out_file[start:end, :, :, 3] = merlin_logI.squeeze(1).cpu().numpy()

            # Free GPU memory for large batches
            del real_b, imag_b, adam_logI, merlin_logI
            torch.cuda.empty_cache() if device.type == "cuda" else None

        per_file_patches.append(out_file)
        # Free memory
        del patches, out_file
        gc.collect()

    # Concatenate and save
    patches_all = np.concatenate(per_file_patches, axis=0)
    del per_file_patches

    with h5py.File(output_dir / "test.h5", "w") as f:
        dset = f.create_dataset(
            "patches",
            data=patches_all,
            maxshape=(None, args.patch_size, args.patch_size, 4),
            dtype="float32",
        )
        add_metadata_to_dataset(
            dset,
            int(patches_all.shape[0]),
            patches_per_image,
        )

    del patches_all
    log.info(f"Saved test split with {total_patches} patches to {output_dir / 'test.h5'}")
    log.info(f"In total {total_patches} patches processed:")


def main():
    """Main function to create the test dataset."""
    # ----- Setup logging -----
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    split_file = Path(args.split_file)
    dataset_name = f"TSX_preprocessed_{split_file.stem}_{args.patch_size}x{args.patch_size}"
    setup_logging(output_dir, dataset_name)

    # ----- Validate script arguments -----
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory does not exist: {output_dir}")

    # ----- Print configuration -----
    log.info(f"{Colors.YELLOW}TSX Test Dataset Creation - Configuration:{Colors.RESET}")
    log.info(f"  Input directory: {args.input_dir}")
    log.info(f"  Output directory: {args.output_dir}")
    log.info(f"  Split file: {args.split_file}")
    log.info(f"  Patch size: {args.patch_size}x{args.patch_size}")
    log.info(f"  Seed: {args.seed}")
    log.info("")

    # ----- Find test images from split_file -----
    with open(args.split_file) as f:
        splits = json.load(f)
    test_files = splits.get("test", [])
    if not test_files:
        raise RuntimeError(f"No test files found in split file: {args.split_file}")

    log.info("Files for test split:")
    filepaths: list[Path] = []
    for file in test_files:
        file_path = input_dir / file
        log.info(f"  - {extract_short_name_from_TSX_filepath(file_path)}")
        if not file_path.is_file():
            raise RuntimeError(f"Expected file for test split does not exist: {file_path}")
        filepaths.append(file_path)

    # ----- Set seeds -----
    np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # ----- Process -----
    start_time = datetime.now()
    log.info(
        f"{Colors.GREEN}Starting the creation of the dataset {dataset_name} at {start_time}{Colors.RESET}",
    )

    process_test_dataset(
        filepaths,
        output_dir / dataset_name,
    )
    end_time = datetime.now()
    log.info(f"{Colors.GREEN}Processing completed successfully!{Colors.RESET}")
    log.info(
        f"Started: {start_time}. Finished: {end_time}. Total duration: {end_time - start_time}."
    )


if __name__ == "__main__":
    main()
