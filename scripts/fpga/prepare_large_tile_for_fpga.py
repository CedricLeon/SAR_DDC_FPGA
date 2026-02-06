#!/usr/bin/env python3
"""Prepare large tiles for FPGA evaluation.

Loads raw patch (CoSAR complex or npy), symmetrizes it, and saves as input.npy.
Copies Ground Truths as gt_merlin.npy and gt_adam.npy.
Example usage:
`python scripts/fpga/prepare_large_tile_for_fpga.py --raw data/visualization/raw_Hamburg_patch_[11000:12024-8500:9524].npy --merlin data/visualization/for_evaluations/denoised_by_MERLIN_[11000:12024-8500:9524]_logI.npy --adam data/visualization/for_evaluations/denoised_by_ADAM-NOC_[11000:12024-8500:9524]_logI.npy`
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Add src to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.sar_utils import symmetrize


def main():
    """Main function to prepare large tile for FPGA evaluation."""
    parser = argparse.ArgumentParser(description="Prepare evaluation data for FPGA.")
    parser.add_argument("--raw", required=True, help="Path to raw patch (.npy or .cos)")
    parser.add_argument("--merlin", help="Path to MERLIN GT (.npy)")
    parser.add_argument("--adam", help="Path to ADAM GT (.npy)")

    args = parser.parse_args()

    # Create output directory
    patch_name_coordinates = (
        os.path.basename(args.raw).replace("raw_", "").replace("patch_", "").replace(".npy", "")
    )
    output_dir = os.path.join("data/fpga_eval/", patch_name_coordinates)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Process Raw Input
    print(f"Processing raw input: {args.raw}")
    if not args.raw.endswith(".npy"):
        raise ValueError("Currently only .npy raw files are supported.")

    input_data = np.load(args.raw, allow_pickle=True)  # [H, W, 2]
    if input_data is None:
        raise ValueError("Failed to load input data.")
    input_data = symmetrize(input_data)

    # Save input.npy
    input_path = os.path.join(output_dir, "raw_input_symmetrized.npy")
    np.save(input_path, input_data)
    print(f"Saved input to {input_path}")

    # 2. Process MERLIN GT
    if args.merlin:
        print(f"Processing MERLIN GT: {args.merlin}")
        merlin_path = os.path.join(output_dir, "GT_MERLIN.npy")

        merlin_data = np.load(args.merlin, allow_pickle=True)
        print(f"{merlin_data.shape=}")
        # Convert from logI to linA
        if "_logI" in args.merlin:
            merlin_data = np.sqrt(np.exp(merlin_data))
            merlin_path = merlin_path.replace(".npy", "_linA.npy")
        else:
            print(
                "Warning: MERLIN GT does not seem to be in logI format, no modifications applied."
            )
        np.save(merlin_path, merlin_data)
        print(f"Saved MERLIN GT to {merlin_path}")

    # 3. Process ADAM GT
    if args.adam:
        print(f"Processing ADAM-NOC GT: {args.adam}")
        adam_path = os.path.join(output_dir, "GT_ADAM-NOC.npy")

        adam_data = np.load(args.adam, allow_pickle=True)
        print(f"{adam_data.shape=}")
        # Convert from logI to linA
        if "_logI" in args.adam:
            adam_data = np.sqrt(np.exp(adam_data))
            adam_path = adam_path.replace(".npy", "_linA.npy")
        else:
            print(
                "Warning: ADAM-NOC GT does not seem to be in logI format, no modifications applied."
            )
        np.save(adam_path, adam_data)
        print(f"Saved ADAM-NOC GT to {adam_path}")

    print(f"Preparation complete. Ready to transfer {output_dir} to FPGA.")


if __name__ == "__main__":
    main()
