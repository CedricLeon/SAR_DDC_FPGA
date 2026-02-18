import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to sys path
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.components.compressai_dpu import get_scale_table  # noqa: E402
from src.models.components.res_scale_hyperprior_dpu import (  # noqa: E402
    ResidualScaleHyperpriorPatched,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to input .ckpt file")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.ckpt}")
    print(f"Saving exported parameters to: {args.output}")

    # ----- 1. Load Config & Instantiate Model -----
    # We need to instantiate the model to call .update() method which populates the tables
    config_path = Path(args.ckpt).parent.parent / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    print(f"Reading config from {config_path}...")
    conf = OmegaConf.load(config_path)
    model_params = conf.model.net
    if "_target_" in model_params:
        model_params.pop("_target_")
    model_params["export_dpu"] = True
    print(f"Instantiating model with: {model_params}")
    model = ResidualScaleHyperpriorPatched(**model_params)

    # ----- 2. Load Weights -----
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:  # Lightning checkpoint format
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # ----- 3. Force Update -----
    print("Force entropy models update (populate tables)...")

    # Explicitly update components because CompressionModel.update() skips "Patched" components (Patched components are no instances of compressai.entropy_models.EntropyModel)

    print("  -> Updating EntropyBottleneck...")
    model.entropy_bottleneck.update(force=True)
    print("  -> Updating GaussianConditional...")
    gc = model.gaussian_conditional

    # Check if scale_table is populated (from checkpoint) or needs initialization
    if gc.scale_table.numel() == 0:
        print("     No scale table found in checkpoint, using default log-scale table.")
        scale_table = (
            get_scale_table()
        )  # Default Log-Scale table from CompressAI see https://interdigitalinc.github.io/CompressAI/models.html
        gc.update_scale_table(scale_table, force=True)
    else:
        print("     Using scale table from checkpoint.")
        gc.update_scale_table(gc.scale_table, force=True)

    # ----- 4. Extract parameters -----
    # We only extract and export parameters necessary for inference

    eb = model.entropy_bottleneck
    eb_medians = (
        eb.quantiles[:, 0, 1].detach().cpu().numpy()
    )  # Ensure medians extraction quantiles is (C, 1, 3). Medians at index 1.
    eb_quantized_cdf = eb._quantized_cdf.detach().cpu().numpy().astype(np.int32)
    eb_offset = eb._offset.detach().cpu().numpy().astype(np.int32)
    eb_cdf_length = eb._cdf_length.detach().cpu().numpy().astype(np.int32)

    if eb_quantized_cdf.size == 0:
        raise ValueError("Error: EntropyBottleneck CDF is empty! Update failed.")

    gc = model.gaussian_conditional
    gc_scale_table = gc.scale_table.detach().cpu().numpy().astype(np.float32)
    # Added for Real Inference (C++ rANS)
    gc_quantized_cdf = gc.quantized_cdf.detach().cpu().numpy().astype(np.int32)
    gc_cdf_length = gc.cdf_length.detach().cpu().numpy().astype(np.int32)
    gc_offset = gc.offset.detach().cpu().numpy().astype(np.int32)

    if (
        gc.scale_table is None or gc_scale_table.size == 0
    ):  # Should never be the case due to update above
        raise RuntimeError("Error: GaussianConditional scale_table is None!")

    print("Checks:")
    print(f"  - EB medians shape: {eb_medians.shape}")
    print(f"  - EB CDF shape: {eb_quantized_cdf.shape}")
    print(f"  - EB offset shape: {eb_offset.shape}")
    print(f"  - EB CDF length shape: {eb_cdf_length.shape}")
    print(f"  - GC scale table shape: {gc_scale_table.shape}")
    print(f"  - GC CDF shape: {gc_quantized_cdf.shape}")

    np.savez(
        args.output,
        # Legacy/DPU keys
        eb_quantized_cdf=eb_quantized_cdf,
        eb_offset=eb_offset,
        eb_cdf_length=eb_cdf_length,
        eb_medians=eb_medians,
        gc_scale_table=gc_scale_table,
        # New Real Inference keys
        gc_quantized_cdf=gc_quantized_cdf,
        gc_cdf_length=gc_cdf_length,
        gc_offset=gc_offset,
    )
    print(f"Entropy models parameters saved to {args.output}.")
