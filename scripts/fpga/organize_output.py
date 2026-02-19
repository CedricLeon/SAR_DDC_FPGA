import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def make_a_nice_run_name(cfg: Any) -> str:
    """Generate a descriptive run name based on important parameters.

    Logic duplicated from src.utils.utils to avoid complex imports in Vitis runtime.
    """

    # Helper to safe get from DictConfig or dict
    def get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return obj.get(key, default)

    model_config = cfg.model.net
    model_target = get(model_config, "_target_", "")

    if "ResidualScaleHyperprior" in model_target:
        model = "ResSHyp"
    elif "Merlin" in model_target:
        model = "Merlin"
    else:
        model = "UnknownModel"

    activation = get(model_config, "activation", "gdn")
    seed = cfg.get("seed", None)

    criterion = cfg.model.criterion
    lmbda = get(criterion, "lmbda", None)

    return f"{model}-{activation}_s{seed}_L{lmbda}"


def create_manifest(path: Path, cfg: Any, run_name: str, original_run_dir: str):
    """Create a JSON manifest with run metadata."""

    # 1. Extract Training Timestamp from Run Directory path
    # Expected format: .../YYYY-MM-DD_HH-MM-SS/N or .../YYYY-MM-DD_HH-MM-SS/
    trained_at = "Unknown"
    try:
        parts = Path(original_run_dir).parts
        for part in reversed(parts):
            # Simple check for timestamp-like string (len 19: 4+1+2+1+2+1+2+1+2+1+2)
            if len(part) == 19 and "_" in part and "-" in part:
                # Verify it parses to date to be sure
                try:
                    datetime.strptime(part, "%Y-%m-%d_%H-%M-%S")
                    trained_at = part
                    break
                except ValueError:
                    continue
    except Exception:
        pass

    # 2. Build Metadata Dictionary
    meta = {
        "model_name": run_name,
        "compiled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_at": trained_at,
        "original_run_dir": original_run_dir,
        "seed": cfg.get("seed", "Unknown"),
        # Flattened Key Params
        "lambda": cfg.model.criterion.get("lmbda", "Unknown"),
        "activation": cfg.model.net.get("activation", "Unknown"),
    }

    # Optional Flattened Params
    loss_target = cfg.model.criterion.get("_target_", None)
    if loss_target:
        meta["loss"] = loss_target.split(".")[-1]  # Short name

    # Check for no_output_padding in model config (used in conv layers)
    # Recursively checking might be too much, checking top level net config
    if "no_output_padding" in cfg.model.net:
        meta["no_output_padding"] = cfg.model.net.no_output_padding

    manifest_path = path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Created manifest at {manifest_path}")

    # Create a visible marker file for file explorer convenience
    marker_path = path / f"MODEL_IS_{run_name}.txt"
    marker_path.touch()


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to the original Hydra Run Directory"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the compiled output directory (e.g. Model_pt)",
    )
    parser.add_argument(
        "--output_mount_point",
        type=str,
        default="DDC_FPGA/results/fpga",
        help="Root directory where to store compiled models and the symlink.",
    )
    # Default behavior is to override. Use --no-override to prevent it.
    parser.add_argument(
        "--no-override",
        dest="override",
        action="store_false",
        help="If set, do NOT override existing model directory (exit with error).",
    )
    parser.set_defaults(override=True)

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    # Ensure we use absolute path for the mount point relative to where script is called or project root
    # Ideally absolute, but here relative to project root (usually cwd)
    output_root = Path(args.output_mount_point).resolve()
    compiled_models_dir = output_root / "compiled_models"

    compiled_models_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        print(f"Error: Run config directory {run_dir} does not exist.")
        sys.exit(1)

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        sys.exit(1)

    print(f"Processing output for run: {run_dir}")

    # 1. Load Hydra Config
    config_path = run_dir / ".hydra" / "config.yaml"

    # Default name if config fails (without timestamp, will rely on force override if needed)
    final_name = f"{source_dir.name}"

    cfg = None

    if not config_path.exists():
        print(f"Warning: .hydra/config.yaml not found in {run_dir}. Cannot generate nice name.")
    else:
        try:
            cfg = OmegaConf.load(config_path)

            nice_name = make_a_nice_run_name(cfg)
            final_name = f"{nice_name}_pt"

            # Save annotated config to source dir
            OmegaConf.save(cfg, source_dir / "train_config.yaml")

        except Exception as e:
            print(f"Error processing config: {e}")
            cfg = None  # invalid config

    # 2. Add Manifest
    if cfg:
        create_manifest(source_dir, cfg, final_name, str(run_dir))
    else:
        # Create a basic manifest if config failed
        with open(source_dir / "manifest.json", "w") as f:
            json.dump(
                {
                    "model_name": final_name,
                    "compiled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "Config load failed, limited metadata.",
                },
                f,
                indent=4,
            )

        # Create marker file also for failed config case
        (source_dir / f"MODEL_IS_{final_name}.txt").touch()

    # 3. Move Directory to Storage
    destination_path = compiled_models_dir / final_name

    if destination_path.exists():
        if not args.override:
            print(f"Error: Destination {destination_path} exists and --no-override was specified.")
            sys.exit(1)
        else:
            print(f"Warning: Destination {destination_path} exists. Removing it (Overwrite Mode).")
            shutil.rmtree(destination_path)

    print(f"Moving {source_dir} -> {destination_path}")
    shutil.move(str(source_dir), str(destination_path))

    # 4. Update Symlink
    symlink_path = output_root / "active_model"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    # Create relative symlink for portability
    # symlink -> compiled_models/name
    relative_target = Path("compiled_models") / final_name
    os.symlink(relative_target, symlink_path)

    print(f"Updated symlink: {symlink_path} -> {relative_target}")
    print(f"Success! Model ready at: {symlink_path}")


if __name__ == "__main__":
    main()
