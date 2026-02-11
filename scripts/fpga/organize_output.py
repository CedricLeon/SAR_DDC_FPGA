import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def make_a_nice_run_name(cfg: Dict[str, Any]) -> str:
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

    return f"{model}-{activation}_s{seed}_ʎ{lmbda}"


def main():
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
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    source_dir = Path(args.source_dir).resolve()

    if not run_dir.exists():
        print(f"Error: Run config directory {run_dir} does not exist.")
        sys.exit(1)

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        sys.exit(1)

    print(f"Processing output for run: {run_dir}")

    # 1. Load Hydra Config
    config_path = run_dir / ".hydra" / "config.yaml"

    if not config_path.exists():
        print(f"Warning: .hydra/config.yaml not found in {run_dir}. Cannot generate nice name.")
        final_name = source_dir.name
    else:
        try:
            cfg = OmegaConf.load(config_path)

            # Set original_run_dir using the passed argument
            OmegaConf.set_struct(cfg, False)
            cfg.original_run_dir = str(run_dir)
            print(f"Set original_run_dir in config: {run_dir}")

            nice_name = make_a_nice_run_name(cfg)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            final_name = f"{nice_name}_pt_{timestamp}"

            # Save annotated config to source dir
            OmegaConf.save(cfg, source_dir / "train_config.yaml")
            print(f"Saved annotated training config to {source_dir / 'train_config.yaml'}")

        except Exception as e:
            print(f"Error processing config: {e}")
            final_name = source_dir.name

    # 2. Rename Directory
    if final_name != source_dir.name:
        new_path = source_dir.parent / final_name
        print(f"Renaming {source_dir.name} -> {new_path.name}")
        shutil.move(str(source_dir), str(new_path))
        print(f"Success! Output is at: {new_path}")
    else:
        print(f"No renaming performed. Output is at: {source_dir}")


if __name__ == "__main__":
    main()
