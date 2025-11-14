"""Re-evaluate past W&B runs on a new test dataset.

For each run matching filtering criteria (e.g., config["model"]["criterion"]["lmbda"] == 100),
this script:
    1. Finds its checkpoint path via config["paths"]["output_dir"].
    2. Loads the corresponding LightningModule (SARDDCModule).
    3. Rebuilds the test dataloader using config["data"].
    4. Runs model.test() on the new dataset.
    5. Updates W&B summary:
        - Moves existing test/* metrics to old_test/*
        - Adds new test/* metrics from the new evaluation
        - Logs a timestamp in config["retested_on"]
"""

import warnings
from datetime import datetime
from pathlib import Path

import hydra
import rootutils
import torch
import wandb
from lightning import LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.cuda.amp.autocast.*",
)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from src.data.sar_datamodule import TSXSSCDataModule  # noqa: E402
# from src.models.sar_ddc_module import SARDDCModule  # noqa: E402

# =============== User settings ===============
ENTITY = "cedric-leonard"
PROJECT = "SAR_DDC-RD-curve"
CHECKPOINT_NAME = "last.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FILTERS_CONFIG = [
    # ("model.criterion.lmbda", "==", 100),
    ("retested_on", "is_none", None),  # skip already re-tested runs
    # ("_timestamp", ">", datetime(2024, 10, 1).timestamp()),  # skip too old runs
]
# ============================================


def check_single_condition(value, op, test_value) -> bool:
    """Check a single condition against a W&B config dict."""
    if op == "==":
        return value == test_value
    elif op == "!=":
        return value != test_value
    elif op == "is_none":
        return value is None
    elif op == "exists":
        return value is not None
    elif op == ">":
        return value is not None and float(value) > float(test_value)
    elif op == "<":
        return value is not None and float(value) < float(test_value)
    else:
        raise ValueError(f"Unsupported filter op: {op}")


def run_matches_config_filters(run_cfg: dict) -> bool:
    """Return True if a run satisfies *all* filters."""
    conf = OmegaConf.create(run_cfg)
    for key, op, test_value in FILTERS_CONFIG:
        value = OmegaConf.select(conf, key)
        if not check_single_condition(value, op, test_value):
            return False
    return True


def instantiate_model_and_load_weights(hydra_cfg: DictConfig, ckpt_path: Path) -> LightningModule:
    """Instantiate the model from training config and load weights from checkpoint."""
    print(f"Instantiating model <{hydra_cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(hydra_cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"Loaded checkpoint state_dict with message: {msg}")
    model.to(DEVICE)
    model.eval()
    return model


def build_test_dataloader(hydra_cfg: DictConfig):
    """Rebuild the test dataloader from the config."""
    datamodule = hydra.utils.instantiate(hydra_cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    return datamodule.test_dataloader()


def evaluate_model(model: LightningModule, test_loader):
    """Run Lightning test loop and return metrics dict."""
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )
    results = trainer.test(model, dataloaders=test_loader, verbose=False)
    # results is a list of dicts (one per dataloader)
    print(f"  Obtained test metrics ({len(results)=}): {results}")
    return results[0] if results else {}


def rename_old_test_metrics(summary_dict: dict) -> dict:
    """Move existing test/* metrics to old_test/* to preserve them."""
    updated = {}
    for k, v in summary_dict.items():
        if k.startswith("test/"):
            updated[f"old_{k}"] = v
        else:
            updated[k] = v
    return updated


def update_wandb_run(run, new_metrics: dict):
    """Update W&B summary and config for the run."""
    print(f"  Updating W&B summary for run {run.id}")
    # Backup old test metrics
    summary = run.summary._json_dict
    summary = rename_old_test_metrics(summary)

    # Add new ones
    summary.update(new_metrics)

    # Update summary and mark the timestamp in config
    run.summary._json_dict = summary
    run.config["retested_on"] = datetime.now().isoformat()
    run.update()
    print("  Updated successfully.\n")


def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    print(f"Found {len(runs)} total runs in {ENTITY}/{PROJECT}")

    matching_runs = [r for r in runs if run_matches_config_filters(r.config)]
    # matching_runs = [r for r in runs if r.id == "tmbzj6t9"]

    print(f"{len(matching_runs)} runs match the filters: {FILTERS_CONFIG}")

    for run in matching_runs:
        print(f"Processing run {run.id} ({run.name})")

        output_dir = Path(run.config["paths"]["output_dir"])
        ckpt_path = Path(output_dir) / "checkpoints" / CHECKPOINT_NAME
        cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
        if isinstance(cfg, DictConfig):
            hydra_cfg = cfg

        model = instantiate_model_and_load_weights(hydra_cfg, ckpt_path)
        test_loader = build_test_dataloader(hydra_cfg)
        metrics = evaluate_model(model, test_loader)
        update_wandb_run(run, metrics)

        # # --- Load model and data ---
        # cfg = run.config
        # try:
        #     model = load_model_from_run(cfg)
        #     test_loader = build_test_dataloader(cfg["data"])
        # except Exception as e:
        #     print(f"  ❌ Failed to load run {run.id}: {e}")
        #     continue

        # # --- Evaluate ---
        # try:
        #     metrics = evaluate_model(model, test_loader)
        # except Exception as e:
        #     print(f"  ❌ Failed to evaluate run {run.id}: {e}")
        #     continue

        # # --- Update W&B ---
        # try:
        #     update_wandb_run(run, metrics)
        # except Exception as e:
        #     print(f"  ❌ Failed to update W&B for run {run.id}: {e}")
        #     continue


if __name__ == "__main__":
    main()
