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
from typing import Any, Mapping

import hydra
import rootutils
import torch
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from src.callbacks.compare_reconstruction_to_gt import CompareReconstructionToGT  # noqa: E402

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.cuda.amp.autocast.*",
)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# =============== User settings ===============
ENTITY = "cedric-leonard"
PROJECT = "SAR_DDC_FPGA"
CHECKPOINT_NAME = "last.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FILTERS_CONFIG = [
    ("model.criterion.lmbda", "in", [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]),
    ("seed", "!=", 42),  # Filter by inequality
    ("model.net.activation", "!=", "gdn1"),  # Filter string inequality
    ("retested_on", "is_none", None),
]


def check_single_condition(value, op, test_value) -> bool:
    """Check a single condition against a W&B config dict."""
    if op == "==":
        return value == test_value
    elif op == "!=":
        return value != test_value
    elif op == "in":
        return value in test_value
    elif op == "not in":
        return value not in test_value
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
    print(f"    Instantiating model <{hydra_cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(hydra_cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"    Loaded checkpoint state_dict with message: {msg}")
    model.to(DEVICE)
    model.eval()
    return model


def build_test_dataloader(hydra_cfg: DictConfig):
    """Rebuild the test dataloader from the config."""
    datamodule = hydra.utils.instantiate(hydra_cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    return datamodule.test_dataloader()


class DictLogger(Logger):
    """A dummy logger that captures metrics in a dictionary."""

    def __init__(self):
        super().__init__()
        self._metrics = {}

    @property
    def name(self) -> str:
        """Name of the logger."""
        return "DictLogger"

    @property
    def version(self) -> str:
        """Version of the logger."""
        return "0.1"

    @property
    def experiment(self) -> Any:
        """The callback calls self.logger.experiment.log(...)"""
        return self

    def log_hyperparams(self, params: Any, *args, **kwargs):
        """DummyLogger."""
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None):
        """Update the internal metrics dictionary with new values."""
        self._metrics.update(metrics)

    def log(self, metrics: Mapping[str, Any]):
        """Custom log method to support .experiment.log({...}) style calls."""
        self._metrics.update(metrics)

    def save(self):
        """DummyLogger."""
        pass

    def finalize(self, status: str):
        """DummyLogger."""
        pass


def evaluate_model_captured(model: LightningModule, test_loader, hydra_cfg: DictConfig):
    """Run Lightning test loop and return metrics dict, capturing callback logs."""

    # Setup dummy logger to capture callback outputs
    dict_logger = DictLogger()

    # Instantiate callbacks from config if available
    callbacks = []
    if "callbacks" in hydra_cfg and "compare_recon_to_gt" in hydra_cfg.callbacks:
        gt_callback = hydra.utils.instantiate(hydra_cfg.callbacks.compare_recon_to_gt)
        callbacks.append(gt_callback)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=dict_logger,  # Pass our dummy logger
        callbacks=callbacks,
        enable_checkpointing=False,
    )
    gt_callback.on_fit_start(trainer, model)  # Manually call to setup any internal state

    results = trainer.test(model, dataloaders=test_loader, verbose=False)

    final_metrics = {}
    if results:
        final_metrics.update(results[0])

    # Add metrics captured by the callback (e.g. from DictLogger)
    if dict_logger._metrics:
        print(f"    Captured {len(dict_logger._metrics)} additional metrics from callbacks.")
        for k, v in dict_logger._metrics.items():
            if not isinstance(v, wandb.Image):
                final_metrics[k] = v

    return final_metrics


def clean_summary_dict(summary_dict: dict) -> dict:
    """Remove 'old_test/*' keys and 'test/*' keys to start fresh."""
    cleaned = {}
    for k, v in summary_dict.items():
        if k.startswith("old_test/"):
            continue  # Remove clutter from previous runs
        if k.startswith("test/"):
            continue  # Remove current wrong metrics (will be replaced)
        cleaned[k] = v
    return cleaned


def update_wandb_run(run, new_metrics: dict):
    """Update W&B summary and config for the run."""
    print(f"    Updating W&B summary for run {run.id}")

    # Get current summary
    summary = run.summary._json_dict

    # Filter test keys for display
    test_keys_before = {k: v for k, v in summary.items() if "test/" in k}
    print(
        f"    [Before] {len(test_keys_before)} test metrics found. test/psnr={test_keys_before.get('test/psnr', 'N/A')}dB, test/psnr_merlin={test_keys_before.get('test/psnr_merlin', 'N/A')}dB."
    )
    summary = clean_summary_dict(summary)
    summary.update(new_metrics)

    # actually summary is dict, let's just look at new_metrics
    print(
        f"    [After] {len(new_metrics)} new test metrics to be saved. test/psnr={new_metrics.get('test/psnr', 'N/A')}dB, test/psnr_merlin={new_metrics.get('test/psnr_merlin', 'N/A')}dB."
    )

    # Update summary and mark the timestamp in config
    run.summary._json_dict = summary
    run.config["retested_on"] = datetime.now().isoformat()
    run.update()


def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    print(f"Found {len(runs)} total runs in {ENTITY}/{PROJECT}")

    # ----- Filtering -----
    # Apply FILTERS_CONFIG
    matching_runs = [r for r in runs if run_matches_config_filters(r.config)]
    # Uncomment the following line to test on a single run (replace ID with a valid one)
    # matching_runs = [r for r in matching_runs if r.id in ["ykihmv1p"]]
    print(f"{len(matching_runs)} runs match the filters: {FILTERS_CONFIG}")

    for i, run in enumerate(matching_runs):
        print(
            f"\n\033[32mProcessing run {i + 1}/{len(matching_runs)}: ID={run.id} ({run.name}), lambda={run.config.get('lambda', None)}, seed={run.config.get('seed', None)}, created {run.created_at}...\033[0m"
        )

        output_dir = Path(run.config["paths"]["output_dir"])
        ckpt_path = Path(output_dir) / "checkpoints" / CHECKPOINT_NAME
        cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
        if isinstance(cfg, DictConfig):
            hydra_cfg = cfg

        try:
            model = instantiate_model_and_load_weights(hydra_cfg, ckpt_path)
            test_loader = build_test_dataloader(hydra_cfg)
            metrics = evaluate_model_captured(model, test_loader, hydra_cfg)
            update_wandb_run(run, metrics)
        except Exception as e:
            print(f"    ERROR processing run {run.id}: {e}")
            import traceback

            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
