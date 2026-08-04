"""Re-evaluate past W&B runs on a new test dataset.

For each run matching filtering criteria (e.g., config["model"]["criterion"]["lmbda"] == 100),
this script:
    1. Finds its checkpoint path via config["paths"]["output_dir"].
    2. Loads the corresponding LightningModule (SARDDCModule).
    3. Rebuilds the test dataloader using config["data"].
    4. Runs model.test() on the new dataset.
    5. Updates W&B summary:
        - Erases existing old_test/* and test/* metrics
        - Adds new test/* metrics from the new evaluation
        - Logs a timestamp in config["retested_on"]
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import hydra
import rootutils
import torch
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.evaluation import run_dual_evaluation

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
    # ("seed", "==", 42),
    ("seed", "in", [0, 1, 2, 3, 4, 5]),  # exclude seed=42 (manual test run)
    # Restrict to DPU-deployable architecture to match FPGA comparison set.
    # Comment these two lines out to re-evaluate all architectures.
    # ("model.net.activation", "==", "relu"),
    # ("model.net.no_output_padding", "==", True),
    # ("retested_on", "is_none", None),
    # ("retested_on", "is_none_or_older_than", datetime(2026, 2, 12, 23, 59, 0)),
    # ("retested_on", "is_after", datetime(2026, 2, 11, 10, 0, 0)),
    # ("retested_on", "is_before", datetime(2026, 2, 11, 10, 0, 0)),
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
    elif op == "is_before":
        if value is None:
            return False
        return datetime.fromisoformat(str(value)) < test_value
    elif op == "is_after":
        if value is None:
            return False
        return datetime.fromisoformat(str(value)) > test_value
    elif op == "is_none_or_is_before":
        if value is None:
            return True
        return datetime.fromisoformat(str(value)) < test_value
    elif op == "is_none_or_is_after":
        if value is None:
            return True
        return datetime.fromisoformat(str(value)) > test_value
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


def instantiate_model_and_load_weights(
    hydra_cfg: DictConfig, ckpt_path: Path, force_patch_version: bool = False
) -> LightningModule:
    """Instantiate the model from training config and load weights from checkpoint."""
    print(f"    Instantiating model <{hydra_cfg.model._target_}>")
    if force_patch_version:
        hydra_cfg.model.net.export_dpu = True
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


def evaluate_model_captured(
    model: LightningModule,
    full_loader,
    hydra_cfg: DictConfig,
    log_dir: Path,
    run_id: str = "unknown",
):
    """Run Lightning test loop and return metrics dict, capturing callback logs."""

    # Setup dummy logger to capture callback outputs
    dict_logger = DictLogger()

    # Use the original run directory for logs so that reconstruction images are saved there
    print(f"    Logs (images) for this run {run_id} will be saved in: {log_dir}")

    # Instantiate callbacks from config if available
    callbacks = []
    if "callbacks" in hydra_cfg and "compare_recon_to_gt" in hydra_cfg.callbacks:
        # Convert to a plain dict so we can freely add/remove keys without hitting OmegaConf struct-mode restrictions (Hydra configs are read-only by default).
        cb_dict: Dict[str, Any] = OmegaConf.to_container(
            hydra_cfg.callbacks.compare_recon_to_gt, resolve=True
        )  # type: ignore[assignment]
        cb_dict["verbose"] = True
        # Migrate old API keys (blend_method/stride) to the new patch_infer API (blend_profile/overlap).
        cb_dict.pop("split_large_patch", None)
        cb_dict.pop("blend_method", None)
        cb_dict.pop("stride", None)
        cb_dict.setdefault("blend_profile", "sigmoid")
        cb_dict.setdefault("overlap", 16)

        gt_callback = hydra.utils.instantiate(cb_dict)
        callbacks.append(gt_callback)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=dict_logger,  # Pass our dummy logger
        callbacks=callbacks,
        enable_checkpointing=False,
        default_root_dir=str(log_dir),
        # limit_test_batches=1, # Quick debug: test on only 1 batch
    )
    # Manually call on_fit_start to set up callback internal state (e.g. load reference images).
    # Guard: gt_callback is only defined when the callback config existed.
    if callbacks:
        gt_callback.on_fit_start(trainer, model)  # Manually call to setup any internal state

    # Run evaluation on full_test and on subset300
    final_metrics = run_dual_evaluation(
        trainer=trainer,
        model=model,
        dataloaders=full_loader,
        hdf5_dir=hydra_cfg.data.get("hdf5_dir", None),
        batch_size=hydra_cfg.data.get("batch_size", 1),
        num_workers=hydra_cfg.data.get("num_workers", 0),
        skip_full_test=False,
    )

    # Add metrics captured by the callback (e.g. from DictLogger)
    media_artifacts = {}
    if dict_logger._metrics:
        print(f"    Captured {len(dict_logger._metrics)} additional metrics from callbacks.")
        for k, v in dict_logger._metrics.items():
            if isinstance(v, wandb.Image):
                media_artifacts[k] = v
            else:
                final_metrics[k] = v

    return final_metrics, media_artifacts


def clean_summary_dict(summary_dict: dict) -> dict:
    """Remove 'old_test/*' keys, 'test/*' keys, and new dual test set prefix keys to start
    fresh."""
    cleaned = {}
    for k, v in summary_dict.items():
        if k.startswith("test_full/") or k.startswith("test/") or k.startswith("test_sub500/"):
            continue
        cleaned[k] = v
    return cleaned


_SHOW_KEYS = [
    # Noisy
    "test_sub500/psnr_noisy",
    "test_sub500/enl_recon",
    "test_sub500/ratio_mean",
    "test_sub500/ratio_enl",
    # Despeckling quality vs references
    "test_sub500/psnr_adam_noc",
    "test_sub500/psnr_merlin",
    "test_sub500/ssim_merlin",
    "test_sub500/epd_merlin",
    # Compression
    "test_sub500/bpp",
    "test_sub500/bpp_bitstream",
    # Noisy
    "test/psnr_noisy",
    "test/enl_recon",
    "test/ratio_mean",
    "test/ratio_enl",
    # Despeckling quality vs references
    "test/psnr_adam_noc",
    "test/psnr_merlin",
    "test/ssim_merlin",
    "test/epd_merlin",
    # Compression
    "test/bpp",
    "test/bpp_bitstream",
]


def _print_before_after(before: dict, after: dict) -> None:
    """Print a compact Before / After table for the keys listed in _SHOW_KEYS."""
    col_w = 12  # width for value column
    print(f"    {'Metric':<35} {'Before':>{col_w}}  {'After':>{col_w}}")
    print(f"    {'-' * 35} {'-' * col_w}  {'-' * col_w}")
    for key in _SHOW_KEYS:
        b_val = before.get(key)
        a_val = after.get(key)
        b_str = f"{b_val:.4f}" if isinstance(b_val, (int, float)) else "—"
        a_str = f"{a_val:.4f}" if isinstance(a_val, (int, float)) else "—"
        changed = "  ✓" if b_str != a_str else ""
        print(f"    {key:<35} {b_str:>{col_w}}  {a_str:>{col_w}}{changed}")


def diff_all_metrics(before: dict, after: dict, rtol: float = 5e-3) -> list:
    """Print a full before/after diff of every test metric; return the material non-SSIM movers.

    Returns a list of ``(key, relative_change)`` for metrics that moved materially and are *not*
    expected to move. An empty list means the run behaved exactly as the AMP_LIN_99 basis change
    predicts.

    Unlike ``_print_before_after`` (a fixed whitelist), this compares **every** ``test/`` and
    ``test_sub500/`` key present in either summary. The AMP_LIN_99 basis change may only move
    ``ssim_*``, ``ms_ssim_*`` and ``epd_*``. A material move anywhere else (PSNR, MSE, bpp, ENL,
    ratios) is worth seeing, because W&B history is not backed up.

    In practice such moves have two very different causes, which the printed magnitude separates:

    * **~1e-4 relative** — re-running the same checkpoint on a different GPU/cuDNN algorithm
      (measured: PSNR 0.0007 dB, MSE 0.016 %). Not a behaviour change.
    * **1e-2 to 1e-1 relative** — the stored summary predates commit ``5f52f98`` (2026-04-30),
      which added ``nan_to_num``/``clamp(0,1)`` sanitisation of ``x_hat`` in ``test_step``. Those
      summaries were computed from unclamped reconstructions and are simply stale; re-evaluating
      corrects them (e.g. ResSHyp λ1000: psnr_merlin 28.14 → 27.68).

    ``rtol`` defaults to 5e-3: above the natural sensitivity of the variance-ratio metrics
    (``enl_recon``, ``ratio_*`` jitter up to ~1e-3) and well below the stale-summary signal.
    """
    expected = ("ssim", "ms_ssim", "epd")

    def _is_expected(key: str) -> bool:
        return key.split("/", 1)[-1].startswith(expected)

    keys = sorted(k for k in set(before) | set(after) if k.startswith(("test/", "test_sub500/")))
    moved, unexpected, worst = [], [], 0.0
    for k in keys:
        b, a = before.get(k), after.get(k)
        if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
            continue
        if a == b:
            continue
        rel = abs(a - b) / max(abs(b), 1e-12)
        moved.append((k, b, a, rel))
        if not _is_expected(k):
            worst = max(worst, rel)
            if rel > rtol:
                unexpected.append((k, rel))

    print(f"\n    {'Metric':<38}{'Before':>13}{'After':>13}{'Δ':>13}{'rel':>10}")
    print(f"    {'-' * 38}{'-' * 13}{'-' * 13}{'-' * 13}{'-' * 10}")
    for k, b, a, rel in moved:
        flag = "" if _is_expected(k) else ("   <-- UNEXPECTED" if rel > rtol else "   (jitter)")
        print(f"    {k:<38}{b:>13.5f}{a:>13.5f}{a - b:>+13.5f}{rel:>10.2e}{flag}")
    print(f"    ({len(moved)} of {len(keys)} metrics moved, {len(keys) - len(moved)} identical)")
    print(f"    largest non-SSIM/EPD deviation: {worst:.2e} relative (blocks above {rtol:.0e})")

    if unexpected:
        keys = [k for k, _ in unexpected]
        print(
            f"\n    \033[33m{len(unexpected)} non-SSIM metric(s) moved materially: {keys}\033[0m"
        )
        print("    Likely a stale pre-2026-04-30 summary (see docstring); re-evaluating fixes it.")
    return unexpected


def update_wandb_run(
    run_obj, cfg, new_metrics: dict, output_dir: Path, media_artifacts: Optional[dict] = None
):
    """Update W&B summary and config for the run using a resumed run context.

    Args:
        run_obj: The run object returned by wandb.Api().run(...) (used for initial config access)
        cfg: The hydra config (used for project/entity info)
        new_metrics: Dictionary of scalar metrics to update.
        media_artifacts: Dictionary of rich media (images) to log.
    """
    print(f"    Updating W&B summary for run {run_obj.id}")

    # Capture before-update values for the comparison print
    current_summary = run_obj.summary._json_dict
    before_metrics = {k: current_summary.get(k) for k in _SHOW_KEYS}

    # We use the API object to clean the summary first (faster than doing it in the run context)
    # This ensures old keys are actually removed, not just overwritten in history
    cleaned_summary = clean_summary_dict(current_summary)
    run_obj.summary._json_dict = cleaned_summary
    run_obj.update()

    # Log new metrics and artifacts using a resumed run
    # Note: We use cfg.logger.entity/project because run.entity might differ if you have access to multiple orgs

    # Look for entity/project in run_obj if not in cfg (fallback)
    entity = cfg.logger.get("entity", run_obj.entity)
    project = cfg.logger.get("project", run_obj.project)

    with wandb.init(
        entity=entity, project=project, id=run_obj.id, dir=str(output_dir), resume="must"
    ) as run:
        # 1. Log media artifacts (Images, etc.)
        if media_artifacts:
            print(f"    Logging {len(media_artifacts)} media artifacts to W&B...")
            run.log(media_artifacts)

        # 2. Update scalar metrics
        # We update the summary directly to ensure these are the "final" values displayed in the dash
        # logging them normally would add a history step, which is also fine, but updating summary is explicit.
        run.summary.update(new_metrics)

        # 3. Mark the run as retested
        run.config.update({"retested_on": datetime.now().isoformat()}, allow_val_change=True)

    # Print verification
    print(f"    [After] {len(new_metrics)} new test metrics saved.")
    after_metrics = {k: new_metrics.get(k) for k in _SHOW_KEYS}
    _print_before_after(before_metrics, after_metrics)


def filter_runs_by_creation_date(runs: list, limit_date: datetime) -> list:
    """Filter runs created after a specific date, handling timezone offsets.

    Args:
        runs: List of W&B runs.
        limit_date: The naive datetime threshold.
    """
    kept_runs = []
    for r in runs:
        # W&B created_at is usually timezone-aware (UTC)
        r_dt = datetime.fromisoformat(str(r.created_at))

        # If run date is aware and limit is naive, strip tz from run date to compare
        if r_dt.tzinfo is not None and limit_date.tzinfo is None:
            r_dt = r_dt.replace(tzinfo=None)

        if r_dt > limit_date:
            kept_runs.append(r)
    return kept_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Re-evaluate and print the full metric diff, but do NOT write to W&B.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N runs.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Skip (do not write) any run whose non-SSIM/EPD metrics moved materially.",
    )
    parser.add_argument("--run-ids", nargs="+", default=None, help="Only these W&B run ids.")
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="Relative change above which a non-SSIM/EPD metric is reported as material.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    print(f"Found {len(runs)} total runs in {ENTITY}/{PROJECT}")

    # ----- Filtering -----
    matching_runs = [r for r in runs if run_matches_config_filters(r.config)]
    if args.run_ids:
        matching_runs = [r for r in matching_runs if r.id in args.run_ids]
    # matching_runs = [r for r in matching_runs if r.id in list_of_runs_id_in_the_filter]
    # One safe + one problematyic run = ['lwks3okq', 'jvsj0ut4']
    # All 24 runs with NaN problems (GDN + output_padding, for lambdas 50,100,2000,1000 all seeds = ['jvsj0ut4', 'we73n9uj', '5h78ir4k', 'zrw08hbz', 'elp40xh9', 'x4qu6p2x', 'vkd9treb', 'n5gsa2fo', '0rru19sn', '4p4ghwww', 'atv9gmhm', '9tfd1snp', 'ltec20ym', '5tj53usr', 'p9dl5f81', 'f5fs0s9d', 'hg6f3jgu', 'cj2n50np', '5wa47ncm', 'w870mauv', 'ljfcdsty', '0fftmoe3', 'gpesw2xn', 'pn1kgfwh']]
    # matching_runs = [r for r in matching_runs if r.id in ["3nmfvbn0"]]

    # Filter by creation date using the helper to avoid timezone errors
    # matching_runs = filter_runs_by_creation_date(matching_runs, datetime(2026, 2, 11, 10, 0, 0))
    print(f"{len(matching_runs)} runs match the filters: {FILTERS_CONFIG}")
    if args.limit is not None:
        matching_runs = matching_runs[: args.limit]
        print(f"Limited to the first {len(matching_runs)} run(s)")
    if args.dry_run:
        print("\033[33mDRY RUN — re-evaluating and diffing only, W&B will not be written\033[0m")

    stale_runs, written, skipped, failed = [], [], [], []
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
            model = instantiate_model_and_load_weights(
                hydra_cfg, ckpt_path, force_patch_version=False
            )
            test_loader = build_test_dataloader(hydra_cfg)
            metrics, media = evaluate_model_captured(
                model, test_loader, hydra_cfg, output_dir, run.id
            )
            # Diff against the live summary *before* touching it — the only chance to notice
            # a metric moving that shouldn't, since the old values are not backed up anywhere.
            movers = diff_all_metrics(dict(run.summary._json_dict), metrics, args.rtol)
            if movers:
                worst_key, worst_rel = max(movers, key=lambda kv: kv[1])
                stale_runs.append((run.id, run.name, len(movers), worst_key, worst_rel))
            if args.dry_run:
                print("    DRY RUN — W&B not modified.")
                continue
            if movers and args.strict:
                print("    --strict: skipping W&B update for this run.")
                skipped.append(run.id)
                continue
            update_wandb_run(run, hydra_cfg, metrics, output_dir, media)
            written.append(run.id)
        except Exception as e:
            print(f"    ERROR processing run {run.id}: {e}")
            import traceback

            traceback.print_exc()
            failed.append(run.id)
            continue

    # ---- End-of-sweep report ----
    print(f"\n{'=' * 100}\nSWEEP SUMMARY")
    print(f"  written : {len(written)}   skipped: {len(skipped)}   failed: {len(failed)}")
    if failed:
        print(f"  failed ids: {failed}")
    if stale_runs:
        print(
            f"\n  {len(stale_runs)} run(s) had non-SSIM/EPD metrics move above {args.rtol:.0e} — "
            "these had stale summaries (pre-2026-04-30 clamp fix) that this sweep also corrected:"
        )
        print(f"    {'run id':<12}{'name':<48}{'#moved':>7}{'worst metric':>26}{'rel':>10}")
        for rid, name, n, key, rel in sorted(stale_runs, key=lambda r: -r[4]):
            print(f"    {rid:<12}{name[:47]:<48}{n:>7}{key[-25:]:>26}{rel:>10.2e}")
    else:
        print("\n  No run moved a non-SSIM/EPD metric materially.")


if __name__ == "__main__":
    main()
