import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torchmetrics
import torchmetrics.functional.image as F
from lightning import LightningModule
from matplotlib.ticker import FuncFormatter
from omegaconf import DictConfig, OmegaConf
from ptflops import get_model_complexity_info  # type: ignore
from torch import nn
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.sar_datamodule import TSXSSCDataModule  # noqa: E402
from src.utils import RankedLogger, extras, process_large_patch  # noqa: E402
from src.utils.constants import amp_max, amp_min  # noqa: E402
from src.utils.sar_utils import load_cosar, symmetrize  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def _best_device() -> torch.device:
    """Select the best available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_run_dir(ckpt_path: Path) -> Path:
    """Get the run directory from the checkpoint path."""
    # ckpt lives in .../runs/<date>/checkpoints/<file>.ckpt
    # run dir = parent.parent
    return ckpt_path.parent.parent


def _load_training_cfg_from_ckpt(ckpt_path: Path) -> Any:
    """Load the original training configuration from the checkpoint's run directory."""
    run_dir = _checkpoint_run_dir(ckpt_path)
    training_config_path = run_dir / ".hydra" / "config.yaml"
    if not training_config_path.exists():
        raise FileNotFoundError(f"Training config not found at {training_config_path}.")
    log.info(f"Loading original training config from {training_config_path}")
    return OmegaConf.load(training_config_path)


def _instantiate_model_and_load_weights(train_cfg: DictConfig, ckpt_path: Path) -> LightningModule:
    """Instantiate the model from training config and load weights from checkpoint."""
    log.info(f"Instantiating model <{train_cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(train_cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
    log.info(f"Loaded checkpoint state_dict with message: {msg}")
    model.eval()
    return model


def _count_layers(module: torch.nn.Module) -> Dict[str, int]:
    """Count the number of layers in the model."""
    counts = {
        "Conv2d": 0,
        "ConvTranspose2d": 0,
        "Linear": 0,
        "GDN": 0,
        "TotalModules": 0,
    }
    for m in module.modules():
        counts["TotalModules"] += 1
        if isinstance(m, nn.Conv2d):
            counts["Conv2d"] += 1
        elif isinstance(m, nn.ConvTranspose2d):
            counts["ConvTranspose2d"] += 1
        elif isinstance(m, nn.Linear):
            counts["Linear"] += 1
        elif m.__class__.__name__ == "GDN":
            counts["GDN"] += 1
    return counts


def _params_and_size_mb(module: torch.nn.Module) -> Tuple[int, int, float]:
    """Compute total and trainable parameters, and size in MB."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    state = module.state_dict()
    total_bytes = sum(t.element_size() * t.nelement() for t in state.values())
    return total, trainable, total_bytes / (1024**2)


def _profile_flops_macs_latency(
    module: torch.nn.Module, device: torch.device, input_size: int
) -> Dict[str, Any]:
    """Compute MACs/FLOPs using ptflops only, and measure latency.

    Fails fast if ptflops is unavailable.
    """
    net = getattr(module, "net", module).to(device)
    net.eval()

    macs, params = get_model_complexity_info(
        net,
        (1, input_size, input_size),  # (C,H,W) no batch
        as_strings=False,
        print_per_layer_stat=False,
        verbose=True,
    )
    assert macs is not None, "ptflops returned None for MACs"

    macs_val = int(float(macs))  # supports numpy scalars
    flops_val = int(2 * macs_val)

    # Latency: warmup + measure``
    def time_forward(iters: int = 50, warmup: int = 10) -> float:
        """Measure average forward pass time over `iters` runs after `warmup` runs."""
        x = torch.randn(1, 1, input_size, input_size, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(warmup):
                _ = net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(iters):
                _ = net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
        return (end - start) / iters

    latency = time_forward()

    return {
        "input": [1, 1, input_size, input_size],
        "device": str(device),
        "latency_s": latency,
        "macs": macs_val,
        "flops": flops_val,
    }


def _compute_model_stats(model: LightningModule, input_size: int) -> Dict[str, Any]:
    """Compute various model statistics: layers, parameters, size, MACs/FLOPs, latency."""
    net = getattr(model, "net", model)
    layer_counts = _count_layers(net)
    params_total, params_trainable, weights_size_mb = _params_and_size_mb(net)

    # MACs/FLOPs + CPU latency
    stats_cpu = _profile_flops_macs_latency(net, torch.device("cpu"), input_size)
    # GPU latency (if available)
    latency_gpu = None
    if torch.cuda.is_available():
        stats_gpu = _profile_flops_macs_latency(net, torch.device("cuda"), input_size)
        latency_gpu = stats_gpu.get("latency_s")

    model_stats = {
        "net_class": net.__class__.__name__,
        "layers": layer_counts,
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "weights_size_mb": weights_size_mb,
        "macs": stats_cpu.get("macs"),
        "flops": stats_cpu.get("flops"),
        "latency_cpu_s": stats_cpu.get("latency_s"),
        "latency_gpu_s": latency_gpu,
        "input": stats_cpu.get("input"),
    }

    macs_str = f"{model_stats['macs']:_}" if model_stats.get("macs") is not None else "N/A"
    flops_str = f"{model_stats['flops']:_}" if model_stats.get("flops") is not None else "N/A"
    log.info(
        f"The model {model_stats['net_class']} has {model_stats['parameters_total']:_} parameters, including {model_stats['parameters_trainable']:_} trainable, for a total memory footprint of {model_stats['weights_size_mb']:.2f}MB."
    )
    gpu_part = (
        f" | GPU: {model_stats['latency_gpu_s'] * 1e3:.2f} ms"
        if model_stats["latency_gpu_s"] is not None
        else ""
    )
    log.info(
        f"Latencies — CPU: {model_stats['latency_cpu_s'] * 1e3:.2f} ms{gpu_part}. MACs≈{macs_str} | FLOPs={flops_str}"
    )

    return model_stats


def _standardize_reference_name(name: str) -> str:
    """Standardize reference method name for filenames."""
    return name.replace(" ", "_")


def _load_reference_tile_outputs(methods: List[str], tile_dir: Path) -> Dict[str, Any]:
    """Load reference denoised outputs for given methods from tile directory."""
    refs = {}
    for m in methods:
        key = _standardize_reference_name(m)
        # Expected format: denoised_by_<KEY>_* .npy containing a dict or array
        matches = list(tile_dir.glob(f"denoised_by_{key}_*.npy"))
        if not matches:
            log.warning(
                f"[REF MISSING] Reference output for '{m}' not found in {tile_dir}. "
                "Expected denoised_by_<METHOD>_*.npy. Skipping."
            )
            continue
        refs[m] = np.load(matches[0], allow_pickle=True)
    return refs


def _evaluate_on_test(
    model: LightningModule,
    eval_cfg: DictConfig,
    data_dir: Path | str,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on the test set and compute metrics."""
    # Build datamodule from training cfg.data
    dm = TSXSSCDataModule(
        hdf5_dir=str(data_dir),
        batch_size=eval_cfg.test_batch_size,
        num_workers=eval_cfg.num_workers,
        pin_memory=True,
    )
    dm.prepare_data()
    dm.setup("test")

    model = model.to(device)

    data_range = (0, 1)

    mse = MeanSquaredError().to(device)
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    bpp_list = []
    if eval_cfg.short_test_set:
        log.info("Using short test set for quick evaluation (first 10 batches only).")
        batch_nb = 0
    with torch.inference_mode():
        for batch in tqdm(dm.test_dataloader()):
            real, imag = batch["real"].to(device), batch["imag"].to(device)
            # SARDDC: forward returns dict; MerlinModule not supported here
            out_r = model(real)
            out_i = model(imag)
            if isinstance(out_r, dict) and "x_hat" in out_r:
                recon_r = out_r["x_hat"]
                recon_i = out_i["x_hat"]
                # bpp from criterion requires target; for aggregates, average criterion outputs
                crit_r = model.criterion(out_r, imag)
                crit_i = model.criterion(out_i, real)
                bpp_list.append(((crit_r["bpp"] + crit_i["bpp"]) / 2).detach())
            else:
                # Unsupported for now
                recon_r = out_r
                recon_i = out_i
            recon = 0.5 * (recon_r + recon_i)
            noisy = real + imag

            mse.update(recon, noisy)
            psnr.update(recon, noisy)
            ssim.update(recon, noisy)
            ms_ssim.update(recon, noisy)

            if eval_cfg.short_test_set:
                batch_nb += 1
                if batch_nb >= 10:
                    break

    def to_float(val: Any) -> float:
        """Convert metric value to float."""
        # Some torchmetrics may return Tensor, or (Tensor, Tensor)
        if isinstance(val, tuple):
            val = val[0]
        if hasattr(val, "detach"):
            return float(val.detach().cpu().item())
        try:
            return float(val)
        except Exception:
            return -1.0

    mse_val = to_float(mse.compute())
    psnr_val = to_float(psnr.compute())
    ssim_val = to_float(ssim.compute())
    ms_ssim_val = to_float(ms_ssim.compute())
    results = {
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "ms_ssim": ms_ssim_val,
        "bpp": float(torch.stack(bpp_list).mean().cpu()) if bpp_list else -1.0,
    }

    log.info(f"Results of {model.__class__.__name__} on test set:")
    for key, value in results.items():
        log.info(f"    - {key}: {value:.4f}")

    return results


def _evaluate_tile_and_visualize(
    model: LightningModule,
    cfg: DictConfig,
    save_dir: Path,
) -> Dict[str, Any]:
    """Evaluate the model on a tile and generate visualization figure."""
    tile_path: Path = Path(cfg.tile_path)
    clip_std_factor: float = float(cfg.clip_std_factor)

    # Load tile or pre-extracted patch and symmetrize
    if tile_path.suffix == ".npy":
        patch = np.load(tile_path)
        if patch is None:
            raise FileNotFoundError(f"Failed to load patch from {tile_path}")
        patch = symmetrize(patch)
    else:
        image = load_cosar(tile_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load SAR tile from {tile_path}")
        image = symmetrize(image)  # [H,W,2]
        patch = image[
            cfg.crop_coordinates[0] : cfg.crop_coordinates[0] + cfg.tile_crop_size,
            cfg.crop_coordinates[1] : cfg.crop_coordinates[1] + cfg.tile_crop_size,
            :,
        ]

    eps = 1e-2
    patch_sq = np.square(patch)
    patch_log = np.log(patch_sq + eps)
    patch_norm = (patch_log - 2 * amp_min) / (2 * amp_max - 2 * amp_min)

    device = next(model.parameters()).device
    real = torch.from_numpy(patch_norm[:, :, 0]).to(device).unsqueeze(0).unsqueeze(0)
    imag = torch.from_numpy(patch_norm[:, :, 1]).to(device).unsqueeze(0).unsqueeze(0)

    log.info(f"Loaded and pre-processed tile patch of size {patch.shape}.")

    with torch.inference_mode():
        # Patch-based forward to be consistent and memory-friendly
        metrics_r, recon_r = process_large_patch(
            model=model,
            input=real,
            target=None,
            model_patch_size=cfg.patch_size,
            stride=cfg.stride,
            blend_method=cfg.blend_method,
        )
        metrics_i, recon_i = process_large_patch(
            model=model,
            input=imag,
            target=None,
            model_patch_size=cfg.patch_size,
            stride=cfg.stride,
            blend_method=cfg.blend_method,
        )
    # Average is done later in linear/log domains for visualization/metrics
    bpp_avg = 0.5 * (metrics_r.get("bpp", -1) + metrics_i.get("bpp", -1))

    # Convert to linear amplitude per channel and build log-intensity like in callback
    recon_r_lin = torch.exp(recon_r.squeeze() * (amp_max - amp_min) + amp_min)
    recon_i_lin = torch.exp(recon_i.squeeze() * (amp_max - amp_min) + amp_min)
    I_recon = 0.5 * (recon_r_lin + recon_i_lin)
    recon_logI = torch.log(I_recon + eps)

    # Load references if available
    tile_vis_dir = Path("data/visualization/for_evaluations/")
    refs = _load_reference_tile_outputs(list(cfg.reference_methods), tile_vis_dir)
    nb_refs = len(refs)

    # Build figure similar to callback (noisy, recon, MERLIN GT + histograms)
    noisy_logI = np.log(patch_sq[:, :, 0] + patch_sq[:, :, 1] + eps)
    recon_np = recon_logI.squeeze().cpu().numpy()
    # Prepare reference maps (per method)
    refs_logI: Dict[str, np.ndarray] = {}
    for m, obj in refs.items():
        key = _standardize_reference_name(m)
        try:
            if isinstance(obj, dict):
                # deepdespeckling format: amplitude in linear scale
                A = obj["denoised"]["full"]
                refs_logI[key] = np.log(np.square(A) + eps)
            else:
                refs_logI[key] = np.array(obj)
        except Exception as e:
            log.warning(f"Reference '{m}' has unexpected format: {e}")

    fig, axes = plt.subplots(2, 2 + nb_refs, figsize=(10 + (nb_refs * 5), 10))

    def clip_image(x):
        """Clip image values to <clip_std_factor> standard deviations around the mean."""
        m, s = x.mean(), x.std()
        x = np.clip(x, m - clip_std_factor * s, m + clip_std_factor * s)
        # x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    def plot_histogram(ax, data, title):
        """Plot histogram of data on given axis with mean and std lines."""
        ax.set_title(title)
        ax.hist(data.flatten(), bins=50, alpha=0.7, color="blue")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, loc: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}")
        )
        mean = data.mean()
        std = data.std()
        ax.axvline(mean, color="red", linestyle="--", label="Mean")
        ax.axvline(
            mean + clip_std_factor * std,
            color="green",
            linestyle="--",
            label=f"Mean + {clip_std_factor}*Std",
        )
        ax.axvline(
            mean - clip_std_factor * std,
            color="green",
            linestyle="--",
            label=f"Mean - {clip_std_factor}*Std",
        )
        ax.legend(fontsize=8)

    im0 = axes[0, 0].imshow(clip_image(noisy_logI), cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Noisy Log-I")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    plot_histogram(axes[1, 0], noisy_logI, "Noisy Log-I Histogram")

    im1 = axes[0, 1].imshow(clip_image(recon_np.squeeze()), cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Recon Log-I")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    plot_histogram(axes[1, 1], recon_np, "Recon Log-I Histogram")

    for i, (m, ref) in enumerate(refs_logI.items()):
        im2 = axes[0, 2 + i].imshow(clip_image(ref), cmap="gray")
        axes[0, 2 + i].axis("off")
        axes[0, 2 + i].set_title(f"{m} Log-I")
        fig.colorbar(im2, ax=axes[0, 2 + i], shrink=0.8)
        plot_histogram(axes[1, 2 + i], ref, f"{m} Log-I Histogram")

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "tile_comparison_logI.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    log.info(f"Saved tile comparison figure to {fig_path}")

    # Save the reconstruction image alone with high quality
    recon_save = clip_image(recon_np)
    plt.imsave(save_dir / "tile_reconstruction_logI.png", recon_save, cmap="gray", dpi=300)

    # Metrics on tile vs all references (MSE/PSNR/SSIM/MS-SSIM)
    data_range = float(recon_logI.max() - recon_logI.min())
    recon_logI = recon_logI.unsqueeze(0).unsqueeze(0).to(device).float()

    metrics: Dict[str, float] = {"bpp": float(bpp_avg)}
    log.info(f"Tile (bpp: {bpp_avg:.4f}) metrics vs references:")

    def _to_scalar(val: Any) -> float:
        """Convert metric value to float."""
        if isinstance(val, (tuple, list)):
            val = val[0]
        if hasattr(val, "detach"):
            return float(val.detach().cpu().item())
        try:
            return float(val)
        except Exception:
            return float("nan")

    for key, ref_np in refs_logI.items():
        ref_t = torch.from_numpy(ref_np).unsqueeze(0).unsqueeze(0).to(device).float()

        log.info(f"    - Against {key}:")
        metrics[f"mse_to_{key}"] = _to_scalar(
            torchmetrics.functional.mean_squared_error(recon_logI, ref_t)
        )
        log.info(f"        - MSE: {metrics[f'mse_to_{key}']:.4f}")

        metrics[f"psnr_to_{key}"] = _to_scalar(
            F.peak_signal_noise_ratio(recon_logI, ref_t, data_range=data_range)
        )
        log.info(f"        - PSNR: {metrics[f'psnr_to_{key}']:.2f} dB")

        metrics[f"ssim_to_{key}"] = _to_scalar(
            F.structural_similarity_index_measure(recon_logI, ref_t, data_range=data_range)
        )
        log.info(f"        - SSIM: {metrics[f'ssim_to_{key}']:.4f}")

        metrics[f"ms_ssim_to_{key}"] = _to_scalar(
            F.multiscale_structural_similarity_index_measure(
                recon_logI, ref_t, data_range=data_range
            )
        )
        log.info(f"        - MS-SSIM: {metrics[f'ms_ssim_to_{key}']:.4f}")

    return metrics


def _write_artifacts(
    out_dir: Path,
    model_stats: Any,
    test_metrics: Any,
    tile_info: Any,
    eval_cfg: Any,
    train_cfg: Any,
):
    """Write evaluation artifacts: metrics logs and config files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a hierarchical structure for all metrics
    combined_metrics = {
        "run_name": eval_cfg.get("run_name"),
        "model_stats": model_stats,
        "test_metrics": test_metrics,
        "tile_metrics": tile_info,
    }

    # Write the combined metrics to a single file
    with open(out_dir / "metrics.logs", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    # Keep the configuration files separate
    with open(out_dir / "evaluate_config_resolved.yaml", "w") as f:
        OmegaConf.save(config=OmegaConf.create(eval_cfg), f=f)
    with open(out_dir / "training_config_used.yaml", "w") as f:
        OmegaConf.save(config=OmegaConf.create(train_cfg), f=f)


def _mark_evaluated(ckpt_path: Path, eval_out_dir: Path, re_evaluate: bool, run_name: str) -> bool:
    """Mark the run as evaluated by creating a marker file in the checkpoint's run directory."""
    run_dir = _checkpoint_run_dir(ckpt_path)
    marker = run_dir / "evaluated.txt"
    if marker.exists() and not re_evaluate:
        log.info(
            f"Run already evaluated per {marker}. Set re_evaluate=true to force new evaluation."
        )
        return False
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(marker, "a") as f:
        f.write(f"Evaluated at {timestamp} by run {run_name}, logs available at: {eval_out_dir}\n")
    return True


@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate.yaml")
def main(cfg: DictConfig) -> None:
    # Validate inputs
    assert cfg.ckpt_path, "Checkpoint path (ckpt_path) must be provided."
    ckpt = Path(cfg.ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Keep minimal logging and tag extras
    extras(cfg)

    # Recover training config from run dir and instantiate model
    train_cfg = _load_training_cfg_from_ckpt(ckpt)
    model = _instantiate_model_and_load_weights(train_cfg, ckpt)

    # Prepare evaluation output dir (Hydra determines working dir); we still compute a concrete path to store artifacts
    # Use current working directory as hydra run dir
    eval_out_dir = Path(cfg.paths.output_dir)

    # Stop early if evaluated already (unless re_evaluate)
    if not _mark_evaluated(ckpt, eval_out_dir, bool(cfg.re_evaluate), cfg.run_name):
        return

    # Choose best device for evaluation (prefer GPU if available)
    device = _best_device()
    model = model.to(device)

    # Model stats (on model.net if available)
    model_stats = _compute_model_stats(model, int(cfg.patch_size))

    # Test-set evaluation
    test_metrics = _evaluate_on_test(model, cfg, train_cfg.data.hdf5_dir, device)

    # Tile evaluation + visuals
    tile_metrics = _evaluate_tile_and_visualize(model, cfg, eval_out_dir)

    # Save artifacts and configs
    eval_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(eval_cfg_dict, dict):
        eval_cfg_dict = {}
    train_cfg_dict = OmegaConf.to_container(train_cfg, resolve=True)
    if not isinstance(train_cfg_dict, dict):
        train_cfg_dict = {}

    _write_artifacts(
        eval_out_dir,
        model_stats,
        test_metrics,
        tile_metrics,
        eval_cfg=eval_cfg_dict,
        train_cfg=train_cfg_dict,
    )


if __name__ == "__main__":
    main()
