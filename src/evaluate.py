import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from matplotlib.ticker import FuncFormatter
from omegaconf import DictConfig, OmegaConf
from ptflops import get_model_complexity_info  # type: ignore
from torch import Tensor, nn
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.sar_datamodule import TSXSSCDataModule  # noqa: E402
from src.models.merlin_module import MerlinModule  # noqa: E402
from src.models.sar_ddc_module import SARDDCModule  # noqa: E402
from src.utils.constants import AMP_MAX, AMP_MIN, EPS  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    estimate_bpp,
    get_all_distortion_metrics,
)

# from src.utils.processing_utils import process_large_patch  # noqa: E402
from src.utils.pylogger import RankedLogger  # noqa: E402
from src.utils.sar_utils import load_cosar, symmetrize  # noqa: E402
from src.utils.utils import extras  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def _checkpoint_run_dir(ckpt_path: Path) -> Path:
    """Get the run directory from the checkpoint path."""
    # ckpt lives in .../runs/<date>/checkpoints/<file>.ckpt
    return ckpt_path.parent.parent


def _load_training_cfg_from_ckpt(ckpt_path: Path) -> DictConfig:
    """Load the original training configuration from the checkpoint's run directory.

    Args:
        ckpt_path (Path): Path to the checkpoint file.
    Returns:
        training_cfg (DictConfig): The training configuration.
    """
    run_dir = _checkpoint_run_dir(ckpt_path)
    training_config_path = run_dir / ".hydra" / "config.yaml"
    if not training_config_path.exists():
        raise FileNotFoundError(f"Training config not found at {training_config_path}.")
    log.info(f"Loading original training config from {training_config_path}")
    training_cfg = OmegaConf.load(training_config_path)
    assert isinstance(training_cfg, DictConfig), "Training config is not a DictConfig."
    return training_cfg


def _instantiate_model_and_load_weights(train_cfg: DictConfig, ckpt_path: Path) -> LightningModule:
    """Instantiate the model from training config and load weights from checkpoint.

    Args:
        train_cfg (DictConfig): The training configuration.
        ckpt_path (Path): Path to the checkpoint file.
    Returns:
        model (LightningModule): The instantiated model with loaded weights.
    """
    log.info(f"Instantiating model <{train_cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(train_cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
    log.info(f"Loaded checkpoint state_dict with message: {msg}")
    model.eval()
    return model


def _count_layers(module: torch.nn.Module) -> Dict[str, int]:
    """Count the number of layers in the model.

    Args:
        module (torch.nn.Module): The model to analyze.
    Returns:
        counts (Dict[str, int]): A dictionary with counts of different layer types.
    """
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
    """Compute total and trainable parameters, and size in MB.

    Args:
        module (torch.nn.Module): The model to analyze.
    Returns:
        total (int): Total number of parameters.
        trainable (int): Number of trainable parameters.
        size_mb (float): Size of model weights in megabytes.
    """
    total: int = sum(p.numel() for p in module.parameters())
    trainable: int = sum(p.numel() for p in module.parameters() if p.requires_grad)
    state: Dict[str, Tensor] = module.state_dict()
    total_bytes: int = sum(t.element_size() * t.nelement() for t in state.values())
    return total, trainable, total_bytes / (1024**2)


def _profile_flops_macs_latency(
    module: torch.nn.Module, device: torch.device, input_size: int
) -> Dict[str, Any]:
    """Compute MACs/FLOPs using ptflops only, and measure latency.

    Fails fast if ptflops is unavailable.
    Args:
        module (torch.nn.Module): The model to analyze.
        device (torch.device): Device to run the model on.
        input_size (int): Input height and width (assumes square input).
    Returns:
        stats (Dict[str, Any]): Dictionary with input shape, device, #MACs, #FLOPs, and latency in seconds.
    """
    net = getattr(module, "net", module).to(device)
    net.eval()

    macs, _ = get_model_complexity_info(
        net,
        (1, input_size, input_size),  # (C,H,W) no batch
        as_strings=False,
        print_per_layer_stat=False,
        verbose=True,
    )
    assert macs is not None, "ptflops returned None for MACs"

    macs_val: int = int(float(macs))  # supports numpy scalars
    flops_val: int = int(2 * macs_val)

    # Latency: warmup + measure``
    def time_forward(iters: int = 50, warmup: int = 10) -> float:
        """Measure average forward pass time over `iters` runs after `warmup` runs.

        Args:
            iters (int): Number of iterations to average over.
            warmup (int): Number of warmup iterations.
         Returns:
            avg_time (float): Average time per forward pass in seconds.
        """
        x: Tensor = torch.randn(1, 1, input_size, input_size, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(warmup):
                _ = net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start: float = time.time()
            for _ in range(iters):
                _ = net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end: float = time.time()
        return (end - start) / iters

    latency: float = time_forward()

    return {
        "input": [1, 1, input_size, input_size],
        "device": str(device),
        "latency_s": latency,
        "macs": macs_val,
        "flops": flops_val,
    }


def _compute_model_stats(model: LightningModule, input_size: int) -> Dict[str, Any]:
    """Compute various model statistics: layers, parameters, size, MACs/FLOPs, latency.
    Args:
        model (LightningModule): The model to analyze.
        input_size (int): Input height and width (assumes square input).
    Returns:
        model_stats (Dict[str, Any]): Dictionary with model statistics.
    """
    net: torch.nn.Module = getattr(model, "net", model)
    layer_counts: Dict[str, int] = _count_layers(net)
    params_total, params_trainable, weights_size_mb = _params_and_size_mb(net)

    # MACs/FLOPs + CPU latency
    stats_cpu: Dict[str, Any] = _profile_flops_macs_latency(net, torch.device("cpu"), input_size)
    # GPU latency (if available)
    latency_gpu: Union[float, None] = None
    if torch.cuda.is_available():
        stats_gpu: Dict[str, Any] = _profile_flops_macs_latency(
            net, torch.device("cuda"), input_size
        )
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

    macs_str: str = f"{model_stats['macs']:_}" if model_stats.get("macs") is not None else "N/A"
    flops_str: str = f"{model_stats['flops']:_}" if model_stats.get("flops") is not None else "N/A"
    log.info(
        f"The model {model_stats['net_class']} has {model_stats['parameters_total']:_} parameters, including {model_stats['parameters_trainable']:_} trainable, for a total memory footprint of {model_stats['weights_size_mb']:.2f}MB."
    )
    gpu_part: str = (
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


def _load_reference_tile_outputs(methods: List[str], tile_dir: Path) -> Dict[str, np.ndarray]:
    """Load reference denoised outputs for given methods from tile directory.

    Args:
        methods (List[str]): List of method names to load references for.
        tile_dir (Path): Directory containing reference output .npy files.
    Returns:
        refs (Dict[str, np.ndarray]): Dictionary with loaded reference arrays.
    """
    refs: Dict[str, np.ndarray] = {}
    for m in methods:
        method_name: str = _standardize_reference_name(m)
        # Expected format: denoised_by_<method_name>_* .npy containing a dict or array
        matches: List[Path] = list(tile_dir.glob(f"denoised_by_{method_name}_*.npy"))
        if not matches:
            log.warning(
                f"[REF MISSING] Reference output for '{m}' not found in {tile_dir}. "
                f"Expected denoised_by_{method_name}_*.npy. Skipping."
            )
            continue
        refs[method_name] = np.load(matches[0], allow_pickle=True)
    return refs


def _evaluate_on_test(
    model: LightningModule,
    eval_cfg: DictConfig,
    data_dir: str,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on the test set and compute metrics.

    Args:
        model (LightningModule): The model to evaluate.
        eval_cfg (DictConfig): Evaluation configuration.
        data_dir (str): Directory containing the test data.
        device (torch.device): Device to run the evaluation on.
    Returns:
        results (Dict[str, float]): Dictionary with evaluation metrics.
    """
    # Build datamodule from training cfg.data
    dm: LightningDataModule = TSXSSCDataModule(
        hdf5_dir=str(data_dir),
        batch_size=eval_cfg.test_batch_size,
        num_workers=eval_cfg.num_workers,
        pin_memory=True,
    )
    dm.prepare_data()
    dm.setup("test")

    model = model.to(device)

    mse_list: List[float] = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []
    ms_ssim_list: List[float] = []
    bpp_list: List[Tensor] = []

    if eval_cfg.short_test_set:
        log.info("Using short test set for quick evaluation (first 10 batches only).")
        batch_nb: int = 0
    with torch.inference_mode():
        for batch in tqdm(dm.test_dataloader()):
            real: Tensor = batch["real"].to(device)
            imag: Tensor = batch["imag"].to(device)
            if isinstance(model, SARDDCModule):
                noisy_lin: Tensor = torch.cat([real, imag], dim=1).contiguous()
                recon: Dict[str, Tensor] = model(noisy_lin)
                bpp_list.append(Tensor(estimate_bpp(recon)))
                recon_real: Tensor = recon["x_hat"]
                recon_imag: Tensor = recon["x_hat"]
            elif isinstance(model, MerlinModule):  # untested so far
                recon_real: Tensor = model(real)
                recon_imag: Tensor = model(imag)
            else:
                raise ValueError(
                    "Model output format not supported for evaluation, the model is probably MerlinModule. @TODO: Implement support."
                )

            # Convert model output to linear amplitude
            recon_real_denorm: Tensor = recon_real * (AMP_MAX - AMP_MIN) + AMP_MIN
            recon_imag_denorm: Tensor = recon_imag * (AMP_MAX - AMP_MIN) + AMP_MIN
            recon_real_lin: Tensor = torch.exp(recon_real_denorm)
            recon_imag_lin: Tensor = torch.exp(recon_imag_denorm)
            recon_linA: Tensor = torch.sqrt(
                0.5 * (torch.square(recon_real_lin) + torch.square(recon_imag_lin))
            )

            # Noisy image linear amplitude
            noisy_linA: Tensor = torch.sqrt(torch.square(real) + torch.square(imag))

            # Compute metrics
            all_metrics: Dict[str, float] = get_all_distortion_metrics(recon_linA, noisy_linA)
            mse_list.append(all_metrics["mse"])
            psnr_list.append(all_metrics["psnr"])

            ssim_list.append(all_metrics["ssim"])
            ms_ssim_list.append(all_metrics["ms_ssim"])
            if eval_cfg.short_test_set:
                batch_nb += 1
                if batch_nb >= 10:
                    break

    mse_val: float = float(np.mean(mse_list))
    psnr_val: float = float(np.mean(psnr_list))
    ssim_val: float = float(np.mean(ssim_list))
    ms_ssim_val: float = float(np.mean(ms_ssim_list))
    results: Dict[str, float] = {
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
) -> Dict[str, float]:
    """Evaluate the model against references on a tile and generate visualization figure.

    Args:
        model (LightningModule): The model to evaluate.
        cfg (DictConfig): Evaluation configuration.
        save_dir (Path): Directory to save visualization outputs.
    Returns:
        metrics (Dict[str, float]): Dictionary with evaluation metrics on the tile.
    """
    tile_path: Path = Path(cfg.tile_path)

    # Load tile or pre-extracted patch and symmetrize
    if tile_path.suffix == ".npy":
        patch: np.ndarray = np.load(tile_path)
        if patch is None:
            raise FileNotFoundError(f"Failed to load patch from {tile_path}")
        patch = symmetrize(patch)
    else:
        image: Union[np.ndarray, None] = load_cosar(tile_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load SAR tile from {tile_path}")
        image = symmetrize(image)  # [H,W,2]
        patch: np.ndarray = image[
            cfg.crop_coordinates[0] : cfg.crop_coordinates[0] + cfg.tile_crop_size,
            cfg.crop_coordinates[1] : cfg.crop_coordinates[1] + cfg.tile_crop_size,
            :,
        ]

    device = next(model.parameters()).device
    input_real = torch.from_numpy(patch[:, :, 0]).to(device).unsqueeze(0).unsqueeze(0)
    input_imag = torch.from_numpy(patch[:, :, 1]).to(device).unsqueeze(0).unsqueeze(0)

    log.info(f"Loaded and pre-processed tile patch of size {patch.shape}.")

    with torch.inference_mode():
        # Patch-based forward to be consistent and memory-friendly
        if isinstance(model, SARDDCModule):
            noisy_lin: Tensor = torch.cat([input_real, input_imag], dim=1).contiguous()
            recon: Dict[str, Tensor] = model(noisy_lin)
            recon_real: Tensor = recon["x_hat"]
            recon_imag: Tensor = recon["x_hat"]
            bpp: Tensor = Tensor(estimate_bpp(recon))
        elif isinstance(model, MerlinModule):  # untested so far
            recon_real: Tensor = model(input_real)
            recon_imag: Tensor = model(input_imag)
            bpp: Tensor = Tensor(-1.0)
        else:
            raise ValueError(
                "Model output format not supported for evaluation, the model is probably MerlinModule. @TODO: Implement support."
            )
    # # Average is done later in linear/log domains for visualization/metrics
    # bpp_avg = 0.5 * (metrics_r.get("bpp", -1) + metrics_i.get("bpp", -1))

    # Convert to linear amplitude per channel and build log-intensity like in callback
    recon_real_lin: Tensor = torch.exp(recon_real.squeeze() * (AMP_MAX - AMP_MIN) + AMP_MIN)
    recon_imag_lin: Tensor = torch.exp(recon_imag.squeeze() * (AMP_MAX - AMP_MIN) + AMP_MIN)
    recon_I: Tensor = 0.5 * (recon_real_lin + recon_imag_lin)
    recon_linA: Tensor = torch.sqrt(recon_I)
    recon_logI: Tensor = torch.log(recon_I + EPS)

    # Load references
    tile_vis_dir: Path = Path("data/visualization/for_evaluations/")
    refs_linA: Dict[str, np.ndarray] = _load_reference_tile_outputs(
        list(cfg.reference_methods), tile_vis_dir
    )
    refs_logI: Dict[str, np.ndarray] = {
        method: np.log(np.square(ref_linA) + EPS) for method, ref_linA in refs_linA.items()
    }
    nb_refs: int = len(refs_linA)

    # ----- Visualization figure (LOG-I) -----
    noisy_logI: np.ndarray = np.log(np.square(patch[:, :, 0]) + np.square(patch[:, :, 1]) + EPS)
    recon_logI_np: np.ndarray = recon_logI.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 2 + nb_refs, figsize=(10 + (nb_refs * 5), 10))

    clip_std_factor: float = float(cfg.clip_std_factor)

    def clip_image(x: np.ndarray) -> np.ndarray:
        """Clip image values to <clip_std_factor> standard deviations around the mean.

        Args:
            x (np.ndarray): Input image array.
        Returns:
            np.ndarray: Clipped image array.
        """
        m, s = x.mean(), x.std()
        x = np.clip(x, m - clip_std_factor * s, m + clip_std_factor * s)
        # x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    def plot_histogram(ax, data: np.ndarray, title: str) -> None:
        """Plot histogram of data on given axis with mean and std lines.

        Args:
            ax (plt.Axes): Matplotlib axis to plot on.
            data (np.ndarray): Data array to plot histogram for.
            title (str): Title for the histogram plot.
        """
        ax.set_title(title)
        ax.hist(data.flatten(), bins=50, alpha=0.7, color="blue")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, loc: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}")
        )
        mean: float = data.mean()
        std: float = data.std()
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

    im1 = axes[0, 1].imshow(clip_image(recon_logI_np), cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Recon Log-I")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    plot_histogram(axes[1, 1], recon_logI_np, "Recon Log-I Histogram")

    for i, (method_name, ref) in enumerate(refs_logI.items()):
        im2 = axes[0, 2 + i].imshow(clip_image(ref), cmap="gray")
        axes[0, 2 + i].axis("off")
        axes[0, 2 + i].set_title(f"{method_name} Log-I")
        fig.colorbar(im2, ax=axes[0, 2 + i], shrink=0.8)
        plot_histogram(axes[1, 2 + i], ref, f"{method_name} Log-I Histogram")

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "tile_comparison_logI.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    log.info(f"Saved tile comparison figure to {fig_path}")

    # Save the reconstruction image alone with high quality
    plt.imsave(
        save_dir / "tile_reconstruction_logI.png", clip_image(recon_logI_np), cmap="gray", dpi=300
    )

    # ----- Metrics on tile vs all references (Lin-A) -----
    metrics: Dict[str, float] = {"bpp": float(bpp)}
    log.info(f"Tile (bpp: {bpp:.4f}) metrics vs references:")

    for method_name, ref_np in refs_logI.items():
        log.info(f"    - Against {method_name}:")
        ref_t: Tensor = torch.from_numpy(ref_np).to(device).float()
        ref_linA: Tensor = torch.sqrt(torch.exp(ref_t) + EPS)

        all_metrics: Dict[str, float] = get_all_distortion_metrics(recon_linA, ref_linA)
        metrics[f"mse_to_{method_name}"] = all_metrics["mse"]
        log.info(f"        - MSE: {metrics[f'mse_to_{method_name}']:.4f}")
        metrics[f"psnr_to_{method_name}"] = all_metrics["psnr"]
        log.info(f"        - PSNR: {metrics[f'psnr_to_{method_name}']:.2f} dB")
        metrics[f"ssim_to_{method_name}"] = all_metrics["ssim"]
        log.info(f"        - SSIM: {metrics[f'ssim_to_{method_name}']:.4f}")
        metrics[f"ms_ssim_to_{method_name}"] = all_metrics["ms_ssim"]
        log.info(f"        - MS-SSIM: {metrics[f'ms_ssim_to_{method_name}']:.4f}")

    return metrics


def _write_artifacts(
    out_dir: Path,
    model_stats: Dict[str, Any],
    test_metrics: Dict[str, float],
    tile_info: Dict[str, float],
    eval_cfg: Any,
    train_cfg: Any,
):
    """Write evaluation artifacts: metrics logs and config files.
    Args:
        out_dir (Path): Output directory to save artifacts.
        model_stats (Dict[str, Any]): Model statistics.
        test_metrics (Dict[str, float]): Test set evaluation metrics.
        tile_info (Dict[str, float]): Tile evaluation metrics.
        eval_cfg (Any): Evaluation configuration.
        train_cfg (Any): Training configuration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a hierarchical structure for all metrics
    combined_metrics: Dict[str, Any] = {
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
    """Mark the run as evaluated by creating a marker file in the checkpoint's run directory.

    Args:
        ckpt_path (Path): Path to the checkpoint file.
        eval_out_dir (Path): Directory where evaluation outputs are saved.
        re_evaluate (bool): Whether to force re-evaluation.
        run_name (str): Name of the evaluation run.
    Returns:
        should_evaluate (bool): True if evaluation should proceed, False if already evaluated.
    """
    marker: Path = _checkpoint_run_dir(ckpt_path) / "evaluated.txt"
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
    ckpt: Path = Path(cfg.ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Keep minimal logging and tag extras
    extras(cfg)

    # Recover training config from run dir and instantiate model
    train_cfg: DictConfig = _load_training_cfg_from_ckpt(ckpt)
    model: LightningModule = _instantiate_model_and_load_weights(train_cfg, ckpt)

    # Use current working directory as hydra run dir
    eval_out_dir: Path = Path(cfg.paths.output_dir)

    # Stop early if evaluated already (unless re_evaluate)
    if not _mark_evaluated(ckpt, eval_out_dir, bool(cfg.re_evaluate), cfg.run_name):
        return

    # Choose best device for evaluation (prefer GPU if available)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Model stats (on model.net if available)
    model_stats: Dict[str, Any] = _compute_model_stats(model, int(cfg.patch_size))

    # Test-set evaluation
    test_metrics: Dict[str, float] = _evaluate_on_test(
        model, cfg, str(train_cfg.data.hdf5_dir), device
    )

    # Tile evaluation + visuals
    tile_metrics: Dict[str, float] = _evaluate_tile_and_visualize(model, cfg, eval_out_dir)

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
