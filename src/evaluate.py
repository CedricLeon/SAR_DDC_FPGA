import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.sar_datamodule import TSXSSCDataModule  # noqa: E402
from src.utils import RankedLogger, extras, process_large_patch  # noqa: E402
from src.utils.sar_utils import load_cosar, symmetrize  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class EvalConfig:
    ckpt_path: str
    re_evaluate: bool
    reference_methods: List[str]
    tile_path: str
    tile_crop_size: int
    crop_coordinates: List[int]
    device: str
    input_size_for_stats: int
    test_batch_size: int
    num_workers: int
    visualize: bool
    clip_std_factor: float


def _resolve_device(policy: str) -> torch.device:
    if policy == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if policy in ("cpu", "cuda"):
        return torch.device(policy)
    return torch.device("cpu")


def _checkpoint_run_dir(ckpt_path: Path) -> Path:
    # ckpt lives in .../runs/<date>/checkpoints/<file>.ckpt
    # run dir = parent.parent
    return ckpt_path.parent.parent


def _load_training_cfg_from_ckpt(ckpt_path: Path) -> Any:
    run_dir = _checkpoint_run_dir(ckpt_path)
    training_config_path = run_dir / ".hydra" / "config.yaml"
    if not training_config_path.exists():
        raise FileNotFoundError(f"Training config not found at {training_config_path}.")
    log.info(f"Loading original training config from {training_config_path}")
    return OmegaConf.load(training_config_path)


def _instantiate_model_and_load_weights(
    train_cfg: DictConfig, ckpt_path: Path
) -> LightningModule:
    log.info(f"Instantiating model <{train_cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(train_cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
    log.info(f"Loaded checkpoint state_dict with message: {msg}")
    model.eval()
    return model


def _count_layers(module: torch.nn.Module) -> Dict[str, int]:
    from torch import nn

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
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    state = module.state_dict()
    total_bytes = sum(t.element_size() * t.nelement() for t in state.values())
    return total, trainable, total_bytes / (1024**2)


def _profile_flops_macs_latency(
    module: torch.nn.Module, device: torch.device, input_size: int
) -> Dict[str, Any]:
    module = module.to(device)
    module.eval()
    x = torch.randn(1, 1, input_size, input_size, device=device)

    # Use torch.profiler to get op-level metrics when available.
    # Note: FLOPs reporting coverage depends on PyTorch version.
    flops_total = None
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                *(
                    [torch.profiler.ProfilerActivity.CUDA]
                    if device.type == "cuda"
                    else []
                ),
            ]
        ) as prof:
            with torch.inference_mode():
                _ = module(x)
        # Aggregate flops if available
        flops_total = 0
        for e in prof.key_averages():
            if hasattr(e, "flops") and e.flops is not None:
                flops_total += int(e.flops)
        if flops_total == 0:
            flops_total = None
    except Exception as e:
        log.warning(f"Profiler FLOPs not available: {e}")

    macs_est = int(flops_total // 2) if flops_total is not None else None

    # Latency: warmup + measure
    def time_forward(iters: int = 50, warmup: int = 10) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(warmup):
                _ = module(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(iters):
                _ = module(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
        return (end - start) / iters

    latency = time_forward()

    return {
        "input": [1, 1, input_size, input_size],
        "device": str(device),
        "latency_s": latency,
        "macs_estimate": macs_est,
        "flops_profiler": flops_total,
    }


def _standardize_reference_name(name: str) -> str:
    return name.replace(" ", "_")


def _load_reference_tile_outputs(methods: List[str], tile_dir: Path) -> Dict[str, Any]:
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
        try:
            obj = np.load(matches[0], allow_pickle=True)
            refs[m] = obj.item() if hasattr(obj, "item") else obj
        except Exception as e:
            log.warning(f"Failed to load reference '{m}' from {matches[0]}: {e}")
    return refs


def _evaluate_on_test(
    model: LightningModule,
    train_cfg: DictConfig,
    overrides: EvalConfig,
) -> Dict[str, float]:
    # Build datamodule from training cfg.data
    data_dir = train_cfg.data.hdf5_dir
    dm = TSXSSCDataModule(
        hdf5_dir=data_dir,
        batch_size=overrides.test_batch_size,
        num_workers=overrides.num_workers,
        pin_memory=True,
    )
    dm.prepare_data()
    dm.setup("test")

    device = _resolve_device(overrides.device)
    model = model.to(device)
    from torchmetrics import MeanSquaredError
    from torchmetrics.image import (
        MultiScaleStructuralSimilarityIndexMeasure,
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )

    from src.utils.constants import amp_max, amp_min

    mse = MeanSquaredError().to(device)
    # Metrics computed on log-intensity domain; data range corresponds to 2*amp_max-2*amp_min
    data_range_logI = float(2 * amp_max - 2 * amp_min)
    psnr = PeakSignalNoiseRatio(data_range=data_range_logI).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range_logI).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range_logI).to(
        device
    )

    bpp_list = []
    with torch.inference_mode():
        for batch in dm.test_dataloader():
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

            # Denormalize back to log-intensity domain like in the callback:
            # normalized = (logI - 2*amp_min) / (2*amp_max - 2*amp_min)
            # => logI = normalized * (2*amp_max - 2*amp_min) + 2*amp_min
            recon_denorm = recon * (2 * amp_max - 2 * amp_min) + 2 * amp_min
            target_denorm = imag * (2 * amp_max - 2 * amp_min) + 2 * amp_min

            mse.update(recon_denorm, target_denorm)
            psnr.update(recon_denorm, target_denorm)
            ssim.update(recon_denorm, target_denorm)
            ms_ssim.update(recon_denorm, target_denorm)

    def to_float(val: Any) -> float:
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
    return results


def _evaluate_tile_and_visualize(
    model: LightningModule,
    tile_path: Path,
    methods: List[str],
    save_dir: Path,
    crop_coordinates: Tuple[int, int],
    tile_crop_size: int,
    clip_std_factor: float,
) -> Dict[str, Any]:
    # Load tile, symmetrize, normalize
    from src.utils.constants import amp_max, amp_min

    sar = load_cosar(tile_path)
    if sar is None:
        raise FileNotFoundError(f"Failed to load SAR tile from {tile_path}")
    sar = symmetrize(sar)  # [H,W,2]

    # Crop a tile using provided coordinates and size
    H, W, _ = sar.shape
    h0, w0 = int(crop_coordinates[0]), int(crop_coordinates[1])
    h1, w1 = h0 + int(tile_crop_size), w0 + int(tile_crop_size)
    h0 = max(0, min(h0, H - 1))
    w0 = max(0, min(w0, W - 1))
    h1 = max(1, min(h1, H))
    w1 = max(1, min(w1, W))
    patch = sar[h0:h1, w0:w1, :]

    eps = 1e-2
    patch_sq = np.square(patch)
    patch_log = np.log(patch_sq + eps)
    patch_norm = (patch_log - 2 * amp_min) / (2 * amp_max - 2 * amp_min)

    device = next(model.parameters()).device
    real = torch.from_numpy(patch_norm[:, :, 0]).to(device).unsqueeze(0).unsqueeze(0)
    imag = torch.from_numpy(patch_norm[:, :, 1]).to(device).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        # Patch-based forward to be consistent and memory-friendly
        metrics_r, recon_r = process_large_patch(
            model=model,
            input=real,
            target=imag,
            model_patch_size=256,
            stride=256,
            blend_method="count",
        )
        metrics_i, recon_i = process_large_patch(
            model=model,
            input=imag,
            target=real,
            model_patch_size=256,
            stride=256,
            blend_method="count",
        )
    # Average is done later in linear/log domains for visualization/metrics

    # Convert to linear amplitude per channel and build log-intensity like in callback
    recon_r_lin = torch.exp(recon_r.squeeze() * (amp_max - amp_min) + amp_min)
    recon_i_lin = torch.exp(recon_i.squeeze() * (amp_max - amp_min) + amp_min)
    I_recon = 0.5 * (recon_r_lin + recon_i_lin)
    recon_logI = torch.log(I_recon + eps)

    # Load references if available
    tile_vis_dir = Path("data/visualization")
    refs = _load_reference_tile_outputs(methods, tile_vis_dir)

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
                arr = np.array(obj)
                if arr.ndim == 2:
                    # Assume already logI
                    refs_logI[key] = arr
                elif arr.ndim == 3 and arr.shape[-1] == 2:
                    # If 2-channel amplitude, build intensity then log
                    intensity = 0.5 * (np.abs(arr[..., 0]) + np.abs(arr[..., 1]))
                    refs_logI[key] = np.log(intensity + eps)
        except Exception as e:
            log.warning(f"Reference '{m}' has unexpected format: {e}")

    fig, axes = plt.subplots(2, 4, figsize=(15, 10))

    def clip_norm(x):
        m, s = x.mean(), x.std()
        x = np.clip(x, m - clip_std_factor * s, m + clip_std_factor * s)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    axes[0, 0].imshow(clip_norm(noisy_logI), cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Noisy Log-I")

    axes[0, 1].imshow(clip_norm(recon_np.squeeze()), cmap="gray")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Recon Log-I")

    if "MERLIN" in refs_logI:
        axes[0, 2].imshow(clip_norm(refs_logI["MERLIN"]), cmap="gray")
        axes[0, 2].axis("off")
        axes[0, 2].set_title("MERLIN Log-I")
    else:
        axes[0, 2].text(0.5, 0.5, "MERLIN Not Available", ha="center", va="center")
        axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    axes[1, 0].hist(noisy_logI.flatten(), bins=50)
    axes[1, 1].hist(recon_np.flatten(), bins=50)
    if "MERLIN" in refs_logI:
        axes[1, 2].hist(refs_logI["MERLIN"].flatten(), bins=50)
    axes[1, 3].axis("off")
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "tile_comparison_logI.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Save the reconstruction image alone with high quality
    recon_save = clip_norm(recon_np)
    plt.imsave(save_dir / "tile_reconstruction_logI.png", recon_save, cmap="gray")

    # Metrics on tile vs all references (MSE/PSNR/SSIM/MS-SSIM)
    from torchmetrics.image import (
        MultiScaleStructuralSimilarityIndexMeasure as TMS_SSIM,
    )
    from torchmetrics.image import StructuralSimilarityIndexMeasure as TSSIM

    metrics: Dict[str, float] = {}
    recon_t = torch.from_numpy(recon_np).float().unsqueeze(0).unsqueeze(0)
    for key, ref_np in refs_logI.items():
        ref_t = torch.from_numpy(ref_np).float().unsqueeze(0).unsqueeze(0)
        mse = float(np.mean((recon_np - ref_np) ** 2))
        peak = float(ref_np.max())
        psnr = 20 * np.log10(peak / np.sqrt(mse)) if mse > 0 else float("inf")
        data_range = float(ref_np.max() - ref_np.min() + 1e-8)
        ssim_metric = TSSIM(data_range=data_range)
        ms_ssim_metric = TMS_SSIM(data_range=data_range)
        with torch.no_grad():
            ssim_val = float(ssim_metric(recon_t, ref_t).item())
            ms_ssim_val = float(ms_ssim_metric(recon_t, ref_t).item())
        metrics[f"mse_to_{key}"] = mse
        metrics[f"psnr_to_{key}"] = float(psnr)
        metrics[f"ssim_to_{key}"] = ssim_val
        metrics[f"ms_ssim_to_{key}"] = ms_ssim_val
    return {"tile_metrics": metrics, "fig": str(fig_path)}


def _write_artifacts(
    out_dir: Path,
    model_stats: Any,
    test_metrics: Any,
    tile_info: Any,
    eval_cfg: Any,
    train_cfg: Any,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_stats.json", "w") as f:
        json.dump(model_stats, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(out_dir / "tile_metrics.json", "w") as f:
        json.dump(tile_info, f, indent=2)
    with open(out_dir / "evaluate_config_resolved.yaml", "w") as f:
        OmegaConf.save(config=OmegaConf.create(eval_cfg), f=f)
    with open(out_dir / "training_config_used.yaml", "w") as f:
        OmegaConf.save(config=OmegaConf.create(train_cfg), f=f)


def _mark_evaluated(ckpt_path: Path, eval_out_dir: Path, re_evaluate: bool) -> bool:
    run_dir = _checkpoint_run_dir(ckpt_path)
    marker = run_dir / "evaluated.txt"
    if marker.exists() and not re_evaluate:
        log.info(
            f"Run already evaluated per {marker}. Set re_evaluate=true to force new evaluation."
        )
        return False
    with open(marker, "a") as f:
        f.write(str(eval_out_dir) + "\n")
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
    eval_out_dir = Path.cwd()

    # Stop early if evaluated already (unless re_evaluate)
    if not _mark_evaluated(ckpt, eval_out_dir, bool(cfg.re_evaluate)):
        return

    # Choose device
    device = _resolve_device(cfg.device)
    model = model.to(device)

    # Model stats (on model.net if available)
    net = getattr(model, "net", model)
    layer_counts = _count_layers(net)
    total_p, train_p, size_mb = _params_and_size_mb(net)
    hw_stats = _profile_flops_macs_latency(net, device, int(cfg.input_size_for_stats))
    model_stats = {
        "net_class": net.__class__.__name__,
        "layers": layer_counts,
        "parameters_total": total_p,
        "parameters_trainable": train_p,
        "weights_size_mb": size_mb,
        **hw_stats,
    }

    # Test-set evaluation
    # Build overrides dataclass from cfg
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}
    overrides = EvalConfig(
        ckpt_path=str(cfg_dict.get("ckpt_path", "")),
        re_evaluate=bool(cfg_dict.get("re_evaluate", False)),
        reference_methods=list(cfg_dict.get("reference_methods", [])),
        tile_path=str(cfg_dict.get("tile_path", "")),
        tile_crop_size=int(cfg_dict.get("tile_crop_size", 1024)),
        crop_coordinates=list(cfg_dict.get("crop_coordinates", [11000, 8500])),
        device=str(cfg_dict.get("device", "auto")),
        input_size_for_stats=int(cfg_dict.get("input_size_for_stats", 256)),
        test_batch_size=int(cfg_dict.get("test_batch_size", 16)),
        num_workers=int(cfg_dict.get("num_workers", 8)),
        visualize=bool(cfg_dict.get("visualize", True)),
        clip_std_factor=float(cfg_dict.get("clip_std_factor", 3.0)),
    )

    test_metrics = _evaluate_on_test(model, train_cfg, overrides)

    # Tile evaluation + visuals
    tile_info = (
        _evaluate_tile_and_visualize(
            model,
            Path(cfg.tile_path),
            list(cfg.reference_methods),
            eval_out_dir,
            crop_coordinates=(
                int(overrides.crop_coordinates[0]),
                int(overrides.crop_coordinates[1]),
            ),
            tile_crop_size=int(overrides.tile_crop_size),
            clip_std_factor=float(overrides.clip_std_factor),
        )
        if bool(cfg.visualize)
        else {"tile_metrics": {}, "fig": None}
    )

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
        tile_info,
        eval_cfg=eval_cfg_dict,
        train_cfg=train_cfg_dict,
    )

    # Console summary
    log.info(
        f"Model: {model_stats['net_class']}, params={model_stats['parameters_total']}, size={model_stats['weights_size_mb']:.2f}MB"
    )
    log.info(
        f"Latency ({model_stats['device']}): {model_stats['latency_s'] * 1e3:.2f} ms | MACs≈{model_stats['macs_estimate']:,} | FLOPs={model_stats['flops_profiler']}"
    )
    log.info(f"Test metrics: {test_metrics}")
    if tile_info.get("fig"):
        log.info(f"Saved tile comparison to {tile_info['fig']}")


if __name__ == "__main__":
    main()
