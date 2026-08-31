"""Resolve + load a trained SAR-DDC checkpoint into a bare inference net.

Ported from scripts/evaluation/benchmark_gpu.py's `resolve_checkpoint_from_model_dir` /
`load_model_from_checkpoint` (same logic, including the `dc18b81` "always test last.ckpt, matching
FPGA quantisation" fix) so ddc_edge has no import dependency on `scripts/`. See
project_jetson_benchmark_scope memory for why that fix matters and why benchmark_gpu.py itself stays a
smoke test only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


def resolve_checkpoint_from_model_dir(
    model_dir: Path, project_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Read `manifest.json` in `model_dir`, return (path to `last.ckpt`, manifest dict).

    Tries the manifest's `original_run_dir` verbatim first, then re-roots relative to `project_root`
    if that path doesn't exist here (manifests are often written on a different host than they're
    later read from — e.g. a Jetson reading a manifest generated on the dev host).
    """
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {model_dir}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())

    original_run_dir = manifest.get("original_run_dir", "")
    if not original_run_dir:
        raise ValueError(f"'original_run_dir' is empty in {manifest_path}")

    run_dir = Path(original_run_dir)
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    if ckpt.exists():
        return ckpt, manifest

    parts = run_dir.parts
    for i, part in enumerate(parts):
        if part == "DDC_FPGA":
            rerooted = project_root / Path(*parts[i + 1 :]) / "checkpoints" / "last.ckpt"
            if rerooted.exists():
                return rerooted, manifest
            break

    raise FileNotFoundError(
        f"Could not find checkpoint from manifest (tried {ckpt}, and a re-rooted path under "
        f"{project_root}). Pass --ckpt explicitly if the run directory has moved."
    )


def load_model_from_checkpoint(ckpt_path: Path) -> torch.nn.Module:
    """Instantiate the model from its training config, load weights, return the bare `net` (not the
    Lightning wrapper), eval mode, entropy tables populated (`net.update(force=True)`) — identical
    sequence to benchmark_gpu.py's loader."""
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    net = model.net
    net.eval()
    net.update(force=True)
    return net
