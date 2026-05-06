#!/usr/bin/env python3
"""Performance Benchmark for GPU and CPU Inference.

Measures per-component latency, throughput, and (optionally) power consumption
for the SAR DDC model on CUDA GPU and/or host CPU.  Designed to produce JSON
output directly comparable to ``scripts/fpga/benchmark_fpga.py``.

Scenarios
---------
  full           : Compress + Decompress  (g_a → h_a → EB → h_s → GC → g_s)
  compress       : Encode only            (g_a → h_a → EB → h_s → GC.compress)
  decompress     : Decode only            (EB.decompress → h_s → GC.decompress → g_s)
  nn_only        : All four NN subgraphs back-to-back, no entropy coding
  entropy_only   : Entropy coding round-trip only (EB + GC)

By default the script runs both GPU and CPU measurements sequentially.
Use ``--no-gpu`` or ``--no-cpu`` to skip one.

Usage:
    python scripts/benchmark_gpu.py --ckpt <path/to/checkpoint.ckpt> --scenario full [options]

Output: ``results/<ckpt_name>/benchmark_<device>_<scenario>.json``
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import torch
from torch import Tensor

PROJECT_ROOT = rootutils.setup_root(
    Path(__file__).resolve().parent.parent, indicator=".project-root", pythonpath=True
)

from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.utils.constants import AMP_MAX, AMP_MIN, EPS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (must match FPGA benchmark / inference_utils.py)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256
C_MAIN = 128
C_HYPER = C_MAIN * 2

SCENARIOS = ["full", "compress", "decompress", "nn_only", "entropy_only"]


# ---------------------------------------------------------------------------
# Model loading (reuses evaluate.py pattern)
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(ckpt_path: Path) -> torch.nn.Module:
    """Instantiate ResidualScaleHyperpriorPatched from a training checkpoint.

    Returns the *inner* ``net`` (not the Lightning wrapper) in eval mode with
    entropy tables populated.
    """
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    # Instantiate the LightningModule, then extract the inner net
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"Loaded checkpoint: {msg}")

    net = model.net
    net.eval()
    net.update(force=True)
    return net


# ---------------------------------------------------------------------------
# Model-dir resolution  (reads manifest.json → original_run_dir → checkpoint)
# ---------------------------------------------------------------------------
def resolve_checkpoint_from_model_dir(model_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a checkpoint path from a compiled FPGA model directory.

    Reads ``manifest.json`` inside *model_dir*, extracts ``original_run_dir``,
    and returns ``<run_dir>/checkpoints/last.ckpt``.

    Because the manifest may have been written on a different machine (e.g.
    ``/mnt/vitisAI/Vitis-AI/DDC_FPGA/…`` vs ``/home/user/dev/Vitis-AI/DDC_FPGA/…``),
    we try the following resolution order:

    1. Absolute path as stored in the manifest.
    2. Re-root: extract the ``DDC_FPGA/…`` suffix and resolve from *PROJECT_ROOT*.

    Returns
    -------
    ckpt_path : Path
        Resolved path to ``last.ckpt``.
    manifest : dict
        The full manifest dict (useful for metadata).
    """
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {model_dir}.\n"
            "Expected a compiled FPGA model directory (e.g. results/fpga/active_model/)."
        )

    with open(manifest_path) as f:
        manifest: dict[str, Any] = json.load(f)

    original_run_dir = manifest.get("original_run_dir", "")
    if not original_run_dir:
        raise ValueError(f"'original_run_dir' is empty in {manifest_path}")

    run_dir = Path(original_run_dir)
    ckpt = run_dir / "checkpoints" / "last.ckpt"

    # Strategy 1: absolute path as-is
    if ckpt.exists():
        return ckpt, manifest

    # Strategy 2: re-root relative to PROJECT_ROOT
    # Look for 'DDC_FPGA/' in the path and re-anchor from there.
    parts = run_dir.parts
    for i, part in enumerate(parts):
        if part == "DDC_FPGA":
            # Reconstruct from PROJECT_ROOT + everything after "DDC_FPGA/"
            relative_suffix = Path(*parts[i + 1 :])
            ckpt_rerooted = PROJECT_ROOT / relative_suffix / "checkpoints" / "last.ckpt"
            if ckpt_rerooted.exists():
                return ckpt_rerooted, manifest
            break

    raise FileNotFoundError(
        f"Could not find checkpoint from manifest.\n"
        f"  original_run_dir : {original_run_dir}\n"
        f"  Tried            : {ckpt}\n"
        f"  Also tried       : {PROJECT_ROOT / 'logs' / '...' }\n"
        f"Provide --ckpt explicitly if the run directory has moved."
    )


# ---------------------------------------------------------------------------
# Timer utility (mirrors FPGA StepTimer exactly)
# ---------------------------------------------------------------------------
class StepTimer:
    """Accumulates per-step wall-clock durations using time.perf_counter.

    For GPU: call ``torch.cuda.synchronize()`` **before** each ``mark()``
    to ensure all GPU ops have finished.
    """

    def __init__(self) -> None:
        self._marks: list[tuple[str, float]] = []
        self._history: dict[str, list[float]] = {}

    def mark(self, label: str) -> None:
        """Record a timestamp for a given step label."""
        self._marks.append((label, time.perf_counter()))

    def commit(self) -> None:
        """Process recorded marks into durations and accumulate in history."""
        for i in range(len(self._marks) - 1):
            name = self._marks[i][0]
            dt = self._marks[i + 1][1] - self._marks[i][1]
            self._history.setdefault(name, []).append(dt)
        self._marks.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for each step label in history."""
        out: dict[str, dict[str, float]] = {}
        for name, vals in self._history.items():
            if name.startswith("_"):
                continue
            out[name] = {
                "mean_s": statistics.mean(vals),
                "std_s": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "median_s": statistics.median(vals),
                "min_s": min(vals),
                "max_s": max(vals),
                "p95_s": sorted(vals)[int(0.95 * len(vals))],
                "n": len(vals),
            }
        return out

    def total_mean(self) -> float:
        """Compute total mean duration across all steps (excluding those starting with '_')."""
        return sum(v["mean_s"] for k, v in self.summary().items() if not k.startswith("_"))


# ---------------------------------------------------------------------------
# CUDA Event Timer (high-precision GPU-only timing)
# ---------------------------------------------------------------------------
class CudaEventTimer:
    """Per-step GPU timing using torch.cuda.Event for microsecond precision.

    Usage mirrors StepTimer: mark() → commit(), but records CUDA events
    instead of perf_counter timestamps.  Requires CUDA.
    """

    def __init__(self) -> None:
        self._marks: list[tuple[str, torch.cuda.Event]] = []
        self._history: dict[str, list[float]] = {}

    def mark(self, label: str) -> None:
        """Record a CUDA event for a given step label."""
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()  # type: ignore[call-arg]
        self._marks.append((label, ev))  # type: ignore[arg-type]

    def commit(self) -> None:
        """Process recorded CUDA events into durations and accumulate in history."""
        # Synchronize to ensure all events are recorded
        torch.cuda.synchronize()
        for i in range(len(self._marks) - 1):
            name = self._marks[i][0]
            dt_ms = self._marks[i][1].elapsed_time(self._marks[i + 1][1])
            self._history.setdefault(name, []).append(dt_ms / 1000.0)  # store in seconds
        self._marks.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for each step label in history."""
        out: dict[str, dict[str, float]] = {}
        for name, vals in self._history.items():
            if name.startswith("_"):
                continue
            out[name] = {
                "mean_s": statistics.mean(vals),
                "std_s": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "median_s": statistics.median(vals),
                "min_s": min(vals),
                "max_s": max(vals),
                "p95_s": sorted(vals)[int(0.95 * len(vals))],
                "n": len(vals),
            }
        return out

    def total_mean(self) -> float:
        """Compute total mean duration across all steps (excluding those starting with '_')."""
        return sum(v["mean_s"] for k, v in self.summary().items() if not k.startswith("_"))


# ---------------------------------------------------------------------------
# Power samplers
# ---------------------------------------------------------------------------
class NvidiaSmiPowerSampler:
    """Background polling of GPU power via nvidia-smi."""

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.1):
        self._gpu_index = gpu_index
        self._poll_interval = poll_interval_s
        self._samples: list[float] = []  # watts
        self._timestamps: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def _read_power_w(self) -> float:
        """Read current GPU power in watts using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _poll_loop(self) -> None:
        """Background loop to poll GPU power at regular intervals."""
        while self._running:
            t = time.perf_counter()
            self._samples.append(self._read_power_w())
            self._timestamps.append(t)
            # Sleep-based polling (nvidia-smi has ~100ms overhead anyway)
            elapsed = time.perf_counter() - t
            remaining = self._poll_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def start(self) -> None:
        """Start background polling thread."""
        self._running = True
        self._samples = []
        self._timestamps = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread and wait for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def results(self) -> dict[str, Any]:
        """Compute average power, total energy, and sample count from collected data."""
        if len(self._timestamps) < 2:
            return {"avg_power_w": 0.0, "energy_j": 0.0, "n_samples": 0, "duration_s": 0.0}
        duration = self._timestamps[-1] - self._timestamps[0]
        avg_w = statistics.mean(self._samples) if self._samples else 0.0
        return {
            "avg_power_w": round(avg_w, 4),
            "energy_j": round(avg_w * duration, 6),
            "n_samples": len(self._samples),
            "duration_s": round(duration, 4),
        }


class RAPLPowerSampler:
    """Background polling of CPU package power via Intel RAPL sysfs.

    Reads energy_uj counters at start/stop and computes average power. Optionally polls at
    intervals for time-series data. By default `energy_uj` files are readable only by root.
    Therefore, to measure power run this file as root, or temporarily change permissions of the relevant `energy_uj` files to be world-readable:
    ```bash
    sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
    sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj
    ```
    """

    RAPL_BASE = "/sys/class/powercap/intel-rapl"

    def __init__(self, poll_interval_s: float = 0.1):
        self._poll_interval = poll_interval_s
        self._domains: dict[str, str] = {}  # name → energy_uj path
        self._samples: dict[str, list[float]] = {}  # name → [energy_uj, ...]
        self._timestamps: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._discover_domains()

    @staticmethod
    def _try_read_energy_uj(path: Path) -> bool:
        """Return True if *path* exists and is readable by the current user."""
        try:
            path.read_text()
            return True
        except (OSError, PermissionError):
            return False

    def _discover_domains(self) -> None:
        """Discover RAPL domains by scanning sysfs and populate self._domains.

        Each ``energy_uj`` file is probe-read during discovery: domains whose
        file exists but is not readable (e.g. ``-r--------`` requiring root on
        kernels ≥ 5.10) are silently skipped so that ``available`` correctly
        returns ``False`` instead of collecting 0.0 W readings.
        """
        base = Path(self.RAPL_BASE)
        if not base.exists():
            return
        for pkg_dir in sorted(base.iterdir()):
            if not pkg_dir.name.startswith("intel-rapl:"):
                continue
            energy_file = pkg_dir / "energy_uj"
            name_file = pkg_dir / "name"
            if (
                energy_file.exists()
                and name_file.exists()
                and self._try_read_energy_uj(energy_file)
            ):
                name = name_file.read_text().strip()
                self._domains[name] = str(energy_file)
                self._samples[name] = []
            # Sub-domains (dram, core, etc.)
            for sub_dir in sorted(pkg_dir.iterdir()):
                if not sub_dir.name.startswith("intel-rapl:"):
                    continue
                sub_energy = sub_dir / "energy_uj"
                sub_name = sub_dir / "name"
                if (
                    sub_energy.exists()
                    and sub_name.exists()
                    and self._try_read_energy_uj(sub_energy)
                ):
                    sname = sub_name.read_text().strip()
                    self._domains[sname] = str(sub_energy)
                    self._samples[sname] = []

    @staticmethod
    def _read_energy_uj(path: str) -> float:
        """Read energy in micro-joules from a given sysfs path."""
        try:
            with open(path) as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            return 0.0

    def _poll_loop(self) -> None:
        """Background loop to poll energy counters at regular intervals."""
        while self._running:
            t = time.perf_counter()
            for name, path in self._domains.items():
                self._samples[name].append(self._read_energy_uj(path))
            self._timestamps.append(t)
            elapsed = time.perf_counter() - t
            remaining = self._poll_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def start(self) -> None:
        """Start background polling thread."""
        self._running = True
        self._samples = {n: [] for n in self._domains}
        self._timestamps = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread and wait for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def available(self) -> bool:
        """Check if any RAPL domains were discovered."""
        return len(self._domains) > 0

    def results(self) -> dict[str, dict[str, Any]]:
        """Compute average power, total energy, and sample count from collected data."""
        out: dict[str, dict[str, Any]] = {}
        if len(self._timestamps) < 2:
            for name in self._domains:
                out[name] = {
                    "avg_power_w": 0.0,
                    "energy_j": 0.0,
                    "n_samples": 0,
                    "duration_s": 0.0,
                }
            return out
        duration = self._timestamps[-1] - self._timestamps[0]
        for name, samples in self._samples.items():
            if len(samples) < 2:
                out[name] = {
                    "avg_power_w": 0.0,
                    "energy_j": 0.0,
                    "n_samples": 0,
                    "duration_s": 0.0,
                }
                continue
            # RAPL energy counters are cumulative (micro-joules)
            energy_delta_uj = samples[-1] - samples[0]
            # Handle counter wraparound
            if energy_delta_uj < 0:
                max_uj_path = Path(self._domains[name]).parent / "max_energy_range_uj"
                if max_uj_path.exists():
                    max_uj = float(max_uj_path.read_text().strip())
                    energy_delta_uj += max_uj
            energy_j = energy_delta_uj / 1e6
            avg_w = energy_j / duration if duration > 0 else 0.0
            out[name] = {
                "avg_power_w": round(avg_w, 4),
                "energy_j": round(energy_j, 6),
                "n_samples": len(samples),
                "duration_s": round(duration, 4),
            }
        return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def make_dummy_input(device: torch.device) -> Tensor:
    """Create a random normalised [1, 2, 256, 256] input (NCHW, already log-normalised)."""
    return torch.randn(1, 2, IMAGE_SIZE, IMAGE_SIZE, device=device, dtype=torch.float32)


def normalize_input(x_lin: Tensor) -> Tensor:
    """Normalise linear-amplitude [B, 2, H, W] to log-scale [0,1] as SARDDCModule.forward does."""
    return (torch.log(torch.square(x_lin) + EPS) - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)


# ---------------------------------------------------------------------------
# Synchronisation helper
# ---------------------------------------------------------------------------
def sync(device: torch.device) -> None:
    """Synchronize CUDA if on GPU (no-op for CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------
def run_scenario_full(
    x: Tensor,
    net: torch.nn.Module,
    timer: StepTimer,
    device: torch.device,
    cuda_timer: CudaEventTimer | None = None,
) -> int:
    """Full compress + decompress.

    Returns total compressed bytes.
    """
    is_gpu = device.type == "cuda"
    prefix = "gpu" if is_gpu else "nn"

    sync(device)
    timer.mark("preprocess")
    if cuda_timer:
        cuda_timer.mark("preprocess")

    # ---- Encode ----
    x_real = x[:, :1, :, :]
    x_imag = x[:, 1:, :, :]

    sync(device)
    timer.mark(f"{prefix}_g_a")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_g_a")
    y_real = net.g_a(x_real)
    y_imag = net.g_a(x_imag)

    sync(device)
    timer.mark("cpu_concat_abs")
    if cuda_timer:
        cuda_timer.mark("cpu_concat_abs")
    y = torch.cat((y_real, y_imag), dim=1)
    y_abs = torch.abs(y)

    sync(device)
    timer.mark(f"{prefix}_h_a")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_h_a")
    z = net.h_a(y_abs)

    sync(device)
    timer.mark("cpu_eb_compress")
    if cuda_timer:
        cuda_timer.mark("cpu_eb_compress")
    z_strings = net.entropy_bottleneck.compress(z)
    z_bytes = sum(
        len(s)
        for s_list in z_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )

    sync(device)
    timer.mark("cpu_eb_decompress")
    if cuda_timer:
        cuda_timer.mark("cpu_eb_decompress")
    z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    sync(device)
    timer.mark(f"{prefix}_h_s")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_h_s")
    scales = net.h_s(z_hat)

    sync(device)
    timer.mark("cpu_gc_compress")
    if cuda_timer:
        cuda_timer.mark("cpu_gc_compress")
    indexes = net.gaussian_conditional.build_indexes(scales)
    y_strings = net.gaussian_conditional.compress(y, indexes)
    y_bytes = sum(
        len(s)
        for s_list in y_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )

    # ---- Decode ----
    sync(device)
    timer.mark("cpu_gc_decompress")
    if cuda_timer:
        cuda_timer.mark("cpu_gc_decompress")
    y_hat = net.gaussian_conditional.decompress(y_strings, indexes)

    sync(device)
    timer.mark("cpu_split_y_hat")
    if cuda_timer:
        cuda_timer.mark("cpu_split_y_hat")
    y_hat_real = y_hat[:, : y_hat.shape[1] // 2, :, :]
    y_hat_imag = y_hat[:, y_hat.shape[1] // 2 :, :, :]

    sync(device)
    timer.mark(f"{prefix}_g_s")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_g_s")
    _recon_real = net.g_s(y_hat_real)
    _recon_imag = net.g_s(y_hat_imag)

    sync(device)
    timer.mark("postprocess")
    if cuda_timer:
        cuda_timer.mark("postprocess")
    timer.mark("_end")
    timer.commit()
    if cuda_timer:
        cuda_timer.mark("_end")
        cuda_timer.commit()

    return z_bytes + y_bytes


def run_scenario_compress(
    x: Tensor,
    net: torch.nn.Module,
    timer: StepTimer,
    device: torch.device,
    cuda_timer: CudaEventTimer | None = None,
) -> int:
    """Compress only (encode path)."""
    is_gpu = device.type == "cuda"
    prefix = "gpu" if is_gpu else "nn"

    sync(device)
    timer.mark("preprocess")
    if cuda_timer:
        cuda_timer.mark("preprocess")

    x_real = x[:, :1, :, :]
    x_imag = x[:, 1:, :, :]

    sync(device)
    timer.mark(f"{prefix}_g_a")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_g_a")
    y_real = net.g_a(x_real)
    y_imag = net.g_a(x_imag)

    sync(device)
    timer.mark("cpu_concat_abs")
    if cuda_timer:
        cuda_timer.mark("cpu_concat_abs")
    y = torch.cat((y_real, y_imag), dim=1)
    y_abs = torch.abs(y)

    sync(device)
    timer.mark(f"{prefix}_h_a")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_h_a")
    z = net.h_a(y_abs)

    sync(device)
    timer.mark("cpu_eb_compress")
    if cuda_timer:
        cuda_timer.mark("cpu_eb_compress")
    z_strings = net.entropy_bottleneck.compress(z)
    z_bytes = sum(
        len(s)
        for s_list in z_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )

    sync(device)
    timer.mark("cpu_eb_decompress")
    if cuda_timer:
        cuda_timer.mark("cpu_eb_decompress")
    z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    sync(device)
    timer.mark(f"{prefix}_h_s")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_h_s")
    scales = net.h_s(z_hat)

    sync(device)
    timer.mark("cpu_gc_compress")
    if cuda_timer:
        cuda_timer.mark("cpu_gc_compress")
    indexes = net.gaussian_conditional.build_indexes(scales)
    y_strings = net.gaussian_conditional.compress(y, indexes)
    y_bytes = sum(
        len(s)
        for s_list in y_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )

    sync(device)
    timer.mark("_end")
    timer.commit()
    if cuda_timer:
        cuda_timer.mark("_end")
        cuda_timer.commit()

    return z_bytes + y_bytes


def run_scenario_decompress(
    x: Tensor,
    net: torch.nn.Module,
    timer: StepTimer,
    device: torch.device,
    cuda_timer: CudaEventTimer | None = None,
    *,
    cached: dict[str, Any] | None = None,
) -> int:
    """Decompress only (decode path)."""
    if cached is None:
        raise ValueError("decompress scenario requires cached data from a prior compress.")

    is_gpu = device.type == "cuda"
    prefix = "gpu" if is_gpu else "nn"

    z_strings = cached["z_strings"]
    y_strings = cached["y_strings"]
    z_shape = cached["z_shape"]
    indexes = cached["indexes"]
    z_bytes = cached["z_bytes"]
    y_bytes = cached["y_bytes"]

    sync(device)
    timer.mark("cpu_eb_decompress")
    if cuda_timer:
        cuda_timer.mark("cpu_eb_decompress")
    z_hat = net.entropy_bottleneck.decompress(z_strings, z_shape)

    sync(device)
    timer.mark(f"{prefix}_h_s")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_h_s")
    scales = net.h_s(z_hat)

    sync(device)
    timer.mark("cpu_gc_decompress")
    if cuda_timer:
        cuda_timer.mark("cpu_gc_decompress")
    indexes_new = net.gaussian_conditional.build_indexes(scales)
    y_hat = net.gaussian_conditional.decompress(y_strings, indexes_new)

    sync(device)
    timer.mark("cpu_split_y_hat")
    if cuda_timer:
        cuda_timer.mark("cpu_split_y_hat")
    y_hat_real = y_hat[:, : y_hat.shape[1] // 2, :, :]
    y_hat_imag = y_hat[:, y_hat.shape[1] // 2 :, :, :]

    sync(device)
    timer.mark(f"{prefix}_g_s")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_g_s")
    _recon_real = net.g_s(y_hat_real)
    _recon_imag = net.g_s(y_hat_imag)

    sync(device)
    timer.mark("_end")
    timer.commit()
    if cuda_timer:
        cuda_timer.mark("_end")
        cuda_timer.commit()

    return z_bytes + y_bytes


def run_scenario_nn_only(
    x: Tensor,
    net: torch.nn.Module,
    timer: StepTimer,
    device: torch.device,
    cuda_timer: CudaEventTimer | None = None,
) -> int:
    """NN subgraphs only — no entropy coding."""
    is_gpu = device.type == "cuda"
    prefix = "gpu" if is_gpu else "nn"

    x_real = x[:, :1, :, :]
    x_imag = x[:, 1:, :, :]

    sync(device)
    timer.mark(f"{prefix}_g_a")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_g_a")
    y_real = net.g_a(x_real)
    y_imag = net.g_a(x_imag)

    sync(device)
    timer.mark("cpu_concat_abs")
    if cuda_timer:
        cuda_timer.mark("cpu_concat_abs")
    y = torch.cat((y_real, y_imag), dim=1)
    y_abs = torch.abs(y)

    sync(device)
    timer.mark(f"{prefix}_h_a")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_h_a")
    z = net.h_a(y_abs)

    sync(device)
    timer.mark(f"{prefix}_h_s")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_h_s")
    # Feed z directly to h_s (bypass entropy)
    scales = net.h_s(z)

    sync(device)
    timer.mark(f"{prefix}_g_s")
    if cuda_timer:
        cuda_timer.mark(f"{prefix}_g_s")
    y_hat_real = y[:, :C_MAIN, :, :]
    y_hat_imag = y[:, C_MAIN:, :, :]
    _recon_real = net.g_s(y_hat_real)
    _recon_imag = net.g_s(y_hat_imag)

    sync(device)
    timer.mark("_end")
    timer.commit()
    if cuda_timer:
        cuda_timer.mark("_end")
        cuda_timer.commit()

    return 0


def run_scenario_entropy_only(
    x: Tensor,
    net: torch.nn.Module,
    timer: StepTimer,
    device: torch.device,
    cuda_timer: CudaEventTimer | None = None,
) -> int:
    """Entropy coding only (produce latents via NN, time only CPU coding)."""
    x_real = x[:, :1, :, :]
    x_imag = x[:, 1:, :, :]

    # Produce latents (untimed)
    with torch.inference_mode():
        y_real = net.g_a(x_real)
        y_imag = net.g_a(x_imag)
        y = torch.cat((y_real, y_imag), dim=1)
        y_abs = torch.abs(y)
        z = net.h_a(y_abs)

    sync(device)

    # ---- Timed section ----
    timer.mark("cpu_eb_compress")
    if cuda_timer:
        cuda_timer.mark("cpu_eb_compress")
    z_strings = net.entropy_bottleneck.compress(z)
    z_bytes = sum(
        len(s)
        for s_list in z_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )

    sync(device)
    timer.mark("cpu_eb_decompress")
    if cuda_timer:
        cuda_timer.mark("cpu_eb_decompress")
    z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    sync(device)
    timer.mark("cpu_gc_compress")
    if cuda_timer:
        cuda_timer.mark("cpu_gc_compress")
    scales = net.h_s(z_hat)
    indexes = net.gaussian_conditional.build_indexes(scales)
    y_strings = net.gaussian_conditional.compress(y, indexes)
    y_bytes = sum(
        len(s)
        for s_list in y_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )

    sync(device)
    timer.mark("cpu_gc_decompress")
    if cuda_timer:
        cuda_timer.mark("cpu_gc_decompress")
    _y_hat = net.gaussian_conditional.decompress(y_strings, indexes)

    sync(device)
    timer.mark("_end")
    timer.commit()
    if cuda_timer:
        cuda_timer.mark("_end")
        cuda_timer.commit()

    return z_bytes + y_bytes


# ---------------------------------------------------------------------------
# Pre-compress helper for decompress scenario
# ---------------------------------------------------------------------------
def _precompress(x: Tensor, net: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Run a single encode pass and cache bitstreams for the decompress scenario."""
    sync(device)
    with torch.inference_mode():
        x_real = x[:, :1, :, :]
        x_imag = x[:, 1:, :, :]
        y_real = net.g_a(x_real)
        y_imag = net.g_a(x_imag)
        y = torch.cat((y_real, y_imag), dim=1)
        y_abs = torch.abs(y)
        z = net.h_a(y_abs)

        z_strings = net.entropy_bottleneck.compress(z)
        z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales = net.h_s(z_hat)
        indexes = net.gaussian_conditional.build_indexes(scales)
        y_strings = net.gaussian_conditional.compress(y, indexes)

    z_bytes = sum(
        len(s)
        for s_list in z_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )
    y_bytes = sum(
        len(s)
        for s_list in y_strings
        for s in (s_list if isinstance(s_list, (list, tuple)) else [s_list])
    )

    return {
        "z_strings": z_strings,
        "y_strings": y_strings,
        "z_shape": z.size()[-2:],
        "indexes": indexes,
        "z_bytes": z_bytes,
        "y_bytes": y_bytes,
    }


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict[str, Any]:
    """Collect GPU hardware info via torch.cuda and nvidia-smi."""
    if not torch.cuda.is_available():
        return {}
    info: dict[str, Any] = {
        "name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda or "",  # type: ignore[attr-defined]
        "cudnn_version": (
            str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""
        ),
        "torch_version": torch.__version__,
        "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2)),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.limit,clocks.max.sm,clocks.max.mem",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 3:
            info["power_limit_w"] = float(parts[0])
            info["max_sm_clock_mhz"] = float(parts[1])
            info["max_mem_clock_mhz"] = float(parts[2])
    except Exception:
        pass
    return info


def get_cpu_info() -> dict[str, Any]:
    """Collect CPU info from /proc/cpuinfo and lscpu."""
    info: dict[str, Any] = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["name"] = line.split(":", 1)[1].strip()
                    break
        result = subprocess.run(["nproc"], capture_output=True, text=True, timeout=5)
        info["n_cores"] = int(result.stdout.strip())
    except Exception:
        pass
    info["torch_threads"] = torch.get_num_threads()
    return info


def get_model_info(net: torch.nn.Module) -> dict[str, Any]:
    """Basic model stats."""
    total_params = sum(p.numel() for p in net.parameters())
    total_bytes = sum(p.element_size() * p.numel() for p in net.parameters())
    return {
        "total_params": total_params,
        "trainable_params": sum(p.numel() for p in net.parameters() if p.requires_grad),
        "weights_size_mb": round(total_bytes / (1024**2), 2),
        "activation": getattr(net, "activation", "unknown"),
        "N": C_MAIN,
        "M": C_HYPER,
    }


# ---------------------------------------------------------------------------
# Main benchmark for one device
# ---------------------------------------------------------------------------
def run_benchmark_on_device(
    net: torch.nn.Module,
    device: torch.device,
    scenario: str,
    n_warmup: int,
    n_iters: int,
    measure_power: bool,
    power_poll_hz: float,
    idle_baseline_s: float = 10.0,
) -> dict[str, Any]:
    """Run the benchmark on a single device and return the results dict."""
    is_gpu = device.type == "cuda"
    platform = f"GPU_{torch.cuda.get_device_name(0).replace(' ', '_')}" if is_gpu else "CPU"
    nn_prefix = "gpu" if is_gpu else "nn"

    print(f"\n{'='*60}")
    print(f"  BENCHMARKING ON: {platform}")
    print(f"{'='*60}")
    print(f"  Scenario : {scenario}")
    print(f"  Warmup   : {n_warmup}")
    print(f"  Measured : {n_iters}")
    print(f"  Power    : {'ON' if measure_power else 'OFF'}")

    net = net.to(device)
    net.eval()

    # ---- Prepare input ----
    x = make_dummy_input(device)
    print(f"  Input    : {list(x.shape)} on {device}")

    # ---- Select scenario ----
    scenario_fn_map = {
        "full": run_scenario_full,
        "compress": run_scenario_compress,
        "decompress": run_scenario_decompress,
        "nn_only": run_scenario_nn_only,
        "entropy_only": run_scenario_entropy_only,
    }
    scenario_fn = scenario_fn_map[scenario]

    # Pre-compress for decompress scenario
    cached: dict[str, Any] | None = None
    if scenario == "decompress":
        print("  Pre-compressing for decompress scenario...")
        cached = _precompress(x, net, device)

    # ---- Power setup ----
    gpu_power: NvidiaSmiPowerSampler | None = None
    rapl_power: RAPLPowerSampler | None = None
    if measure_power:
        if is_gpu:
            gpu_power = NvidiaSmiPowerSampler(poll_interval_s=1.0 / power_poll_hz)
            print(f"  GPU power sampling at {power_poll_hz} Hz")
        rapl_power = RAPLPowerSampler(poll_interval_s=1.0 / power_poll_hz)
        if rapl_power.available:
            print(f"  RAPL CPU power sampling at {power_poll_hz} Hz")
        else:
            rapl_power = None
            print(
                "  RAPL not available — CPU power disabled\n"
                "  (energy_uj files require root; run with sudo or:\n"
                "   sudo chmod o+r /sys/class/powercap/intel-rapl/*/energy_uj)"
            )

    # ---- Idle baseline ----
    idle_gpu_results: dict[str, Any] | None = None
    idle_rapl_results: dict[str, dict[str, Any]] | None = None
    if idle_baseline_s > 0 and measure_power:
        print(f"\n  Capturing idle baseline ({idle_baseline_s:.0f}s)...")
        if gpu_power:
            idle_gpu_sampler = NvidiaSmiPowerSampler(poll_interval_s=1.0 / power_poll_hz)
            idle_gpu_sampler.start()
        if rapl_power:
            idle_rapl_sampler = RAPLPowerSampler(poll_interval_s=1.0 / power_poll_hz)
            idle_rapl_sampler.start()
        time.sleep(idle_baseline_s)
        if gpu_power:
            idle_gpu_sampler.stop()
            idle_gpu_results = idle_gpu_sampler.results()
        if rapl_power:
            idle_rapl_sampler.stop()
            idle_rapl_results = idle_rapl_sampler.results()
        print("  Idle baseline captured.")

    # ---- Warmup ----
    print(f"\n  Warmup ({n_warmup} iterations)...")
    warmup_timer = StepTimer()
    with torch.inference_mode():
        for _ in range(n_warmup):
            kwargs: dict[str, Any] = {}
            if scenario == "decompress":
                kwargs["cached"] = cached
            scenario_fn(x, net, warmup_timer, device, **kwargs)

    # ---- Measured runs ----
    print(f"  Benchmarking ({n_iters} iterations)...")
    timer = StepTimer()
    cuda_timer = CudaEventTimer() if is_gpu else None
    total_bytes_list: list[int] = []

    if gpu_power:
        gpu_power.start()
    if rapl_power:
        rapl_power.start()

    wall_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_iters):
            kwargs = {}
            if scenario == "decompress":
                kwargs["cached"] = cached
            nbytes = scenario_fn(x, net, timer, device, cuda_timer, **kwargs)
            total_bytes_list.append(nbytes)
    wall_end = time.perf_counter()

    if gpu_power:
        gpu_power.stop()
    if rapl_power:
        rapl_power.stop()

    wall_total = wall_end - wall_start

    # ---- Collect results ----
    step_summary = timer.summary()
    cuda_step_summary = cuda_timer.summary() if cuda_timer else None

    nn_steps = [k for k in step_summary if k.startswith(f"{nn_prefix}_")]
    cpu_steps = [k for k in step_summary if k.startswith("cpu_")]
    nn_total_mean = sum(step_summary[k]["mean_s"] for k in nn_steps)
    cpu_total_mean = sum(step_summary[k]["mean_s"] for k in cpu_steps)

    iter_mean = timer.total_mean()

    # Map FPGA-style field names
    nn_label = "gpu" if is_gpu else "nn"
    results: dict[str, Any] = {
        "platform": platform,
        "device": str(device),
        "scenario": scenario,
        "timestamp": datetime.now().isoformat(),
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        # Latency breakdown (wall-clock via perf_counter + sync)
        "latency_breakdown": step_summary,
        "latency_total_mean_s": iter_mean,
        "latency_total_mean_ms": iter_mean * 1000,
        f"latency_{nn_label}_total_mean_ms": nn_total_mean * 1000,
        "latency_cpu_total_mean_ms": cpu_total_mean * 1000,
        "latency_wall_total_s": wall_total,
        # Throughput
        "throughput_fps": n_iters / wall_total,
        # Compression
        "avg_compressed_bytes": (
            statistics.mean(total_bytes_list)
            if total_bytes_list and total_bytes_list[0] > 0
            else None
        ),
    }

    # CUDA event timing (GPU only, more precise for NN steps)
    if cuda_step_summary:
        results["latency_breakdown_cuda_events"] = cuda_step_summary

    # Power
    power_data: dict[str, Any] = {}
    if gpu_power:
        power_data["gpu"] = gpu_power.results()
    if rapl_power:
        power_data["cpu_rapl"] = rapl_power.results()
    if idle_gpu_results:
        power_data["idle_gpu"] = idle_gpu_results
    if idle_rapl_results:
        power_data["idle_cpu_rapl"] = idle_rapl_results

    if power_data:
        # Compute totals
        if "gpu" in power_data:
            power_data["gpu_avg_w"] = power_data["gpu"]["avg_power_w"]
        if "cpu_rapl" in power_data:
            rapl_total = sum(v["avg_power_w"] for v in power_data["cpu_rapl"].values())
            power_data["cpu_rapl_total_avg_w"] = round(rapl_total, 4)
        results["power"] = power_data

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Entrypoint."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model source: either a checkpoint or a compiled FPGA model directory
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", help="Path to training checkpoint (.ckpt).")
    source.add_argument(
        "--model-dir",
        help=(
            "Path to a compiled FPGA model directory containing manifest.json "
            "(e.g. results/fpga/active_model/). The checkpoint is resolved from "
            "the manifest's 'original_run_dir' key."
        ),
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=SCENARIOS,
        help="Which pipeline stage(s) to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (default: 20)")
    parser.add_argument(
        "--iters", type=int, default=100, help="Measured iterations (default: 100)"
    )
    parser.add_argument("--power", action="store_true", help="Enable power sampling.")
    parser.add_argument(
        "--power-hz", type=float, default=10.0, help="Power sampling frequency (default: 10 Hz)."
    )
    parser.add_argument(
        "--idle-baseline",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Capture idle power for N seconds before benchmarking (requires --power).",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmark.")
    parser.add_argument("--no-cpu", action="store_true", help="Skip CPU benchmark.")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: results/<ckpt_name>/)"
    )
    args = parser.parse_args()

    # ---- Resolve checkpoint ----
    manifest: dict[str, Any] | None = None
    model_name: str | None = None

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        print(f"Resolving checkpoint from model directory: {model_dir}")
        ckpt_path, manifest = resolve_checkpoint_from_model_dir(model_dir)
        model_name = manifest.get("model_name")
        print(f"  Model name : {model_name}")
        print(f"  Checkpoint : {ckpt_path}")
    else:
        ckpt_path = Path(args.ckpt).resolve()
        if not ckpt_path.exists():
            print(f"Error: checkpoint not found at {ckpt_path}")
            sys.exit(1)

    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif model_name:
        out_dir = Path("results") / "benchmark" / model_name
    else:
        run_name = ckpt_path.parent.parent.name  # e.g. "2026-02-07_00-51-42"
        out_dir = Path("results") / "benchmark" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model from {ckpt_path}...")
    net = load_model_from_checkpoint(ckpt_path)
    print(f"Model loaded. Params: {sum(p.numel() for p in net.parameters()):,}")

    # Collect model info once
    model_info = get_model_info(net)
    if manifest:
        model_info["manifest"] = manifest

    # ---- GPU benchmark ----
    if not args.no_gpu and torch.cuda.is_available():
        gpu_results = run_benchmark_on_device(
            net=net,
            device=torch.device("cuda"),
            scenario=args.scenario,
            n_warmup=args.warmup,
            n_iters=args.iters,
            measure_power=args.power,
            power_poll_hz=args.power_hz,
            idle_baseline_s=args.idle_baseline,
        )
        gpu_results["hw_info"] = get_gpu_info()
        gpu_results["model_info"] = model_info

        gpu_path = out_dir / f"benchmark_gpu_{args.scenario}.json"
        with open(gpu_path, "w") as f:
            json.dump(gpu_results, f, indent=2, default=str)
        print(f"\n  GPU results saved to {gpu_path}")
        _print_summary("GPU", gpu_results)
    elif not args.no_gpu:
        print("\nNo CUDA GPU available — skipping GPU benchmark.")

    # ---- CPU benchmark ----
    if not args.no_cpu:
        cpu_results = run_benchmark_on_device(
            net=net,
            device=torch.device("cpu"),
            scenario=args.scenario,
            n_warmup=args.warmup,
            n_iters=args.iters,
            measure_power=args.power,
            power_poll_hz=args.power_hz,
            idle_baseline_s=args.idle_baseline,
        )
        cpu_results["hw_info"] = get_cpu_info()
        cpu_results["model_info"] = model_info

        cpu_path = out_dir / f"benchmark_cpu_{args.scenario}.json"
        with open(cpu_path, "w") as f:
            json.dump(cpu_results, f, indent=2, default=str)
        print(f"\n  CPU results saved to {cpu_path}")
        _print_summary("CPU", cpu_results)

    print("\nDone.")


def _print_summary(label: str, results: dict[str, Any]) -> None:
    """Print a human-readable summary matching the FPGA benchmark style."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY — {label} — {results['scenario']}")
    print(f"{'='*60}")
    print(f"  Total latency  : {results['latency_total_mean_ms']:.2f} ms / patch")

    # Find the NN total key
    for key in ("latency_gpu_total_mean_ms", "latency_nn_total_mean_ms"):
        if key in results:
            nn_label = "GPU" if "gpu" in key else "NN"
            print(f"    {nn_label} time     : {results[key]:.2f} ms")
    print(f"    CPU time     : {results['latency_cpu_total_mean_ms']:.2f} ms")
    print(f"  Throughput     : {results['throughput_fps']:.2f} patches/s")

    if results.get("avg_compressed_bytes"):
        bpp = results["avg_compressed_bytes"] * 8 / (IMAGE_SIZE * IMAGE_SIZE)
        print(f"  Avg BPP        : {bpp:.4f}")

    if "power" in results:
        pdata = results["power"]
        if "gpu_avg_w" in pdata:
            print(f"  GPU power      : {pdata['gpu_avg_w']:.2f} W")
        if "cpu_rapl_total_avg_w" in pdata:
            print(f"  CPU power      : {pdata['cpu_rapl_total_avg_w']:.2f} W")
        if "idle_gpu" in pdata:
            print(f"  GPU idle       : {pdata['idle_gpu']['avg_power_w']:.2f} W")

    print("\n  Per-step breakdown:")
    for step, stats in results["latency_breakdown"].items():
        print(f"    {step:25s} : {stats['mean_s']*1000:8.3f} ms  (std={stats['std_s']*1000:.3f})")

    if "latency_breakdown_cuda_events" in results:
        print("\n  Per-step breakdown (CUDA events):")
        for step, stats in results["latency_breakdown_cuda_events"].items():
            print(
                f"    {step:25s} : {stats['mean_s']*1000:8.3f} ms  (std={stats['std_s']*1000:.3f})"
            )

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
