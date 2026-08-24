"""Device-agnostic power sampling: tegrastats (Jetson, whole-SoC) -> nvidia-smi (any NVIDIA GPU,
board power only) -> unavailable (reported as `null`, never a silent 0 — mirrors the convention already
used by scripts/evaluation/benchmark_gpu.py's RAPL/nvidia-smi samplers).

`tegrastats`'s rail-line format is board/JetPack-version-specific (confirmed by direct comparison: Orin
exposes VDD_GPU_SOC / VDD_CPU_CV / VIN_SYS_5V0, other Jetson generations expose different rail names) —
so the parser below is generic (`<NAME> curmW/avgmW/maxmW`, any name) rather than hardcoding rails, and
reports every rail it finds alongside a best-effort sum. That sum can double-count if two reported rails
are nested in scope (e.g. one rail's power is a subset of another's) — this code cannot tell from the
labels alone, so `per_rail_mw` is always included so the number can be sanity-checked or corrected by
hand instead of trusted blindly.
"""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Optional

_RAIL_RE = re.compile(r"([A-Za-z0-9_]+)\s+(\d+)mW/(\d+)mW/(\d+)mW")


class TegrastatsPowerSampler:
    def __init__(self, interval_ms: int = 200):
        """Whole-SoC power sampling via `tegrastats` (Jetson only)."""
        self._interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._samples: dict[str, list[int]] = {}
        self._timestamps: list[float] = []
        self._running = False
        self.available = shutil.which("tegrastats") is not None

    def _read_loop(self) -> None:
        """Read lines from the `tegrastats` subprocess and parse power samples."""
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if not self._running:
                break
            t = time.perf_counter()
            found = _RAIL_RE.findall(line)
            if not found:
                continue
            for name, cur, _avg, _mx in found:
                self._samples.setdefault(name, []).append(int(cur))
            self._timestamps.append(t)

    def start(self) -> None:
        """Start the `tegrastats` subprocess and begin reading power samples in a background
        thread."""
        if not self.available:
            return
        self._samples, self._timestamps, self._running = {}, [], True
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the `tegrastats` subprocess and background thread, cleaning up resources."""
        if not self.available:
            return
        self._running = False
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def results(self) -> dict[str, Any] | None:
        """Return a summary of the collected power samples, including average power, energy, and
        per-rail breakdown."""
        if not self.available or len(self._timestamps) < 2 or not self._samples:
            return None
        duration = self._timestamps[-1] - self._timestamps[0]
        per_rail_mw = {name: statistics.mean(v) for name, v in self._samples.items() if v}
        total_w = sum(per_rail_mw.values()) / 1000.0
        return {
            "avg_power_w": round(total_w, 4),
            "energy_j": round(total_w * duration, 6),
            "n_samples": len(self._timestamps),
            "duration_s": round(duration, 4),
            "per_rail_mw": {k: round(v, 1) for k, v in per_rail_mw.items()},
            "source": "tegrastats",
            "scope_caveat": "sum of all rails tegrastats reports; may double-count nested rails",
        }


class NvidiaSmiPowerSampler:
    """GPU-board power only (narrower scope than tegrastats' whole-SoC view) — fallback for non-
    Jetson CUDA devices (a desktop GPU, or a future board without tegrastats)."""

    def __init__(self, interval_s: float = 0.2):
        """Sample power draw from `nvidia-smi` at a given interval (seconds)."""
        self._interval_s = interval_s
        self._samples: list[float] = []
        self._timestamps: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    def _poll_loop(self) -> None:
        """"""
        while self._running:
            t = time.perf_counter()
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self._samples.append(float(out.stdout.strip().splitlines()[0]))
                self._timestamps.append(t)
            except Exception:
                pass
            elapsed = time.perf_counter() - t
            remaining = self._interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def start(self) -> None:
        """Start the background thread that polls `nvidia-smi` for power draw."""
        if not self.available:
            return
        self._samples, self._timestamps, self._running = [], [], True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and clean up resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    def results(self) -> dict[str, Any] | None:
        """Return a summary of the collected power samples, including average power and energy."""
        if not self.available or len(self._timestamps) < 2:
            return None
        duration = self._timestamps[-1] - self._timestamps[0]
        avg_w = statistics.mean(self._samples)
        return {
            "avg_power_w": round(avg_w, 4),
            "energy_j": round(avg_w * duration, 6),
            "n_samples": len(self._samples),
            "duration_s": round(duration, 4),
            "source": "nvidia-smi",
        }


def make_power_sampler():
    """Best available sampler for this device: tegrastats > nvidia-smi > None (never guess)."""
    tg = TegrastatsPowerSampler()
    if tg.available:
        return tg
    sm = NvidiaSmiPowerSampler()
    if sm.available:
        return sm
    return None
