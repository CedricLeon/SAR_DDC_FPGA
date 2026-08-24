"""Device-agnostic power sampling: tegrastats (Jetson, whole-SoC) -> nvidia-smi (any NVIDIA GPU,
board power only) -> unavailable (reported as `null`, never a silent 0 — mirrors the convention already
used by scripts/evaluation/benchmark_gpu.py's RAPL/nvidia-smi samplers).

`tegrastats`'s rail-line format is board/JetPack-version-specific (Orin: 3-value `cur/avg/max` fields,
rails `VDD_GPU_SOC`/`VDD_CPU_CV`/`VIN_SYS_5V0`; Thor: 2-value `cur/avg` fields, rails `VDD_GPU`/
`VDD_CPU_SOC_MSS`/`VIN_SYS_5V0`/`VIN` — confirmed by live capture on both, 2026-08-24) — so the parser
is generic (`<NAME> <mW-fields>`, 2 or 3 fields, any rail name) rather than hardcoding a field count.

**Both a whole-board total and a scope-matched compute-only figure are reported — neither is dropped.**
`avg_power_w` is the whole-board total and is the headline figure for now (project decision, 2026-08-24:
report both, default to the total); `avg_power_w_compute_only` is the on-chip-compute subset, comparable
to the FPGA sampler's own deliberately-excluded `peripherals` group (`DDR4_DIMM_VDDQ` etc., see
`inference_cpp/src/benchmark/power_sampler.cpp`) — Orin's `VIN_SYS_5V0` is the analogous board I/O + DRAM
rail (confirmed via NVIDIA's Jetson Linux Developer Guide r39.2; live idle-vs-active capture: ~3.0 W idle
vs ~4.0 W during a compress run, mostly baseline infrastructure not workload-driven). Docs:
docs/onboard_pipeline.md §12, project_jetson_edge_pipeline memory.

**"Whole-board total" is a per-board policy too, not just "sum everything found"** — Thor's `VIN` rail is
itself a superset of its other rails (unlike Orin, where all three rails are independent/non-nested), so
blindly summing every rail found would double-count on Thor. `_RAIL_POLICY` below is keyed by the SoC
compatible string (`/proc/device-tree/compatible`) and states both `total` (which rail(s) sum to the
whole-board figure) and `compute` (the scope-matched subset) explicitly per board. An unrecognized board
falls back to summing everything found for both figures (same value for each) but flags `rail_policy`
loudly rather than presenting it as verified — `per_rail_mw` always has every individual rail so either
figure can be checked or recomputed by hand.
"""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

_RAIL_RE = re.compile(r"([A-Za-z0-9_]+)\s+(\d+)mW(?:/\d+mW){1,2}")

# SoC compatible string -> per-board rail policy. "compute" = on-chip-compute rails (the scope-matched
# figure, comparable to the FPGA's peripherals-excluded number). "total" = the rail(s) that sum to the
# whole-board figure — NOT always "every rail found": on Orin the 3 rails are independent so total is
# all of them, but on Thor `VIN` is itself a superset of the others, so total is `VIN` alone (summing it
# together with the other rails would double-count).
# Thor's entry is informed by one live capture (2026-08-24) but NOT independently verified the way
# Orin's is (Thor work is deferred — see inference_edge/README.md) — treat as a documented best guess,
# not a confirmed policy, until Thor is actually worked on again.
_RAIL_POLICY: dict[str, dict[str, list[str]]] = {
    "tegra234": {  # Orin — 3 independent (non-nested) rails, confirmed via NVIDIA Jetson Linux
        # Developer Guide r39.2 + live idle-vs-active capture (see module docstring).
        "compute": ["VDD_GPU_SOC", "VDD_CPU_CV"],
        "excluded": ["VIN_SYS_5V0"],
        "total": ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"],
    },
    "tegra264": {  # Thor — unverified, see docstring above. VIN is a superset of the other rails.
        "compute": ["VDD_GPU", "VDD_CPU_SOC_MSS"],
        "excluded": ["VIN_SYS_5V0"],
        "total": ["VIN"],
    },
}


def _detect_soc() -> str | None:
    """Best-effort SoC compatible string (e.g. `tegra234`) from the device tree, for `_RAIL_POLICY`
    lookup.

    None off-Jetson or if unreadable — callers must treat that as "policy unknown", not error.
    """
    try:
        compat = Path("/proc/device-tree/compatible").read_bytes().split(b"\x00")
        for c in compat:
            s = c.decode(errors="ignore")
            if s.startswith("nvidia,tegra"):
                return s.split(",", 1)[1]
    except Exception:
        pass
    return None


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
            for name, cur in found:
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
        """Return a summary of the collected power samples.

        `avg_power_w` is the whole-board total
        (headline figure for now, per project decision 2026-08-24) — summed per `_RAIL_POLICY["total"]`,
        which is NOT always "every rail found" (Thor's `VIN` rail is itself a superset of the others;
        summing everything there would double-count). `avg_power_w_compute_only` is the scope-matched
        on-chip-compute figure (`_RAIL_POLICY["compute"]`), comparable to the FPGA sampler's own
        peripherals-excluded convention. `per_rail_mw` always has every individual rail tegrastats
        reported, so either figure can be checked or recomputed by hand.
        """
        if not self.available or len(self._timestamps) < 2 or not self._samples:
            return None
        duration = self._timestamps[-1] - self._timestamps[0]
        per_rail_mw = {name: statistics.mean(v) for name, v in self._samples.items() if v}

        soc = _detect_soc()
        policy = _RAIL_POLICY.get(soc) if soc else None
        if policy is not None:
            total_rails = [r for r in policy["total"] if r in per_rail_mw]
            compute_rails = [r for r in policy["compute"] if r in per_rail_mw]
            missing = sorted(set(policy["total"] + policy["compute"]) - set(per_rail_mw))
            total_w = sum(per_rail_mw[r] for r in total_rails) / 1000.0
            compute_w = sum(per_rail_mw[r] for r in compute_rails) / 1000.0
            rail_policy = f"soc={soc}: total={total_rails}, compute_only={compute_rails}" + (
                f" (MISSING from tegrastats output: {missing}!)" if missing else ""
            )
        else:
            total_rails = list(per_rail_mw)
            compute_rails = total_rails
            total_w = sum(per_rail_mw.values()) / 1000.0
            compute_w = total_w
            rail_policy = (
                f"unrecognized board (soc={soc!r}, not in _RAIL_POLICY) — summed ALL rails found for "
                "both figures; may double-count if any rail is a superset of another (seen on Thor), "
                "and may include board-I/O/DDR rails the FPGA comparison excludes — verify before citing"
            )

        return {
            "avg_power_w": round(total_w, 4),
            "energy_j": round(total_w * duration, 6),
            "avg_power_w_compute_only": round(compute_w, 4),
            "energy_j_compute_only": round(compute_w * duration, 6),
            "n_samples": len(self._timestamps),
            "duration_s": round(duration, 4),
            "per_rail_mw": {k: round(v, 1) for k, v in per_rail_mw.items()},
            "total_rails_summed": total_rails,
            "compute_rails_summed": compute_rails,
            "source": "tegrastats",
            "rail_policy": rail_policy,
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
            # No board-I/O-rail split is visible to nvidia-smi (single number) — same value in both
            # fields so downstream consumers can read avg_power_w_compute_only unconditionally.
            "avg_power_w_compute_only": round(avg_w, 4),
            "energy_j_compute_only": round(avg_w * duration, 6),
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
