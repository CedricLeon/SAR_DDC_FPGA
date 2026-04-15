#!/usr/bin/env python3
"""Performance Benchmark for FPGA Hybrid Inference.

Measures per-component latency, throughput, and (optionally) power consumption
for the SAR DDC model running on the Vitis-AI DPU + ARM CPU.

Scenarios
---------
  full         : Compress + Decompress (g_a -> h_a -> EB -> h_s -> GC -> g_s)
  compress     : Encode only          (g_a -> h_a -> EB.compress -> GC.compress)
  decompress   : Decode only          (EB.decompress -> h_s -> GC.decompress -> g_s)
  dpu_only     : All four DPU subgraphs back-to-back, no entropy coding
  entropy_only : Entropy coding round-trip only (EB + GC compress/decompress)

By default all scenarios that call g_a or g_s process real and imag channels
in parallel via Python threads.  ``execute_async`` releases the GIL while
waiting on hardware, so two threads genuinely overlap on separate DPU cores
when ≥2 cores are available (confirmed on ZCU102 B4096).
Use --no-parallel to revert to sequential single-core execution for comparison.

Usage (on ZCU102):
    python3 benchmark_fpga.py --xmodel model.xmodel --scenario full [options]

The script outputs a standardised JSON file that can be merged with GPU/CPU
benchmarks for publication-ready comparison tables.

Requires Python >= 3.8 (Vitis-AI container / PetaLinux constraint).
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import statistics
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import vart  # type: ignore
    import xir  # type: ignore
except ImportError:
    print("ERROR: Vitis-AI libraries (vart, xir) not found.")
    sys.exit(1)

from entropy_models_inference import EntropyBottleneck, GaussianConditional
from inference_utils import AMP_MAX, AMP_MIN, EPS, DPUSubgraphRunner, identify_subgraphs

# ---------------------------------------------------------------------------
# Constants (must match inference_hybrid.py)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256
C_MAIN = 128
C_HYPER = C_MAIN * 2

SCENARIOS = ["full", "compress", "decompress", "dpu_only", "entropy_only"]

# When True (default), g_a and g_s subgraphs run real and imag channels in
# parallel via Python threads.  execute_async releases the Python GIL while
# waiting on hardware, so two threads genuinely overlap at the DPU level when
# ≥2 physical cores are available.
# Set to False via --no-parallel for a sequential single-core baseline.
_PARALLEL: bool = True


# ---------------------------------------------------------------------------
# Power sampler (INA226 via sysfs)
# ---------------------------------------------------------------------------
class INA226PowerSampler:
    """Polls INA226 sensors via /sys/class/hwmon at configurable frequency.

    Runs in a background thread.  Call ``start()`` before the workload and
    ``stop()`` after.  ``results()`` returns average power (W) and total
    energy (J) for each tracked rail.

    Parameters
    ----------
    rail_map : dict
        Mapping of human-readable rail name to hwmon path, e.g.
        ``{"VCCINT": "/sys/class/hwmon/hwmon10/power1_input"}``.
        The sysfs file returns micro-watts.
    poll_interval_s : float
        Seconds between consecutive reads (default 10 ms = 100 Hz).
    """

    def __init__(
        self,
        rail_map: dict[str, str],
        poll_interval_s: float = 0.01,
    ):
        self._rail_map = rail_map
        self._poll_interval = poll_interval_s
        self._samples: dict[str, list[float]] = {r: [] for r in rail_map}
        self._timestamps: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def _read_power_uw(self, path: str) -> float:
        """Read instantaneous power in micro-watts from sysfs."""
        try:
            with open(path) as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            return 0.0

    def _poll_loop(self) -> None:
        """Poll loop running in background thread."""
        while self._running:
            t = time.perf_counter()
            for rail, path in self._rail_map.items():
                self._samples[rail].append(self._read_power_uw(path))
            self._timestamps.append(t)
            # Busy-wait for higher precision than time.sleep allows
            while time.perf_counter() - t < self._poll_interval:
                pass

    def start(self) -> None:
        """Start the background polling thread."""
        self._running = True
        self._samples = {r: [] for r in self._rail_map}
        self._timestamps = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread and wait for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def results(self) -> dict[str, dict[str, float]]:
        """Return per-rail statistics.

        Returns
        -------
        dict
            ``{rail_name: {"avg_power_w": ..., "energy_j": ..., "n_samples": ...}}``
        """
        out: dict[str, dict[str, float]] = {}
        if len(self._timestamps) < 2:
            for rail in self._rail_map:
                out[rail] = {"avg_power_w": 0.0, "energy_j": 0.0, "n_samples": 0}
            return out

        duration = self._timestamps[-1] - self._timestamps[0]
        for rail, samples in self._samples.items():
            watts = [s / 1e6 for s in samples]  # uW -> W
            avg_w = statistics.mean(watts) if watts else 0.0
            energy_j = avg_w * duration
            out[rail] = {
                "avg_power_w": round(avg_w, 4),
                "energy_j": round(energy_j, 6),
                "n_samples": len(samples),
                "duration_s": round(duration, 4),
            }
        return out


# ---------------------------------------------------------------------------
# Sensor auto-discovery
# ---------------------------------------------------------------------------
# ZCU102 INA226 chip reference designator → power rail mapping.
#
# Derived by cross-referencing two tables from UG1182 (v1.7):
#   • Table 3-56 "ZCU102 Power Rails with INA226 Power Monitors"
#     Lists each monitored power rail, its regulator, and INA226 I2C address
#     (e.g. VCCINT → PL:0x40).
#   • Table 3-22 "I2C0 U60 (Addr. 0x75) Mux Target Bus Connections"
#     Lists each physical INA226 chip by its PCB reference designator and
#     I2C address (e.g. U79 → PL_PMBUS:0x40).
#
# Matching on I2C bus + address gives the definitive link:
#   sysfs "ina226_u79" → (Table 3-22) PL:0x40 → (Table 3-56) VCCINT.
ZCU102_SENSOR_MAP: dict[str, str] = {
    # ---- PL_PMBUS (I2C mux channel 2) ----
    "u79": "VCCINT",  # PL core supply — dominant DPU power (0x40)
    "u81": "VCCBRAM",  # PL Block RAM supply (0x41)
    "u80": "VCCAUX",  # PL auxiliary supply (0x42)
    "u84": "VCC1V2",  # PL 1.2 V supply (0x43)
    "u16": "VCC3V3",  # PL 3.3 V general I/O (0x44)
    "u65": "VADJ_FMC",  # FMC adjustable VCCO (0x45)
    "u74": "MGTAVCC",  # PL MGT transceiver analog core (0x46)
    "u75": "MGTAVTT",  # PL MGT transceiver termination (0x47)
    # ---- PS_PMBUS (I2C mux channel 1) ----
    "u76": "VCCPSINTFP",  # PS full-power domain (APU + interconnect) (0x40)
    "u77": "VCCPSINTLP",  # PS low-power domain (RPU) (0x41)
    "u78": "VCCPSAUX",  # PS auxiliary supply (0x42)
    "u87": "VCCPSPLL",  # PS PLL supply (0x43)
    "u85": "MGTRAVCC",  # PS MGT transceiver analog core (0x44)
    "u86": "MGTRAVTT",  # PS MGT transceiver termination (0x45)
    "u93": "VCCO_PSDDR_504",  # PS DDR I/O supply (bank 504) (0x46)
    "u88": "VCCOPS",  # PS operational supply (0x47)
    "u15": "VCCOPS3",  # PS operational supply 3 (0x4A)
    "u92": "VCCPSDDRPLL",  # PS DDR PLL supply (0x4B)
}

# Semantic groupings for power analysis reporting.
# DPU workload power is dominated by VCCINT + VCCBRAM (PL fabric + BRAM).
POWER_GROUPS: dict[str, list[str]] = {
    "PL_total": [
        "VCCINT",
        "VCCBRAM",
        "VCCAUX",
        "VCC1V2",
        "VCC3V3",
        "VADJ_FMC",
        "MGTAVCC",
        "MGTAVTT",
    ],
    "PS_total": [
        "VCCPSINTFP",
        "VCCPSINTLP",
        "VCCPSAUX",
        "VCCPSPLL",
        "MGTRAVCC",
        "MGTRAVTT",
        "VCCO_PSDDR_504",
        "VCCOPS",
        "VCCOPS3",
        "VCCPSDDRPLL",
    ],
    "DPU_fabric": ["VCCINT", "VCCBRAM"],  # Directly driven by DPU activity
    "PS_compute": ["VCCPSINTFP", "VCCPSINTLP"],  # ARM A53 (entropy coding)
    "peripherals": ["DDR4_DIMM_VDDQ", "UTIL_3V3", "UTIL_5V0"],  # PMBus rails (not in INA226)
}

# PMBus-accessible rails not covered by INA226 sensors.
# Read via /dev/i2c-4 (MAXIM_PMBUS virtual bus) using raw I2C_RDWR ioctl.
# The three MAX15303 controllers do not implement READ_POUT (0x97).
# Power is computed as P = V * I from READ_VOUT (0x8b) and READ_IOUT (0x8c).
#
# Typical idle values on ZCU102 (measured):
#   DDR4_DIMM_VDDQ : ~0.58 W   (1.2 V DDR4 SODIMM core supply)
#   UTIL_3V3       : ~2.19 W   (Ethernet PHY, USB hubs, FMC digital)
#   UTIL_5V0       : ~0.00 W   (5 V USB VBUS — ~0 W when no USB device attached)
PMBUS_RAIL_MAP: dict[str, int] = {
    "DDR4_DIMM_VDDQ": 0x1D,
    "UTIL_3V3": 0x1A,
    "UTIL_5V0": 0x1B,
}
PMBUS_BUS: str = "/dev/i2c-4"  # MAXIM_PMBUS virtual bus (i2c mux channel 2)


class PMBusRailSampler:
    """Polls DDR4_DIMM_VDDQ, UTIL_3V3, UTIL_5V0 via /dev/i2c-4 (I2C_RDWR).

    These three rails are powered by MAX15303 (Maxim InTune) controllers that
    expose PMBus but have no INA226 companion chip.  MAX15303 does not
    implement READ_POUT (0x97), so power is computed as V x I using
    READ_VOUT (0x8b, Linear16) and READ_IOUT (0x8c, Linear11).

    VOUT_MODE (0x20) is read once at init; for all three rails on ZCU102 it
    returns 0x14 -> exponent = 20 - 32 = -12.

    Transport: I2C_RDWR with a two-message write+read transaction.
    No i2cset/i2cget needed; raw ioctl bypasses the kernel driver lock.

    Poll latency: ~3.4 ms for all 3 rails x 2 registers -> max ~290 Hz.
    Recommend poll_interval_s=0.04 (25 Hz) to keep overhead below 20 percent.

    Requires Python >= 3.8 (ctypes + fcntl only).
    """

    _I2C_RDWR = 0x0707
    _I2C_M_RD = 0x0001

    # PMBus command codes
    _CMD_VOUT_MODE = 0x20  # 1 byte: bits[4:0] = signed 5-bit exponent for Linear16
    _CMD_READ_VOUT = 0x8B  # 2 bytes LE: Linear16  V = raw * 2^exp
    _CMD_READ_IOUT = 0x8C  # 2 bytes LE: Linear11  val = mant * 2^exp (embedded)

    def __init__(
        self,
        rail_map: dict[str, int],
        bus: str = PMBUS_BUS,
        poll_interval_s: float = 0.04,
    ):
        self._rail_map = rail_map
        self._bus = bus
        self._poll_interval = poll_interval_s
        self._samples: dict[str, list[float]] = {r: [] for r in rail_map}
        self._timestamps: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

        # Read VOUT_MODE once at init to determine Linear16 exponent per rail.
        # On ZCU102 all three rails return 0x14 (exp = 20-32 = -12).
        self._vout_exp: dict[str, int] = {}
        try:
            fd = os.open(bus, os.O_RDWR)
            try:
                for rail, addr in rail_map.items():
                    raw = self._rdwr_read(fd, addr, self._CMD_VOUT_MODE, 1)[0]
                    exp = raw & 0x1F
                    if exp > 15:
                        exp -= 32
                    self._vout_exp[rail] = exp
            finally:
                os.close(fd)
        except OSError as exc:
            print(f"PMBusRailSampler: cannot open {bus}: {exc}")
            self._vout_exp = {r: -12 for r in rail_map}  # safe fallback

    # ------------------------------------------------------------------
    # Internal I2C helpers (Python 3.8 compatible, ctypes only)
    # ------------------------------------------------------------------

    def _rdwr_read(self, fd: int, addr: int, cmd: int, n: int) -> bytes:
        """Issue a PMBus read: write <cmd> then repeated-start read <n> bytes."""
        # Build two i2c_msg structs packed as a ctypes array.
        # Layout: addr(u16), flags(u16), len(u16), __pad(u16), buf(u64 ptr)
        # (matches struct i2c_msg in <linux/i2c.h> on a 64-bit ARM system)
        fmt = "HHHH Q"  # H=uint16, Q=uint64

        wb = (ctypes.c_uint8 * 1)(cmd)
        rb = (ctypes.c_uint8 * n)(*([0] * n))

        msg_write = struct.pack(fmt, addr, 0, 1, 0, ctypes.addressof(wb))
        msg_read = struct.pack(fmt, addr, self._I2C_M_RD, n, 0, ctypes.addressof(rb))
        msgs_buf = ctypes.create_string_buffer(msg_write + msg_read)

        # struct i2c_rdwr_ioctl_data { struct i2c_msg *msgs; __u32 nmsgs; }
        rdwr_fmt = "QI"  # ptr(u64) + nmsgs(u32)
        rdwr_buf = ctypes.create_string_buffer(
            struct.pack(rdwr_fmt, ctypes.addressof(msgs_buf), 2)
        )
        fcntl.ioctl(fd, self._I2C_RDWR, rdwr_buf)
        return bytes(rb)

    @staticmethod
    def _linear11(raw16: int) -> float:
        """Decode a PMBus Linear11 word into a float."""
        exp = (raw16 >> 11) & 0x1F
        mant = raw16 & 0x7FF
        if exp > 15:
            exp -= 32
        if mant > 1023:
            mant -= 2048
        return mant * (2.0**exp)

    def _read_power_w(self, fd: int, rail: str, addr: int) -> float:
        """Read V_out and I_out for one rail and return V*I (W)."""
        raw_v = struct.unpack("<H", self._rdwr_read(fd, addr, self._CMD_READ_VOUT, 2))[0]
        raw_i = struct.unpack("<H", self._rdwr_read(fd, addr, self._CMD_READ_IOUT, 2))[0]
        v = raw_v * (2.0 ** self._vout_exp[rail])
        i = self._linear11(raw_i)
        return max(0.0, v * i)

    # ------------------------------------------------------------------
    # Sampler interface (mirrors INA226PowerSampler)
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Poll loop running in background thread."""
        fd = os.open(self._bus, os.O_RDWR)
        try:
            while self._running:
                t = time.perf_counter()
                for rail, addr in self._rail_map.items():
                    try:
                        p = self._read_power_w(fd, rail, addr)
                    except OSError:
                        p = 0.0
                    self._samples[rail].append(p)
                self._timestamps.append(t)
                while time.perf_counter() - t < self._poll_interval:
                    pass
        finally:
            os.close(fd)

    def start(self) -> None:
        """Start the background polling thread."""
        self._running = True
        self._samples = {r: [] for r in self._rail_map}
        self._timestamps = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread and wait for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def results(self) -> dict[str, dict[str, float]]:
        """Return per-rail statistics (same schema as INA226PowerSampler.results)."""
        out: dict[str, dict[str, float]] = {}
        if len(self._timestamps) < 2:
            for rail in self._rail_map:
                out[rail] = {
                    "avg_power_w": 0.0,
                    "energy_j": 0.0,
                    "n_samples": 0,
                    "duration_s": 0.0,
                }
            return out
        duration = self._timestamps[-1] - self._timestamps[0]
        for rail, samples in self._samples.items():
            avg_w = statistics.mean(samples) if samples else 0.0
            out[rail] = {
                "avg_power_w": round(avg_w, 4),
                "energy_j": round(avg_w * duration, 6),
                "n_samples": len(samples),
                "duration_s": round(duration, 4),
            }
        return out


def discover_ina226_sensors() -> dict[str, str]:
    """Scan /sys/class/hwmon for INA226 sensors matching ZCU102 known rails.

    Returns a dict mapping rail names to sysfs ``power1_input`` paths.
    """
    found: dict[str, str] = {}
    hwmon_root = Path("/sys/class/hwmon")
    if not hwmon_root.exists():
        return found

    for hwmon_dir in sorted(hwmon_root.iterdir()):
        name_file = hwmon_dir / "name"
        power_file = hwmon_dir / "power1_input"
        if not name_file.exists() or not power_file.exists():
            continue
        try:
            sensor_name = name_file.read_text().strip()  # e.g. "ina226_u79"
        except OSError:
            continue
        if not sensor_name.startswith("ina226_"):
            continue

        u_ref = sensor_name.split("_", 1)[1]  # e.g. "u79"
        if u_ref in ZCU102_SENSOR_MAP:
            rail = ZCU102_SENSOR_MAP[u_ref]
            found[rail] = str(power_file)

    return found


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------
class StepTimer:
    """Accumulates per-step wall-clock durations using time.perf_counter().

    Usage::

        timer = StepTimer()
        # ... iteration loop ...
        timer.mark("g_a")          # start
        runner_g_a.run(data)
        timer.mark("h_a")          # end of g_a, start of h_a
        runner_h_a.run(data)
        timer.mark("_end")         # end of h_a
        timer.commit()             # save this iteration's splits
    """

    def __init__(self) -> None:
        self._marks: list[tuple[str, float]] = []
        self._history: dict[str, list[float]] = {}

    def mark(self, label: str) -> None:
        """Record a timestamp with the given label."""
        self._marks.append((label, time.perf_counter()))

    def commit(self) -> None:
        """Compute durations between consecutive marks and store them."""
        for i in range(len(self._marks) - 1):
            name = self._marks[i][0]
            dt = self._marks[i + 1][1] - self._marks[i][1]
            self._history.setdefault(name, []).append(dt)
        self._marks.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        """Return per-step statistics (seconds)."""
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
        """Sum of mean durations across all (non-internal) steps."""
        return sum(v["mean_s"] for k, v in self.summary().items() if not k.startswith("_"))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def make_dummy_patch() -> np.ndarray:
    """Create a random 256x256x2 float32 patch mimicking raw SAR complex data."""
    return np.random.randn(IMAGE_SIZE, IMAGE_SIZE, 2).astype(np.float32)


def load_real_patch(dataset_path: Path, index: int = 0) -> np.ndarray:
    """Load a single patch from the .npy test set."""
    data = np.load(dataset_path)
    return data[index, :, :, :2].astype(np.float32)


def preprocess_patch(noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a raw [H,W,2] patch into two [1, H, W, 1] NHWC inputs for g_a."""
    noisy_sq = np.square(noisy)
    noisy_logI = np.log(noisy_sq + EPS)
    noisy_norm = (noisy_logI - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)
    real = noisy_norm[:, :, 0][np.newaxis, :, :, np.newaxis]  # [1, H, W, 1]
    imag = noisy_norm[:, :, 1][np.newaxis, :, :, np.newaxis]
    return real.astype(np.float32), imag.astype(np.float32)


# ---------------------------------------------------------------------------
# Parallel execution helper
# ---------------------------------------------------------------------------
def run_dual(
    runner_main: DPUSubgraphRunner,
    runner_aux: DPUSubgraphRunner | None,
    data_a: np.ndarray,
    data_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one DPU subgraph twice for two independent inputs.

    When ``_PARALLEL`` is True and ``runner_aux`` is provided, both calls are
    dispatched simultaneously via Python threads.  ``execute_async`` releases
    the GIL while waiting on hardware, so two threads genuinely overlap at
    the DPU hardware level when ≥2 physical cores are available.

    Falls back to sequential calls on ``runner_main`` when ``_PARALLEL`` is
    False (``--no-parallel``) or when ``runner_aux`` is None.
    """
    if _PARALLEL and runner_aux is not None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(runner_main.run, data_a)
            fut_b = pool.submit(runner_aux.run, data_b)
            return fut_a.result(), fut_b.result()
    return runner_main.run(data_a), runner_main.run(data_b)


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------
def run_scenario_full(
    real: np.ndarray,
    imag: np.ndarray,
    runners: dict[str, DPUSubgraphRunner],
    eb: EntropyBottleneck,
    gc: GaussianConditional,
    timer: StepTimer,
) -> int:
    """Full compress + decompress.

    Returns total compressed bytes.
    """
    timer.mark("preprocess")

    # ---- Encode ----
    timer.mark("dpu_g_a")
    y_real, y_imag = run_dual(runners["g_a"], runners.get("g_a_1"), real, imag)

    timer.mark("cpu_concat_abs")
    y = np.concatenate((y_real, y_imag), axis=-1)
    y_abs = np.abs(y)

    timer.mark("dpu_h_a")
    z = runners["h_a"].run(y_abs)

    timer.mark("cpu_eb_compress")
    z_strings = eb.compress(z)
    z_bytes = sum(len(s) for s in z_strings)

    timer.mark("cpu_eb_decompress")
    z_hat = eb.decompress(z_strings, (z.shape[1], z.shape[2]))

    timer.mark("dpu_h_s")
    scales = runners["h_s"].run(z_hat)

    timer.mark("cpu_gc_compress")
    means = np.zeros_like(y)
    y_strings = gc.compress(y, scales, means)
    y_bytes = sum(len(s) for s in y_strings)

    # ---- Decode ----
    timer.mark("cpu_gc_decompress")
    y_hat = gc.decompress(y_strings, scales, means)

    timer.mark("cpu_split_y_hat")
    y_hat_real = y_hat[..., :C_MAIN]
    y_hat_imag = y_hat[..., C_MAIN:]

    timer.mark("dpu_g_s")
    run_dual(runners["g_s"], runners.get("g_s_1"), y_hat_real, y_hat_imag)

    timer.mark("postprocess")
    # (postprocess placeholder — no denorm needed for timing)
    timer.mark("_end")
    timer.commit()

    return z_bytes + y_bytes


def run_scenario_compress(
    real: np.ndarray,
    imag: np.ndarray,
    runners: dict[str, DPUSubgraphRunner],
    eb: EntropyBottleneck,
    gc: GaussianConditional,
    timer: StepTimer,
) -> int:
    """Compress only (encode path).

    Returns compressed bytes.
    """
    timer.mark("preprocess")

    timer.mark("dpu_g_a")
    y_real, y_imag = run_dual(runners["g_a"], runners.get("g_a_1"), real, imag)

    timer.mark("cpu_concat_abs")
    y = np.concatenate((y_real, y_imag), axis=-1)
    y_abs = np.abs(y)

    timer.mark("dpu_h_a")
    z = runners["h_a"].run(y_abs)

    timer.mark("cpu_eb_compress")
    z_strings = eb.compress(z)
    z_bytes = sum(len(s) for s in z_strings)

    timer.mark("cpu_eb_decompress")
    z_hat = eb.decompress(z_strings, (z.shape[1], z.shape[2]))

    timer.mark("dpu_h_s")
    scales = runners["h_s"].run(z_hat)

    timer.mark("cpu_gc_compress")
    means = np.zeros_like(y)
    y_strings = gc.compress(y, scales, means)
    y_bytes = sum(len(s) for s in y_strings)

    timer.mark("_end")
    timer.commit()
    return z_bytes + y_bytes


def run_scenario_decompress(
    _real: np.ndarray,
    _imag: np.ndarray,
    runners: dict[str, DPUSubgraphRunner],
    eb: EntropyBottleneck,
    gc: GaussianConditional,
    timer: StepTimer,
    *,
    cached_strings: dict[str, Any] | None = None,
) -> int:
    """Decompress only (decode path).

    Needs pre-compressed bitstreams; pass via ``cached_strings``.
    """
    if cached_strings is None:
        raise ValueError("decompress scenario requires cached_strings from a prior compress.")

    z_strings = cached_strings["z_strings"]
    y_strings = cached_strings["y_strings"]
    z_shape = cached_strings["z_shape"]
    scales_shape = cached_strings["scales_shape"]
    z_bytes = cached_strings["z_bytes"]
    y_bytes = cached_strings["y_bytes"]

    timer.mark("cpu_eb_decompress")
    z_hat = eb.decompress(z_strings, z_shape)

    timer.mark("dpu_h_s")
    scales = runners["h_s"].run(z_hat)

    timer.mark("cpu_gc_decompress")
    means = np.zeros(scales_shape, dtype=np.float32)
    y_hat = gc.decompress(y_strings, scales, means)

    timer.mark("cpu_split_y_hat")
    y_hat_real = y_hat[..., :C_MAIN]
    y_hat_imag = y_hat[..., C_MAIN:]

    timer.mark("dpu_g_s")
    run_dual(runners["g_s"], runners.get("g_s_1"), y_hat_real, y_hat_imag)

    timer.mark("_end")
    timer.commit()
    return z_bytes + y_bytes


def run_scenario_dpu_only(
    real: np.ndarray,
    imag: np.ndarray,
    runners: dict[str, DPUSubgraphRunner],
    eb: EntropyBottleneck,
    gc: GaussianConditional,
    timer: StepTimer,
) -> int:
    """DPU subgraphs only — no entropy coding, no CPU pre/postprocessing."""
    timer.mark("dpu_g_a")
    y_real, y_imag = run_dual(runners["g_a"], runners.get("g_a_1"), real, imag)

    timer.mark("cpu_concat_abs")
    y = np.concatenate((y_real, y_imag), axis=-1)
    y_abs = np.abs(y)

    timer.mark("dpu_h_a")
    z = runners["h_a"].run(y_abs)

    timer.mark("dpu_h_s")
    # For dpu_only we bypass entropy and feed z directly to h_s
    # (this is not physically meaningful but isolates DPU latency).
    scales = runners["h_s"].run(z)

    timer.mark("dpu_g_s")
    # Feed y directly (skip quantise/dequantise through entropy)
    y_hat_real = y[..., :C_MAIN]
    y_hat_imag = y[..., C_MAIN:]
    run_dual(runners["g_s"], runners.get("g_s_1"), y_hat_real, y_hat_imag)

    timer.mark("_end")
    timer.commit()
    return 0


def run_scenario_entropy_only(
    real: np.ndarray,
    imag: np.ndarray,
    runners: dict[str, DPUSubgraphRunner],
    eb: EntropyBottleneck,
    gc: GaussianConditional,
    timer: StepTimer,
) -> int:
    """Entropy coding only — produce latents via DPU then time only the CPU coding."""
    # We need real latents for meaningful entropy coding, so run encoders once (untimed)
    y_real, y_imag = run_dual(runners["g_a"], runners.get("g_a_1"), real, imag)
    y = np.concatenate((y_real, y_imag), axis=-1)
    y_abs = np.abs(y)
    z = runners["h_a"].run(y_abs)

    # --- Timed section ---
    timer.mark("cpu_eb_compress")
    z_strings = eb.compress(z)
    z_bytes = sum(len(s) for s in z_strings)

    timer.mark("cpu_eb_decompress")
    z_hat = eb.decompress(z_strings, (z.shape[1], z.shape[2]))

    # Need scales for GC
    scales = runners["h_s"].run(z_hat)

    timer.mark("cpu_gc_compress")
    means = np.zeros_like(y)
    y_strings = gc.compress(y, scales, means)
    y_bytes = sum(len(s) for s in y_strings)

    timer.mark("cpu_gc_decompress")
    _y_hat = gc.decompress(y_strings, scales, means)

    timer.mark("_end")
    timer.commit()
    return z_bytes + y_bytes


# ---------------------------------------------------------------------------
# Pre-compress helper for decompress scenario
# ---------------------------------------------------------------------------
def _precompress(
    real: np.ndarray,
    imag: np.ndarray,
    runners: dict[str, DPUSubgraphRunner],
    eb: EntropyBottleneck,
    gc: GaussianConditional,
) -> dict[str, Any]:
    """Run a single encode pass and cache everything the decompress scenario needs."""
    y_real, y_imag = run_dual(runners["g_a"], runners.get("g_a_1"), real, imag)
    y = np.concatenate((y_real, y_imag), axis=-1)
    y_abs = np.abs(y)
    z = runners["h_a"].run(y_abs)

    z_strings = eb.compress(z)
    z_hat = eb.decompress(z_strings, (z.shape[1], z.shape[2]))
    scales = runners["h_s"].run(z_hat)
    means = np.zeros_like(y)
    y_strings = gc.compress(y, scales, means)

    return {
        "z_strings": z_strings,
        "y_strings": y_strings,
        "z_shape": (z.shape[1], z.shape[2]),
        "scales_shape": scales.shape,
        "z_bytes": sum(len(s) for s in z_strings),
        "y_bytes": sum(len(s) for s in y_strings),
    }


# ---------------------------------------------------------------------------
# xdputil metadata (parsed from xmodel -l output)
# ---------------------------------------------------------------------------
def get_xmodel_metadata(xmodel_path: Path) -> dict[str, Any]:
    """Run 'xdputil xmodel -l' and parse the JSON output for workload/memory info."""
    try:
        result = subprocess.run(
            ["xdputil", "xmodel", str(xmodel_path), "-l"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
    except Exception as e:
        print(f"Warning: could not parse xdputil xmodel -l output: {e}")
        return {}

    subgraphs = data.get("subgraphs", [])
    meta: dict[str, Any] = {"subgraphs": {}}
    for sg in subgraphs:
        if sg.get("device") != "DPU":
            continue
        name = sg.get("name", "")
        role = None
        for r in ("g_a", "h_a", "h_s", "g_s"):
            if r in name:
                role = r
                break
        if role is None:
            continue

        regs = {ri["name"]: ri for ri in sg.get("reg info", [])}
        meta["subgraphs"][role] = {
            "workload_ops": sg.get("workload", 0),
            "const_bytes": regs.get("REG_0", {}).get("size", 0),
            "workspace_bytes": regs.get("REG_1", {}).get("size", 0),
            "input_bytes": regs.get("REG_2", {}).get("size", 0),
            "output_bytes": regs.get("REG_3", {}).get("size", 0),
            "fixpos_in": sg.get("input_tensor", [{}])[0].get("fixpos", None),
            "fixpos_out": sg.get("output_tensor", [{}])[0].get("fixpos", None),
        }

    # Total workload across all DPU subgraphs
    meta["total_workload_ops"] = sum(s["workload_ops"] for s in meta["subgraphs"].values())
    meta["total_const_bytes"] = sum(s["const_bytes"] for s in meta["subgraphs"].values())

    return meta


def get_dpu_query_info() -> dict[str, Any]:
    """Run 'xdputil query' and extract DPU frequency, arch, core count."""
    try:
        result = subprocess.run(
            ["xdputil", "query"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Filter only the JSON part (skip stderr warnings)
        lines = result.stdout.strip().split("\n")
        json_start = next(i for i, l in enumerate(lines) if l.strip().startswith("{"))
        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)
    except Exception as e:
        print(f"Warning: could not parse xdputil query output: {e}")
        return {}

    kernels = data.get("kernels", [])
    dpu_cores = [k for k in kernels if k.get("IP Type") == "DPU"]
    info: dict[str, Any] = {
        "n_dpu_cores": len(dpu_cores),
        "vai_version": data.get("VAI Version", {}),
        "dpu_ip_spec": data.get("DPU IP Spec", {}),
    }
    if dpu_cores:
        info["dpu_arch"] = dpu_cores[0].get("DPU Arch", "")
        info["dpu_freq_mhz"] = dpu_cores[0].get("DPU Frequency (MHz)", 0)
        info["fingerprint"] = dpu_cores[0].get("fingerprint", "")
    return info


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    xmodel_path: Path,
    scenario: str,
    n_warmup: int,
    n_iters: int,
    data_path: Path | None,
    measure_power: bool,
    power_poll_hz: float,
    idle_baseline_s: float = 0.0,
) -> dict[str, Any]:
    """Run the benchmark and return a results dictionary."""
    print(f"Scenario : {scenario}")
    print(f"Warmup   : {n_warmup} iterations")
    print(f"Measured : {n_iters} iterations")
    print(f"Power    : {'ON' if measure_power else 'OFF'}")

    # ---- Load model ----
    entropy_path = xmodel_path.parent / "entropy_params.npz"
    graph = xir.Graph.deserialize(str(xmodel_path))
    sg_map = identify_subgraphs(graph, xmodel_path.parent / "meta.json", verbose=False)

    data = np.load(entropy_path)
    eb = EntropyBottleneck(
        channels=data["eb_cdf_length"].shape[0],
        quantized_cdf=data["eb_quantized_cdf"],
        cdf_length=data["eb_cdf_length"],
        offset=data["eb_offset"],
        medians=data.get("eb_medians"),
    )
    gc = GaussianConditional(
        scale_table=data["gc_scale_table"],
        quantized_cdf=data["gc_quantized_cdf"],
        cdf_length=data["gc_cdf_length"],
        offset=data["gc_offset"],
    )

    runners: dict[str, DPUSubgraphRunner] = {}
    for key, sg in sg_map.items():
        runners[key] = DPUSubgraphRunner(
            vart.Runner.create_runner(sg, "run"),
            sg,
            key,
        )

    # Create auxiliary runners for g_s and g_a so run_dual() can dispatch real and imag channels to separate DPU cores simultaneously.
    # execute_async releases the GIL, so two Python threads overlap at the hardware level when ≥2 physical cores are available.
    #
    # CRITICAL — creation ORDER determines VART's static core assignment (round-robin):
    #   xmodel topo order to cores:  h_s→0, h_a→1, g_s→2, g_a→0  (from xdputil xmodel -l)
    #   aux runners:        g_s_1→1, g_a_1→2
    # This ensures g_s (core 2) ‖ g_s_1 (core 1)  — different cores ✓
    #               g_a (core 0) ‖ g_a_1 (core 2)  — different cores ✓
    # If the order were reversed (g_a_1 then g_s_1), g_s_1 would land on core 2
    # (same as g_s) and get NO speedup — as was observed empirically.
    for key in ("g_s", "g_a"):  # g_s MUST come first
        if key in sg_map:
            runners[f"{key}_1"] = DPUSubgraphRunner(
                vart.Runner.create_runner(sg_map[key], "run"),
                sg_map[key],
                f"{key}_1",
            )
    print(f"Parallel : {'ON (threaded real‖imag)' if _PARALLEL else 'OFF (sequential)'}")

    # ---- Prepare input ----
    if data_path is not None and data_path.exists():
        noisy = load_real_patch(data_path, index=0)
        print(f"Using real data from {data_path}")
    else:
        noisy = make_dummy_patch()
        print("Using random dummy patch (no --data provided)")

    real, imag = preprocess_patch(noisy)

    # ---- Select scenario function ----
    scenario_fn_map = {
        "full": run_scenario_full,
        "compress": run_scenario_compress,
        "decompress": run_scenario_decompress,
        "dpu_only": run_scenario_dpu_only,
        "entropy_only": run_scenario_entropy_only,
    }
    scenario_fn = scenario_fn_map[scenario]

    # For decompress, pre-compress once to get bitstreams
    cached_strings: dict[str, Any] | None = None
    if scenario == "decompress":
        print("Pre-compressing to obtain bitstreams for decompress scenario...")
        cached_strings = _precompress(real, imag, runners, eb, gc)

    # ---- Power setup ----
    power_sampler: INA226PowerSampler | None = None
    pmbus_sampler: PMBusRailSampler | None = None
    if measure_power:
        rails = discover_ina226_sensors()
        if rails:
            print(f"Discovered {len(rails)} power rails: {list(rails.keys())}")
            power_sampler = INA226PowerSampler(rails, poll_interval_s=1.0 / power_poll_hz)
        else:
            print("Warning: No INA226 sensors found — power measurement disabled.")
        if Path(PMBUS_BUS).exists():
            pmbus_sampler = PMBusRailSampler(PMBUS_RAIL_MAP, poll_interval_s=0.04)
            print(f"PMBus rails enabled: {list(PMBUS_RAIL_MAP.keys())}")
        else:
            print(f"Warning: {PMBUS_BUS} not found — PMBus rail measurement disabled.")

    # ---- Idle baseline (power only, no inference) ----
    idle_baseline_results: dict[str, dict[str, float]] | None = None
    idle_pmbus_results: dict[str, dict[str, float]] | None = None
    if (power_sampler is not None or pmbus_sampler is not None) and idle_baseline_s > 0:
        print(f"\nCapturing idle baseline ({idle_baseline_s:.0f}s, no inference)...")
        idle_sampler = INA226PowerSampler(
            {r: p for r, p in discover_ina226_sensors().items()},
            poll_interval_s=1.0 / power_poll_hz,
        )
        idle_pmbus = (
            PMBusRailSampler(PMBUS_RAIL_MAP, poll_interval_s=0.04)
            if pmbus_sampler is not None
            else None
        )
        idle_sampler.start()
        if idle_pmbus is not None:
            idle_pmbus.start()
        time.sleep(idle_baseline_s)
        idle_sampler.stop()
        if idle_pmbus is not None:
            idle_pmbus.stop()
        idle_baseline_results = idle_sampler.results()
        if idle_pmbus is not None:
            idle_pmbus_results = idle_pmbus.results()
        print("Idle baseline captured.")

    # ---- Warmup ----
    print(f"\nWarmup ({n_warmup} iterations)...")
    warmup_timer = StepTimer()
    for _ in range(n_warmup):
        kwargs = {}
        if scenario == "decompress":
            kwargs["cached_strings"] = cached_strings
        scenario_fn(real, imag, runners, eb, gc, warmup_timer, **kwargs)  # type: ignore

    # ---- Measured runs ----
    print(f"Benchmarking ({n_iters} iterations)...")
    timer = StepTimer()
    total_bytes_list: list[int] = []

    if power_sampler is not None:
        power_sampler.start()
    if pmbus_sampler is not None:
        pmbus_sampler.start()

    wall_start = time.perf_counter()
    for _ in range(n_iters):
        kwargs = {}
        if scenario == "decompress":
            kwargs["cached_strings"] = cached_strings
        nbytes = scenario_fn(real, imag, runners, eb, gc, timer, **kwargs)  # type: ignore
        total_bytes_list.append(nbytes)
    wall_end = time.perf_counter()

    if power_sampler is not None:
        power_sampler.stop()
    if pmbus_sampler is not None:
        pmbus_sampler.stop()

    wall_total = wall_end - wall_start

    # ---- Collect results ----
    step_summary = timer.summary()

    # Aggregate DPU-only and CPU-only times
    dpu_steps = [k for k in step_summary if k.startswith("dpu_")]
    cpu_steps = [k for k in step_summary if k.startswith("cpu_")]
    dpu_total_mean = sum(step_summary[k]["mean_s"] for k in dpu_steps)
    cpu_total_mean = sum(step_summary[k]["mean_s"] for k in cpu_steps)

    # Per-iteration total from step sums
    iter_mean = timer.total_mean()

    results: dict[str, Any] = {
        # Metadata
        "platform": "FPGA_ZCU102",
        "scenario": scenario,
        "timestamp": datetime.now().isoformat(),
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        # Latency breakdown (seconds)
        "latency_breakdown": step_summary,
        "latency_total_mean_s": iter_mean,
        "latency_total_mean_ms": iter_mean * 1000,
        "latency_dpu_total_mean_ms": dpu_total_mean * 1000,
        "latency_cpu_total_mean_ms": cpu_total_mean * 1000,
        "latency_wall_total_s": wall_total,
        # Throughput
        "throughput_fps": n_iters / wall_total,
        # Compression (only meaningful for full/compress/decompress)
        "avg_compressed_bytes": (
            statistics.mean(total_bytes_list)
            if total_bytes_list and total_bytes_list[0] > 0
            else None
        ),
    }

    # Power
    if power_sampler is not None:
        per_rail = power_sampler.results()
        power_data: dict[str, Any] = {"per_rail": per_rail}

        # Aggregate by semantic group (INA226 rails only for PL/PS groups)
        groups_agg: dict[str, float] = {}
        for group_name, group_rails in POWER_GROUPS.items():
            if group_name == "peripherals":
                continue  # handled separately via PMBus
            total_w = sum(per_rail[r]["avg_power_w"] for r in group_rails if r in per_rail)
            groups_agg[group_name] = round(total_w, 4)
        power_data["groups_avg_w"] = groups_agg

        # Board total (INA226 rails)
        board_total_w = sum(v["avg_power_w"] for v in per_rail.values())
        power_data["board_total_avg_w"] = round(board_total_w, 4)

        # PMBus rails (DDR4_DIMM_VDDQ, UTIL_3V3, UTIL_5V0)
        if pmbus_sampler is not None:
            pmbus_rail_results = pmbus_sampler.results()
            power_data["pmbus_rails"] = pmbus_rail_results
            pmbus_total_w = sum(v["avg_power_w"] for v in pmbus_rail_results.values())
            power_data["board_total_extended_avg_w"] = round(board_total_w + pmbus_total_w, 4)
            # Add peripherals group to groups_avg_w
            groups_agg["peripherals"] = round(pmbus_total_w, 4)

        # Idle baseline (if captured)
        if idle_baseline_results is not None:
            power_data["idle_baseline"] = idle_baseline_results
            idle_total_w = sum(v["avg_power_w"] for v in idle_baseline_results.values())
            power_data["idle_board_total_avg_w"] = round(idle_total_w, 4)
            if idle_pmbus_results is not None:
                power_data["idle_pmbus_rails"] = idle_pmbus_results
                idle_pmbus_total_w = sum(v["avg_power_w"] for v in idle_pmbus_results.values())
                power_data["idle_board_total_extended_avg_w"] = round(
                    idle_total_w + idle_pmbus_total_w, 4
                )

        results["power"] = power_data

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--xmodel", required=True, help="Path to compiled .xmodel")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=SCENARIOS,
        help="Which pipeline stage(s) to benchmark.",
    )
    parser.add_argument("--data", default=None, help="Path to .npy test set (uses first patch)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (default: 20)")
    parser.add_argument(
        "--iters", type=int, default=100, help="Measured iterations (default: 100)"
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Enable INA226 power sampling during benchmark.",
    )
    parser.add_argument(
        "--power-hz",
        type=float,
        default=50.0,
        help="Power sampling frequency in Hz (default: 50).",
    )
    parser.add_argument(
        "--collect-hw-meta",
        action="store_true",
        help="Also run xdputil query / xmodel -l and include HW metadata in output.",
    )
    parser.add_argument(
        "--idle-baseline",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Capture idle power baseline for N seconds before benchmarking (requires --power).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/benchmark_fpga_<scenario>.json)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help=(
            "Run g_a and g_s sequentially on a single DPU core (single-core baseline). "
            "By default all scenarios process real and imag channels in parallel via "
            "Python threads, exploiting multiple DPU cores. "
            "Output is saved with a '_sequential' suffix when this flag is set."
        ),
    )
    args = parser.parse_args()

    global _PARALLEL
    _PARALLEL = not getattr(args, "no_parallel", False)

    xmodel_path = Path(args.xmodel).resolve()
    if not xmodel_path.exists():
        print(f"Error: {xmodel_path} not found.")
        sys.exit(1)

    data_path = Path(args.data).resolve() if args.data else None

    # Run benchmark
    results = run_benchmark(
        xmodel_path=xmodel_path,
        scenario=args.scenario,
        n_warmup=args.warmup,
        n_iters=args.iters,
        data_path=data_path,
        measure_power=args.power,
        power_poll_hz=args.power_hz,
        idle_baseline_s=args.idle_baseline,
    )

    # Optional HW metadata
    if args.collect_hw_meta:
        print("\nCollecting hardware metadata...")
        results["hw_dpu_info"] = get_dpu_query_info()
        results["hw_xmodel_meta"] = get_xmodel_metadata(xmodel_path)

    # Save
    output_dir = xmodel_path.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_suffix = "_sequential" if not _PARALLEL else ""
    out_path = (
        Path(args.output)
        if args.output
        else output_dir / f"benchmark_fpga_{args.scenario}{seq_suffix}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK SUMMARY — {args.scenario}")
    print(f"{'=' * 60}")
    print(f"  Total latency  : {results['latency_total_mean_ms']:.2f} ms / patch")
    print(f"    DPU time     : {results['latency_dpu_total_mean_ms']:.2f} ms")
    print(f"    CPU time     : {results['latency_cpu_total_mean_ms']:.2f} ms")
    print(f"  Throughput     : {results['throughput_fps']:.2f} patches/s")
    if results.get("avg_compressed_bytes"):
        bpp = results["avg_compressed_bytes"] * 8 / (IMAGE_SIZE * IMAGE_SIZE)
        print(f"  Avg BPP        : {bpp:.4f}")
    if "power" in results:
        pdata = results["power"]
        print(f"  Board total    : {pdata.get('board_total_avg_w', 0):.3f} W  (INA226 rails)")
        for group, watts in pdata.get("groups_avg_w", {}).items():
            print(f"    {group:14s} : {watts:.3f} W")
        if "board_total_extended_avg_w" in pdata:
            pmbus_total = pdata["board_total_extended_avg_w"] - pdata.get("board_total_avg_w", 0)
            print(f"  PMBus rails    : {pmbus_total:.3f} W  (DDR4+UTIL_3V3+UTIL_5V0)")
            print(f"  Extended total : {pdata['board_total_extended_avg_w']:.3f} W")
        if "idle_board_total_avg_w" in pdata:
            print(f"  Idle baseline  : {pdata['idle_board_total_avg_w']:.3f} W  (INA226)")
            if "idle_board_total_extended_avg_w" in pdata:
                print(f"  Idle extended  : {pdata['idle_board_total_extended_avg_w']:.3f} W")
            delta = pdata["board_total_avg_w"] - pdata["idle_board_total_avg_w"]
            print(f"  Dynamic delta  : {delta:.3f} W")

    print("\n  Per-step breakdown:")
    for step, stats in results["latency_breakdown"].items():
        print(f"    {step:25s} : {stats['mean_s']*1000:8.3f} ms  (std={stats['std_s']*1000:.3f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
