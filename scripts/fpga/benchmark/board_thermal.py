"""Board thermal telemetry + cooldown-gate for the ZCU102 streaming sweep.

Reads the AMS system-monitor die temperature (PS + PL) and the A53 clock (a throttle detector) over
SSH, and idle-waits between measured runs until the die drops below a target — so sweep runs don't
drift or thermally throttle with cumulative heat, which otherwise biased the later configs
(docs/onboard_pipeline.md §8 caveat). Reused by stream_benchmark.py and the overlap sweep.

Sensors (probed on the board): `iio:device0` = ams (in_temp0_ps_temp, in_temp2_pl_temp; °C =
(raw+offset)*scale/1000); A53 scaling_cur_freq idles at 1.2 GHz, so any lower reading during a run
means the CPU thermally throttled and that run's throughput/energy are suspect.
"""

import subprocess
import time

BOARD = "ZCU102"

# One round-trip: PS + PL die-temp raw/scale/offset, then the A53 current + max clock (kHz). A53
# throttling shows as cur < max, compared in kHz (exact — avoids MHz-rounding false positives: the
# nominal 1199999 kHz rounds to 1199 MHz, so a "< 1200 MHz" test would flag every healthy run).
_READ_CMD = (
    "d=/sys/bus/iio/devices/iio:device0; "
    "for ch in in_temp0_ps_temp in_temp2_pl_temp; do "
    "echo $ch $(cat $d/${ch}_raw) $(cat $d/${ch}_scale) $(cat $d/${ch}_offset); done; "
    "f=/sys/devices/system/cpu/cpu0/cpufreq; "
    "echo freq $(cat $f/scaling_cur_freq) $(cat $f/scaling_max_freq)"
)


def read_thermal() -> dict:
    """SoC die temp (°C, the hotter of PS/PL) + A53 clock (MHz), via one SSH round-trip."""
    r = subprocess.run(["ssh", BOARD, _READ_CMD], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"thermal read failed ({r.returncode}): {r.stderr.strip()}")
    temps: dict = {}
    cur_khz = max_khz = None
    for line in r.stdout.split("\n"):
        p = line.split()
        if not p:
            continue
        if p[0] == "freq":
            cur_khz, max_khz = int(p[1]), int(p[2])
        else:  # ch raw scale offset
            temps[p[0]] = (float(p[1]) + float(p[3])) * float(p[2]) / 1000.0
    if not temps or cur_khz is None:
        raise RuntimeError(f"could not parse thermal read:\n{r.stdout}")
    return {
        "die_c": round(max(temps.values()), 1),
        "ps_c": round(temps["in_temp0_ps_temp"], 1),
        "pl_c": round(temps["in_temp2_pl_temp"], 1),
        "a53_mhz": cur_khz // 1000,
        "a53_throttled": cur_khz < max_khz,
    }


def cooldown(target_c: float, poll_s: float = 5.0, max_wait_s: float = 300.0) -> dict:
    """Idle-wait until the die temp is at/below ``target_c`` (or ``max_wait_s`` elapses).

    Returns the wait + before/after temps so the sweep can log how long cooldowns actually take
    (and flag a capped wait, meaning the board never cooled — investigate cooling).
    """
    t0 = time.monotonic()
    start = read_thermal()
    th = start
    while th["die_c"] > target_c and (time.monotonic() - t0) < max_wait_s:
        time.sleep(poll_s)
        th = read_thermal()
    waited = time.monotonic() - t0
    return {
        "wait_s": round(waited, 1),
        "target_c": target_c,
        "start_die_c": start["die_c"],
        "end_die_c": th["die_c"],
        "capped": waited >= max_wait_s,
    }
