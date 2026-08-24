#!/usr/bin/env python3
"""jetson_power_arch_sweep.py — 4 archs x 4 nvpmodel power modes, full-scene compress-only sweep on
Orin.

Orchestrates over SSH: for each nvpmodel mode (MAXN, MODE_50W, MODE_30W, MODE_15W, high->low power),
switches mode + `jetson_clocks`, settles, then runs `ddc-edge compress` on the full Hamburg scene
(7,540 patches, overlap=2) for each of the 4 deployed architectures (FP, ResFP, SHyp, ResSHyp —
results/fpga/<arch>-relu_s0_L20_pt/ on Orin), fetching each mode's JSONs back before moving to the next
(a crash partway through the sweep does not lose earlier modes).

Compress-only (no quality/verify pass): this sweep is about timing/throughput/power, not PSNR/SSIM —
quality is already validated once per arch's checkpoint elsewhere (docs/onboard_pipeline.md §12).

Needs the Orin sudo password (for `nvpmodel -m` / `jetson_clocks`) in $ORIN_SUDO_PASSWORD — never
hardcode it here, and never pass it as a CLI arg (visible in `ps`).

    ORIN_SUDO_PASSWORD=... python scripts/evaluation/jetson_power_arch_sweep.py
    ORIN_SUDO_PASSWORD=... python scripts/evaluation/jetson_power_arch_sweep.py --modes MAXN,MODE_30W --archs FP

Results land in results/benchmark_jetson/orin/power_sweep/<arch>_<mode>.json. Build the summary table
afterwards with jetson_power_arch_table.py. Restores MAXN at the end regardless of where the sweep
started or stopped, so the board is left in a known, fully-powered default state.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

SSH_HOST = "Orin"
REMOTE_REPO = "~/ddc_edge/repo"
REMOTE_VENV_ACTIVATE = "~/ddc_edge/.venv/bin/activate"  # NOT repo/.venv -- confirmed 2026-08-24
FULL_TILE = (
    "data/cache/symstudy/Hamburg_TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_"
    "IMAGE_HH_SRA_strip_004_full_raw.npy"
)  # relative to REMOTE_REPO; same full scene as the earlier SHyp production_overlap2 run

MODES = [
    ("MAXN", 0),
    ("MODE_50W", 3),
    ("MODE_30W", 2),
    ("MODE_15W", 1),
]  # /etc/nvpmodel.conf on Orin
ARCHS = ["FP", "ResFP", "SHyp", "ResSHyp"]

REMOTE_RESULTS_DIR = f"{REMOTE_REPO}/sweep_results"
LOCAL_RESULTS_DIR = REPO_ROOT / "results" / "benchmark_jetson" / "orin" / "power_sweep"

SETTLE_S = 20  # after mode switch + jetson_clocks, before the first compress run of that mode
PER_RUN_TIMEOUT_S = (
    1800  # 30 min ceiling per (arch, mode) -- generous vs. the ~4-11 min observed range
)


def ssh(cmd: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run `cmd` on the Orin host over ssh, capturing output."""
    return subprocess.run(["ssh", SSH_HOST, cmd], capture_output=True, text=True, timeout=timeout)


def sudo_ssh(cmd: str, password: str, timeout: float | None = 30) -> subprocess.CompletedProcess:
    """Run a sudo command on Orin, feeding the password to `sudo -S` via this process's own stdin
    pipe (never argv, never a remote heredoc) so it never shows up in `ps` and never collides with
    a wrapped command that also wants stdin (nvpmodel/jetson_clocks don't, so this is safe
    here)."""
    return subprocess.run(
        ["ssh", SSH_HOST, f"sudo -S -p '' {cmd}"],
        input=password + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def set_power_mode(mode_name: str, mode_id: int, password: str) -> None:
    """Switch nvpmodel to the given mode, lock clocks with jetson_clocks, verify, and settle."""
    print(f"[sweep] switching to {mode_name} (id={mode_id})...", flush=True)
    r = sudo_ssh(f"nvpmodel -m {mode_id}", password, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"nvpmodel -m {mode_id} failed (exit {r.returncode}): {r.stderr}")
    r = sudo_ssh("jetson_clocks", password, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"jetson_clocks failed (exit {r.returncode}): {r.stderr}")
    time.sleep(SETTLE_S)
    q = ssh("nvpmodel -q", timeout=10)
    print(f"[sweep]   confirmed: {q.stdout.strip()!r}", flush=True)
    if mode_name not in q.stdout:
        raise RuntimeError(f"mode switch did not take: expected {mode_name!r}, got {q.stdout!r}")


def run_compress(arch: str, mode_name: str) -> bool:
    """Run one ddc-edge compress on Orin for (arch, mode_name); return True on success."""
    model_dir = f"results/fpga/{arch}-relu_s0_L20_pt"
    out_ddc = "/tmp/sweep_current.ddc"  # scratch, overwritten each run -- only the JSON is kept
    remote_json = f"{REMOTE_RESULTS_DIR}/{arch}_{mode_name}.json"
    cmd = (
        f"mkdir -p {REMOTE_RESULTS_DIR} && cd {REMOTE_REPO} && "
        f"source {REMOTE_VENV_ACTIVATE} && "
        f"ddc-edge compress --model-dir {model_dir} --tile {FULL_TILE} --out {out_ddc} "
        f"--overlap 2 --power --json {remote_json}"
    )
    print(f"[sweep]   {arch} @ {mode_name} ...", flush=True)
    t0 = time.time()
    try:
        r = ssh(cmd, timeout=PER_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(
            f"[sweep]   {arch} @ {mode_name}: TIMED OUT after {PER_RUN_TIMEOUT_S}s -- skipping",
            flush=True,
        )
        return False
    dt = time.time() - t0
    if r.returncode != 0:
        print(
            f"[sweep]   {arch} @ {mode_name}: FAILED (exit {r.returncode}, {dt:.0f}s)\n"
            f"{r.stderr[-2000:]}",
            flush=True,
        )
        return False
    print(f"[sweep]   {arch} @ {mode_name}: OK ({dt:.0f}s)", flush=True)
    return True


def fetch_results() -> None:
    """Rsync all JSONs collected so far back to the local repo (safe to call
    repeatedly/incrementally)."""
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-av", f"{SSH_HOST}:{REMOTE_RESULTS_DIR}/", f"{LOCAL_RESULTS_DIR}/"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default=",".join(m for m, _ in MODES))
    parser.add_argument("--archs", default=",".join(ARCHS))
    args = parser.parse_args()

    password = os.environ.get("ORIN_SUDO_PASSWORD")
    if not password:
        raise SystemExit("ORIN_SUDO_PASSWORD not set -- required to switch nvpmodel modes on Orin")

    want_mode_names = set(args.modes.split(","))
    want_archs = args.archs.split(",")
    modes = [(n, i) for n, i in MODES if n in want_mode_names]
    if not modes:
        raise SystemExit(
            f"no matching modes in --modes={args.modes!r} (known: {[m for m,_ in MODES]})"
        )
    unknown_archs = set(want_archs) - set(ARCHS)
    if unknown_archs:
        raise SystemExit(f"unknown archs {unknown_archs} (known: {ARCHS})")

    print(f"[sweep] modes={[m for m, _ in modes]} archs={want_archs}", flush=True)
    print(f"[sweep] tile={FULL_TILE} (full scene, overlap=2)", flush=True)

    results: dict[tuple[str, str], bool] = {}
    try:
        for mode_name, mode_id in modes:
            set_power_mode(mode_name, mode_id, password)
            for arch in want_archs:
                results[(arch, mode_name)] = run_compress(arch, mode_name)
            fetch_results()
            print(
                f"[sweep] mode {mode_name} done, results fetched -> {LOCAL_RESULTS_DIR}",
                flush=True,
            )
    finally:
        print("[sweep] restoring MAXN as the default mode...", flush=True)
        set_power_mode("MAXN", 0, password)
        fetch_results()

    n_ok = sum(results.values())
    print(f"[sweep] done: {n_ok}/{len(results)} runs succeeded", flush=True)
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"[sweep] FAILED: {failed}", flush=True)


if __name__ == "__main__":
    main()
