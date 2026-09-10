#!/usr/bin/env python3
"""jetson_power_arch_sweep.py — Jetson Orin sweep over architectures x power modes x batch-size x
precision x real/imag-fusion, full-scene compress-only.

Orchestrates over SSH: for each nvpmodel mode (MAXN, MODE_50W, MODE_30W, MODE_15W, high->low power),
switches mode + `jetson_clocks`, settles, then for each architecture runs the requested
`batch-size x precision x fuse` grid of `ddc-edge compress` calls on the full Hamburg scene
(7,540 patches, overlap=2 by default), fetching each mode's JSONs back before moving on (a crash
partway through does not lose earlier modes).

Two typical invocations:

    # Stage A — full batch/precision/fuse grid at MAXN only:
    ORIN_SUDO_PASSWORD=... python scripts/evaluation/jetson_power_arch_sweep.py \
        --modes MAXN --batch-sizes 1,4,8,16,32 --precisions fp32,fp16,bf16 --fuse both \
        --results-subdir batch_precision_sweep

    # Stage B — one chosen config across the other power modes:
    ORIN_SUDO_PASSWORD=... python scripts/evaluation/jetson_power_arch_sweep.py \
        --modes MODE_50W,MODE_30W,MODE_15W --batch-sizes 16 --precisions bf16 --fuse off \
        --results-subdir batch_precision_sweep

With the defaults (`--batch-sizes 1 --precisions fp32 --fuse off`) each (arch, mode) reduces to a
single b1/fp32 run.

**Some mode switches reboot the board.** MODE_30W/MODE_15W change the online-CPU-core count vs.
MAXN/MODE_50W (`/etc/nvpmodel.conf`), and this nvpmodel build (1.1.4) requires a reboot to apply that.
`set_power_mode` always switches with `--force` (auto-reboots, no interactive prompt) and polls for ssh
to come back (~49s observed). Expect ~1-2 extra minutes around each MODE_30W/MODE_15W entry or exit.

**Power is always measured** (`--power`); results carry throughput, avg power (total + compute-only)
and energy/patch. Compress-only — no quality/verify pass (quality is validated separately, is
batch/mode-independent, and a batch>1 hyperprior `.ddc` is not decodable anyway).

**Fail-fast.** Any run that exits non-zero, or whose bpp deviates >1% from that architecture's first
run (bpp is ~batch/precision-invariant), aborts the whole sweep with a clear message so the operator
can investigate rather than collect silently-wrong numbers. Results fetched so far are kept.

`--no-clocks` skips `jetson_clocks` (and, when the board is already in the only requested mode, the
whole sudo path) — a fallback for when the sudo password is unavailable; the numbers are then at
whatever the mode clocks to under load, so prefer running with sudo for the manuscript.

Needs the Orin sudo password in $ORIN_SUDO_PASSWORD (unless every requested mode is already active and
--no-clocks is set) — never hardcode it, never pass it as a CLI arg (visible in `ps`).

Results land in results/benchmark_jetson/orin/<results-subdir>/<arch>_<mode>_b<batch>_<prec>_<fuse>.json.
Build the summary tables afterwards with jetson_batch_precision_table.py. Restores MAXN at the end.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

SSH_HOST = "Orin"
REMOTE_REPO = "~/ddc_edge/repo"
REMOTE_VENV_ACTIVATE = "~/ddc_edge/.venv/bin/activate"  # NOT repo/.venv -- confirmed 2026-08-24
FULL_TILE = (
    "data/cache/symstudy/Hamburg_TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_"
    "IMAGE_HH_SRA_strip_004_full_raw.npy"
)  # relative to REMOTE_REPO; the full scene, snap grid, overlap=2 => 7,540 patches

MODES = [
    ("MAXN", 0),
    ("MODE_50W", 3),
    ("MODE_30W", 2),
    ("MODE_15W", 1),
]  # /etc/nvpmodel.conf on Orin
ARCHS = ["FP", "ResFP", "SHyp", "ResSHyp"]
PRECISIONS = ["fp32", "fp16", "bf16"]

SETTLE_S = 20  # after mode switch + jetson_clocks, before the first compress run of that mode
PER_RUN_TIMEOUT_S = 1800  # 30 min ceiling per run -- generous vs. the ~1-10 min observed range
SSH_RECONNECT_TIMEOUT_S = 300  # ceiling to wait for Orin to come back after a mode-switch reboot
SSH_RECONNECT_POLL_S = 5  # observed reboot-to-ssh-ready: ~49s (2026-08-24 live test)
BPP_TOL = (
    0.01  # a run's bpp must be within this fraction of the arch's first run (batch-invariant)
)

_SUMMARY_BPP = re.compile(r"bpp=([\d.]+)")

# results dirs are set from --results-subdir in main()
REMOTE_RESULTS_DIR = ""
LOCAL_RESULTS_DIR: Path = (
    REPO_ROOT / "results" / "benchmark_jetson" / "orin" / "batch_precision_sweep"
)


def ssh(cmd: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run `cmd` on the Orin host over ssh, capturing output."""
    return subprocess.run(["ssh", SSH_HOST, cmd], capture_output=True, text=True, timeout=timeout)


def sudo_ssh(cmd: str, password: str, timeout: float | None = 30) -> subprocess.CompletedProcess:
    """Run a sudo command on Orin, feeding the password to `sudo -S` via this process's own stdin
    pipe (never argv, never a remote heredoc) so it never shows up in `ps`."""
    return subprocess.run(
        ["ssh", SSH_HOST, f"sudo -S -p '' {cmd}"],
        input=password + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh_alive(timeout: float = 3) -> bool:
    """True if Orin answers ssh right now -- used to detect when a reboot has completed."""
    try:
        r = subprocess.run(
            [
                "ssh",
                "-o",
                f"ConnectTimeout={int(timeout)}",
                "-o",
                "BatchMode=yes",
                SSH_HOST,
                "echo alive",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def wait_for_reboot() -> None:
    """Poll until Orin's ssh is back up.

    A no-op if no reboot was actually triggered.
    """
    print("[sweep]   waiting for Orin (reboot, if one was triggered)...", flush=True)
    t0 = time.time()
    while time.time() - t0 < SSH_RECONNECT_TIMEOUT_S:
        if ssh_alive():
            print(f"[sweep]   ssh back after {time.time() - t0:.0f}s", flush=True)
            return
        time.sleep(SSH_RECONNECT_POLL_S)
    raise RuntimeError(f"Orin ssh did not come back within {SSH_RECONNECT_TIMEOUT_S}s")


def set_power_mode(mode_name: str, mode_id: int, password: str | None, no_clocks: bool) -> None:
    """Switch nvpmodel to the given mode, lock clocks with jetson_clocks (unless --no-clocks),
    verify.

    With `no_clocks` and the board already in `mode_name`, this needs no sudo at all (pure verify).
    A mode *switch* always needs the password regardless of `no_clocks`.
    """
    print(f"[sweep] selecting {mode_name} (id={mode_id})...", flush=True)
    q0 = ssh("nvpmodel -q", timeout=10)
    already = mode_name in q0.stdout
    if not already:
        if not password:
            raise SystemExit(
                f"switching to {mode_name} needs sudo, but ORIN_SUDO_PASSWORD is not set "
                f"(board is at: {q0.stdout.strip()!r})"
            )
        try:
            r = sudo_ssh(f"nvpmodel -m {mode_id} --force", password, timeout=20)
            print(
                f"[sweep]   nvpmodel --force: exit={r.returncode} stdout={r.stdout.strip()!r}",
                flush=True,
            )
        except subprocess.TimeoutExpired:
            print(
                "[sweep]   nvpmodel --force: local ssh timeout (likely mid-reboot disconnect)",
                flush=True,
            )
        wait_for_reboot()
    else:
        print(f"[sweep]   already at {mode_name}", flush=True)

    if not no_clocks:
        if not password:
            raise SystemExit(
                "jetson_clocks needs sudo; set ORIN_SUDO_PASSWORD or pass --no-clocks"
            )
        r = sudo_ssh("jetson_clocks", password, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"jetson_clocks failed (exit {r.returncode}): {r.stderr}")
    else:
        print("[sweep]   --no-clocks: skipping jetson_clocks (clocks unpinned)", flush=True)

    time.sleep(SETTLE_S)
    q = ssh("nvpmodel -q", timeout=10)
    if mode_name not in q.stdout:
        raise RuntimeError(f"mode did not take: expected {mode_name!r}, got {q.stdout!r}")
    print(f"[sweep]   confirmed: {q.stdout.strip()!r}", flush=True)


def run_label(arch: str, mode: str, batch: int, prec: str, fuse: bool) -> str:
    """Filename stem that uniquely identifies a run: arch, mode, batch, precision, fuse."""
    return f"{arch}_{mode}_b{batch}_{prec}_{'fuse' if fuse else 'nofuse'}"


def run_compress(arch: str, mode: str, batch: int, prec: str, fuse: bool, args) -> float:
    """Run one ddc-edge compress on Orin; return its bpp.

    Raises on failure.
    """
    model_dir = f"results/fpga/{arch}-relu_s0_L20_pt"
    out_ddc = "/tmp/sweep_current.ddc"  # scratch, overwritten each run -- only the JSON is kept
    label = run_label(arch, mode, batch, prec, fuse)
    remote_json = f"{REMOTE_RESULTS_DIR}/{label}.json"
    fuse_flag = " --fuse-reim" if fuse else ""
    cmd = (
        f"mkdir -p {REMOTE_RESULTS_DIR} && cd {REMOTE_REPO} && source {REMOTE_VENV_ACTIVATE} && "
        f"ddc-edge compress --model-dir {model_dir} --tile {args.tile} --out {out_ddc} "
        f"--overlap {args.overlap} --batch-size {batch} --precision {prec}{fuse_flag} "
        f"--warmup-rows {args.warmup_rows} --power --json {remote_json}"
    )
    print(f"[sweep]   run {label} ...", flush=True)
    t0 = time.time()
    try:
        r = ssh(cmd, timeout=PER_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"[sweep] ABORT: {label} TIMED OUT after {PER_RUN_TIMEOUT_S}s")
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(
            f"[sweep] ABORT: {label} FAILED (exit {r.returncode}, {dt:.0f}s)\n{r.stderr[-2000:]}"
        )
    m = _SUMMARY_BPP.search(r.stdout)
    if not m:
        raise SystemExit(
            f"[sweep] ABORT: {label} printed no bpp (stale binary?)\n{r.stdout[-1500:]}"
        )
    bpp = float(m.group(1))
    print(f"[sweep]   run {label}: OK ({dt:.0f}s, bpp={bpp:.4f})", flush=True)
    return bpp


def fetch_results() -> None:
    """Rsync all JSONs collected so far back to the local repo (safe to call repeatedly)."""
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-av", f"{SSH_HOST}:{REMOTE_RESULTS_DIR}/", f"{LOCAL_RESULTS_DIR}/"], check=True
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--modes", default="MAXN", help="comma list from MAXN,MODE_50W,MODE_30W,MODE_15W"
    )
    p.add_argument("--archs", default=",".join(ARCHS))
    p.add_argument("--batch-sizes", default="1", help="comma list, e.g. 1,4,8,16,32")
    p.add_argument("--precisions", default="fp32", help="comma list from fp32,fp16,bf16")
    p.add_argument("--fuse", choices=["off", "on", "both"], default="off", help="real/imag fusion")
    p.add_argument(
        "--results-subdir",
        default="batch_precision_sweep",
        help="under results/benchmark_jetson/orin/",
    )
    p.add_argument(
        "--warmup-rows", type=int, default=8, help="ddc-edge --warmup-rows (JIT absorb)"
    )
    p.add_argument(
        "--overlap", type=int, default=2, help="patch overlap (spec: always 2 / snap grid)"
    )
    p.add_argument("--tile", default=FULL_TILE, help="board-relative tile path")
    p.add_argument(
        "--no-clocks", action="store_true", help="skip jetson_clocks (no-sudo fallback)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print the config plan and one command, exit"
    )
    return p.parse_args()


def main() -> None:
    global REMOTE_RESULTS_DIR, LOCAL_RESULTS_DIR
    args = parse_args()

    want_mode_names = args.modes.split(",")
    modes = [(n, i) for n, i in MODES if n in want_mode_names]
    if not modes:
        raise SystemExit(
            f"no matching modes in --modes={args.modes!r} (known: {[m for m, _ in MODES]})"
        )
    archs = args.archs.split(",")
    if set(archs) - set(ARCHS):
        raise SystemExit(f"unknown archs {set(archs) - set(ARCHS)} (known: {ARCHS})")
    batches = [int(b) for b in args.batch_sizes.split(",")]
    precs = args.precisions.split(",")
    if set(precs) - set(PRECISIONS):
        raise SystemExit(
            f"unknown precisions {set(precs) - set(PRECISIONS)} (known: {PRECISIONS})"
        )
    fuses = {"off": [False], "on": [True], "both": [False, True]}[args.fuse]

    REMOTE_RESULTS_DIR = f"{REMOTE_REPO}/sweep_results/{args.results_subdir}"
    LOCAL_RESULTS_DIR = REPO_ROOT / "results" / "benchmark_jetson" / "orin" / args.results_subdir

    # config product per (mode, arch): batch x precision x fuse, in a stable order
    cfgs = [(b, p, f) for b in batches for p in precs for f in fuses]
    n_total = len(modes) * len(archs) * len(cfgs)
    print(f"[sweep] modes={[m for m, _ in modes]} archs={archs}", flush=True)
    print(
        f"[sweep] {len(cfgs)} configs/arch (batch={batches} prec={precs} fuse={fuses}); "
        f"{n_total} runs total; overlap={args.overlap}; results -> {LOCAL_RESULTS_DIR}",
        flush=True,
    )

    if args.dry_run:
        for b, p, f in cfgs:
            print(f"    {run_label(archs[0], modes[0][0], b, p, f)}")
        return

    password = os.environ.get("ORIN_SUDO_PASSWORD")
    ref_bpp: dict[str, float] = {}  # per-arch first-run bpp, for the invariance sanity check
    n_ok = 0
    try:
        for mode_name, mode_id in modes:
            set_power_mode(mode_name, mode_id, password, args.no_clocks)
            for arch in archs:
                for b, p, f in cfgs:
                    bpp = run_compress(arch, mode_name, b, p, f, args)
                    ref = ref_bpp.setdefault(arch, bpp)
                    if ref > 0 and abs(bpp - ref) / ref > BPP_TOL:
                        fetch_results()
                        raise SystemExit(
                            f"[sweep] ABORT: {arch} bpp={bpp:.4f} deviates >{BPP_TOL:.0%} from this "
                            f"arch's reference {ref:.4f} ({run_label(arch, mode_name, b, p, f)}) — "
                            f"bpp should be batch/precision-invariant. Stopping to investigate."
                        )
                    n_ok += 1
                fetch_results()  # incremental: safe against a later crash
            print(f"[sweep] mode {mode_name} done ({n_ok}/{n_total} ok so far)", flush=True)
    finally:
        if password and not args.no_clocks:
            print("[sweep] restoring MAXN...", flush=True)
            try:
                set_power_mode("MAXN", 0, password, args.no_clocks)
            except Exception as e:  # never mask the original error with a restore failure
                print(f"[sweep]   (MAXN restore failed: {e})", flush=True)
        fetch_results()

    print(f"[sweep] DONE: {n_ok}/{n_total} runs OK", flush=True)


if __name__ == "__main__":
    main()
