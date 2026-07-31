#!/usr/bin/env python3
"""stream_benchmark.py — cold/warm-controlled runner for the onboard streaming compressor.

Runs on the HOST; drives ``stream_pipeline`` on the ZCU102 over SSH and reports steady-state
throughput (patch/s, SLC MB/s) and full-tile latency for a chosen schedule.

COLD is the default and the honest number: the page cache is dropped
(``sync; echo 3 > /proc/sys/vm/drop_caches``; the board runs as root) *before each timed run*, so
the SD read is real and reproducible — matching the acquire -> focus -> store -> read flow (the
focused SLC genuinely sits on persistent storage, not our RAM). ``--keep-cache`` runs WARM (skips
the drop) to simulate a much faster persistent store (read served from RAM), i.e. the compute
ceiling. Dropping caches by hand is error-prone (we forget); doing it in the runner is the point.

Usage:
    conda activate DDC_FPGA
    python scripts/fpga/benchmark/stream_benchmark.py --schedule p0 --s1 --threads 4 --prefetch
    python scripts/fpga/benchmark/stream_benchmark.py --schedule seq --keep-cache --max-rows 8
    python scripts/fpga/benchmark/stream_benchmark.py --schedule p0 --s1 --threads 4 --dry-run

Prerequisites: SSH alias ZCU102 configured; the model already deployed to
``ZCU102:/home/root/SAR_DDC/active_model/``; stream_pipeline built (or pass --rebuild-cpp).
"""

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
BOARD = "ZCU102"
BOARD_ROOT = "/home/root/SAR_DDC"
# int16 complex SLC patch: 256x256 px x (I + Q) x 2 B = 256 KiB. The objective doc's ingest unit.
PATCH_BYTES = 256 * 256 * 4

_SUMMARY = re.compile(
    r"(\d+) patches \((\d+) x (\d+)\) \| bpp=([\d.]+) \| ([\d.]+) patch/s \| total=([\d.]+) s"
)
_READ = re.compile(r"read=([\d.]+)")


def ssh_capture(remote_cmd: str) -> str:
    """Run a command on the board via SSH, returning its stdout (raises on non-zero exit)."""
    r = subprocess.run(["ssh", BOARD, remote_cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {remote_cmd}\n{r.stderr.strip()}")
    return r.stdout


def parse_run(stdout: str) -> dict:
    """Extract the metrics from one stream_pipeline invocation's stdout."""
    m = _SUMMARY.search(stdout)
    if not m:
        raise RuntimeError(f"could not parse stream_pipeline output:\n{stdout}")
    n, grid_a, grid_r, bpp, patch_s, total_s = m.groups()
    read = _READ.search(stdout)
    return {
        "n_patches": int(n),
        "grid_a": int(grid_a),
        "grid_r": int(grid_r),
        "bpp": float(bpp),
        "patch_s": float(patch_s),
        "total_s": float(total_s),
        "read_ms": float(read.group(1)) if read else None,
    }


def schedule_flags(args) -> list:
    """Map the CLI knobs to stream_pipeline flags."""
    flags = []
    if args.schedule == "p0":
        flags += ["--p0", "--threads", str(args.threads)]
    if args.s1:
        flags.append("--s1")
    if not args.whole:
        flags.append("--windowed")
    if args.prefetch:
        flags.append("--prefetch")
    if args.max_rows >= 0:
        flags += ["--max-rows", str(args.max_rows)]
    return flags


def remote_cmd(args, cold: bool) -> str:
    """Build the full remote shell command for one run (with the cold-drop prefix if cold)."""
    drop = "sync && echo 3 > /proc/sys/vm/drop_caches && " if cold else ""
    flags = " ".join(schedule_flags(args))
    return (
        f"cd {BOARD_ROOT} && {drop}./build_cpp/stream_pipeline "
        f"--xmodel active_model/*.xmodel --params active_model/entropy_params "
        f"--tile {args.tile} --out /tmp/stream_bench.ddc {flags}"
    )


def label(args, cold: bool) -> str:
    """Compact config label for the result filename."""
    parts = [args.schedule]
    if args.s1:
        parts.append("s1")
    if args.schedule == "p0":
        parts.append(f"t{args.threads}")
    if args.prefetch:
        parts.append("pf")
    parts.append("cold" if cold else "warm")
    if args.max_rows >= 0:
        parts.append(f"r{args.max_rows}")
    return "_".join(parts)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--schedule", choices=["seq", "p0"], default="p0", help="base schedule")
    p.add_argument("--s1", action="store_true", help="channel-parallel g_a (composes with both)")
    p.add_argument("--threads", type=int, default=4, help="worker count for --schedule p0")
    p.add_argument("--prefetch", action="store_true", help="double-buffer row-block reads")
    p.add_argument(
        "--tile", default="data/stream_tile_1k_i16.npy", help="board-relative tile path"
    )
    p.add_argument(
        "--whole", action="store_true", help="load whole tile (default: windowed stream)"
    )
    p.add_argument("--max-rows", type=int, default=-1, help="cap azimuth patch-rows (-1 = full)")
    p.add_argument("--keep-cache", action="store_true", help="WARM: skip the cache drop")
    p.add_argument("--iters", type=int, default=3, help="timed iterations (median reported)")
    p.add_argument("--warmup", type=int, default=1, help="discarded warmup iterations")
    p.add_argument(
        "--rebuild-cpp", action="store_true", help="rsync + rebuild stream_pipeline first"
    )
    p.add_argument("--out", default=None, help="result JSON path (default: auto under results/)")
    p.add_argument("--dry-run", action="store_true", help="print the remote command and exit")
    return p.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    if args.prefetch and args.whole:
        raise SystemExit("--prefetch requires windowed streaming (drop --whole)")
    cold = not args.keep_cache

    if args.dry_run:
        print(f"[{label(args, cold)}] remote command:\n  {remote_cmd(args, cold)}")
        return

    if args.rebuild_cpp:
        print("[host] rsync + rebuild stream_pipeline on board...")
        subprocess.run(
            [
                "rsync",
                "-az",
                str(REPO_ROOT / "inference_cpp" / "src") + "/",
                f"{BOARD}:{BOARD_ROOT}/inference_cpp/src/",
            ],
            check=True,
        )
        print(ssh_capture(f"cd {BOARD_ROOT}/build_cpp && make stream_pipeline -j4 2>&1 | tail -3"))

    manifest = json.loads(ssh_capture(f"cat {BOARD_ROOT}/active_model/manifest.json"))
    model_name = manifest["model_name"]
    cmd = remote_cmd(args, cold)

    print("=" * 68)
    print(f" stream_benchmark : {model_name}  [{label(args, cold)}]")
    print(f" mode             : {'COLD (drop cache per run)' if cold else 'WARM (--keep-cache)'}")
    print(f" tile             : {args.tile}")
    print(f" iters/warmup     : {args.iters} / {args.warmup}")
    print("=" * 68)

    for i in range(args.warmup):
        parse_run(ssh_capture(cmd))  # discard (stabilise DPU/thermal; populate cache when warm)
        print(f"  warmup {i + 1}/{args.warmup} done")

    runs = []
    for i in range(args.iters):
        r = parse_run(ssh_capture(cmd))
        runs.append(r)
        print(f"  iter {i + 1}: {r['patch_s']:.2f} patch/s  total={r['total_s']:.2f} s")

    n_patches = runs[0]["n_patches"]
    med_total = statistics.median(r["total_s"] for r in runs)
    med_patch_s = statistics.median(r["patch_s"] for r in runs)
    # SLC ingest rate: how fast we consume the focused int16 SLC (the objective doc's "data/s").
    slc_mb_s = n_patches * PATCH_BYTES / med_total / 1e6 if med_total > 0 else 0.0

    result = {
        "model_name": model_name,
        "label": label(args, cold),
        "schedule": args.schedule,
        "s1": args.s1,
        "threads": args.threads if args.schedule == "p0" else None,
        "prefetch": args.prefetch,
        "windowed": not args.whole,
        "cold": cold,
        "tile": args.tile,
        "max_rows": args.max_rows,
        "n_patches": n_patches,
        "grid": [runs[0]["grid_a"], runs[0]["grid_r"]],
        "bpp": runs[0]["bpp"],
        "iters": args.iters,
        "totals_s": [r["total_s"] for r in runs],
        "median_total_s": med_total,
        "median_patch_s": med_patch_s,
        "slc_mb_s": slc_mb_s,
        "read_ms": runs[0]["read_ms"],
    }

    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / "benchmark_stream" / model_name / f"{label(args, cold)}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("-" * 68)
    print(
        f" median: {med_patch_s:.2f} patch/s | {slc_mb_s:.1f} MB/s SLC | "
        f"full-tile {med_total:.2f} s | bpp {runs[0]['bpp']:.4f}"
    )
    print(f" -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
