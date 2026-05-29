#!/usr/bin/env python3
"""run_benchmarks.py — run all benchmark configs for the currently-active model.

Run ON THE BOARD.

Usage:
    python3 run_benchmarks.py <model_name> [options]

Options:
    --iters N           timed iterations for s0/s1 runs       [default: 50]
    --ceiling-iters N   timed iterations for ceiling runs     [default: 100]
    --subset N          patches to load and cycle             [default: 20]
    --warmup N          warmup iterations                     [default: 5]
    --skip-ceiling      skip nn_only / entropy_only sweeps
    --force             overwrite existing JSON files

Output: /home/root/SAR_DDC/bench_results/<model_name>/*.json
"""

import argparse
import subprocess
import sys
from pathlib import Path

BOARD_ROOT = Path("/home/root/SAR_DDC")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("model_name")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--ceiling-iters", type=int, default=100)
    p.add_argument("--subset", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--skip-ceiling", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    model_name = args.model_name

    binary = BOARD_ROOT / "build_cpp" / "benchmark_hardware"
    data = BOARD_ROOT / "data" / "test_sub500_seed42.npy"
    params = BOARD_ROOT / "active_model" / "entropy_params"
    outdir = BOARD_ROOT / "bench_results" / model_name

    xmodels = list((BOARD_ROOT / "active_model").glob("*.xmodel"))
    if len(xmodels) != 1:
        sys.exit(f"ERROR: expected exactly one .xmodel in active_model/, found: {xmodels}")
    xmodel = xmodels[0]

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 54)
    print(f" model : {model_name}")
    print(f" xmodel: {xmodel.name}")
    print(f" output: {outdir}")
    print("=" * 54)
    print()

    def run(config, scenario, iters, extra_flags=()):
        suffix = "".join(
            f"_dpu{v}" if f == "--dpu-cores" else f"_ent{v}" if f == "--entropy-threads" else ""
            for f, v in zip(extra_flags[::2], extra_flags[1::2])
        )
        outfile = outdir / f"{config}_{scenario}{suffix}.json"
        if outfile.exists() and not args.force:
            print(f"[skip] {outfile.name} already exists (--force to overwrite)")
            return
        print(f"--- {config} / {scenario}{' ' + suffix if suffix else ''} ---")
        subprocess.run(
            [
                str(binary),
                "--xmodel",
                str(xmodel),
                "--params",
                str(params),
                "--data",
                str(data),
                "--config",
                config,
                "--scenario",
                scenario,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(iters),
                "--subset",
                str(args.subset),
                "--power",
                *[str(a) for a in extra_flags],
                "--output",
                str(outfile),
            ],
            check=True,
        )
        print()

    # S0/S1 × compress/full
    run("s0", "compress", args.iters)
    run("s0", "full", args.iters)
    run("s1", "compress", args.iters)
    run("s1", "full", args.iters)

    if not args.skip_ceiling:
        for n in range(1, 4):  # DPU cores 1..3
            run("nn_only", "compress", args.ceiling_iters, ("--dpu-cores", n))
        for n in range(1, 5):  # entropy threads 1..4
            run("entropy_only", "compress", args.ceiling_iters, ("--entropy-threads", n))

    print("=" * 54)
    print(f" Done: {model_name}")
    for f in sorted(outdir.glob("*.json")):
        print(f"  {f.name}")
    print("=" * 54)


if __name__ == "__main__":
    main()
