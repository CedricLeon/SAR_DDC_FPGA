#!/usr/bin/env python3
"""stream_sweep.py — full-scene ablation sweep for the onboard streaming compressor.

For each (arch, lambda): deploy the model, then run the CUMULATIVE optimization configs on the full
scene, COLD, via stream_benchmark.py — seq -> +s1 -> +p0 -> +prefetch -> +neon — plus one WARM run
of the most-optimized config (the persistent-memory ceiling). Per-config results land in
results/benchmark_stream/<model>/<label>.json; build the table afterwards with stream_table.py.

conda activate DDC_FPGA python scripts/fpga/benchmark/stream_sweep.py --archs FP,ResSHyp --lambdas
1000,20

A single full-scene run is the measurement (7,296-patch average is stable) so iters=1, warmup=0.
ResSHyp seq/s1 runs are ~10-12 min each — expect a few hours; run detached.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
DEPLOY = REPO_ROOT / "scripts" / "fpga" / "deploy" / "deploy.py"
BENCH = REPO_ROOT / "scripts" / "fpga" / "benchmark" / "stream_benchmark.py"

# Cumulative optimization ladder — each row = the previous flags + one more optimization.
CONFIGS = [
    ("seq", ["--schedule", "seq"]),
    ("+s1", ["--schedule", "seq", "--s1"]),
    ("+p0", ["--schedule", "p0", "--s1", "--threads", "4"]),
    ("+prefetch", ["--schedule", "p0", "--s1", "--threads", "4", "--prefetch"]),
    ("+neon", ["--schedule", "p0", "--s1", "--threads", "4", "--prefetch", "--neon"]),
]


def run(cmd, **kw) -> int:
    """Run a subprocess, streaming output; return its exit code (does not raise)."""
    print(f"[sweep] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], **kw).returncode


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--archs", default="FP,ResSHyp")
    ap.add_argument("--lambdas", default="1000,20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", default="data/full_scene_i16.npy", help="board-relative full scene")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--power", action="store_true", help="sample board power (INA226) on every run"
    )
    ap.add_argument(
        "--cooldown", action="store_true", help="thermal cooldown-gate + telemetry per run"
    )
    ap.add_argument(
        "--cooldown-c", type=float, default=58.0, help="cool die to <= this °C before each run"
    )
    ap.add_argument("--warm-final", action="store_true", default=True)
    ap.add_argument("--no-warm-final", dest="warm_final", action="store_false")
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    lambdas = [int(x) for x in args.lambdas.split(",")]
    power = ["--power"] if args.power else []
    cool = ["--cooldown", "--cooldown-c", str(args.cooldown_c)] if args.cooldown else []
    combos = [(a, lam) for a in archs for lam in lambdas]
    print(f"[sweep] {len(combos)} models x {len(CONFIGS)} configs; warm={args.warm_final}")

    failures = []
    for arch, lam in combos:
        model = f"{arch}-relu_s{args.seed}_L{lam}_pt"
        print(f"\n{'=' * 68}\n  {model}\n{'=' * 68}", flush=True)
        rc = run(
            [
                sys.executable,
                DEPLOY,
                "--model-name",
                model,
                "--skip-compile",
                "--skip-infer",
                "--skip-fetch",
            ]
        )
        if rc != 0:
            print(f"[sweep] DEPLOY FAILED for {model} — skipping its configs", flush=True)
            failures.append(f"deploy:{model}")
            continue

        for label, flags in CONFIGS:
            print(f"\n[sweep] {model}  {label}  (cold)", flush=True)
            rc = run(
                [
                    sys.executable,
                    BENCH,
                    *flags,
                    *power,
                    *cool,
                    "--tile",
                    args.tile,
                    "--threads",
                    str(args.threads),
                    "--iters",
                    "1",
                    "--warmup",
                    "0",
                ]
            )
            if rc != 0:
                failures.append(f"{model}:{label}")

        if args.warm_final:
            print(f"\n[sweep] {model}  {CONFIGS[-1][0]}  (WARM ceiling)", flush=True)
            rc = run(
                [
                    sys.executable,
                    BENCH,
                    *CONFIGS[-1][1],
                    *power,
                    *cool,
                    "--tile",
                    args.tile,
                    "--threads",
                    str(args.threads),
                    "--iters",
                    "1",
                    "--warmup",
                    "0",
                    "--keep-cache",
                ]
            )
            if rc != 0:
                failures.append(f"{model}:+neon:warm")

    print(f"\n{'=' * 68}\n[sweep] done. failures: {failures or 'none'}")
    print(f"[sweep] build the table: python {BENCH.parent / 'stream_table.py'}")


if __name__ == "__main__":
    main()
