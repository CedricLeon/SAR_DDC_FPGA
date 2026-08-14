#!/usr/bin/env python3
"""stream_fanout_sweep.py — N1 DPU fan-out lane sweep for the onboard streaming compressor.

For each (arch, lambda): deploy the model, then run the fan-out schedule
(``--p0 --fanout --threads L --prefetch --neon``) on the full scene at each lane count L, COLD and
WARM, via stream_benchmark.py. This isolates DPU data-parallel scaling: L independent pipelines, each
on its own DPU core (VART round-robin), the p0 DPU mutex dropped. Lanes 1/2/3 are the real cores; L=4
oversubscribes (round-robin wraps onto a used core) — a deliberate data point, not a speedup.

Per-config JSON lands in results/benchmark_stream/<model>/<label>.json (label carries ``fo`` +
``tL``); build the scaling table afterwards with fanout_table.py. Each run is COOLDOWN-gated (die
<= --cooldown-c) so cumulative heat doesn't bias later lanes (docs/onboard_pipeline.md §8 caveat).

For each lane, COLD runs first (drops the page cache, real SD read) and WARM (``--keep-cache``)
immediately after, reusing the tile the cold run just paged in — so warm is the compute ceiling.

    conda activate DDC_FPGA
    python scripts/fpga/benchmark/stream_fanout_sweep.py --archs ResSHyp,FP --lambdas 1000

The ~7.5k-patch within-run average is stable, but the per-lane placement diagnosis needs run-to-run
robustness, so the default is ``--iters 3 --warmup 1`` (median reported; drop to ``--iters 1 --warmup 0``
for a quick look). ResSHyp lane-1 is ~13 min/iter, so the full ResSHyp matrix at the default is a few
hours — run detached.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
DEPLOY = REPO_ROOT / "scripts" / "fpga" / "deploy" / "deploy.py"
BENCH = REPO_ROOT / "scripts" / "fpga" / "benchmark" / "stream_benchmark.py"

# The fan-out config, fixed across the sweep (only the lane count varies). Matches the §8 "best rung"
# (prefetch + neon) so fan-out is compared like-for-like against p0+s1.
FANOUT_FLAGS = ["--schedule", "p0", "--fanout", "--prefetch", "--neon"]


def run(cmd) -> int:
    """Run a subprocess, streaming output; return its exit code (does not raise)."""
    print(f"[fanout-sweep] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def bench_cmd(args, lane: int, warm: bool) -> list:
    """One stream_benchmark.py invocation for a given lane count and cold/warm mode."""
    cmd = [
        sys.executable,
        BENCH,
        *FANOUT_FLAGS,
        "--threads",
        str(lane),
        "--tile",
        args.tile,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]
    if args.power:
        cmd.append("--power")
    if args.cooldown:
        cmd += ["--cooldown", "--cooldown-c", str(args.cooldown_c)]
    if warm:
        cmd.append("--keep-cache")
    return cmd


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--archs", default="ResSHyp,FP")
    ap.add_argument("--lambdas", default="1000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", default="data/full_scene_i16.npy", help="board-relative full scene")
    ap.add_argument(
        "--lanes", default="1,2,3,4", help="DPU lane counts to sweep (1..3 real, 4 oversubscribes)"
    )
    ap.add_argument(
        "--power", action="store_true", default=True, help="sample board power (J/patch)"
    )
    ap.add_argument("--no-power", dest="power", action="store_false")
    ap.add_argument(
        "--cooldown", action="store_true", default=True, help="thermal cooldown-gate each run"
    )
    ap.add_argument("--no-cooldown", dest="cooldown", action="store_false")
    ap.add_argument(
        "--cooldown-c", type=float, default=58.0, help="cool die to <= this °C before each run"
    )
    ap.add_argument(
        "--iters", type=int, default=3, help="timed iterations per config (median reported)"
    )
    ap.add_argument("--warmup", type=int, default=1, help="discarded warmup iterations per config")
    ap.add_argument("--warm", action="store_true", default=True, help="run a warm pass per lane")
    ap.add_argument("--no-warm", dest="warm", action="store_false")
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    lambdas = [int(x) for x in args.lambdas.split(",")]
    lanes = [int(x) for x in args.lanes.split(",")]
    combos = [(a, lam) for a in archs for lam in lambdas]
    modes = "cold+warm" if args.warm else "cold"
    print(
        f"[fanout-sweep] {len(combos)} models × lanes {lanes} × ({modes}); "
        f"power={args.power} cooldown={args.cooldown}"
    )

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
            print(f"[fanout-sweep] DEPLOY FAILED for {model} — skipping", flush=True)
            failures.append(f"deploy:{model}")
            continue

        for lane in lanes:
            # COLD first so the tile is paged in for the WARM run that follows at the same lane.
            print(f"\n[fanout-sweep] {model}  lane={lane}  (cold)", flush=True)
            if run(bench_cmd(args, lane, warm=False)) != 0:
                failures.append(f"{model}:t{lane}:cold")
            if args.warm:
                print(f"\n[fanout-sweep] {model}  lane={lane}  (warm)", flush=True)
                if run(bench_cmd(args, lane, warm=True)) != 0:
                    failures.append(f"{model}:t{lane}:warm")

    print(f"\n{'=' * 68}\n[fanout-sweep] done. failures: {failures or 'none'}")
    print(f"[fanout-sweep] build the table: python {BENCH.parent / 'fanout_table.py'}")


if __name__ == "__main__":
    main()
