#!/usr/bin/env python3
"""benchmark_sweep.py — deploy and benchmark all 4 architectures in sequence.

Runs on the HOST.  For each model:
  1. Deploys the compiled model to the board  (transfer only, no re-compile)
  2. Rebuilds the C++ binary if --rebuild-cpp is set
  3. Pushes run_benchmarks.py and collect_roofline.py to the board
  4. Runs the benchmark sweep via SSH
  5. Collects xdputil roofline peaks via SSH
  6. Fetches all results to results/benchmark_hardware/<model>/

Usage:
    conda activate DDC_FPGA
    python scripts/fpga/benchmark/benchmark_sweep.py [options]

Options:
    --models M1,M2,...  comma-separated model names  [default: all 4 below]
    --iters N           s0/s1 timed iterations       [default: 50]
    --ceiling-iters N   ceiling timed iterations     [default: 100]
    --subset N          patch subset size            [default: 20]
    --skip-ceiling      skip nn_only/entropy_only sweeps
    --skip-roofline     skip xdputil peak collection
    --rebuild-cpp       push C++ sources and rebuild on board before first model
    --force             overwrite existing JSON files on board

Prerequisites:
    - SSH alias ZCU102 is configured (~/.ssh/config)
    - All models listed in --models are compiled under results/fpga/compiled_models/
    - Run from the repo root with conda activate DDC_FPGA
"""

import argparse
import subprocess
import sys
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
SCRIPTS_DIR = (
    Path(__file__).resolve().parent
)  # scripts/fpga/benchmark/ (run_benchmarks.py, collect_roofline.py)
DEPLOY_DIR = SCRIPTS_DIR.parent / "deploy"  # scripts/fpga/deploy/ (deploy.py)
BOARD = "ZCU102"
BOARD_ROOT = "/home/root/SAR_DDC"

DEFAULT_MODELS = [
    "ResSHyp-relu_s0_L1000_pt",
    "SHyp-relu_s0_L1000_pt",
    "ResFP-relu_s0_L1000_pt",
    "FP-relu_s0_L1000_pt",
]


def run(cmd, **kwargs):
    """Run a subprocess command, streaming output, raising on failure."""
    print(f"[host] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def ssh(remote_cmd: str):
    """Run a command on the board via SSH."""
    run(["ssh", BOARD, remote_cmd])


def scp_to_board(local: Path, remote: str):
    """Copy a local file to the board via SCP."""
    run(["scp", str(local), f"{BOARD}:{remote}"])


def scp_from_board(remote: str, local: Path):
    """Copy a file from the board to the local machine via SCP."""
    run(["scp", remote, str(local)])


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--ceiling-iters", type=int, default=100)
    p.add_argument("--subset", type=int, default=20)
    p.add_argument("--skip-ceiling", action="store_true")
    p.add_argument("--skip-roofline", action="store_true")
    p.add_argument("--rebuild-cpp", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 60)
    print(" benchmark_sweep.py")
    print(f" models        : {models}")
    print(f" iters/ceiling : {args.iters} / {args.ceiling_iters}")
    print(f" subset        : {args.subset}")
    print(f" skip-ceiling  : {args.skip_ceiling}")
    print(f" skip-roofline : {args.skip_roofline}")
    print("=" * 60)
    print()

    # Push board-side scripts once (same scripts work for all models)
    print("[host] Pushing run_benchmarks.py and collect_roofline.py to board...")
    scp_to_board(SCRIPTS_DIR / "run_benchmarks.py", f"{BOARD_ROOT}/run_benchmarks.py")
    scp_to_board(SCRIPTS_DIR / "collect_roofline.py", f"{BOARD_ROOT}/collect_roofline.py")
    print()

    # Optionally rebuild C++ binary
    if args.rebuild_cpp:
        print("[host] Pushing C++ sources and rebuilding benchmark_hardware on board...")
        run(
            [
                "rsync",
                "-av",
                str(REPO_ROOT / "inference_cpp" / "src") + "/",
                f"{BOARD}:{BOARD_ROOT}/inference_cpp/src/",
            ]
        )
        ssh(f"cd {BOARD_ROOT}/build_cpp && make -j4 benchmark_hardware 2>&1 | tail -8")
        print()

    for model in models:
        print("=" * 60)
        print(f" Processing: {model}")
        print("=" * 60)

        # 1. Deploy model to board (transfer only — model must be compiled)
        print(f"\n[host] Deploying {model}...")
        run(
            [
                sys.executable,
                str(DEPLOY_DIR / "deploy.py"),
                "--model-name",
                model,
                "--skip-compile",
                "--skip-infer",
                "--skip-fetch",
            ]
        )
        print()

        # 2. Run benchmark sweep on board
        bench_cmd = (
            f"python3 {BOARD_ROOT}/run_benchmarks.py {model}"
            f" --iters {args.iters}"
            f" --ceiling-iters {args.ceiling_iters}"
            f" --subset {args.subset}"
            + (" --skip-ceiling" if args.skip_ceiling else "")
            + (" --force" if args.force else "")
        )
        print("\n[host] Running benchmark sweep on board...")
        ssh(bench_cmd)
        print()

        # 3. Collect xdputil roofline peaks (while model is still active)
        if not args.skip_roofline:
            print("\n[host] Collecting xdputil roofline peaks (~60s per DPU subgraph)...")
            ssh(f"python3 {BOARD_ROOT}/collect_roofline.py {model}")
            print()

        # 4. Fetch benchmark results
        local_dir = REPO_ROOT / "results" / "benchmark_hardware" / model
        roof_dir = REPO_ROOT / "results" / "benchmark_hardware" / "_roofline"
        local_dir.mkdir(parents=True, exist_ok=True)
        roof_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[host] Fetching benchmark results → {local_dir.relative_to(REPO_ROOT)}")
        run(["scp", f"{BOARD}:{BOARD_ROOT}/bench_results/{model}/*.json", str(local_dir)])

        if not args.skip_roofline:
            roof_src = f"{BOARD}:{BOARD_ROOT}/bench_results/{model}_xmodel_info.json"
            roof_dest = roof_dir / f"{model}_xmodel_info.json"
            print(
                f"[host] Fetching xmodel info (peaks + ops/bytes) → {roof_dest.relative_to(REPO_ROOT)}"
            )
            scp_from_board(roof_src, roof_dest)

        print()

    print("=" * 60)
    print(" Sweep complete.  Results:")
    bh = REPO_ROOT / "results" / "benchmark_hardware"
    for f in sorted(bh.rglob("*.json")):
        print(f"  {f.relative_to(REPO_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
