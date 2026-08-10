#!/usr/bin/env python3
"""overlap_sweep.py — Table B of the overlap study: reconstructed-tile quality vs the latency /
bitrate / energy cost of patch overlap, on the full scene at the common deployment config.

Per arch (deployed once from the host's compiled_models), for each overlap in {0,2,4,8,16} at the
COMMON config (p0 + s1 + threads 4 + prefetch + neon, cold, thermal-gated, power — the same config
for both archs so they're comparable):

  1. compress the full scene -> .ddc              (stream_benchmark.py; measured latency/bitrate/energy)
  2. decode it on-board (real INT8 h_s+g_s+rANS)  -> per-patch linear-amplitude stack
  3. fetch the .ddc + the decoded stack to the host
  4. stitch on host + score vs the full-scene MERLIN GT   (stitch_ddc.py)

One combined JSON per (arch, overlap) under results/benchmark_stream_overlap/<model>/ov<N>.json.
The overlap latency cost is measured on the same pipeline as §8 (not estimated); reconstruction uses
the board's real INT8 decode. See docs/onboard_pipeline.md §10. Long-running — run detached.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
BOARD = "ZCU102"
BOARD_ROOT = "/home/root/SAR_DDC"
# SD-backed scratch: board /tmp is a 2 GB tmpfs, too small for a ~2.2 GB full-scene decode.
BOARD_TMP = f"{BOARD_ROOT}/tmp"
DEPLOY = REPO_ROOT / "scripts" / "fpga" / "deploy" / "deploy.py"
BENCH = REPO_ROOT / "scripts" / "fpga" / "benchmark" / "stream_benchmark.py"
STITCH = REPO_ROOT / "scripts" / "evaluation" / "stitch_ddc.py"
# Common config = everything except warm; identical for both archs so they stay comparable.
COMMON = ["--schedule", "p0", "--s1", "--threads", "4", "--prefetch", "--neon"]


def run(cmd, **kw) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output unless capture_output is set; never raises."""
    print(f"[ovsweep] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def main() -> None:
    """Deploy each model, then compress/decode/stitch/score across the overlap set."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--archs", default="FP,ResSHyp")
    ap.add_argument(
        "--lam", type=int, default=20, help="lambda (one value; throughput is λ-indep)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlaps", default="0,2,4,8,16")
    ap.add_argument("--tile", default="data/full_scene_i16.npy", help="board-relative full scene")
    ap.add_argument(
        "--gt",
        default="data/visualization/MERLIN/linA_MERLIN_full_Hamburg.npy",
        help="host-relative MERLIN full-tile GT (from merlin_full_gt.py --full)",
    )
    ap.add_argument("--cooldown-c", type=float, default=58.0)
    ap.add_argument(
        "--keep-tiles", action="store_true", help="keep stitched tiles (else score only)"
    )
    ap.add_argument(
        "--warm",
        action="store_true",
        help="warm compress (tile served from cache) — exposes the compute-bound cost",
    )
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    overlaps = [int(x) for x in args.overlaps.split(",")]
    warm_flags = ["--keep-cache", "--warmup", "1"] if args.warm else ["--warmup", "0"]
    sfx = "_warm" if args.warm else ""
    gt = REPO_ROOT / args.gt
    if not gt.exists():
        raise SystemExit(f"MERLIN GT not found: {gt} (run merlin_full_gt.py --full first)")
    outdir = REPO_ROOT / "results" / "benchmark_stream_overlap"
    work = outdir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh", BOARD, f"mkdir -p {BOARD_TMP}"], check=False)

    failures = []
    for arch in archs:
        model = f"{arch}-relu_s{args.seed}_L{args.lam}_pt"
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
        ).returncode
        if rc != 0:
            failures.append(f"deploy:{model}")
            continue

        mdir = outdir / model
        mdir.mkdir(parents=True, exist_ok=True)
        for ov in overlaps:
            tag = f"{model}_ov{ov}{sfx}"
            board_ddc = f"{BOARD_TMP}/ovsweep_ov{ov}.ddc"
            board_recon = f"{BOARD_TMP}/ovsweep_ov{ov}_recon.npy"
            cj = mdir / f"ov{ov}{sfx}_compress.json"
            qj = mdir / f"ov{ov}{sfx}_quality.json"

            # 1. compress (measured latency/bitrate/energy, thermal-gated), keeping the .ddc on-board
            rc = run(
                [
                    sys.executable,
                    BENCH,
                    *COMMON,
                    "--overlap",
                    str(ov),
                    "--power",
                    "--cooldown",
                    "--cooldown-c",
                    str(args.cooldown_c),
                    "--tile",
                    args.tile,
                    "--iters",
                    "1",
                    *warm_flags,
                    "--out-ddc",
                    board_ddc,
                    "--out",
                    str(cj),
                ]
            ).returncode
            if rc != 0:
                failures.append(f"{tag}:compress")
                continue

            # 2. decode on-board -> per-patch linear amplitude (real INT8 h_s+g_s+rANS)
            dec = (
                f"cd {BOARD_ROOT} && ./build_cpp/stream_pipeline --xmodel active_model/*.xmodel "
                f"--params active_model/entropy_params --decode {board_ddc} --out {board_recon}"
            )
            r = run(["ssh", BOARD, dec], capture_output=True, text=True)
            if r.returncode != 0:
                failures.append(f"{tag}:decode")
                print(r.stderr, flush=True)
                continue

            # 3. fetch .ddc + decoded stack
            host_ddc = work / f"{tag}.ddc"
            host_recon = work / f"{tag}_recon.npy"
            run(["rsync", "-a", f"{BOARD}:{board_ddc}", str(host_ddc)])
            run(["rsync", "-a", f"{BOARD}:{board_recon}", str(host_recon)])
            run(
                ["ssh", BOARD, f"rm -f {board_ddc} {board_recon}"]
            )  # free board SD (~2.2 GB recon)

            # 4. stitch on host + score vs the full-scene MERLIN GT
            tile_out = work / f"{tag}_tile.npy"
            rc = run(
                [
                    sys.executable,
                    STITCH,
                    "--ddc",
                    str(host_ddc),
                    "--patches",
                    str(host_recon),
                    "--out",
                    str(tile_out),
                    "--gt",
                    str(gt),
                    "--metrics",
                    str(qj),
                ]
            ).returncode
            host_recon.unlink(missing_ok=True)  # ~2 GB per-patch stack — drop once stitched
            if not args.keep_tiles:
                tile_out.unlink(missing_ok=True)
            if rc != 0:
                failures.append(f"{tag}:stitch")
                continue

            # 5. combine compress + quality into the per-(arch, overlap) record
            comp = json.loads(cj.read_text())
            qual = json.loads(qj.read_text())
            combined = {
                "model": model,
                "arch": arch,
                "lambda": args.lam,
                "overlap": ov,
                "n_patches": comp["n_patches"],
                "bpp": comp["bpp"],
                "median_total_s": comp["median_total_s"],
                "median_patch_s": comp["median_patch_s"],
                "slc_mb_s": comp["slc_mb_s"],
                "avg_power_w": comp.get("avg_power_w"),
                "j_per_patch": comp.get("j_per_patch"),
                "thermal": comp.get("thermal"),
                "quality": qual,
            }
            (mdir / f"ov{ov}{sfx}.json").write_text(json.dumps(combined, indent=2))
            print(
                f"[ovsweep] {tag}: {comp['median_total_s']:.1f}s  bpp={comp['bpp']:.4f}  "
                f"PSNR={qual.get('psnr', float('nan')):.2f}  SSIM={qual.get('ssim', float('nan')):.4f}",
                flush=True,
            )

    print(f"\n{'=' * 68}\n[ovsweep] done. failures: {failures or 'none'}")


if __name__ == "__main__":
    main()
