"""Ddc-edge CLI — compress a real tile end-to-end and report timing/throughput/energy, mirroring
`stream_pipeline`'s printed summary (`main_stream.cpp`) so the two are easy to eyeball against each
other.

See `inference_edge/README.md` for scope.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rootutils
import torch

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.constants import AMP_MAX, AMP_MIN, EPS
from src.utils.ddc_format import (
    ARCH_IDS,
    DDCHeader,
    fnv1a_64_bytes,
    write_ddc,
)

from .model_loading import (
    load_model_from_checkpoint,
    resolve_checkpoint_from_model_dir,
)
from .pipeline import PATCH, compress_tile, load_tile
from .power import make_power_sampler
from .timing import StageTimer
from .verify import score_ddc_against_gt


def _device_info(device: torch.device) -> dict[str, Any]:
    """Return a dict of device info for the given torch.device, including CUDA version and device
    name if applicable."""
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or ""
    try:
        # Jetson-specific identity, if present; silently absent elsewhere (not an error — this field is
        # informational provenance, not a required input, so a missing file is not an errors-over-
        # fallbacks case).
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            info["board_model"] = (
                model_path.read_bytes().split(b"\x00")[0].decode(errors="replace")
            )
        rel = Path("/etc/nv_tegra_release")
        if rel.exists():
            info["l4t_release"] = rel.read_text().splitlines()[0].strip()
    except Exception:
        pass
    return info


def cmd_compress(args: argparse.Namespace) -> None:
    """Compress a tile into a .ddc, timed + optionally power-sampled."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[ddc-edge] device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    t0 = time.perf_counter()

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        ckpt_path, manifest = resolve_checkpoint_from_model_dir(model_dir, PROJECT_ROOT)
    else:
        ckpt_path = Path(args.ckpt).resolve()
        manifest = {"model_name": args.arch_override}
        if not manifest["model_name"]:
            raise SystemExit(
                "--ckpt requires --arch-override (no manifest.json to read arch from)"
            )

    model_name = manifest["model_name"]
    arch = model_name.split("-", 1)[0]
    if arch not in ARCH_IDS:
        raise SystemExit(f"unknown arch '{arch}' derived from model_name '{model_name}'")

    print(f"[ddc-edge] loading checkpoint: {ckpt_path}")
    net = load_model_from_checkpoint(ckpt_path).to(device)
    has_hyper = hasattr(net, "h_a")
    print(f"[ddc-edge] model: {model_name} (arch={arch}, hyper={has_hyper})")

    tile_path = Path(args.tile).resolve()
    timer = StageTimer(device)
    t_read0 = time.perf_counter()
    tile = load_tile(tile_path)
    timer.add("read", time.perf_counter() - t_read0)
    H, W, _ = tile.shape
    print(f"[ddc-edge] tile: {tile_path.name} [{H}x{W}x2]")

    sampler = make_power_sampler() if args.power else None
    if args.power and sampler is None:
        print(
            "[ddc-edge] --power requested but neither tegrastats nor nvidia-smi is available — "
            "power will be reported as null, not 0."
        )
    if sampler is not None:
        sampler.start()

    t_compress0 = time.perf_counter()
    records, grid_r, grid_a = compress_tile(
        net, tile, args.overlap, device, timer, max_rows=args.max_rows
    )
    t_compress_wall = time.perf_counter() - t_compress0

    if sampler is not None:
        sampler.stop()
    power_result = sampler.results() if sampler is not None else None

    n_patches = len(records)
    payload_bytes = sum(len(z) + len(y) for z, y in records)

    tile_id = args.tile_id or tile_path.stem
    header = DDCHeader(
        arch_id=ARCH_IDS[arch],
        N=net.nb_channels_main,
        M=net.nb_channels_main * 2,
        patch=PATCH,
        stride=PATCH - args.overlap,
        scene_H=H,
        scene_W=W,
        grid_r=grid_r,
        grid_a=grid_a,
        amp_min=AMP_MIN,
        amp_max=AMP_MAX,
        eps=EPS,
        # NOT the same guard as the FPGA's (that hashes entropy_params/*.npy, INT8-specific CDF
        # tables this FP32 pipeline doesn't have as a separate directory). Hashing the checkpoint
        # instead: identifies "which trained weights produced this .ddc" — comparable across ddc-edge
        # runs of the same checkpoint, but NOT cross-decodable against the board's decoder by design
        # (different precision anyway — see inference_edge/README.md).
        params_sha=fnv1a_64_bytes(ckpt_path.read_bytes()),
        tile_id=tile_id,
        model_id=model_name,
    )
    t_write0 = time.perf_counter()
    out_path = Path(args.out).resolve()
    write_ddc(out_path, header, records)
    timer.add("write", time.perf_counter() - t_write0)

    t_total = time.perf_counter() - t0
    stages_ms = timer.totals_ms()
    nn_ms = timer.total_ms("g_a", "h_a", "h_s")
    entropy_ms = timer.total_ms("eb_compress", "eb_decompress", "gc_compress")
    bpp = payload_bytes * 8.0 / (n_patches * PATCH * PATCH) if n_patches else 0.0
    throughput = n_patches / t_compress_wall if t_compress_wall > 0 else 0.0

    result: dict[str, Any] = {
        "tool": "ddc-edge",
        "mode": "compress",
        "arch": arch,
        "model_name": model_name,
        "tile_id": tile_id,
        "overlap": args.overlap,
        "max_rows": args.max_rows,
        "grid_r": grid_r,
        "grid_a": grid_a,
        "n_patches": n_patches,
        "scene_H": H,
        "scene_W": W,
        "payload_bytes": payload_bytes,
        "file_bytes": out_path.stat().st_size,
        "bpp": bpp,
        "t_total_s": t_total,
        "t_compress_wall_s": t_compress_wall,
        "throughput_patch_s": throughput,
        "stages_ms": stages_ms,
        "stage_counts": timer.counts(),
        "nn_ms_total": nn_ms,
        "entropy_ms_total": entropy_ms,
        "nn_ms_per_patch": nn_ms / n_patches if n_patches else 0.0,
        "entropy_ms_per_patch": entropy_ms / n_patches if n_patches else 0.0,
        "power": power_result,
        "hw_info": _device_info(device),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
    }

    print(
        f"[ddc-edge] compress: {n_patches} patches ({grid_a} x {grid_r}) | bpp={bpp:.4f} | "
        f"{throughput:.2f} patch/s | total={t_total:.1f}s"
    )
    print(
        f"  timing(ms/patch): patchify={stages_ms.get('patchify', 0) / max(n_patches, 1):.3f} "
        f"normalize={stages_ms.get('normalize', 0) / max(n_patches, 1):.3f} "
        f"nn={nn_ms / max(n_patches, 1):.3f} entropy={entropy_ms / max(n_patches, 1):.3f}"
    )
    if power_result:
        print(
            f"  [power:{power_result['source']}] {power_result['avg_power_w']:.2f} W | "
            f"{power_result['energy_j']:.1f} J | "
            f"{power_result['energy_j'] / n_patches * 1000:.3f} mJ/patch"
        )
    else:
        print("  [power] unavailable (null, not 0)")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str))
        print(f"[ddc-edge] results written to {args.json}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Decode a .ddc and score it against a MERLIN GT patch stack."""
    from src.utils.ddc_format import read_ddc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    header, records = read_ddc(Path(args.ddc).resolve())
    print(
        f"[ddc-edge] verify: {args.ddc} — arch_id={header.arch_id} n_patches={header.n_patches} "
        f"grid={header.grid_a}x{header.grid_r}"
    )

    if args.model_dir:
        ckpt_path, manifest = resolve_checkpoint_from_model_dir(
            Path(args.model_dir).resolve(), PROJECT_ROOT
        )
    else:
        ckpt_path = Path(args.ckpt).resolve()
    net = load_model_from_checkpoint(ckpt_path).to(device)

    gt = np.load(args.gt)
    result = score_ddc_against_gt(net, records, gt, device, sample=args.sample)
    print(
        f"[ddc-edge] scored {result['n_scored']}/{result['n_total_records']} patches vs GT: "
        f"PSNR {result['psnr_mean']:.2f} ± {result['psnr_std']:.2f} dB (min {result['psnr_min']:.2f}) | "
        f"SSIM {result['ssim_mean']:.4f} ± {result['ssim_std']:.4f}"
    )
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str))
        print(f"[ddc-edge] verify results written to {args.json}")


def _git_sha() -> str:
    """Return the current git commit SHA of the project root, or an empty string if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ddc-edge."""
    p = argparse.ArgumentParser(prog="ddc-edge", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "compress", help="Compress a tile into a .ddc, timed + optionally power-sampled."
    )
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", help="Compiled-model directory containing manifest.json.")
    src.add_argument("--ckpt", help="Direct path to a training checkpoint (.ckpt).")
    c.add_argument(
        "--arch-override",
        default="",
        help="Arch name (FP/ResFP/SHyp/ResSHyp) — required with --ckpt.",
    )
    c.add_argument(
        "--tile", required=True, help="[H,W,2] float32 tile: .npy or raw TerraSAR-X .cos."
    )
    c.add_argument("--out", required=True, help="Output .ddc path.")
    c.add_argument("--tile-id", default="", help="Header tile_id (default: tile filename stem).")
    c.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Patch overlap in px (default: 2, the streaming default).",
    )
    c.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Cap azimuth patch-rows for a quick test (-1 = full scene).",
    )
    c.add_argument(
        "--power",
        action="store_true",
        help="Sample power during compression (tegrastats > nvidia-smi > null).",
    )
    c.add_argument("--json", default="", help="Write full results to this JSON path.")
    c.set_defaults(func=cmd_compress)

    v = sub.add_parser(
        "verify", help="Decode a .ddc and score it against a MERLIN GT patch stack."
    )
    vsrc = v.add_mutually_exclusive_group(required=True)
    vsrc.add_argument("--model-dir", help="Compiled-model directory containing manifest.json.")
    vsrc.add_argument("--ckpt", help="Direct path to a training checkpoint (.ckpt).")
    v.add_argument(
        "--ddc", required=True, help="Path to the .ddc to verify (must be an overlap=0 run)."
    )
    v.add_argument(
        "--gt",
        required=True,
        help="Path to the matching [n,256,256] MERLIN GT .npy (same grid, same order).",
    )
    v.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Patches to score (-1 = all; default 200, evenly spaced).",
    )
    v.add_argument("--json", default="", help="Write full results to this JSON path.")
    v.set_defaults(func=cmd_verify)

    return p


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
