#!/usr/bin/env python3
"""rescore_hamburg_tiles.py — quantify the SSIM/EPD ``data_range`` + clipping artifact on the
Hamburg tile, for every matched float32 (GPU) / INT8 (FPGA) model pair.

The saved Hamburg reconstructions are already on disk for both backends — GPU under
``<run_dir>/recon_<tile>_linA.npy`` and INT8 under ``<model_dir>/results/<tile>_recon_linA.npy``
— so the whole float-vs-INT8 comparison can be re-scored off-board, at no deploy cost. Every
pair is scored twice:

  * ``old``  — SSIM/MS-SSIM at ``data_range = max(predicted)``, EPD unclipped (the convention
    behind the current manuscript numbers);
  * ``new``  — everything clipped to ``AMP_LIN_99`` with ``data_range = AMP_LIN_99``
    (``src.utils.metrics``).

Writes a tidy per-model CSV plus a per-architecture summary of how much of the reported
float32→INT8 quality drop was convention rather than quantization. Reads only; it never
touches the canonical board-written ``metrics.json``.

    python scripts/evaluation/rescore_hamburg_tiles.py
    python scripts/evaluation/rescore_hamburg_tiles.py --seeds 0 --lambdas 1 20 1000   # quick
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rootutils
import torch

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from _plotkit import load_fpga_quality, load_quality_runs  # noqa: E402
from torchmetrics.functional.image import (  # noqa: E402
    structural_similarity_index_measure as _tm_ssim,
)

from src.utils.constants import AMP_LIN_99  # noqa: E402
from src.utils.metrics import epd, psnr, ssim  # noqa: E402

TILE = "Hamburg_[11000:12024-8500:9524]"
GT_PATH = REPO_ROOT / "data" / "visualization" / TILE / "linA_MERLIN.npy"
OUT_DIR = REPO_ROOT / "results" / "ssim_convention"


def _old_ssim(recon: np.ndarray, gt: np.ndarray) -> float:
    """SSIM as it was computed before the fix: unclipped, data_range = max(predicted)."""
    r = torch.from_numpy(recon)[None, None]
    g = torch.from_numpy(gt)[None, None]
    return float(_tm_ssim(r, g, data_range=float(r.max())))


def _old_epd(recon: np.ndarray, gt: np.ndarray) -> float:
    """EPD as it was computed before the fix: no AMP_LIN_99 clip.

    ``src.utils.metrics.epd`` now clips, so the pre-fix behaviour is reproduced by scaling both
    images below the clip — EPD is a ratio of gradient products, hence scale-invariant.
    """
    scale = np.float32(AMP_LIN_99 / max(float(recon.max()), float(gt.max())))
    return epd(recon * scale, gt * scale)


def _tile_paths(row, backend: str) -> Optional[Path]:
    if backend == "gpu":
        return Path(row.run_dir) / f"recon_{TILE}_linA.npy" if row.run_dir else None
    return Path(row.model_dir) / "results" / f"{TILE}_recon_linA.npy"


def collect(seeds: List[int], lambdas: Optional[List[float]]) -> List[Dict]:
    """Score every (arch, λ, seed) pair that has both a GPU and an INT8 tile on disk."""
    gt = np.load(GT_PATH).astype(np.float32)
    gpu = load_quality_runs(seeds=seeds, verbose=False)
    fpga = load_fpga_quality(seeds=seeds, verbose=False)

    def _keys(df):  # "lambda" is a keyword, so index the columns rather than itertuples
        return set(zip(df["arch"], df["lambda"], df["seed"]))

    rows: List[Dict] = []
    keys = sorted(_keys(gpu) & _keys(fpga))
    for arch, lam, seed in keys:
        if lambdas is not None and lam not in lambdas:
            continue
        g = gpu[(gpu.arch == arch) & (gpu["lambda"] == lam) & (gpu.seed == seed)]
        f = fpga[(fpga.arch == arch) & (fpga["lambda"] == lam) & (fpga.seed == seed)]
        if g.empty or f.empty:
            continue
        p_gpu, p_int8 = _tile_paths(g.iloc[0], "gpu"), _tile_paths(f.iloc[0], "fpga")
        if p_gpu is None or not p_gpu.exists() or not p_int8.exists():
            print(f"  skip {arch} L{int(lam)} s{seed}: tile missing")
            continue

        r_gpu = np.load(p_gpu).astype(np.float32)
        r_int8 = np.load(p_int8).astype(np.float32)
        t_gpu, t_int8, t_gt = (torch.from_numpy(a)[None, None] for a in (r_gpu, r_int8, gt))
        row = {
            "arch": arch,
            "lambda": lam,
            "seed": seed,
            "max_gpu": float(r_gpu.max()),
            "max_int8": float(r_int8.max()),
            "psnr_gpu": psnr(t_gpu, t_gt),
            "psnr_int8": psnr(t_int8, t_gt),
            "ssim_old_gpu": _old_ssim(r_gpu, gt),
            "ssim_old_int8": _old_ssim(r_int8, gt),
            "ssim_new_gpu": ssim(t_gpu, t_gt),
            "ssim_new_int8": ssim(t_int8, t_gt),
            "epd_old_gpu": _old_epd(r_gpu, gt),
            "epd_old_int8": _old_epd(r_int8, gt),
            "epd_new_gpu": epd(r_gpu, gt),
            "epd_new_int8": epd(r_int8, gt),
        }
        for m in ("ssim_old", "ssim_new", "epd_old", "epd_new"):
            row[f"d_{m}"] = row[f"{m}_gpu"] - row[f"{m}_int8"]
        rows.append(row)
        print(
            f"  {arch:8s} L{int(lam):<5d} s{seed}  ΔSSIM {row['d_ssim_old']:+.4f} → "
            f"{row['d_ssim_new']:+.4f}   ΔEPD {row['d_epd_old']:+.3f} → {row['d_epd_new']:+.3f}",
            flush=True,
        )
    return rows


def summarise(rows: List[Dict]) -> None:
    """Print the float32→INT8 gap per architecture under both conventions."""
    if not rows:
        raise SystemExit("no matched tile pairs found")
    print(
        f"\n{'arch':10s}{'n':>4s}{'ΔSSIM old':>12s}{'ΔSSIM new':>12s}{'ΔEPD old':>11s}{'ΔEPD new':>11s}"
    )
    for arch in sorted({r["arch"] for r in rows}) + ["ALL"]:
        sub = rows if arch == "ALL" else [r for r in rows if r["arch"] == arch]
        m = {
            k: float(np.mean([r[k] for r in sub]))
            for k in ("d_ssim_old", "d_ssim_new", "d_epd_old", "d_epd_new")
        }
        print(
            f"{arch:10s}{len(sub):>4d}{m['d_ssim_old']:>+12.4f}{m['d_ssim_new']:>+12.4f}"
            f"{m['d_epd_old']:>+11.3f}{m['d_epd_new']:>+11.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--lambdas", type=float, nargs="+", default=None, help="default: all")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "hamburg_tile_rescore.csv")
    args = ap.parse_args()

    if not GT_PATH.exists():
        raise SystemExit(f"MERLIN GT not found: {GT_PATH}")
    rows = collect(args.seeds, args.lambdas)
    summarise(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[rescore] {len(rows)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
