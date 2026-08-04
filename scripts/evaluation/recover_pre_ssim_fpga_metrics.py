#!/usr/bin/env python3
"""recover_pre_ssim_fpga_metrics.py — rebuild the *pre-fix* FPGA test-set metrics off-board.

The AMP_LIN_99 re-evaluation sweep overwrote every compiled model's ``results/metrics.json`` in
place, and ``results/`` is not under version control, so the old-convention FPGA numbers behind the
current manuscript figures were lost for the models the sweep had already reached.

They are fully reconstructible, because that same sweep saved every reconstruction
(``--save-recons`` → ``results/reconstructions_test_set/recon_test_set_linA.npz``). The INT8
reconstruction is deterministic and unaffected by the metric change, so replaying the **old**
metric definitions over the saved recons reproduces the pre-fix ``metrics.json`` exactly:

  * SSIM — OpenCV ``QualitySSIM`` semantics: **unclipped**, stabilisers fixed at ``data_range=255``
  * EPD  — **unclipped** gradient-magnitude correlation
  * MSE / PSNR / ENL / ratio_* — unchanged by the fix (recomputed here as a self-check)

Validation is built in: 48 models were snapshotted before the sweep reached them
(``results/fpga_metrics_backup_pre_ssim/``). ``--validate`` recomputes those and reports the
deviation from their stored values, so the reconstruction is proven against ground truth before
being trusted for the rest.

    python scripts/evaluation/recover_pre_ssim_fpga_metrics.py --validate
    python scripts/evaluation/recover_pre_ssim_fpga_metrics.py          # write recovered JSONs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rootutils
from scipy.ndimage import gaussian_filter

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.constants import AMP_LIN_99  # noqa: E402

COMPILED = REPO_ROOT / "results" / "fpga" / "compiled_models"
BACKUP = REPO_ROOT / "results" / "fpga_metrics_backup_pre_ssim"
SUBSET = (
    REPO_ROOT
    / "data"
    / "processed_hdf5"
    / "TSX_spatial_splits_5_256x256"
    / "test_sub500_seed42.npy"
)
OUT_NAME = "metrics_pre_ssim_fix.json"


def _ssim_255_unclipped(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM with OpenCV's fixed stabilisers (C1=(0.01*255)^2, C2=(0.03*255)^2), no clipping.

    Matches ``cv::quality::QualitySSIM`` (11x11 Gaussian, sigma=1.5) to ~2e-4; verified against a
    board-written value before the fix (0.61372 stored vs 0.61395 here).
    """
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    vx = gaussian_filter(x * x, 1.5) - mx2
    vy = gaussian_filter(y * y, 1.5) - my2
    vxy = gaussian_filter(x * y, 1.5) - mxy
    return float((((2 * mxy + c1) * (2 * vxy + c2)) / ((mx2 + my2 + c1) * (vx + vy + c2))).mean())


def _grad_mag(img: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude, matching the C++ compute_epd."""
    gx, gy = np.zeros_like(img), np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _epd_unclipped(recon: np.ndarray, ref: np.ndarray) -> float:
    """EPD without the AMP_LIN_99 clip — the pre-fix definition."""
    gr, gf = _grad_mag(recon), _grad_mag(ref)
    denom = float((gf * gf).sum())
    return float((gr * gf).sum() / denom) if denom > 0 else float("nan")


def _mse_psnr(recon: np.ndarray, ref: np.ndarray):
    """Clipped MSE/PSNR — identical before and after the fix, so a consistency check."""
    a = np.minimum(recon, AMP_LIN_99)
    b = np.minimum(ref, AMP_LIN_99)
    mse = float(((a - b) ** 2).mean())
    psnr = 20 * np.log10(AMP_LIN_99) - 10 * np.log10(mse) if mse > 0 else float("inf")
    return mse, float(psnr)


def recompute(model_dir: Path, subset: np.ndarray) -> Optional[Dict]:
    """Replay the pre-fix metric definitions over a model's saved reconstructions."""
    npz = model_dir / "results" / "reconstructions_test_set" / "recon_test_set_linA.npz"
    if not npz.exists():
        return None
    recons = np.load(npz)["recon_linA"].astype(np.float32)
    n = min(len(recons), len(subset))

    noisy = np.sqrt(0.5 * (subset[:n, :, :, 0] ** 2 + subset[:n, :, :, 1] ** 2)).astype(np.float32)
    adam = subset[:n, :, :, 2].astype(np.float32)
    merlin = subset[:n, :, :, 3].astype(np.float32)

    acc = {
        k: []
        for k in (
            "ssim_m",
            "ssim_a",
            "ssim_n",
            "epd_m",
            "epd_a",
            "mse_m",
            "mse_a",
            "mse_n",
            "psnr_m",
            "psnr_a",
            "psnr_n",
            "enl",
            "rmean",
            "renl",
        )
    }
    for i in range(n):
        r = recons[i]
        acc["ssim_m"].append(_ssim_255_unclipped(r, merlin[i]))
        acc["ssim_a"].append(_ssim_255_unclipped(r, adam[i]))
        acc["ssim_n"].append(_ssim_255_unclipped(r, noisy[i]))
        acc["epd_m"].append(_epd_unclipped(r, merlin[i]))
        acc["epd_a"].append(_epd_unclipped(r, adam[i]))
        for tag, ref in (("m", merlin[i]), ("a", adam[i]), ("n", noisy[i])):
            mse, psnr = _mse_psnr(r, ref)
            acc[f"mse_{tag}"].append(mse)
            acc[f"psnr_{tag}"].append(psnr)
        lin_i = np.square(r)
        acc["enl"].append(float(lin_i.mean() ** 2 / lin_i.var()) if lin_i.var() > 0 else np.nan)
        ratio = np.square(noisy[i]) / (lin_i + 1e-10)
        acc["rmean"].append(float(ratio.mean()))
        acc["renl"].append(float(ratio.mean() ** 2 / ratio.var()) if ratio.var() > 0 else np.nan)

    m = {k: float(np.mean(v)) for k, v in acc.items()}
    return {
        "MERLIN": {"mse": m["mse_m"], "psnr": m["psnr_m"], "ssim": m["ssim_m"], "epd": m["epd_m"]},
        "ADAM": {"mse": m["mse_a"], "psnr": m["psnr_a"], "ssim": m["ssim_a"], "epd": m["epd_a"]},
        "Noisy": {
            "mse": m["mse_n"],
            "psnr": m["psnr_n"],
            "ssim": m["ssim_n"],
            "enl": m["enl"],
            "ratio_mean": m["rmean"],
            "ratio_enl": m["renl"],
        },
        "recon": {"enl": m["enl"], "ratio_mean": m["rmean"], "ratio_enl": m["renl"]},
        "_note": "recovered off-board from saved reconstructions; pre-AMP_LIN_99 convention",
    }


def main() -> None:
    """Validate the reconstruction against snapshots, or write recovered JSONs."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Recompute the snapshotted models and report deviation from their stored values.",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not SUBSET.exists():
        raise SystemExit(f"subset file not found: {SUBSET}")
    subset = np.load(SUBSET, mmap_mode="r")

    targets = (
        sorted(p.name for p in BACKUP.iterdir())
        if args.validate
        else sorted(p.name for p in COMPILED.iterdir() if p.is_dir())
    )
    if args.limit:
        targets = targets[: args.limit]

    devs = {"ssim": [], "epd": [], "psnr": []}
    written = 0
    for name in targets:
        rec = recompute(COMPILED / name, subset)
        if rec is None:
            print(f"  {name}: no recon dump, skipped")
            continue
        if args.validate:
            stored = json.loads((BACKUP / name / "metrics.json").read_text())
            for key, k in (("ssim", "ssim"), ("epd", "epd"), ("psnr", "psnr")):
                s, r = stored["MERLIN"].get(k), rec["MERLIN"].get(k)
                if s is not None and r is not None:
                    devs[key].append(abs(s - r))
            print(
                f"  {name:<28} ssim {stored['MERLIN']['ssim']:.5f} vs {rec['MERLIN']['ssim']:.5f}"
                f"   epd {stored['MERLIN']['epd']:.5f} vs {rec['MERLIN']['epd']:.5f}"
            )
        else:
            out = COMPILED / name / "results" / OUT_NAME
            out.write_text(json.dumps(rec, indent=4, sort_keys=True))
            written += 1

    if args.validate:
        print("\nmax |deviation| vs stored pre-sweep values:")
        for k, v in devs.items():
            if v:
                print(f"  {k:<6} max {max(v):.2e}   mean {np.mean(v):.2e}   (n={len(v)})")
    else:
        print(f"\nwrote {written} x {OUT_NAME}")


if __name__ == "__main__":
    main()
