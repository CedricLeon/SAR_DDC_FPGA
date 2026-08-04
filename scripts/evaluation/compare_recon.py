#!/usr/bin/env python3
"""compare_recon.py — compare two decoded reconstructions (e.g. scalar vs NEON denorm) on linear
amplitude: distortion metrics (against each other and an optional MERLIN GT) + a few log-intensity
patch panels. Reassembles [n,P,P] patch stacks (azimuth-outer, range-inner) into the full image.

    python scripts/evaluation/compare_recon.py --a scalar.npy --b neon.npy \
        --labels scalar,neon --gt merlin.npy --noisy sym_Noisy.npy --grid 4,4 --out fig.png
"""

import argparse

import matplotlib
import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.metrics import get_all_distortion_metrics  # noqa: E402


def reassemble(patches: np.ndarray, grid_a: int, grid_r: int) -> np.ndarray:
    """[n,P,P] patch stack (azimuth pa outer, range pr inner) -> [grid_a*P, grid_r*P] image."""
    p = patches.shape[-1]
    img = np.zeros((grid_a * p, grid_r * p), np.float32)
    for k in range(grid_a * grid_r):
        pa, pr = divmod(k, grid_r)
        img[pa * p : (pa + 1) * p, pr * p : (pr + 1) * p] = patches[k]
    return img


def to_db(lin_a: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Linear amplitude -> log-intensity in dB (20*log10|A|), the project's visualisation scale."""
    return 20.0 * np.log10(np.maximum(lin_a, eps))


def metric_row(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Distortion metrics of pred vs ref on linear amplitude (project-standard PSNR/SSIM)."""
    return get_all_distortion_metrics(pred.astype(np.float32), ref.astype(np.float32))


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="recon A .npy [n,P,P] linA")
    ap.add_argument("--b", required=True, help="recon B .npy [n,P,P] linA")
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--gt", help="MERLIN GT .npy [H,W] linA (optional)")
    ap.add_argument("--noisy", help="noisy tile .npy [H,W,2] complex (optional, for the panels)")
    ap.add_argument("--grid", default="4,4", help="grid_a,grid_r")
    ap.add_argument("--patches", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    la, lb = args.labels.split(",")
    ga, gr = (int(x) for x in args.grid.split(","))
    a = reassemble(np.load(args.a), ga, gr)
    b = reassemble(np.load(args.b), ga, gr)
    gt = np.load(args.gt) if args.gt else None
    noisy = None
    if args.noisy:
        z = np.load(args.noisy).astype(np.float32)  # [H,W,2]
        noisy = np.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2)

    print(f"== distortion metrics (linA) — {lb} vs {la}, and each vs GT ==")
    hdr = f"{'pair':<22}{'PSNR (dB)':>12}{'SSIM':>10}{'MS-SSIM':>10}{'MSE':>12}"
    print(hdr)
    rows = [(f"{lb} vs {la}", b, a)]
    if gt is not None:
        rows += [(f"{la} vs MERLIN", a, gt), (f"{lb} vs MERLIN", b, gt)]
    for name, p, t in rows:
        m = metric_row(p, t)
        print(
            f"{name:<22}{m['psnr']:>12.3f}{m['ssim']:>10.5f}"
            f"{m['ms_ssim']:>10.5f}{m['mse']:>12.4g}"
        )
    dif = np.abs(a - b)
    print(
        f"\n{lb}-{la} linA diff: max={dif.max():.4g}  mean={dif.mean():.4g}  "
        f"(range {a.min():.1f}..{a.max():.1f})"
    )

    # ---- log-intensity panels for a few patches ----
    p = np.load(args.a).shape[-1]
    idx = np.unique(np.linspace(0, ga * gr - 1, args.patches).astype(int))
    cols = [("noisy", noisy), ("MERLIN", gt), (la, a), (lb, b)]
    cols = [(nm, im) for nm, im in cols if im is not None]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(3 * len(cols), 3 * len(idx)))
    axes = np.atleast_2d(axes)
    for ri, k in enumerate(idx):
        pa, pr = divmod(int(k), gr)
        sl = (slice(pa * p, (pa + 1) * p), slice(pr * p, (pr + 1) * p))
        ref_db = to_db(a[sl])
        vmin, vmax = np.percentile(ref_db, 1), np.percentile(ref_db, 99)
        for ci, (nm, im) in enumerate(cols):
            ax = axes[ri, ci]
            ax.imshow(to_db(im[sl]), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(nm, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"patch {k}", fontsize=10)
    fig.suptitle(f"log-intensity (dB) — {lb} vs {la} denorm", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nfigure -> {args.out}")


if __name__ == "__main__":
    main()
