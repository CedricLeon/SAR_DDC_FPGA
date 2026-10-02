#!/usr/bin/env python3
"""Overlap figure (DATE'27): the per-patch seam, qualitatively + quantitatively.

Three independent panels, each saved as its own ``{pdf,png}`` so the manuscript can
lay them out as LaTeX subfigures and cross-reference them separately:
  overlap_ov0   -- overlap 0 reconstruction crop (visible seams at patch boundaries)
  overlap_ov2   -- overlap 2 reconstruction crop (same scene window, seams gone)
  overlap_psnr  -- seam-band PSNR vs overlap, all arch x lambda (the quantitative closure)

The two crops share one scene window (ResSHyp lambda1000, biggest seam) and one
grayscale mapping, so the seam is the only thing that changes between them; every
patch boundary (red edge ticks) is at the identical spot. Display = log-intensity.
The "overlap = 0 / 2" panel titles live in the LaTeX subcaptions, not the images.

Data sources (the overlap study A3 is frozen -- not part of the date27 campaign):
  seam PSNR  <- results/benchmark_stream_overlap/overlap_table.csv   (in tree, survived P0.C)
  crops      <- results/date27/overlap/ResSHyp_L1000_ov{N}_crop1024.npy

**Crop provenance (P1.OV, 2026-09-01).** The decoded reconstruction tiles the two
image panels need are 1.93 GB each (full tile 32901x14686 float32) and were archived
out of the tree on 2026-08-31 to
`/mnt/vitisAI/DDC_results_archive/2026-08-31/benchmark_stream_overlap_work.tar.gz`
(members `_work/ResSHyp-relu_s0_L1000_pt_ov{0,2,4,8,16}_warm_tile.npy`). P1.OV
salvaged a 1024x1024 window from each of the five overlap settings -- origin
(row 7788, col 6144) in the full tile, centred on the hand-pinned textured block
below -- into `results/date27/overlap/` (~20 MB total, tracked). The figure renders
only ov0 and ov2; the other three (and the extra margin beyond the 340px render
window) are kept so the window can be moved, zoomed, or shown as a five-across
strip without touching the 23 GB archive again. See results/date27/MANIFEST.md.

Run:  conda activate DDC_FPGA && python scripts/figures/overlap_crop.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/overlap_{ov0,ov2,psnr}.{pdf,png}
"""
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import ARCH_COLORS, DATE27, DISPLAY, REPO_ROOT, save_figure

OVERLAP = REPO_ROOT / "results" / "benchmark_stream_overlap"
TABLE = OVERLAP / "overlap_table.csv"
CROPS = DATE27 / "overlap"
CROP_STEM = "ResSHyp_L1000_ov{ov}_crop1024"

# Geometry of the salvaged crops, in full-tile pixel coordinates (P1.OV). The 1024x1024
# window each crop file holds starts here; every crop uses the identical origin so the
# same scene appears in all five.
CROP_ROW0, CROP_COL0, CROP_SIZE = 7788, 6144, 1024

ROW_C, COL_B, HALF = 8300, 6656, 170  # hand-pinned textured block (Hamburg), full-tile coords
PATCH = 256

# --- per-panel figure sizes (inches). The two crops are square and identical so they
# read as a matched pair when scaled to the same subfigure width; the PSNR panel is a
# touch wider/shorter to match the wider subfigure it goes in. ---
CROP_FIGSIZE = (2.6, 2.6)
PSNR_FIGSIZE = (4.0, 3.0)


def load_crop(ov):
    """The 1024x1024 salvaged crop for overlap ``ov`` (hard-error + regen steps if absent)."""
    p = CROPS / f"{CROP_STEM.format(ov=ov)}.npy"
    if not p.is_file():
        raise FileNotFoundError(
            f"overlap crop is missing: {p}\n"
            "The 1024x1024 crops are salvaged (P1.OV) from the archived reconstruction tiles\n"
            "  archive: /mnt/vitisAI/DDC_results_archive/2026-08-31/"
            "benchmark_stream_overlap_work.tar.gz\n"
            "  member:  _work/ResSHyp-relu_s0_L1000_pt_ov<N>_warm_tile.npy\n"
            f"  window:  rows [{CROP_ROW0}:{CROP_ROW0 + CROP_SIZE}], "
            f"cols [{CROP_COL0}:{CROP_COL0 + CROP_SIZE}] (float32)\n"
            "Regenerate: extract the member, slice that window, save float32 .npy\n"
            "  (full procedure in results/date27/MANIFEST.md, P1.OV block)."
        )
    a = np.load(p)
    if a.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"{p}: expected {(CROP_SIZE, CROP_SIZE)} crop, got {a.shape}")
    return a


def logimg(x):
    """Log10 intensity of ``x``, clipped at 1.0 (the display transform)."""
    return np.log10(np.clip(np.asarray(x, np.float32), 1.0, None))


def patch_boundaries(lo, hi):
    """Window-local offsets of PATCH-grid lines strictly inside the render window ``[lo, hi)``."""
    return [x - lo for x in range(0, hi, PATCH) if lo < x < hi]


def edge_ticks(ax, verticals, horizontals, H, W, L=16, c="#D62728", lw=1.9):
    """Draw inward patch-boundary ticks on all four edges of an ``H``×``W`` image axis."""
    for xc in verticals:  # top + bottom, pointing in
        ax.plot([xc, xc], [-0.5, -0.5 + L], color=c, lw=lw, clip_on=False)
        ax.plot([xc, xc], [H - 0.5, H - 0.5 - L], color=c, lw=lw, clip_on=False)
    for yc in horizontals:  # left + right
        ax.plot([-0.5, -0.5 + L], [yc, yc], color=c, lw=lw, clip_on=False)
        ax.plot([W - 0.5, W - 0.5 - L], [yc, yc], color=c, lw=lw, clip_on=False)


# --- crops: pull the rendered 340x340 sub-window out of the 1024x1024 salvaged crop ---
# Sub-window position inside the crop = its full-tile position minus the crop origin.
lr0, lr1 = ROW_C - HALF - CROP_ROW0, ROW_C + HALF - CROP_ROW0
lc0, lc1 = COL_B - HALF - CROP_COL0, COL_B + HALF - CROP_COL0
crop0 = logimg(load_crop(0)[lr0:lr1, lc0:lc1])
crop2 = logimg(load_crop(2)[lr0:lr1, lc0:lc1])
Hc, Wc = crop0.shape
vmin, vmax = np.percentile(crop0, [2, 98])  # one mapping for both crops (fair comparison)
# patch boundaries inside the render window (window-local coords), both axes
verts = patch_boundaries(COL_B - HALF, COL_B + HALF)
horis = patch_boundaries(ROW_C - HALF, ROW_C + HALF)


def save_crop(img, name):
    """One square crop panel: image + red patch-boundary ticks, no axes, tight margin."""
    fig, ax = plt.subplots(figsize=CROP_FIGSIZE)
    fig.subplots_adjust(0, 0, 1, 1)  # axes fill the figure; the red ticks then stick out
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    edge_ticks(ax, verts, horis, Hc, Wc)
    ax.set_xticks([])
    ax.set_yticks([])
    save_figure(fig, name, pad_inches=0.02)


save_crop(crop0, "overlap_ov0")
save_crop(crop2, "overlap_ov2")

# --- seam PSNR vs overlap (dedupe: prefer warm; quality is mode-independent) ---
series = {}
with open(TABLE) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["lambda"]))
        ov = int(row["overlap"])
        rec = series.setdefault(key, {})
        if ov not in rec or row["mode"] == "warm":
            rec[ov] = float(row["psnr_seam"])

# FP light blue / ResSHyp intense orange, matching the other DATE'27 figures.
STYLE = {"FP": ARCH_COLORS["FP"], "ResSHyp": ARCH_COLORS["ResSHyp"]}
# A short, tight dash period (rather than mpl's default "--") so at least one full
# dash+gap cycle is visible even inside a short legend handle.
LS = {20: (0, (2, 1.3)), 1000: "-"}

fig, axp = plt.subplots(figsize=PSNR_FIGSIZE)
# Plot/legend order: both archs at lambda=1000 first, then both at lambda=20, so a
# 2-column legend fills as one row per lambda (top=1000, bottom=20).
for lam in (1000, 20):
    for arch in ("FP", "ResSHyp"):
        rec = series.get((arch, lam))
        if not rec:
            continue
        ovs = sorted(rec)
        axp.plot(
            ovs,
            [rec[o] for o in ovs],
            marker="o",
            ms=4,
            color=STYLE[arch],
            ls=LS.get(lam, "-"),
            lw=1.6,
            label=f"{DISPLAY[arch]} $\\lambda${lam}",
        )
axp.set_xlabel("overlap [px]")
axp.set_ylabel("seam-band PSNR [dB]")
axp.set_xticks([0, 2, 4, 8, 16])
axp.spines[["top", "right"]].set_visible(False)
axp.legend(
    fontsize=9, frameon=False, loc="lower right", ncol=2, handlelength=2.6, columnspacing=1.0
)
fig.tight_layout()
save_figure(fig, "overlap_psnr")

print(f"  verts={verts} horis={horis}")
for k, rec in sorted(series.items()):
    print(f"  {k}: " + " ".join(f"ov{o}={rec[o]:.2f}" for o in sorted(rec)))
