#!/usr/bin/env python3
"""Overlap figure (DATE'27): the per-patch seam, qualitatively + quantitatively.

Panels:
  (left)   overlap 0 reconstruction crop  -- visible seams at patch boundaries
  (middle) overlap 2 reconstruction crop  -- same scene window, seams gone
  (right)  seam-band PSNR vs overlap, all arch x lambda -- the quantitative closure

Same scene window in both crops (ResSHyp lambda1000, biggest seam), so every patch
boundary (red ticks, both axes) is at the identical spot. Display = log-intensity.

Data sources (the overlap study A3 is frozen — not part of the date27 campaign):
  seam PSNR  <- results/benchmark_stream_overlap/overlap_table.csv        (in tree)
  crops      <- results/benchmark_stream_overlap/_work/<STEM>_ov{0,2}_warm_tile.npy

**P1.0 note — this script does not run green.** The `_work/` decoded tiles were
archived out of the tree on 2026-08-31 to
`/mnt/vitisAI/DDC_results_archive/2026-08-31/benchmark_stream_overlap_work.tar.gz`
(~23 GB) and must NOT be pulled back for a date27 figure (one-source rule). The
last good render, `figures/images/overlap_crop.{pdf,png}` (3 panels), is kept
as-is in the manuscript repo. To regenerate: restore
`ResSHyp-relu_s0_L1000_pt_ov{0,2}_warm_tile.npy` into
`results/benchmark_stream_overlap/_work/` and re-run. The seam-PSNR panel's data
(`overlap_table.csv`) is still in the tree, so only the two image panels are blocked.

Run:  conda activate DDC_FPGA && python scripts/figures/overlap_crop.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/overlap_crop.{pdf,png}
"""
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import ARCH_COLORS, DISPLAY, REPO_ROOT, save_figure

OVERLAP = REPO_ROOT / "results" / "benchmark_stream_overlap"
WORK = OVERLAP / "_work"
TABLE = OVERLAP / "overlap_table.csv"
STEM = "ResSHyp-relu_s0_L1000_pt"
OV0 = WORK / f"{STEM}_ov0_warm_tile.npy"
OV2 = WORK / f"{STEM}_ov2_warm_tile.npy"

ROW_C, COL_B, HALF = 8300, 6656, 170  # hand-pinned textured block (Hamburg)
PATCH = 256


def _require_tiles():
    """Hard-error (with the archive path + regen steps) if the crop tiles are not in the tree."""
    missing = [p for p in (OV0, OV2) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "overlap reconstruction tiles are not in the tree:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\n\nThey were archived on 2026-08-31 to /mnt/vitisAI/DDC_results_archive/"
            "2026-08-31/benchmark_stream_overlap_work.tar.gz (~23 GB) and must not be "
            "restored for a date27 figure (one-source rule, DATE27_paper_plan.md §P1.0). "
            "The rendered figures/images/overlap_crop.{pdf,png} are the last good render "
            "and are kept as-is. To regenerate, restore the two .npy tiles into "
            f"{WORK}/ and re-run."
        )


def logimg(x):
    """Log10 intensity of ``x``, clipped at 1.0 (the display transform)."""
    return np.log10(np.clip(np.asarray(x, np.float32), 1.0, None))


def edge_ticks(ax, verticals, horizontals, H, W, L=16, c="#D62728", lw=1.9):
    """Draw inward patch-boundary ticks on all four edges of an ``H``×``W`` image axis."""
    for xc in verticals:  # top + bottom, pointing in
        ax.plot([xc, xc], [-0.5, -0.5 + L], color=c, lw=lw, clip_on=False)
        ax.plot([xc, xc], [H - 0.5, H - 0.5 - L], color=c, lw=lw, clip_on=False)
    for yc in horizontals:  # left + right
        ax.plot([-0.5, -0.5 + L], [yc, yc], color=c, lw=lw, clip_on=False)
        ax.plot([W - 0.5, W - 0.5 - L], [yc, yc], color=c, lw=lw, clip_on=False)


_require_tiles()

# --- crops ---
a0 = np.load(OV0, mmap_mode="r")
a2 = np.load(OV2, mmap_mode="r")
r0, r1, c0, c1 = ROW_C - HALF, ROW_C + HALF, COL_B - HALF, COL_B + HALF
crop0, crop2 = logimg(a0[r0:r1, c0:c1]), logimg(a2[r0:r1, c0:c1])
Hc, Wc = crop0.shape
vmin, vmax = np.percentile(crop0, [2, 98])
# patch boundaries inside the window (local coords), both axes
verts = [x - c0 for x in range(0, a0.shape[1], PATCH) if c0 < x < c1]
horis = [y - r0 for y in range(0, a0.shape[0], PATCH) if r0 < y < r1]

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

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
for ax, img, label in ((axes[0], crop0, "overlap=0"), (axes[1], crop2, "overlap=2")):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    edge_ticks(ax, verts, horis, Hc, Wc)
    ax.set_xlabel(label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

axp = axes[2]
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
axp.set_xlabel("overlap (px)")
axp.set_ylabel("seam-band PSNR (dB)")
axp.set_xticks([0, 2, 4, 8, 16])
axp.spines[["top", "right"]].set_visible(False)
axp.legend(
    fontsize=10, frameon=False, loc="lower right", ncol=2, handlelength=2.6, columnspacing=1.0
)

fig.tight_layout()
save_figure(fig, "overlap_crop")
print(f"  verts={verts} horis={horis}")
for k, rec in sorted(series.items()):
    print(f"  {k}: " + " ".join(f"ov{o}={rec[o]:.2f}" for o in sorted(rec)))
