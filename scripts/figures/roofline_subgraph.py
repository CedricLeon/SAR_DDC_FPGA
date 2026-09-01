#!/usr/bin/env python3
"""Per-subgraph roofline (DATE'27) — why topology predicts the binding resource.

The characterization's *prediction* evidence: on the fixed-overlay B4096 DPU, the
decider is the main encoder g_a. Its plain (4.51 GOP) vs residual (39.76 GOP) form
differ ~9x in work while both run compute-bound near the peak, so the residual archs
(ResFP/ResSHyp) roof ~9x lower than the plain ones (FP/SHyp) — readable off the roofline,
before any streaming run. The tiny hyperprior kernels h_a/h_s sit low and are
weight-load-bound, not DDR-bound: LdWB is 97-99% of their DDR traffic (vs 16% for
residual g_a, which is feature-map-dominated), and their measured DDR bandwidth
(~5.7-6.0 GB/s, vaitrace AvgBw) is only ~34% of the raw DDR peak below -- the second,
dashed ceiling makes that gap explicit. g_s is excluded: it never runs in this
(compression-only) pipeline (onboard_pipeline.md: "the never-run g_s is dropped").

Williams roofline for one B4096 core:
  - horizontal ceiling  = 1229 GOP/s  (4096 MACs*2 ops/cycle x 0.30 GHz; PG338 + xdputil clock)
  - solid diagonal (DDR)= 17.06 GB/s PS-DDR4 peak, whole-chip  (UG1182); knee at 72 OP/byte
  - solid diagonal (AXI)= 9.6 GB/s per-core AXI interface peak: 2x 128-bit M_AXI_DATA ports per
    DPUCZDX8G core (PG338) x 300 MHz DPU clock / 8. This is the tighter, arguably more correct
    bound for a SINGLE core (this figure's scope) -- a single core cannot exceed its own
    interface width regardless of how much more the shared DDR controller could supply. Shown
    alongside the DDR line rather than replacing it, pending co-author decision on which to use
    in the final figure. Knee at 1229/9.6 = 128 OP/byte.
  - dashed diagonal      = 6.11 GB/s measured DPU weight-load bandwidth ceiling.
    IMPORTANT byte-accounting note: this is NOT vaitrace's raw AvgBw field (5.74/6.01 GB/s
    for h_a/h_s -- that uses vaitrace's own DYNAMIC byte count, LdWB+LdFM+StFM). The x-axis
    (arithmetic intensity) instead uses the STATIC xmodel_info.json byte estimate
    (const+input+output, no workspace) -- a *different* bytes denominator that disagrees
    with vaitrace's by 3-5% for h_a/h_s. Mixing the two would put each point 2.6-5.5% ABOVE
    a line drawn at vaitrace's own AvgBw, which looks like a violated ceiling. Fixed here by
    using achieved_GOPs/AI (h_a: 6.02, h_s: 6.20 GB/s, mean 6.11) -- i.e., the SAME static
    bytes-basis as the x-axis, so the line is self-consistent with where the points are
    actually plotted, and both sit at/below it as a ceiling should.
    Provenance of the underlying measurement is still vaitrace HW_RT + AvgBw-dominance
    (LdWB >=97% of DDR traffic for h_a/h_s) -- docs/onboard_pipeline.md Sec.6 -- only the
    bandwidth NUMBER used for the line differs from the raw AvgBw column, for consistency.
  - each subgraph plotted at (arithmetic intensity, achieved GOP/s):
      AI  = workload_ops / (const + input + output bytes)   [DDR traffic per inference]
      GOP/s = workload_ops / HW_RT                          [vaitrace hardware counter]

Data (provenance in-line):
  ops / bytes <- results/date27/s0/<arch>/<arch>-relu_s0_L20_pt_xmodel_info.json  (via _figutils)
  HW_RT (ms), LdWB, AvgBw  <- docs/onboard_pipeline.md Sec.6 (vaitrace), canonical DPU-time table
Both are architecture properties (lambda-independent); the xmodel_info JSONs are the L20
export but the shapes/ops are identical across lambda.

P1.0 migration: repointed to results/date27/ only; the visual design is unchanged.
The redesign (F3, batch P1.2) removes the 6.11 GB/s weight-load line and adds 3-core
ceiling pairs + aggregate dots from results/date27/vaitrace/ — not done here.

Run:  conda activate DDC_FPGA && python scripts/figures/roofline_subgraph.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/roofline_subgraph.{pdf,png}
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import load_xmodel_subgraphs, save_figure

PEAK_GOPS = 1229.0  # B4096 @ 300 MHz, one core          (PG338)
DDR_GBPS = 17.06  # PS DDR4-2133, 64-bit, raw peak      (UG1182)
WEIGHT_BW_GBPS = 6.11  # DPU effective (weight-load) bandwidth ceiling -- mean of
# achieved_GOPs/AI for h_a (6.02) and h_s (6.20), i.e. computed on
# the SAME static bytes-basis as the x-axis (see docstring above for
# why this differs from vaitrace's raw AvgBw of 5.74/6.01 GB/s)
AXI_GBPS = 9.6  # per-core AXI interface peak: 2x 128-bit M_AXI_DATA ports (PG338)
# x 300 MHz DPU clock / 8 -- the tighter, single-core-correct bound
RIDGE = PEAK_GOPS / DDR_GBPS  # = 72.0 OP/byte  (DDR knee)
RIDGE_AXI = PEAK_GOPS / AXI_GBPS  # = 128.0 OP/byte (AXI knee)
RIDGE_W = PEAK_GOPS / WEIGHT_BW_GBPS  # = 201.1 OP/byte (weight-load knee)

ANN_COLOR = "#333333"  # single color/font/size for all three ceiling-line annotations
ANN_SIZE = 7.8
POINT_COLOR = "black"  # crosses + their name/efficiency labels

# vaitrace HW_RT (ms), per-subgraph — docs/onboard_pipeline.md Sec.6 (canonical DPU time)
HW_RT_MS = {"g_a_plain": 4.10, "g_a_res": 33.51, "h_a": 0.83, "h_s": 0.52}


def ai(ops, const, ib, ob):
    """Arithmetic intensity [OP/byte] = ops / (const + input + output bytes)."""
    return ops / (const + ib + ob)


def load_sub(arch, sg):
    """(workload_ops, const_bytes, input_bytes, output_bytes) for subgraph ``sg`` of ``arch``."""
    d = load_xmodel_subgraphs(arch)[sg]
    return d["workload_ops"], d["const_bytes"], d["input_bytes"], d["output_bytes"]


# plain g_a from FP, residual g_a from ResFP, hyper kernels from ResSHyp
o, c, i, ob = load_sub("FP", "g_a")
ai_gap = ai(o, c, i, ob)
gops_gap = o / (HW_RT_MS["g_a_plain"] * 1e6)
o, c, i, ob = load_sub("ResFP", "g_a")
ai_gar = ai(o, c, i, ob)
gops_gar = o / (HW_RT_MS["g_a_res"] * 1e6)
o, c, i, ob = load_sub("ResSHyp", "h_a")
ai_ha = ai(o, c, i, ob)
gops_ha = o / (HW_RT_MS["h_a"] * 1e6)
o, c, i, ob = load_sub("ResSHyp", "h_s")
ai_hs = ai(o, c, i, ob)
gops_hs = o / (HW_RT_MS["h_s"] * 1e6)

# subgraph point: (short label, AI, achieved GOP/s, label placement)
# placement: "below" (g_a variants) or "right" (h_a/h_s); h_s sits slightly ABOVE h_a
# (338.9 vs 331.7 GOP/s) so h_s's label offsets up and h_a's offsets down -- keeps the
# label order matching the data order instead of crossing over it.
POINTS = [
    (r"$g_a$ residual", ai_gar, gops_gar, "below", (0, -9)),
    (r"$g_a$", ai_gap, gops_gap, "below", (0, -9)),
    (r"$h_a$", ai_ha, gops_ha, "right", (8, -5)),
    (r"$h_s$", ai_hs, gops_hs, "right", (8, 5)),
]

fig, ax = plt.subplots(figsize=(5.4, 3.7))
xlo, xhi = 20.0, 2.5e4
xs = np.geomspace(xlo, xhi, 400)

roof = np.minimum(PEAK_GOPS, DDR_GBPS * xs)
ax.plot(xs, roof, color="#333333", lw=1.6, zorder=3)
ax.fill_between(xs, roof, 1, color="#333333", alpha=0.04, zorder=0)

roof_axi = np.minimum(PEAK_GOPS, AXI_GBPS * xs)
ax.plot(xs, roof_axi, color="#333333", lw=1.6, zorder=3)

# weight-load ceiling: truncated past its own knee -- beyond that it's redundant with the
# solid DPU-peak ceiling and would just overdraw it for the rest of the x-range
xs_w = np.geomspace(xlo, RIDGE_W * 1.15, 200)
roof_w = np.minimum(PEAK_GOPS, WEIGHT_BW_GBPS * xs_w)
ax.plot(xs_w, roof_w, color="#333333", lw=1.4, ls="--", zorder=3)

for lab, x, y, *_ in POINTS:
    ax.scatter([x], [y], s=45, color=POINT_COLOR, marker="x", linewidth=1.3, zorder=5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(xlo, xhi)
ax.set_ylim(80, 2100)
ax.set_xlabel("arithmetic intensity [OP/byte]")
ax.set_ylabel("throughput [GOP/s]")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, which="both", ls="-", lw=0.3, color="#EEEEEE", zorder=0)

fig.tight_layout()
fig.canvas.draw()  # finalize layout so transData below reflects the real axes position


def visual_angle_deg(ax, x0, y0, x1, y1):
    """True on-screen angle (deg) of the segment (x0,y0)-(x1,y1), given the current (possibly log-
    scaled, non-square) axes -- so rotated text can visually align with a plotted line regardless
    of axes aspect ratio."""
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    return np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


# both diagonals have log-log slope 1 (y = B*x) -> identical visual angle on these axes
theta = visual_angle_deg(ax, 10, DDR_GBPS * 10, 1000, DDR_GBPS * 1000)

# --- DPU peak: horizontal, centered (log-space) between the DDR knee and g_a plain ---
x_peak_lab = np.sqrt(RIDGE * ai_gar)
ax.text(
    x_peak_lab,
    PEAK_GOPS * 1.03,
    f"DPU peak {PEAK_GOPS:.0f} GOP/s",
    rotation=0,
    fontsize=ANN_SIZE,
    color=ANN_COLOR,
    ha="center",
    va="bottom",
)

# --- DDR peak: along the solid diagonal, close above it ---
x0 = 21.5
ax.text(
    x0,
    DDR_GBPS * x0 * 1.16,
    f"DDR peak {DDR_GBPS:.1f} GB/s",
    rotation=theta,
    fontsize=ANN_SIZE,
    color=ANN_COLOR,
    ha="left",
    va="bottom",
)

# --- DPU AXI peak: along its own solid diagonal, close above it ---
x2 = 23.0
ax.text(
    x2,
    AXI_GBPS * x2 * 1.16,
    f"DPU AXI peak {AXI_GBPS:.1f} GB/s",
    rotation=theta,
    fontsize=ANN_SIZE,
    color=ANN_COLOR,
    ha="left",
    va="bottom",
)

# --- DPU weight-load: along the dashed diagonal, close above it ---
x1 = 22.5
ax.text(
    x1,
    WEIGHT_BW_GBPS * x1 * 1.17,
    f"measured DPU weight-load {WEIGHT_BW_GBPS:.2f} GB/s",
    rotation=theta,
    fontsize=ANN_SIZE,
    color=ANN_COLOR,
    ha="left",
    va="bottom",
)

# --- per-point labels: subgraph name + efficiency, placed below (g_a) or right (h_a/h_s) ---
for lab, x, y, where, (dx, dy) in POINTS:
    eff = y / PEAK_GOPS * 100
    va = "top" if where == "below" else "center"
    ha = "center" if where == "below" else "left"
    ax.annotate(
        f"{lab} ({eff:.0f}%)",
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.4,
        color=POINT_COLOR,
        ha=ha,
        va=va,
        zorder=6,
    )

save_figure(fig, "roofline_subgraph")
print(f"  visual angle of diagonals: {theta:.2f} deg")
for lab, x, y, *_ in POINTS:
    eff = y / PEAK_GOPS * 100
    line_y = min(PEAK_GOPS, WEIGHT_BW_GBPS * x) if "h_" in lab else min(PEAK_GOPS, DDR_GBPS * x)
    print(f"  AI={x:8.1f} OP/byte  {y:7.1f} GOP/s  eff={eff:4.1f}%   {lab}")
print(
    f"  DDR knee={RIDGE:.1f} OP/byte   AXI knee={RIDGE_AXI:.1f} OP/byte   weight-load knee={RIDGE_W:.1f} OP/byte"
)
print(f"  DPU-peak label x={x_peak_lab:.1f}")
print(
    f"  weight-load line at h_a AI: {WEIGHT_BW_GBPS * ai_ha:.1f} GOP/s (actual {gops_ha:.1f}, {'OK sits below' if gops_ha <= WEIGHT_BW_GBPS * ai_ha else 'STILL ABOVE by %.1f%%' % ((gops_ha / (WEIGHT_BW_GBPS * ai_ha) - 1) * 100)})"
)
print(
    f"  weight-load line at h_s AI: {WEIGHT_BW_GBPS * ai_hs:.1f} GOP/s (actual {gops_hs:.1f}, {'OK sits below' if gops_hs <= WEIGHT_BW_GBPS * ai_hs else 'STILL ABOVE by %.1f%%' % ((gops_hs / (WEIGHT_BW_GBPS * ai_hs) - 1) * 100)})"
)
