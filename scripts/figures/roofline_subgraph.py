#!/usr/bin/env python3
"""F3 — per-subgraph roofline (DATE'27): topology predicts the binding resource.

The characterization's *prediction* evidence. On the fixed-overlay B4096 DPU the
decider is the main encoder g_a: its plain (4.51 GOP) and residual (39.76 GOP)
forms differ ~9x in work while both run compute-bound near the peak, so the
residual archs roof ~9x lower than the plain ones -- readable off the roofline
before any streaming run. The hyperprior kernels h_a/h_s carry ~0.18-0.28 GOP,
sit at arithmetic intensity ~55 OP/byte (left of every ridge), and are
weight-load-bound: LdWB is 97-99 % of their DDR traffic. g_s is excluded -- it
never runs in this compression-only pipeline.

Ceilings (design F3, P1.1 -- Dirk's caption feedback):
  * 1 core, SOLID:  compute 1229 GOP/s (4096 MAC x 2 op/cy x 0.30 GHz, PG338);
    memory 9.6 GB/s = 2 x 128-bit M_AXI_DATA ports @ 300 MHz (PG338) -- a single
    core cannot exceed its own interface width. Ridge 1229/9.6 = 128 OP/byte.
  * 3 cores, DASHED: compute 3 x 1229 = 3687 GOP/s; memory capped at the whole-chip
    PS-DDR4 peak 17.06 GB/s (DDR4-2133 x 64-bit, UG1182), because 3 x 9.6 = 28.8
    exceeds it -- the three cores share one DDR controller. Ridge 3687/17.06 = 216.
  The old 6.11 GB/s "measured weight-load" line is removed (a measured point, not a
  ceiling). The ~13.7 GB/s achievable-DDR figure (R2) is cite-only in the text, not
  drawn.

Points:
  * single core (x marker): (arithmetic intensity, achieved GOP/s) at 1 lane, from
    results/date27/vaitrace/<arch>/vaitrace_1lane.txt (HW_RT hardware counter).
  * 3-core aggregate (o marker): sum of the three cores' achieved GOP/s at the
    per-arch fan-out knee, from results/date27/vaitrace/<arch>/vaitrace_knee{K}.txt
    -- the operating point when all three cores run that subgraph concurrently. FP's
    resolve to the 12 L knee (vaitrace_knee12.txt, P0.7 re-trace); the 32 L capture
    is provenance only. The g_a-residual and h_a/h_s aggregates are read from the
    ResSHyp knee-20 trace (E4), where all three subgraphs run in one condition.
  arithmetic intensity uses the static xmodel_info byte estimate (const+input+output),
  the same denominator for the x-axis and any bandwidth read off the plot.

Data:
  ops / bytes  <- results/date27/s0/<arch>/<arch>-relu_s0_L20_pt_xmodel_info.json  (via _figutils)
  HW_RT / eff / AvgBw  <- results/date27/vaitrace/<arch>/vaitrace_{1lane,knee<K>}.txt
Both are architecture properties (lambda-independent).

Run:  conda activate DDC_FPGA && python scripts/figures/roofline_subgraph.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/roofline_subgraph.{pdf,png}
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import DATE27, KNEE, PALETTE, load_xmodel_subgraphs, save_figure
from matplotlib.lines import Line2D

PEAK_GOPS = 1229.0  # one B4096 core @ 300 MHz                             (PG338)
AXI_GBPS = 9.6  # per-core: 2 x 128-bit M_AXI_DATA @ 300 MHz / 8          (PG338)
DDR_GBPS = 17.06  # whole-chip PS-DDR4-2133 x 64-bit peak                  (UG1182)
NCORE = 3  # usable B4096 cores on this ZCU102

PEAK_3 = NCORE * PEAK_GOPS  # 3-core compute ceiling
MEM_3 = DDR_GBPS  # 3-core memory ceiling: shared DDR, NOT 3 x AXI
RIDGE_1 = PEAK_GOPS / AXI_GBPS  # 1-core ridge  [OP/byte]
RIDGE_3 = PEAK_3 / MEM_3  # 3-core ridge  [OP/byte]

CEIL_COLOR = PALETTE["ceiling"]
SC_COLOR = "#1a1a1a"  # single-core markers
AGG_COLOR = PALETTE["dpu"]  # 3-core aggregate markers (DPU work, under fan-out)
ANN_SIZE = 7.6


def ai(sg):
    """Arithmetic intensity [OP/byte] = workload_ops / (const + input + output bytes)."""
    return sg["workload_ops"] / (sg["const_bytes"] + sg["input_bytes"] + sg["output_bytes"])


def _num(tok):
    """Parse one vaitrace numeric cell; the tool prints '~0' for sub-0.001 values."""
    tok = tok.strip()
    return 0.0 if tok in ("~0", "") else float(tok)


def vaitrace_rows(path):
    """Per-core DPU rows of a vaitrace txt dump -> list of dicts (subgraph classified)."""
    rows = []
    for line in open(path):
        if not line.startswith("DPUCZDX8G_"):
            continue
        p = [c.strip() for c in line.split("|")]
        sg = p[2]
        name = "h_a" if "h_a" in sg else "h_s" if "h_s" in sg else "g_a" if "g_a" in sg else sg
        rows.append(
            dict(
                core=int(p[0].split("_")[1]),
                name=name,
                wl=_num(p[3]),
                hw=_num(p[5]),
                eff=_num(p[6]),
                avgbw=_num(p[10]),
            )
        )
    return rows


def vt(arch, which):
    """Vaitrace rows for ``arch``; ``which`` = 'vaitrace_1lane' or 'vaitrace_knee<K>'."""
    return vaitrace_rows(DATE27 / "vaitrace" / arch / f"{which}.txt")


def gops(wl_gop, hw_ms):
    """Achieved GOP/s from a vaitrace workload (GOP) and hardware runtime (ms)."""
    return wl_gop / (hw_ms * 1e-3)


# --- single-core points: (label, AI, GOP/s, 1-core efficiency %) -----------------------
SUB = {a: load_xmodel_subgraphs(a) for a in ("FP", "ResFP", "ResSHyp")}
one = {a: vt(a, "vaitrace_1lane") for a in ("FP", "ResFP", "ResSHyp")}


def sc_point(arch, name):
    """(AI, achieved GOP/s, vaitrace Effic %) for subgraph ``name`` of ``arch`` at 1 lane."""
    r = [x for x in one[arch] if x["name"] == name][0]
    return ai(SUB[arch][name]), gops(r["wl"], r["hw"]), r["eff"]


SC = {
    "g_a": (r"$g_a$", *sc_point("FP", "g_a")),
    "g_a_res": (r"$g_a$+Res", *sc_point("ResFP", "g_a")),
    "h_a": (r"$h_a$", *sc_point("ResSHyp", "h_a")),
    "h_s": (r"$h_s$", *sc_point("ResSHyp", "h_s")),
}

# --- 3-core aggregate points: sum of per-core GOP/s at the fan-out knee -----------------
knee_fp = vt("FP", f"vaitrace_knee{KNEE['FP']}")
knee_rsh = vt("ResSHyp", f"vaitrace_knee{KNEE['ResSHyp']}")


def agg(rows, name):
    """3-core aggregate for ``name``: (sum GOP/s, per-core eff %-range, per-core GB/s-range)."""
    rs = [x for x in rows if x["name"] == name]
    per = [gops(x["wl"], x["hw"]) for x in rs]
    effs = [x["eff"] for x in rs]
    bws = [x["avgbw"] / 1000 for x in rs]  # MB/s -> GB/s
    return sum(per), (min(effs), max(effs)), (min(bws), max(bws))


AGG = {
    "g_a": agg(knee_fp, "g_a"),  # FP plain g_a, 3 cores @ 12 L knee
    "g_a_res": agg(knee_rsh, "g_a"),  # ResSHyp residual g_a, 3 cores @ 20 L knee
    "h_a": agg(knee_rsh, "h_a"),
    "h_s": agg(knee_rsh, "h_s"),
}

# ======================================================================================
fig, ax = plt.subplots(figsize=(5.6, 4.2))
xlo, xhi = 20.0, 2.6e4
ylo, yhi = 80.0, 5200.0
xs = np.geomspace(xlo, xhi, 400)


def roofline(mem_bw, peak, **kw):
    """Draw one roof: horizontal at ``peak`` GOP/s, diagonal at ``mem_bw`` GB/s below the ridge."""
    ax.plot(xs, np.minimum(peak, mem_bw * xs), **kw)


roofline(AXI_GBPS, PEAK_GOPS, color=CEIL_COLOR, lw=1.7, zorder=3)  # 1 core, solid
roofline(MEM_3, PEAK_3, color=CEIL_COLOR, lw=1.7, ls=(0, (5, 2)), zorder=3)  # 3 cores, dashed
ax.fill_between(
    xs, np.minimum(PEAK_GOPS, AXI_GBPS * xs), ylo, color=CEIL_COLOR, alpha=0.05, zorder=0
)

# --- markers: single-core x, its 3-core aggregate o, a dotted connector between -------
for k, (_, x, y_sc, _) in SC.items():
    y_agg = AGG[k][0]
    ax.plot([x, x], [y_sc, y_agg], color=AGG_COLOR, lw=0.8, ls=":", zorder=4)
    ax.scatter([x], [y_sc], s=52, color=SC_COLOR, marker="x", linewidth=1.5, zorder=6)
    ax.scatter(
        [x],
        [y_agg],
        s=44,
        facecolor="none",
        edgecolor=AGG_COLOR,
        linewidth=1.6,
        marker="o",
        zorder=6,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)
ax.set_xlabel("arithmetic intensity  [OP/byte]")
ax.set_ylabel("throughput  [GOP/s]")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, which="both", ls="-", lw=0.3, color="#EEEEEE", zorder=0)
fig.tight_layout()
fig.canvas.draw()


def angle(x0, y0, x1, y1):
    """On-screen angle (deg) of a data segment, for text rotated to lie along a plotted line."""
    p0, p1 = ax.transData.transform((x0, y0)), ax.transData.transform((x1, y1))
    return np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


th_axi = angle(25, AXI_GBPS * 25, 90, AXI_GBPS * 90)
th_ddr = angle(25, DDR_GBPS * 25, 90, DDR_GBPS * 90)

# --- ceiling labels: flat labels on the left arm of each horizontal, diagonal labels
#     riding each sloped arm in the empty lower-left --------------------------------
ax.text(
    360,
    PEAK_GOPS * 1.07,
    f"1 core  ·  {PEAK_GOPS:.0f} GOP/s",
    ha="left",
    va="bottom",
    fontsize=ANN_SIZE,
    color=CEIL_COLOR,
)
ax.text(
    360,
    PEAK_3 * 1.07,
    f"3 cores  ·  {PEAK_3:.0f} GOP/s",
    ha="left",
    va="bottom",
    fontsize=ANN_SIZE,
    color=CEIL_COLOR,
)
ax.text(
    23,
    AXI_GBPS * 23 * 1.14,
    f"AXI {AXI_GBPS:.1f} GB/s (1 core)",
    rotation=th_axi,
    rotation_mode="anchor",
    ha="left",
    va="bottom",
    fontsize=ANN_SIZE,
    color=CEIL_COLOR,
)
ax.text(
    23,
    DDR_GBPS * 23 * 1.14,
    f"DDR {DDR_GBPS:.2f} GB/s (3 cores, shared)",
    rotation=th_ddr,
    rotation_mode="anchor",
    ha="left",
    va="bottom",
    fontsize=ANN_SIZE,
    color=CEIL_COLOR,
)

# --- single-core point labels: name + 1-core efficiency (vaitrace Effic column) ------
ax.annotate(
    f"{SC['g_a'][0]}  {SC['g_a'][3]:.1f}%",
    (SC["g_a"][1], SC["g_a"][2]),
    xytext=(-4, -13),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=7.8,
    color=SC_COLOR,
    zorder=7,
)
ax.annotate(
    f"{SC['g_a_res'][0]}  {SC['g_a_res'][3]:.1f}%",
    (SC["g_a_res"][1], SC["g_a_res"][2]),
    xytext=(0, -13),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=7.8,
    color=SC_COLOR,
    zorder=7,
)
ax.annotate(
    f"{SC['h_a'][0]} {SC['h_a'][3]:.1f}%",
    (SC["h_a"][1], SC["h_a"][2]),
    xytext=(11, 2),
    textcoords="offset points",
    ha="left",
    va="center",
    fontsize=7.8,
    color=SC_COLOR,
    zorder=7,
)
ax.annotate(
    f"{SC['h_s'][0]} {SC['h_s'][3]:.1f}%",
    (SC["h_s"][1], SC["h_s"][2]),
    xytext=(-11, -2),
    textcoords="offset points",
    ha="right",
    va="center",
    fontsize=7.8,
    color=SC_COLOR,
    zorder=7,
)

# --- story annotation 1: g_a is compute-bound and scales ~3x to the 3-core roof -----
#   sits in the gap between the single-core row and the aggregate row, spanning both g_a
gar = AGG["g_a_res"][1]
gap = AGG["g_a"][1]
ax.text(
    5200,
    1950,
    f"$g_a$: compute-bound, 3 cores $\\approx$ 3×\n"
    f"residual {gar[0]:.0f}–{gar[1]:.0f}% / core, plain {gap[0]:.0f}–{gap[1]:.0f}%",
    fontsize=ANN_SIZE,
    color=AGG_COLOR,
    ha="center",
    va="center",
)

# --- story annotation 2: the hyperprior kernels, weight-load-bound ------------------
#   (the LdWB 97-99% and the "small wall-time share, not the system bottleneck" caveat
#    live in the caption -- P1.5)
ha_eff, ha_bw = AGG["h_a"][1], AGG["h_a"][2]
ax.annotate(
    r"$h_a,h_s$: weight-load-bound."
    "\n"
    f"1 core {SC['h_s'][3]:.0f}–{SC['h_a'][3]:.0f}% eff → "
    f"{ha_eff[0]:.0f}–{ha_eff[1]:.0f}% / core under fan-out\n"
    f"at {ha_bw[0]:.1f}–{ha_bw[1]:.1f} GB/s — shared-DDR contention",
    xy=(SC["h_a"][1], (SC["h_s"][2] * AGG["h_a"][0]) ** 0.5),
    xytext=(120, 112),
    textcoords="data",
    fontsize=ANN_SIZE,
    color=SC_COLOR,
    ha="left",
    va="bottom",
    arrowprops=dict(arrowstyle="-", color=SC_COLOR, lw=0.7),
)

# --- legend: marker meaning (upper-left, above the dashed diagonal) -----------------
ax.legend(
    handles=[
        Line2D([], [], color=SC_COLOR, marker="x", ls="none", ms=7, mew=1.5, label="1 lane"),
        Line2D(
            [],
            [],
            color=AGG_COLOR,
            marker="o",
            ls="none",
            ms=7,
            mfc="none",
            mew=1.6,
            label="3-core knee (aggregate)",
        ),
    ],
    loc="upper left",
    bbox_to_anchor=(0.005, 0.995),
    fontsize=ANN_SIZE,
    frameon=False,
    handletextpad=0.4,
    labelspacing=0.5,
    borderpad=0.2,
)

save_figure(fig, "roofline_subgraph")

print(
    f"  ridge: 1-core {RIDGE_1:.1f} OP/byte   3-core {RIDGE_3:.1f} OP/byte"
    f"   (3*AXI={NCORE * AXI_GBPS:.1f} > DDR {DDR_GBPS})"
)
print("  single core (1 lane):")
for lab, x, y, eff in SC.values():
    print(f"    AI={x:9.1f}  {y:8.1f} GOP/s  eff={eff:5.1f}%   {lab}")
print("  3-core aggregate (knee):")
for k, (s, effr, bwr) in AGG.items():
    print(
        f"    {k:8s} sum={s:8.1f} GOP/s  per-core eff {effr[0]:.1f}-{effr[1]:.1f}%"
        f"  AvgBw {bwr[0]:.2f}-{bwr[1]:.2f} GB/s"
    )
