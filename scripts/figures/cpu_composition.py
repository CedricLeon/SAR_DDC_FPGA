#!/usr/bin/env python3
"""cpu_composition.py — A53 CPU-time composition (%usr / %sys / %idle) at each arch's knee.

The old ``fp_cpu_stack`` figure stacked mpstat %usr→%sys→%idle over a *lane sweep*;
that sweep (``results/benchmark_stream/cpu_probe/mpL_<L>.log``, 20 lane counts) was
deleted in P0.C. The P0.7 re-probe covers **only the knee lane** of each arch
(FP 12 / SH 24 / ResFP 6 / ResSH 20 — §4.0), so this is now a 4-bar comparison at
those operating points rather than a curve.

One stacked bar per architecture: %usr (compute) + %sys (kernel) + %idle = 100 % of
the four cores, steady-state mean (``_figutils.cpu_probe_mpstat``; first/last 10 % of
samples trimmed). The split is the point — SH runs the A53s hardest (77 % usr), ResFP
least (13 %, ~83 % idle); %sys stays a manageable 4–8 % throughout.

Not a manuscript float (decided P1.2) — the numbers go straight into W5's prose.
Colours are the original figure's (``%usr`` orange, ``%sys`` red, ``%idle`` grey).

Run:  conda activate DDC_FPGA && python scripts/figures/cpu_composition.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/cpu_composition.{pdf,png}
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import ARCHS, DISPLAY, KNEE, cpu_probe_mpstat, save_figure

C_USR, C_SYS, C_IDLE = "#E67E22", "#D62728", "#D3D3D3"

comp = {a: cpu_probe_mpstat(a) for a in ARCHS}
usr = [comp[a]["usr"] for a in ARCHS]
sysp = [comp[a]["sys"] for a in ARCHS]
idle = [comp[a]["idle"] for a in ARCHS]
x = list(range(len(ARCHS)))

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.grid(axis="y", which="major", alpha=0.28, lw=0.5, color="#9a9a9a", zorder=0)

b_usr = ax.bar(x, usr, 0.62, color=C_USR, zorder=3, label="%usr (compute)")
b_sys = ax.bar(x, sysp, 0.62, bottom=usr, color=C_SYS, zorder=3, label="%sys (kernel)")
ax.bar(
    x,
    idle,
    0.62,
    bottom=[u + s for u, s in zip(usr, sysp)],
    color=C_IDLE,
    zorder=3,
    label="%idle",
)

# %usr and %idle values on the bar (the two that carry the story; %sys is the thin band)
for xi, u, i in zip(x, usr, idle):
    ax.text(xi, u / 2, f"{u:.0f}", ha="center", va="center", fontsize=6.5, color="white")
    ax.text(xi, 100 - i / 2, f"{i:.0f}", ha="center", va="center", fontsize=6.5, color="#555555")

ax.set_xticks(x)
ax.set_xticklabels([f"{DISPLAY[a]}\n{KNEE[a]} L" for a in ARCHS], fontsize=7)
ax.tick_params(axis="both", length=2)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylabel("A53 cores occupancy  [%]", fontsize=8)
ax.set_ylim(0, 100)
ax.set_xlim(-0.6, len(ARCHS) - 0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    framealpha=1.0,
    fontsize=6.0,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    handlelength=1.0,
    handletextpad=0.4,
    columnspacing=1.0,
)

fig.tight_layout()
save_figure(fig, "cpu_composition")
for a in ARCHS:
    d = comp[a]
    print(
        f"  {a:8s} {KNEE[a]:>3} L   usr {d['usr']:5.1f}  sys {d['sys']:4.1f}  idle {d['idle']:5.1f}"
    )
