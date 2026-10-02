#!/usr/bin/env python3
"""F1 — per-patch sequential breakdown (DATE'27): where the per-patch time goes.

Full-scene (7,540-patch) sequential run, per-patch ms, all four architectures. The
stack (and the legend) is in pipeline-temporal order, bottom -> top:
  row-block read -> patchify -> Log-normalization -> g_a(real)+g_a(imag)
  -> h_a -> Side EE -> h_s -> Main EE -> .ddc write
h_a, h_s and Side EE are 0 for the factorized archs (FP/ResFP), where Side EE's
stage (eb_compress) is the Main EE instead. Side EE = eb_compress + eb_decompress
(the z round-trip); Main EE = gc_compress (hyperprior) or eb_compress (factorized).
Slices are tinted by the resource that runs each stage -- I/O (grey), CPU (blue),
DPU (orange), _figutils.PALETTE.

Two topology axes are visible: the residual axis (g_a 10 -> 73 ms, FP->ResFP /
SHyp->ResSHyp) and the hyperprior axis (adds h_a/h_s on the DPU plus a heavier
Main EE, ~11 ms vs ~6.6 ms).

Numbers (lambda=20; per-stage compute is lambda-independent). One instrumented
full-scene run per arch -- read / patchify / write are single-run I/O, storage-
and tile-dependent.
  stages  <- results/date27/ladder/<arch>/r0_seq_warm.json
            (stage_ms_per_patch, via _figutils.load_seq_stages)

Run:  conda activate DDC_FPGA && python scripts/figures/stacked_time.py
Out:  LaTeX/SAR_DDC_FPGA_DATE27/figures/images/stacked_time.{pdf,png}
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figutils import (
    ARCHS,
    DISPLAY,
    PALETTE,
    PALETTE_FILL,
    blend,
    load_seq_stages,
    save_figure,
)
from matplotlib.patches import Patch

# category -> (resource, display name). Stack + legend order, bottom -> top, is the
# temporal order of the per-patch pipeline. h_a / h_s / Side EE are their own slices
# (0 for FP/ResFP). Legend label = f(name, resource) -- see LABEL below.
CATS = [
    ("read", "io", "Row-block read"),
    ("patchify", "cpu", "Patchify"),
    ("normalize", "cpu", "Log-normalization"),
    ("g_a", "dpu", r"$g_a$(real)$+g_a$(imag)"),
    ("h_a", "dpu", r"$h_a$"),
    ("side_ee", "cpu", "Side EE"),
    ("h_s", "dpu", r"$h_s$"),
    ("main_ee", "cpu", "Main EE"),
    ("write", "io", ".ddc write"),
]
RES = {c: res for c, res, _ in CATS}
RESTAG = {"cpu": "CPU", "dpu": "DPU", "io": "I/O"}


def LABEL(name, res):
    """SMall label offset."""
    # tag-prefixed so every row's name starts at the same x; "(I/O)" reads visually
    # narrower than "(CPU)"/"(DPU)" in a proportional font (the slash), so it gets an
    # extra space to land the names back in line.
    sep = "   " if res == "io" else " "
    return f"({RESTAG[res]}){sep}{name}"


# saturated tone for the primary stage on each resource (g_a on DPU, Main/Side EE on
# CPU, shared); lighter tints of the same hue for the rest. read / write share the
# storage grey.
_CPU_TINT = blend(PALETTE_FILL["cpu"], PALETTE["cpu"], 0.33)
FACE = {
    "read": PALETTE["storage"],
    "patchify": blend(PALETTE_FILL["cpu"], PALETTE["cpu"], 0.66),
    "normalize": _CPU_TINT,
    "g_a": PALETTE["dpu"],
    "h_a": blend(PALETTE_FILL["dpu"], PALETTE["dpu"], 0.55),
    "side_ee": PALETTE["cpu"],
    "h_s": blend(PALETTE_FILL["dpu"], PALETTE["dpu"], 0.30),
    "main_ee": PALETTE["cpu"],
    "write": PALETTE["storage"],
}


def load_arch(arch):
    """(per-category ms/patch, per-patch total ms) for ``arch``."""
    st = load_seq_stages(arch)
    hyper = st.get("gc_compress", 0.0) > 0.0

    def g(k):
        return float(st.get(k, 0.0))

    vals = {
        "read": g("read"),
        "patchify": g("patchify"),
        "normalize": g("normalize"),
        "g_a": g("g_a"),
        "h_a": g("h_a") if hyper else 0.0,
        "side_ee": (g("eb_compress") + g("eb_decompress")) if hyper else 0.0,
        "h_s": g("h_s") if hyper else 0.0,
        "main_ee": g("gc_compress") if hyper else g("eb_compress"),
        "write": g("write"),
    }
    return vals, sum(vals.values())


data = {a: load_arch(a) for a in ARCHS}

xpos = {a: i for i, a in enumerate(ARCHS)}
BW = 0.60

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for a in ARCHS:
    vals, total = data[a]
    bottom = 0.0
    for cat, _, _ in CATS:
        v = vals[cat]
        if v <= 0:
            continue
        ax.bar(
            xpos[a],
            v,
            BW,
            bottom=bottom,
            color=FACE[cat],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        if v >= 6.0:
            ax.text(
                xpos[a],
                bottom + v / 2,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#1a1a1a",
                zorder=3,
            )
        bottom += v
    # per-bar headline: absolute per-patch total
    ax.text(
        xpos[a],
        total + 1.2,
        f"{total:.0f} ms",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#000000",
    )

ax.set_ylim(0, 100)
ax.set_xticks([xpos[a] for a in ARCHS])
ax.set_xticklabels([DISPLAY[a] for a in ARCHS], fontsize=10)
ax.set_ylabel("latency [ms]")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", length=0)

# legend in reversed stack order -> row-block read at the bottom, .ddc write at the top
handles = [
    Patch(facecolor=FACE[c], edgecolor="white", label=LABEL(name, res)) for c, res, name in CATS
]
ax.legend(
    handles=list(reversed(handles)),
    fontsize=9.0,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    borderaxespad=0,
    handlelength=1.2,
    labelspacing=0.3,
)

save_figure(fig, "stacked_time")
for a in ARCHS:
    vals, total = data[a]
    print(
        f"  {a:8s}: "
        + ", ".join(f"{k}={v:.2f}" for k, v in vals.items())
        + f"  total={total:.1f} ms"
    )
