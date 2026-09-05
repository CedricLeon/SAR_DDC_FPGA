#!/usr/bin/env python3
"""_figutils.py — shared kit for the DATE'27 figure scripts.

Single source of truth (leading underscore matches the notebooks' ``_plotkit`` /
``_benchmark_loader`` house convention) for everything the figure scripts must
agree on:

* **palette** — one hue for CPU work, one for DPU work, a neutral grey for
  storage/read, a cool neutral for combined-throughput bars, one more for
  reference lines. Kept in ONE place so "make the figures look uniform" is
  structural, not a per-script promise. ``ARCH_COLORS`` is a *separate* mapping so
  a later redesign can recolour the architecture lines without touching the
  resource palette.
* **names** — results-tree dir names (``FP SHyp ResFP ResSHyp``) → paper display
  names (``FP SH ResFP ResSH``).
* **rung labels** — the cumulative optimization ladder r0-r7
  (``docs/DATE27_paper_plan.md`` §4.0). Paper-level names only; the CLI flags they
  map to are fixed and live in ``LADDER_STEMS``.
* **knee lanes** — per-arch fan-out operating point (§4.0 gate review, 2026-09-01).
* **loaders** — one helper per ``results/date27/`` subdir, resolving the
  flag-encoded filenames.
* **occupancy attribution** — re-exported from ``fanout_occupancy`` (moved into
  this directory in P1.0) so callers go through one module.
* **``save_figure()``** — writes BOTH ``.pdf`` and ``.png`` into
  ``REPO_ROOT/LaTeX/SAR_DDC_FPGA_DATE27/figures/images/`` (env override
  ``DATE27_FIG_OUT``); hard-errors if that directory is missing, never writes to
  cwd.

This is deliberately *not* a plotting-style module — no rcParams, no figure sizes.
Each script keeps its own design; P1.0 only rewired the data plumbing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)
DATE27 = REPO_ROOT / "results" / "date27"

# make sibling modules (fanout_occupancy) importable whether this file is imported
# by a script in this dir or pulled in from elsewhere
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ======================================================================================
# Palette
# ======================================================================================
# Resource colours — the semantic anchor of every DATE'27 figure. These values match
# what the scripts already used (and the ``fig:system_dataflow`` TikZ block), so
# importing them here is a no-op on the rendered output.
PALETTE: dict[str, str] = {
    "cpu": "#2E86C1",  # ARM Cortex-A53 CPU work
    "dpu": "#E67E22",  # DPU work
    "storage": "#7A7A7A",  # SD-card read / storage-bound
    "combined": "#5D6D7E",  # bars that aggregate CPU+DPU throughput (clash with neither)
    "ceiling": "#555555",  # roofs, cold-read ceiling, saturation lines
}

# Light fills for large stacked areas/bars (the TikZ ``loc_*_fill`` tones).
PALETTE_FILL: dict[str, str] = {
    "cpu": "#D6EAF8",
    "dpu": "#FDEBD0",
    "storage": "#ECECEC",
}


def blend(hex_a: str, hex_b: str, t: float = 0.5) -> str:
    """``hex_a`` moved fraction ``t`` toward ``hex_b`` (both ``#rrggbb``)."""
    a = tuple(int(hex_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02X}" for i in range(3))


# Per-architecture line/marker colours. SEPARATE from PALETTE on purpose: today they
# echo the resource palette (blue pair = CPU-bound FP/SH, orange pair = DPU-bound
# ResFP/ResSH; light = factorized, intense = hyperprior) but a redesign may want four
# unrelated hues — change them here only.
ARCH_COLORS: dict[str, str] = {
    "FP": "#82B8DC",
    "SHyp": "#2E86C1",
    "ResFP": "#F2B479",
    "ResSHyp": "#E67E22",
}

# ======================================================================================
# Names
# ======================================================================================
ARCHS: list[str] = ["FP", "SHyp", "ResFP", "ResSHyp"]  # canonical left→right order
DISPLAY: dict[str, str] = {"FP": "FP", "SHyp": "SH", "ResFP": "ResFP", "ResSHyp": "ResSH"}

# The CPU-bound / DPU-bound split the paper's Section III forks on.
CPU_BOUND: list[str] = ["FP", "SHyp"]
DPU_BOUND: list[str] = ["ResFP", "ResSHyp"]

# ======================================================================================
# Optimization ladder — cumulative rungs r0–r7 (docs/DATE27_paper_plan.md §4.0)
# ======================================================================================
RUNGS: list[int] = list(range(8))

# Paper-level rung labels. Adjustable — feedback may rename them again; the CLI flags
# they map to are fixed (see LADDER_STEMS). Paper renames: pool→mt, prefetch→dbuf.
RUNG_LABELS: dict[int, str] = {
    0: "seq",  # sequential baseline
    1: "mt",  # 4-worker pool                       (--p0 --threads 4)
    2: "fo3",  # fan-out 3 lanes, naive placement    (--fanout --lane-major --threads 3)
    3: "fo3p",  # + pinned subgraph→core placement    (--fanout --threads 3)
    4: "knee",  # fan-out at the per-arch knee, pinned (--fanout --threads KNEE[arch])
    5: "+neon",  # + NEON log-approx normalize         (--neon)
    6: "+dbuf",  # + double-buffered row-block read    (--prefetch)
    7: "+ent",  # + optimized rANS                    (--entropy)
}

# The two shaded bands the paper draws over the ladder (F2's job, not P1.0's — kept
# here as the SSOT for the rung→band split). Ranges are inclusive rung indices.
LADDER_BANDS: dict[str, tuple] = {"scheduling": (1, 4), "cpu kernels": (5, 7)}

# ======================================================================================
# Knee lanes — per-arch fan-out operating point (§4.0 gate review, 2026-09-01)
# ======================================================================================
KNEE: dict[str, int] = {"FP": 12, "SHyp": 24, "ResFP": 6, "ResSHyp": 20}

# ======================================================================================
# results/date27/ loaders — one per subdir, resolving the flag-encoded filenames
# ======================================================================================


def _load_json(path: Path) -> dict:
    """Load ``path`` as JSON, hard-erroring (with the attempted path) if it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"expected results/date27 file is missing: {path}")
    with open(path) as fh:
        return json.load(fh)


# ---- ladder/<arch>/ ------------------------------------------------------------------
# rung → filename-stem template ({k} = that arch's knee lane count, {mode} = warm|cold).
# TRAP (§4.1a): FP also has r4–r7 at 32 L for provenance only — resolving through
# KNEE["FP"] == 12 always takes the 12 L files, never the 32 L ones.
LADDER_STEMS: dict[int, str] = {
    0: "r0_seq_{mode}",
    1: "r1_p0_t4_warm",
    2: "r2_fo_t3_lanemaj_warm",
    3: "r3_fo_t3_warm",
    4: "r4_fo_t{k}_warm",
    5: "r5_fo_t{k}_neon_warm",
    6: "r6_fo_t{k}_neon_pf_warm",
    7: "r7_fo_t{k}_neon_pf_ent_warm",
}


def ladder_path(arch: str, rung: int, mode: str = "warm") -> Path:
    """Path to the ``arch`` ladder JSON for ``rung`` (r0 also takes ``mode`` = warm|cold)."""
    stem = LADDER_STEMS[rung].format(k=KNEE[arch], mode=mode)
    return DATE27 / "ladder" / arch / f"{stem}.json"


def load_ladder(arch: str, rung: int, mode: str = "warm") -> dict:
    """The ``arch`` ladder run JSON for ``rung``."""
    return _load_json(ladder_path(arch, rung, mode))


def ladder_series(arch: str, field: str = "median_patch_s") -> list[float]:
    """``field`` for rungs r0…r7 (warm basis), in order."""
    return [load_ladder(arch, r)[field] for r in RUNGS]


def cold_read_ceiling(arch: str) -> float:
    """SD cold-read throughput ceiling (patch/s) = 1000 / (read_ms / n_patches) from the r0 ``seq``
    *cold* run — the storage-bound floor if the tile were not page-cached (every warm rung sits
    above it)."""
    d = load_ladder(arch, 0, mode="cold")
    return 1000.0 / (d["read_ms"] / d["n_patches"])


# ---- lanes/<arch>/ -------------------------------------------------------------------
LANE_GRID: list[int] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]


def lane_json_path(arch: str, n: int) -> Path:
    """Path to the ``arch`` fan-out run JSON at ``n`` lanes (full stack, entropy-on)."""
    return DATE27 / "lanes" / arch / f"t{n}_fo_neon_pf_ent_warm.json"


def load_lane(arch: str, n: int) -> dict:
    """The ``arch`` fan-out run JSON at ``n`` lanes."""
    return _load_json(lane_json_path(arch, n))


def lane_trace_path(arch: str, n: int) -> Path:
    """Path to the ``arch`` subsampled occupancy trace CSV at ``n`` lanes."""
    return DATE27 / "lanes" / arch / f"t{n}_occtrace.csv"


def lane_counts(arch: str) -> list[int]:
    """Lane counts from LANE_GRID that have a run JSON on disk for this arch."""
    return [n for n in LANE_GRID if lane_json_path(arch, n).is_file()]


# ---- sequential per-stage timing -----------------------------------------------------
def load_seq_stages(arch: str) -> dict[str, float]:
    """**Canonical** per-patch sequential stage breakdown, in ms (P0.8).

    ``{read, patchify, normalize, g_a, h_a, h_s, entropy, write, dpu, residual, total}``
    from ``ladder/<arch>/r0_seq_warm.json`` — one instrument, the full 7,540-patch scene,
    the same run that anchors the ladder, and the stages close on ``total`` to within 0.2 %.

    Prefer this over ``load_s0_stages`` for anything sequential: it is the only source
    carrying ``read``/``patchify``/``write`` (together 1.0–3.4 % of per-patch time) and so
    the only one whose stages sum to the measured throughput.
    """
    return load_ladder(arch, 0)["stage_ms_per_patch"]


# ---- s0/<arch>/ ---------------------------------------------------------------------
def load_s0_stages(arch: str) -> dict[str, dict]:
    """Per-stage breakdown from ``benchmark_hardware`` s0, entropy OFF (E3).

    Returns the ``stages`` dict: ``{stage: {mean_ms, median_ms, …}}``.

    A **20-patch, 50-iteration subset of a different binary**, kept for its per-subgraph DPU
    detail and as an independent cross-check (it tracks ``load_seq_stages`` within
    −2.0…+2.4 %) — not as the sequential baseline. It has no read/patchify/write stage, so
    its stages sum to only 96–99 % of measured per-patch time.
    """
    return _load_json(DATE27 / "s0" / arch / "s0_compress_entoff.json")["stages"]


def load_xmodel_subgraphs(arch: str) -> dict[str, dict]:
    """Per-subgraph ops + byte counts (roofline, Tab.

    2) from the ``collect_roofline.py``
    export. Architecture properties — λ-independent.
    """
    hits = sorted((DATE27 / "s0" / arch).glob("*_xmodel_info.json"))
    if not hits:
        raise FileNotFoundError(f"no *_xmodel_info.json in {DATE27 / 's0' / arch}")
    return _load_json(hits[0])["subgraphs"]


# ---- vaitrace/<arch>/, occupancy/<arch>/, checks/<arch>/ ---------------------------
def vaitrace_path(arch: str, which: str) -> Path:
    """Per-core DPU-counter text dump. ``which`` selects the operating point.

    - ``"1lane"`` — one lane, uncontended. **The architecture-independent layer**: at
      one lane every arch measures identically (``g_a`` 89.5 % plain / 96.6 % residual,
      ``h_a`` 27.1 %, ``h_s`` 19.9 %), because efficiency there is a property of the
      subgraph alone. This is the roofline's characterization layer.
    - ``"r7_knee"`` — **the deployed configuration** (P0.9): fan-out at ``KNEE[arch]``
      lanes *with* all CPU optimizations, i.e. the r7 rung the paper reports everywhere
      else. Use this for the 3-core layer.
    - ``"r7_t3"`` — same stack at 3 lanes, one per core. The control that isolates lane
      count from the optimization stack.
    - ``"knee"`` — legacy P0.2/P0.7 capture, fan-out only (**no** CPU optimizations), so
      it sampled a lower DPU duty cycle than the system actually ships. Kept for
      provenance; prefer ``"r7_knee"``.
    """
    d = DATE27 / "vaitrace" / arch
    if which == "1lane":
        return d / "vaitrace_1lane.txt"
    if which == "r7_t3":
        return d / "vaitrace_r7_t3.txt"
    if which == "r7_knee":
        return d / f"vaitrace_r7_knee{KNEE[arch]}.txt"
    if which == "knee":
        exact = d / f"vaitrace_knee{KNEE[arch]}.txt"
        if exact.is_file():
            return exact
        hits = sorted(d.glob("vaitrace_knee*.txt"))
        if not hits:
            raise FileNotFoundError(f"no vaitrace_knee*.txt in {d}")
        return hits[0]
    raise ValueError(
        f"vaitrace_path: which must be '1lane', 'r7_knee', 'r7_t3' or 'knee', got {which!r}"
    )


def occupancy_full_trace_path(arch: str, rung: int) -> Path:
    """Full-scene occupancy trace at ladder rung 4 or 7 (the only two captured).

    Resolved at ``KNEE[arch]`` lanes when a lane-tagged capture exists
    (``r{rung}_full_t{K}.csv`` — FP's P0.7 re-trace at its 12 L knee), else the
    untagged ``r{rung}_full.csv`` (captured at the as-run knee, 32 L for FP).
    """
    if rung not in (4, 7):
        raise ValueError(f"occupancy/ only has r4 and r7 full-scene traces, not r{rung}")
    d = DATE27 / "occupancy" / arch
    tagged = d / f"r{rung}_full_t{KNEE[arch]}.csv"
    return tagged if tagged.is_file() else d / f"r{rung}_full.csv"


def checks_dir(arch: str) -> Path:
    """Directory of ``arch`` cheap-verification artifacts (sha256, SD cold read, 64 L RSS/CMA)."""
    return DATE27 / "checks" / arch


def cpu_probe_mpstat(arch: str) -> dict[str, float]:
    """Steady-state mean ``{usr, sys, idle}`` % of the 4 A53 cores at ``arch``'s knee lane (P0.7
    probe, ``cpu_probe/<arch>/knee_<K>_mpstat.log``).

    Only the knee was probed —
    there is no lane sweep (§4.0). ``all``-row samples, first/last 10 % trimmed as
    ramp/drain; reproduces the §4.0 gate-review table.
    """
    hits = sorted((DATE27 / "cpu_probe" / arch).glob("knee_*_mpstat.log"))
    if not hits:
        raise FileNotFoundError(f"no knee_*_mpstat.log in {DATE27 / 'cpu_probe' / arch}")
    rows = []
    with open(hits[0]) as fh:
        for line in fh:
            p = line.split()
            # "HH:MM:SS all %usr %nice %sys %iowait %irq %soft %steal %guest %gnice %idle"
            if len(p) >= 12 and p[1] == "all" and p[0][2] == ":":
                rows.append((float(p[2]), float(p[4]), float(p[11])))
    if len(rows) < 5:
        raise ValueError(f"{hits[0]}: only {len(rows)} mpstat 'all' rows — cannot trim")
    k = max(1, len(rows) // 10)
    win = rows[k:-k]
    n = len(win)
    return {
        "usr": sum(r[0] for r in win) / n,
        "sys": sum(r[1] for r in win) / n,
        "idle": sum(r[2] for r in win) / n,
    }


# ======================================================================================
# Saving
# ======================================================================================


def figures_dir() -> Path:
    """``REPO_ROOT/LaTeX/SAR_DDC_FPGA_DATE27/figures/images/``, or ``$DATE27_FIG_OUT``.

    Hard-errors if the directory does not exist — never creates it, never falls back to cwd (a
    figure silently written to the wrong place is worse than a failed run).
    """
    override = os.environ.get("DATE27_FIG_OUT")
    d = (
        Path(override)
        if override
        else REPO_ROOT / "LaTeX" / "SAR_DDC_FPGA_DATE27" / "figures" / "images"
    )
    if not d.is_dir():
        raise FileNotFoundError(
            f"figure output directory does not exist: {d}\n"
            f"  expected the DATE'27 manuscript repo checked out at "
            f"{REPO_ROOT / 'LaTeX' / 'SAR_DDC_FPGA_DATE27'}\n"
            f"  (override the location with the DATE27_FIG_OUT env var)"
        )
    return d


def save_figure(fig, name: str, **savefig_kw) -> Path:
    """Write ``name.pdf`` and ``name.png`` into the ``figures_dir()`` location.

    Returns that dir.
    """
    d = figures_dir()
    kw = {"bbox_inches": "tight", "dpi": 200}
    kw.update(savefig_kw)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", **kw)
    print(f"wrote {name}.{{pdf,png}} → {d}")
    return d
