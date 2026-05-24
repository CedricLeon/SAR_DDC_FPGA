#!/usr/bin/env python3
"""compare_benchmark_s0.py — Side-by-side Python vs C++ S0 benchmark comparison.

Usage:
    python scripts/fpga/compare_benchmark_s0.py \\
        --py  /tmp/bench_py_ResSHyp_compress_s0.json \\
        --cpp /tmp/bench_cpp_ResSHyp_compress_s0.json

Reads the JSON output of benchmark_fpga.py (--scenario compress --no-parallel)
and benchmark_hardware (--config s0 --scenario compress) and prints a
stage-by-stage latency comparison table with speedup factors.

Scenario auto-detected from the C++ JSON.  Supports compress and full.

Stage mapping (SHyp compress):
  C++ stage      |  Python stages aggregated
  ---------------|------------------------------------------
  normalize      |  preprocess
  g_a            |  dpu_g_a + cpu_concat_abs
  h_a            |  dpu_h_a
  eb_compress    |  cpu_eb_compress
  eb_decompress  |  cpu_eb_decompress
  h_s            |  dpu_h_s
  gc_compress    |  cpu_gc_compress

Additional stages (full scenario only):
  gc_decompress  |  cpu_gc_decompress
  g_s            |  dpu_g_s (sequential DPU runs, with --no-parallel)
  denorm         |  (no Python equivalent — Python does not time this separately)

[†] Python preprocess_patch() is called once before the benchmark loop and
    reused for all iterations; C++ stage_normalize runs per-iteration.
"""

from __future__ import annotations

import argparse
import json

# ---------------------------------------------------------------------------
# Stage name mapping (C++ stage → list of Python stage keys to sum)
# None means no Python equivalent (shows "—" in that column).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FP (FactorizedPrior) mappings — no h_a/h_s/GC; g_a has no cpu_concat_abs;
# g_s has no cpu_split_y_hat (Python FP path doesn't time that separately).
# ---------------------------------------------------------------------------

CPP_TO_PY_FP_COMPRESS: dict[str, list[str] | None] = {
    "normalize": ["preprocess"],
    "g_a": ["dpu_g_a"],  # FP: no |y| computation, no cpu_concat_abs
    "eb_compress": ["cpu_eb_compress"],
}

CPP_TO_PY_FP_FULL: dict[str, list[str] | None] = {
    "normalize": ["preprocess"],
    "g_a": ["dpu_g_a"],
    "eb_compress": ["cpu_eb_compress"],
    "eb_decompress": ["cpu_eb_decompress"],
    "g_s": ["dpu_g_s"],  # FP: no cpu_split_y_hat in Python timing
    "denorm": ["postprocess"],
}

CPP_ORDER_FP_COMPRESS = ["normalize", "g_a", "eb_compress"]
CPP_ORDER_FP_FULL = ["normalize", "g_a", "eb_compress", "eb_decompress", "g_s", "denorm"]

# ---------------------------------------------------------------------------
# SHyp (ScaleHyperprior) mappings
# ---------------------------------------------------------------------------

CPP_TO_PY_SHYP_COMPRESS: dict[str, list[str] | None] = {
    "normalize": ["preprocess"],
    "g_a": ["dpu_g_a", "cpu_concat_abs"],
    "h_a": ["dpu_h_a"],
    "eb_compress": ["cpu_eb_compress"],
    "eb_decompress": ["cpu_eb_decompress"],
    "h_s": ["dpu_h_s"],
    "gc_compress": ["cpu_gc_compress"],
}

CPP_TO_PY_SHYP_FULL: dict[str, list[str] | None] = {
    "normalize": ["preprocess"],
    "g_a": ["dpu_g_a", "cpu_concat_abs"],
    "h_a": ["dpu_h_a"],
    "eb_compress": ["cpu_eb_compress"],
    "eb_decompress": ["cpu_eb_decompress"],
    "h_s": ["dpu_h_s"],
    "gc_compress": ["cpu_gc_compress"],
    "gc_decompress": ["cpu_gc_decompress"],
    # C++ stage_gs = deinterleave + 2×DPU g_s runs + pack → matches cpu_split_y_hat + dpu_g_s
    "g_s": ["cpu_split_y_hat", "dpu_g_s"],
    # C++ denorm ≈ Python postprocess (0.005 ms vs C++ ~10 ms — Python caches log-amp, so negligible)
    "denorm": ["postprocess"],
}

# C++ stage pipeline order per scenario
CPP_ORDER_COMPRESS = [
    "normalize",
    "g_a",
    "h_a",
    "eb_compress",
    "eb_decompress",
    "h_s",
    "gc_compress",
]
CPP_ORDER_FULL = [
    "normalize",
    "g_a",
    "h_a",
    "eb_compress",
    "eb_decompress",
    "h_s",
    "gc_compress",
    "gc_decompress",
    "g_s",
    "denorm",
]

# Stages whose C++ vs Python comparison is apples-to-apples
EXACT_MATCH_STAGES = {
    "g_a",
    "h_a",
    "eb_compress",
    "eb_decompress",
    "h_s",
    "gc_compress",
    "gc_decompress",
    "g_s",
}

# DPU stage sets (for DPU vs CPU summary)
DPU_STAGES_CPP = {"g_a", "h_a", "h_s", "g_s"}
DPU_STAGES_PY = {"dpu_g_a", "dpu_h_a", "dpu_h_s", "dpu_g_s"}

# ---------------------------------------------------------------------------
# Power helpers
# ---------------------------------------------------------------------------

POWER_GROUPS_DISPLAY = ["DPU_fabric", "PL", "PS", "MPSoC", "peripherals"]

# Used to recompute Python idle groups from per-rail INA226 data
POWER_GROUPS_RAILS: dict[str, list[str]] = {
    "PL": ["VCCINT", "VCCBRAM", "VCCAUX", "VCC1V2", "VCC3V3"],
    "PS": [
        "VCCPSINTFP",
        "VCCPSINTLP",
        "VCCPSAUX",
        "VCCPSPLL",
        "VCCO_PSDDR_504",
        "VCCOPS",
        "VCCOPS3",
        "VCCPSDDRPLL",
    ],
    "DPU_fabric": ["VCCINT", "VCCBRAM"],
    "PS_compute": ["VCCPSINTFP", "VCCPSINTLP"],
}


def extract_py_power(data: dict) -> tuple[dict, dict]:
    """Return (idle_groups_w, active_groups_w) from a Python benchmark JSON.

    Python JSON layout:
      power.groups_avg_w        — active group totals (already aggregated)
      power.idle_baseline       — idle per-rail INA226 {rail: {avg_power_w, ...}}
      power.idle_pmbus_rails    — idle PMBus {rail: {avg_power_w, ...}}
    """
    pdata = data.get("power")
    if not pdata:
        return {}, {}

    active_groups = dict(pdata.get("groups_avg_w", {}))

    # Recompute idle groups from per-rail data
    idle_rails = pdata.get("idle_baseline", {})
    idle_groups: dict = {}
    for group, members in POWER_GROUPS_RAILS.items():
        total = sum(idle_rails.get(r, {}).get("avg_power_w", 0.0) for r in members)
        if total > 0.0:
            idle_groups[group] = total
    if "PL" in idle_groups and "PS" in idle_groups:
        idle_groups["MPSoC"] = idle_groups["PL"] + idle_groups["PS"]
    idle_pmbus = pdata.get("idle_pmbus_rails", {})
    if idle_pmbus:
        idle_groups["peripherals"] = sum(v.get("avg_power_w", 0.0) for v in idle_pmbus.values())

    return idle_groups, active_groups


def extract_cpp_power(data: dict) -> tuple[dict, dict]:
    """Return (idle_groups_w, active_groups_w) from a C++ benchmark JSON.

    C++ JSON layout:   power.idle.groups   — idle group totals   power.active.groups — active group
    totals
    """
    pdata = data.get("power")
    if not pdata:
        return {}, {}
    idle_groups = dict(pdata.get("idle", {}).get("groups", {}))
    active_groups = dict(pdata.get("active", {}).get("groups", {}))
    return idle_groups, active_groups


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------


def py_mean_ms(data: dict, stage: str) -> float | None:
    """Return mean latency in ms for a Python stage, or None if not found."""
    lb = data.get("latency_breakdown", {})
    if stage not in lb:
        return None
    return lb[stage]["mean_s"] * 1000.0


def py_mean_ms_sum(data: dict, stages: list[str]) -> float | None:
    """Return sum of mean latencies in ms for a list of Python stages, or None if any not found."""
    total = 0.0
    for s in stages:
        v = py_mean_ms(data, s)
        if v is None:
            return None
        total += v
    return total


def cpp_mean_ms(data: dict, stage: str) -> float | None:
    """Return mean latency in ms for a C++ stage, or None if not found."""
    stages = data.get("stages", {})
    if stage not in stages:
        return None
    return stages[stage]["mean_ms"]


def speedup(py_ms: float | None, cpp_ms: float | None) -> str:
    """Return a formatted speedup string (e.g. "3.25×") comparing Python vs C++ latencies, or "—"
    if not computable."""
    if py_ms is None or cpp_ms is None or cpp_ms <= 0:
        return "  —  "
    return f"{py_ms / cpp_ms:5.2f}×"


# ---------------------------------------------------------------------------
# Print sections
# ---------------------------------------------------------------------------


def print_latency_section(
    py_data: dict,
    cpp_data: dict,
    stage_map: dict[str, list[str] | None],
    cpp_ordered: list[str],
    cpp_stages_present: set[str],
) -> None:
    """Print the latency comparison section, showing Python vs C++ mean latencies for each stage
    and speedup factors."""
    hdr = f"  {'Stage':<16} {'Python (ms)':>12} {'C++ (ms)':>10} {'Speedup':>8}  {'Note'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cpp_stage in cpp_ordered:
        if cpp_stage not in cpp_stages_present:
            continue
        py_stages = stage_map.get(cpp_stage)
        if py_stages is None:
            py_ms = None
        elif isinstance(py_stages, list):
            py_ms = py_mean_ms_sum(py_data, py_stages)
        else:
            py_ms = None

        c_ms = cpp_mean_ms(cpp_data, cpp_stage)

        py_str = f"{py_ms:>10.2f}" if py_ms is not None else "         —"
        cpp_str = f"{c_ms:>10.2f}" if c_ms is not None else "         —"

        note = ""
        if cpp_stage not in EXACT_MATCH_STAGES:
            if cpp_stage == "normalize":
                note = "[†] Python preprocess_patch() called once before loop — not timed per-iteration"
            elif cpp_stage == "denorm":
                note = (
                    "[‡] Python 'postprocess' ~0.005 ms (log-amp cached); C++ ~10 ms (recomputed)"
                )

        spd = speedup(py_ms, c_ms)
        print(f"  {cpp_stage:<16} {py_str}  {cpp_str}  {spd}  {note}")

    # Total
    py_total = py_data.get("latency_total_mean_ms")
    cpp_total = cpp_data.get("total_latency_mean_ms")
    print("  " + "-" * (len(hdr) - 2))
    py_t_str = f"{py_total:>10.2f}" if py_total is not None else "         —"
    cpp_t_str = f"{cpp_total:>10.2f}" if cpp_total is not None else "         —"
    print(f"  {'TOTAL':<16} {py_t_str}  {cpp_t_str}  {speedup(py_total, cpp_total)}")
    print()

    # DPU vs CPU split
    print("  --- DPU vs CPU split ---")
    py_lb = py_data.get("latency_breakdown", {})
    py_dpu = sum(py_lb[s]["mean_s"] * 1000.0 for s in DPU_STAGES_PY if s in py_lb)
    cpp_stages_d = cpp_data.get("stages", {})
    cpp_dpu = sum(cpp_stages_d[s]["mean_ms"] for s in DPU_STAGES_CPP if s in cpp_stages_d)
    py_cpu = (py_total or 0.0) - py_dpu
    cpp_cpu = (cpp_total or 0.0) - cpp_dpu
    print(f"  {'DPU total':<16} {py_dpu:>10.2f}  {cpp_dpu:>10.2f}  {speedup(py_dpu, cpp_dpu)}")
    print(f"  {'CPU total':<16} {py_cpu:>10.2f}  {cpp_cpu:>10.2f}  {speedup(py_cpu, cpp_cpu)}")
    print()

    # Throughput
    py_fps = py_data.get("throughput_fps")
    cpp_fps = cpp_data.get("throughput_fps")
    print("  --- Throughput ---")
    if py_fps is not None:
        print(f"  Python : {py_fps:.2f} fps  ({1000 / py_fps:.1f} ms/patch)")
    if cpp_fps is not None:
        print(f"  C++    : {cpp_fps:.2f} fps  ({1000 / cpp_fps:.1f} ms/patch)")
    if py_fps and cpp_fps:
        print(f"  Speedup: {cpp_fps / py_fps:.2f}× throughput")
    print()


def print_raw_breakdowns(py_data: dict, cpp_data: dict, cpp_ordered: list[str]) -> None:
    """Print raw stage breakdowns for Python and C++ if available, showing all stages in descending
    order of latency with percentage of total time."""
    py_lb = py_data.get("latency_breakdown", {})
    if py_lb:
        print("  --- Python raw stage breakdown (all stages) ---")
        py_all = sum(v["mean_s"] * 1000.0 for k, v in py_lb.items() if not k.startswith("_"))
        for stage, stats in sorted(py_lb.items(), key=lambda x: -x[1]["mean_s"]):
            if stage.startswith("_"):
                continue
            ms = stats["mean_s"] * 1000.0
            pct = ms / py_all * 100 if py_all > 0 else 0
            print(f"    {stage:<26} {ms:>8.2f} ms  ({pct:4.1f}%)")
        print()

    cpp_stages = cpp_data.get("stages", {})
    if cpp_stages:
        print("  --- C++ raw stage breakdown (all stages) ---")
        cpp_all = sum(v["mean_ms"] for v in cpp_stages.values())
        for stage in cpp_ordered:
            if stage not in cpp_stages:
                continue
            ms = cpp_stages[stage]["mean_ms"]
            pct = ms / cpp_all * 100 if cpp_all > 0 else 0
            print(f"    {stage:<26} {ms:>8.2f} ms  ({pct:4.1f}%)")
        print()


def print_power_section(py_data: dict, cpp_data: dict) -> None:
    """Print the power comparison section, showing idle vs active power for key groups and energy
    per patch estimates if available."""
    py_idle, py_active = extract_py_power(py_data)
    cpp_idle, cpp_active = extract_cpp_power(cpp_data)

    if not py_active and not cpp_active:
        print("  (no power data — rerun with --power --idle-baseline 10)")
        print()
        return

    hdr = (
        f"  {'Group':<14} "
        f"{'Py idle':>8} {'Py actv':>8} {'Py Δ':>7}  "
        f"{'C++ idle':>8} {'C++ actv':>9} {'C++ Δ':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for group in POWER_GROUPS_DISPLAY:
        pi = py_idle.get(group)
        pa = py_active.get(group)
        ci = cpp_idle.get(group)
        ca = cpp_active.get(group)

        def fmt(v: float | None, w: int) -> str:
            """Simple helper to format a power value in watts with 3 decimal places, or "—" if
            None, padded to width w."""
            return f"{v:{w}.3f}" if v is not None else " " * (w - 1) + "—"

        pd = (pa - pi) if pa is not None and pi is not None else None
        cd = (ca - ci) if ca is not None and ci is not None else None

        print(
            f"  {group:<14} "
            f"{fmt(pi, 8)} {fmt(pa, 8)} {fmt(pd, 7)}  "
            f"{fmt(ci, 8)} {fmt(ca, 9)} {fmt(cd, 7)}"
        )
    print()

    # Energy per patch (MPSoC active × mean latency)
    py_total = py_data.get("latency_total_mean_ms")
    cpp_total = cpp_data.get("total_latency_mean_ms")
    py_mpsoc = py_active.get("MPSoC")
    cpp_mpsoc = cpp_active.get("MPSoC")

    print("  --- Energy per patch (MPSoC active power × mean latency) ---")
    py_ej = cpp_ej = None
    if py_mpsoc and py_total:
        py_ej = py_mpsoc * (py_total / 1000.0)
        print(f"  Python : {py_ej * 1000:.2f} mJ  ({py_mpsoc:.3f} W × {py_total:.1f} ms)")
    if cpp_mpsoc and cpp_total:
        cpp_ej = cpp_mpsoc * (cpp_total / 1000.0)
        print(f"  C++    : {cpp_ej * 1000:.2f} mJ  ({cpp_mpsoc:.3f} W × {cpp_total:.1f} ms)")
    if py_ej and cpp_ej:
        print(
            f"  Savings: {(py_ej - cpp_ej) * 1000:.2f} mJ/patch  ({py_ej / cpp_ej:.2f}× less energy)"
        )
    print()

    # VCCINT deep-dive (main DPU/fabric supply)
    print("  --- VCCINT (main DPU supply) ---")
    py_rails = py_data.get("power", {})
    cpp_rails = cpp_data.get("power", {})

    def vccint_w(data_power: dict, key_active: str, key_rails: str) -> float | None:
        """Helper to extract VCCINT power from either Python or C++ power data structures."""
        per_rail = data_power.get(key_active, {})
        if isinstance(per_rail, dict):
            entry = per_rail.get("VCCINT")
            if isinstance(entry, dict):
                return entry.get("avg_power_w")
        return None

    py_vcc_idle = py_rails.get("idle_baseline", {}).get("VCCINT", {}).get("avg_power_w")
    py_vcc_active = py_rails.get("per_rail", {}).get("VCCINT", {}).get("avg_power_w")
    cpp_vcc_idle = cpp_rails.get("idle", {}).get("rails", {}).get("VCCINT", {}).get("avg_power_w")
    cpp_vcc_active = (
        cpp_rails.get("active", {}).get("rails", {}).get("VCCINT", {}).get("avg_power_w")
    )

    def vfmt(v: float | None) -> str:
        """Format a power value in watts with 3 decimal places, or "—" if None."""
        return f"{v:.3f} W" if v is not None else "    —"

    print(f"  Python  idle:   {vfmt(py_vcc_idle)}")
    print(f"  Python  active: {vfmt(py_vcc_active)}")
    print(f"  C++     idle:   {vfmt(cpp_vcc_idle)}")
    print(f"  C++     active: {vfmt(cpp_vcc_active)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--py", required=True, help="Python benchmark JSON")
    parser.add_argument("--cpp", required=True, help="C++ benchmark JSON")
    args = parser.parse_args()

    with open(args.py) as f:
        py_data = json.load(f)
    with open(args.cpp) as f:
        cpp_data = json.load(f)

    py_scenario = py_data.get("scenario", "?")
    cpp_scenario = cpp_data.get("scenario", "?")
    py_iters = py_data.get("n_iters", "?")
    cpp_iters = cpp_data.get("iters", "?")
    cpp_arch = cpp_data.get("arch", "?")

    print()
    print("=" * 72)
    print("  Python vs C++  —  S0 sequential benchmark comparison")
    print("=" * 72)
    print(f"  Python : scenario={py_scenario}  iters={py_iters}  parallel=OFF")
    print(f"  C++    : scenario={cpp_scenario}  iters={cpp_iters}  arch={cpp_arch}")
    print()

    # Choose stage map based on arch and scenario
    is_full = cpp_scenario == "full"
    is_fp = cpp_arch in ("FP", "ResFP")
    if is_fp:
        stage_map = CPP_TO_PY_FP_FULL if is_full else CPP_TO_PY_FP_COMPRESS
        cpp_ordered = CPP_ORDER_FP_FULL if is_full else CPP_ORDER_FP_COMPRESS
    else:
        stage_map = CPP_TO_PY_SHYP_FULL if is_full else CPP_TO_PY_SHYP_COMPRESS
        cpp_ordered = CPP_ORDER_FULL if is_full else CPP_ORDER_COMPRESS

    # Extend ordered list with any unexpected stages from the C++ output
    cpp_stages_present = set(cpp_data.get("stages", {}).keys())
    for s in sorted(cpp_stages_present):
        if s not in cpp_ordered:
            cpp_ordered = list(cpp_ordered) + [s]

    # -----------------------------------------------------------------------
    print("  --- Latency (ms) ---")
    print_latency_section(py_data, cpp_data, stage_map, cpp_ordered, cpp_stages_present)

    # -----------------------------------------------------------------------
    print("  --- DPU+entropy only (normalize excluded for fair comparison) ---")
    cpp_dpu_entropy = sum(
        cpp_mean_ms(cpp_data, s) or 0.0 for s in cpp_stages_present if s != "normalize"
    )
    py_total = py_data.get("latency_total_mean_ms", 0.0)
    if cpp_dpu_entropy > 0 and py_total > 0:
        print(f"  {'Python total':<22} {py_total:>10.2f} ms  (normalize not timed)")
        print(f"  {'C++ DPU+entropy':<22} {cpp_dpu_entropy:>10.2f} ms  (normalize excluded)")
        print(f"  {'Speedup':<22} {py_total / cpp_dpu_entropy:>10.2f}×")
    print()

    # -----------------------------------------------------------------------
    print_raw_breakdowns(py_data, cpp_data, cpp_ordered)

    # -----------------------------------------------------------------------
    print("  --- Power ---")
    print_power_section(py_data, cpp_data)

    # -----------------------------------------------------------------------
    print("  [†] normalize: Python calls preprocess_patch() (numpy vectorized log) once")
    print("      before the benchmark loop; C++ stage_normalize runs per-iteration (~9 ms).")
    print("  [‡] denorm: C++ denorm_to_lina runs per-iteration (~9-10 ms); Python times")
    print("      this as 'postprocess' (~0.005 ms) because log-amplitude is cached from")
    print("      the DPU output — no recomputation. The C++ cost is a real bottleneck.")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
