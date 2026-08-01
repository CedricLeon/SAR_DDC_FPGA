#!/usr/bin/env python3
"""stream_table.py — build the optimization ablation table(s) from stream_sweep.py JSON results.

One markdown table per lambda: rows = the cumulative optimization ladder (seq -> +s1 -> +p0 ->
+prefetch -> +neon), columns = throughput "patch/s (SLC MB/s)" and full-tile latency, for FP and
ResSHyp. The warm (+neon) ceiling and bpp are noted below each table. Prints to stdout and writes
results/benchmark_stream/ablation_table.md.
"""

import argparse
import json

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
RESULTS = REPO_ROOT / "results" / "benchmark_stream"

# Cumulative ladder: (display, flag signature to match a result JSON).
ROWS = [
    ("seq", dict(schedule="seq", s1=False, prefetch=False, neon=False)),
    ("+ s1", dict(schedule="seq", s1=True, prefetch=False, neon=False)),
    ("+ p0", dict(schedule="p0", s1=True, prefetch=False, neon=False)),
    ("+ prefetch", dict(schedule="p0", s1=True, prefetch=True, neon=False)),
    ("+ neon", dict(schedule="p0", s1=True, prefetch=True, neon=True)),
]
ARCHS = ["FP", "ResSHyp"]


def parse_model(name: str):
    """'FP-relu_s0_L1000_pt' -> ('FP', 1000)."""
    return name.split("-")[0], int(name.split("_L")[1].split("_")[0])


def match(d: dict, sig: dict, cold: bool = True) -> bool:
    """True if result d has the given flag signature and cold/warm mode."""
    return (
        d["schedule"] == sig["schedule"]
        and d["s1"] == sig["s1"]
        and d.get("prefetch", False) == sig["prefetch"]
        and d.get("neon", False) == sig["neon"]
        and d["cold"] == cold
    )


def main():
    """Entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    recs = [json.loads(f.read_text()) for f in RESULTS.rglob("*.json")]
    if not recs:
        raise SystemExit(f"no results under {RESULTS} — run stream_sweep.py first")
    lambdas = sorted({parse_model(d["model_name"])[1] for d in recs})

    def find(arch, lam, sig, cold=True):
        hits = (r for r in recs if parse_model(r["model_name"]) == (arch, lam))
        return next((r for r in hits if match(r, sig, cold)), None)

    def tput(d):
        return f"{d['median_patch_s']:.1f} ({d['slc_mb_s']:.1f})" if d else "—"

    def lat(d):
        return f"{d['median_total_s'] / 60:.2f} min" if d else "—"

    out = ["# Streaming optimization ablation (full scene, cold)\n"]
    for lam in lambdas:
        out.append(f"## λ = {lam}\n")
        out.append(
            "| optimization | FP patch/s (MB/s) | FP latency | "
            "ResSHyp patch/s (MB/s) | ResSHyp latency |"
        )
        out.append("| --- | --- | --- | --- | --- |")
        for disp, sig in ROWS:
            c = []
            for arch in ARCHS:
                d = find(arch, lam, sig)
                c += [tput(d), lat(d)]
            out.append(f"| {disp} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
        notes = []
        for arch in ARCHS:
            w = find(arch, lam, ROWS[-1][1], cold=False)
            if w:
                notes.append(
                    f"{arch} warm (+neon) ceiling: {w['median_patch_s']:.1f} patch/s "
                    f"({w['slc_mb_s']:.1f} MB/s), {w['median_total_s'] / 60:.2f} min"
                )
            d = find(arch, lam, ROWS[0][1])
            if d:
                notes.append(f"{arch} bpp={d['bpp']:.3f} (n={d['n_patches']})")
        out.append("\n" + "  \n".join(notes) + "\n")

    text = "\n".join(out)
    print(text)
    dest = RESULTS / "ablation_table.md"
    dest.write_text(text)
    print(f"-> {dest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
