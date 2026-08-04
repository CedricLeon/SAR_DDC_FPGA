#!/usr/bin/env python3
"""rescore_overlap_tiles.py — re-score the saved overlap-study tiles with the coherent,
``AMP_LIN_99``-clipped, seam-band metrics and rebuild the overlap table.

The stitched full-scene tiles in ``results/benchmark_stream_overlap/_work/*_tile.npy`` were scored at
stitch time with an earlier metric convention (unclipped SSIM/EPD, no seam split). This re-scores
them all against the MERLIN full-tile GT — loaded **once** — via ``stitch_ddc.score_arrays``, then:

  * overwrites each ``ov{N}{_warm}_quality.json`` with the new metric set;
  * rebuilds each combined ``ov{N}{_warm}.json``, also fixing the stale, overlap-inflated
    ``slc_mb_s`` to ``scene_bytes / median_total_s`` (scene_bytes = H*W*4, overlap-independent);
  * writes ``overlap_table.md`` + ``overlap_table.csv`` (full / seam / interior PSNR & SSIM + cost).

The cost fields (bpp, latency, power, energy, thermal) come from the saved ``*_compress.json`` and
are NOT recomputed — only the GT-scored quality and the SLC ingest rate change. See §10.

    python scripts/evaluation/rescore_overlap_tiles.py             # re-score all tiles + table
    python scripts/evaluation/rescore_overlap_tiles.py --table-only    # rebuild table from jsons

Sequential and memory-heavy (~20 GB / tile for the scipy SSIM map, ~90 s each) — run detached.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.evaluation.stitch_ddc import score_arrays  # noqa: E402

ROOT = REPO_ROOT / "results" / "benchmark_stream_overlap"
WORK = ROOT / "_work"
GT = REPO_ROOT / "data" / "visualization" / "MERLIN" / "linA_MERLIN_full_Hamburg.npy"
TILE_RE = re.compile(r"^(.+)_ov(\d+)(_warm)?_tile\.npy$")
MODEL_RE = re.compile(r"^(\w+)-relu_s\d+_L(\d+)_pt$")


def parse_tile(name: str):
    """(model, overlap:int, warm:bool) from a ``*_tile.npy`` filename, or None if it doesn't
    match."""
    m = TILE_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), bool(m.group(3))


def rescore_all() -> None:
    """Score every saved tile against the GT (loaded once); rewrite quality + combined JSONs."""
    if not GT.exists():
        raise SystemExit(f"MERLIN GT not found: {GT}")
    tiles = sorted(WORK.glob("*_tile.npy"))
    if not tiles:
        raise SystemExit(f"no tiles in {WORK}")
    print(f"[rescore] loading GT {GT.name} ...", flush=True)
    gt = np.load(GT).astype(np.float32)
    scene_bytes = (
        int(gt.size) * 4
    )  # focused SLC payload H*W*(I+Q)*int16 = H*W*4, overlap-independent
    print(f"[rescore] GT {gt.shape}  scene_bytes={scene_bytes}  {len(tiles)} tiles\n", flush=True)

    for i, tile in enumerate(tiles, 1):
        parsed = parse_tile(tile.name)
        if not parsed:
            print(f"[rescore] SKIP unrecognised {tile.name}", flush=True)
            continue
        model, ov, warm = parsed
        sfx = "_warm" if warm else ""
        mdir = ROOT / model
        comp_path = mdir / f"ov{ov}{sfx}_compress.json"
        if not comp_path.exists():
            print(f"[rescore] SKIP {tile.name}: no {comp_path.name}", flush=True)
            continue

        recon = np.load(tile)
        q = score_arrays(recon, gt)
        del recon
        q["overlap"] = ov
        (mdir / f"ov{ov}{sfx}_quality.json").write_text(json.dumps(q, indent=2))

        comp = json.loads(comp_path.read_text())
        med = comp["median_total_s"]
        arch, lam = MODEL_RE.match(model).groups()
        combined = {
            "model": model,
            "arch": arch,
            "lambda": int(lam),
            "overlap": ov,
            "warm": warm,
            "n_patches": comp["n_patches"],
            "bpp": comp["bpp"],
            "median_total_s": med,
            "median_patch_s": comp["median_patch_s"],
            "scene_bytes": scene_bytes,
            "slc_mb_s": scene_bytes / med / 1e6 if med > 0 else 0.0,
            "avg_power_w": comp.get("avg_power_w"),
            "energy_j": comp.get("energy_j"),
            "j_per_patch": comp.get("j_per_patch"),
            "thermal": comp.get("thermal"),
            "quality": q,
        }
        (mdir / f"ov{ov}{sfx}.json").write_text(json.dumps(combined, indent=2))
        print(
            f"[rescore] {i:2d}/{len(tiles)} {model} ov{ov}{sfx}: "
            f"PSNR full={q['psnr']:.2f} seam={q['psnr_seam']:.2f} int={q['psnr_interior']:.2f} | "
            f"SSIM full={q['ssim']:.4f} seam={q['ssim_seam']:.4f} int={q['ssim_interior']:.4f}",
            flush=True,
        )
    print("\n[rescore] done — building table", flush=True)


def build_table() -> None:
    """Aggregate the combined ``ov{N}{_warm}.json`` files into overlap_table.{md,csv}."""
    rows = []
    for cj in sorted(ROOT.glob("*/ov*.json")):
        if cj.name.endswith(("_compress.json", "_quality.json")):
            continue
        d = json.loads(cj.read_text())
        q = d["quality"]
        rows.append(
            {
                "arch": d["arch"],
                "lambda": d["lambda"],
                "mode": "warm" if d.get("warm") else "cold",
                "overlap": d["overlap"],
                "bpp": d["bpp"],
                "psnr": q["psnr"],
                "psnr_seam": q["psnr_seam"],
                "psnr_interior": q["psnr_interior"],
                "ssim": q["ssim"],
                "ssim_seam": q["ssim_seam"],
                "ssim_interior": q["ssim_interior"],
                "seam_deficit": q["psnr_interior"] - q["psnr_seam"],
                "total_s": d["median_total_s"],
                "slc_mb_s": d["slc_mb_s"],
                "avg_power_w": d.get("avg_power_w"),
                "j_per_patch": d.get("j_per_patch"),
            }
        )
    rows.sort(key=lambda r: (r["arch"], -r["lambda"], r["mode"], r["overlap"]))

    csv_path = ROOT / "overlap_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = (
        "| ov | bpp | PSNR | PSNR seam | PSNR int | seam Δ | SSIM | SSIM seam | SSIM int | "
        "s | MB/s | W | J/patch |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [
        "# Overlap study — reconstructed-tile quality vs cost (coherent AMP_LIN_99 metrics)",
        "",
        "Full-scene Hamburg tile, board INT8 decode, scored vs the project MERLIN full-tile GT.",
        "`seam Δ` = PSNR_interior − PSNR_seam (dB the seam is worse than the interior — what overlap",
        "buys). All PSNR/SSIM clipped to AMP_LIN_99. Cost from the streaming compress run (§8).",
        "",
    ]

    def fmt(v, p):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"

    last = None
    for r in rows:
        grp = (r["arch"], r["lambda"], r["mode"])
        if grp != last:
            lam_note = "" if r["mode"] == "cold" else " (warm)"
            lines += ["", f"## {r['arch']}  λ={r['lambda']}{lam_note}", "", hdr, sep]
            last = grp
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["overlap"]),
                    fmt(r["bpp"], 4),
                    fmt(r["psnr"], 2),
                    fmt(r["psnr_seam"], 2),
                    fmt(r["psnr_interior"], 2),
                    fmt(r["seam_deficit"], 2),
                    fmt(r["ssim"], 4),
                    fmt(r["ssim_seam"], 4),
                    fmt(r["ssim_interior"], 4),
                    fmt(r["total_s"], 1),
                    fmt(r["slc_mb_s"], 2),
                    fmt(r["avg_power_w"], 1),
                    fmt(r["j_per_patch"], 4),
                ]
            )
            + " |"
        )
    (ROOT / "overlap_table.md").write_text("\n".join(lines) + "\n")
    print(
        f"[rescore] wrote {csv_path.relative_to(REPO_ROOT)} and overlap_table.md ({len(rows)} rows)"
    )


def main() -> None:
    """Re-score all tiles (unless --table-only) then rebuild the table."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--table-only", action="store_true", help="skip scoring; rebuild table from JSONs"
    )
    args = ap.parse_args()
    if not args.table_only:
        rescore_all()
    build_table()


if __name__ == "__main__":
    sys.exit(main())
