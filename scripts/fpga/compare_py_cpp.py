#!/usr/bin/env python3
"""compare_py_cpp.py — Compare Python and C++ inference outputs patch-by-patch.

Usage:
    python scripts/fpga/compare_py_cpp.py --py <py_compare_dir> --cpp <cpp_compare_dir>

Both directories must have been produced with --compare-out:
  Python:  python inference_hybrid.py ... --compare-out <py_dir>
  C++:     ./inference_hybrid        ... --compare-out <cpp_dir>

Each directory contains:
  per_patch.json               {"n": N, "patches": [{"bpp": ...}, ...]}
  patch_0000_recon_linA.npy    shape [H, W] float32
  patch_0001_recon_linA.npy
  ...

Pass criteria:
  ALL | bpp_diff |   < 1e-6   (same rANS byte count — must be exact)
  ALL max|px_diff| < 0.5    (default; float32 save-precision noise is < 0.001, real bugs show > 9)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_BYTES_PER_PIXEL = 8 / (256 * 256)  # 1 byte expressed in BPP for a 256x256 patch


def load_dir(p: Path):
    with open(p / "per_patch.json") as f:
        meta = json.load(f)
    n = meta["n"]
    bpps = [patch["bpp"] for patch in meta["patches"]]
    z_bytes = [patch.get("z_bytes", 0) for patch in meta["patches"]]
    y_bytes = [patch.get("y_bytes", 0) for patch in meta["patches"]]
    psnr_merlin = [patch.get("psnr_merlin", None) for patch in meta["patches"]]
    psnr_adam = [patch.get("psnr_adam", None) for patch in meta["patches"]]
    arrays = [np.load(p / f"patch_{i:04d}_recon_linA.npy") for i in range(n)]
    return n, bpps, z_bytes, y_bytes, psnr_merlin, psnr_adam, arrays


def _pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p))


def _array_stats(a: np.ndarray) -> str:
    return f"min={a.min():.3f}  mean={a.mean():.3f}  max={a.max():.3f}  std={a.std():.3f}"


def _diff_stats(diff: np.ndarray) -> str:
    return (
        f"mean={diff.mean():.3f}  "
        f"p50={_pct(diff, 50):.3f}  p95={_pct(diff, 95):.3f}  "
        f"p99={_pct(diff, 99):.3f}  max={diff.max():.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare Python vs C++ inference outputs.")
    parser.add_argument(
        "--py", required=True, metavar="DIR", help="Python --compare-out directory"
    )
    parser.add_argument("--cpp", required=True, metavar="DIR", help="C++ --compare-out directory")
    parser.add_argument(
        "--bpp-tol",
        type=float,
        default=1e-6,
        metavar="F",
        help="Max allowed |bpp_py - bpp_cpp| [default: 1e-6]",
    )
    parser.add_argument(
        "--pixel-tol",
        type=float,
        default=0.5,
        metavar="F",
        help="Max allowed pixel |diff| in linear amplitude [default: 0.5]",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-array stats and signed diff histogram for each patch",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=None,
        metavar="I",
        help="Deep-dive a single patch index (implies --verbose for that patch)",
    )
    args = parser.parse_args()

    py_dir = Path(args.py)
    cpp_dir = Path(args.cpp)

    print(f"Python dir : {py_dir}")
    print(f"C++ dir    : {cpp_dir}")
    print()

    n_py, bpps_py, z_py, y_py, pm_py, pa_py, arrs_py = load_dir(py_dir)
    n_cpp, bpps_cpp, z_cpp, y_cpp, pm_cpp, pa_cpp, arrs_cpp = load_dir(cpp_dir)

    if n_py != n_cpp:
        print(f"ERROR: patch count mismatch: Python={n_py}, C++={n_cpp}")
        sys.exit(1)
    N = n_py
    global_peak = max(a.max() for a in arrs_py)
    print(f"Comparing {N} patches  (global_peak={global_peak:.1f})\n")

    # ------------------------------------------------------------------ #
    # Per-patch table
    bpp_pass = True
    pixel_pass = True
    bpp_diffs = []
    px_diffs = []
    psnrs = []
    z_diffs = []
    y_diffs = []
    delta_psnr_merlin = []

    has_zy = any(z > 0 for z in z_py)
    has_merlin = any(v is not None for v in pm_py) and any(v is not None for v in pm_cpp)
    col = (
        f"{'patch':>6}  {'bpp_py':>9}  {'bpp_cpp':>9}  {'Δbpp':>9}  {'Δbytes':>7}  "
        + (f"{'Δz':>7}  {'Δy':>7}  " if has_zy else "")
        + f"{'py↔cpp':>7}  {'px_max':>8}  {'px_p99':>8}  {'px_mean':>8}"
        + (f"  {'pm_py':>7}  {'pm_cpp':>7}  {'Δpm':>7}" if has_merlin else "")
    )
    print(col)
    print("-" * len(col))

    for i in range(N):
        bpp_diff = abs(bpps_py[i] - bpps_cpp[i])
        byte_diff = round(bpp_diff / _BYTES_PER_PIXEL)
        z_diff = z_cpp[i] - z_py[i]
        y_diff = y_cpp[i] - y_py[i]
        px_diff = np.abs(arrs_py[i].astype(np.float64) - arrs_cpp[i].astype(np.float64))
        max_px = float(px_diff.max())
        p99_px = _pct(px_diff, 99)
        mean_px = float(px_diff.mean())
        mse_i = float(np.mean(px_diff**2))
        psnr_i = (
            20.0 * math.log10(global_peak / math.sqrt(mse_i)) if mse_i > 1e-30 else float("inf")
        )

        bpp_ok = bpp_diff < args.bpp_tol
        pixel_ok = max_px < args.pixel_tol
        marker = "" if (bpp_ok and pixel_ok) else "  FAIL"

        # PSNR vs MERLIN: positive Δpm means C++ is closer to MERLIN than Python
        pm_py_i = pm_py[i]
        pm_cpp_i = pm_cpp[i]
        dpm = (pm_cpp_i - pm_py_i) if (pm_py_i is not None and pm_cpp_i is not None) else None
        delta_psnr_merlin.append(dpm)

        zy_str = f"{z_diff:>+7}  {y_diff:>+7}  " if has_zy else ""
        merlin_str = (
            f"  {pm_py_i:>7.2f}  {pm_cpp_i:>7.2f}  {dpm:>+7.3f}"
            if (has_merlin and dpm is not None)
            else ""
        )
        print(
            f"{i:>6}  {bpps_py[i]:>9.5f}  {bpps_cpp[i]:>9.5f}  {bpp_diff:>9.2e}  {byte_diff:>+7}  "
            + zy_str
            + f"{psnr_i:>7.1f}  {max_px:>8.2f}  {p99_px:>8.2f}  {mean_px:>8.3f}{marker}"
            + merlin_str
        )

        if not bpp_ok:
            bpp_pass = False
        if not pixel_ok:
            pixel_pass = False
        bpp_diffs.append(bpp_diff)
        px_diffs.append(px_diff)
        psnrs.append(psnr_i)
        z_diffs.append(z_diff)
        y_diffs.append(y_diff)

        verbose_this = args.verbose or (args.patch == i)
        if verbose_this:
            print(f"  py  values : {_array_stats(arrs_py[i])}")
            print(f"  cpp values : {_array_stats(arrs_cpp[i])}")
            signed = arrs_cpp[i].astype(np.float64) - arrs_py[i].astype(np.float64)
            print(
                f"  signed diff: mean={signed.mean():.3f}  std={signed.std():.3f}  "
                f"min={signed.min():.3f}  max={signed.max():.3f}"
            )
            print(f"  |diff| dist: {_diff_stats(px_diff)}")
            ratio = arrs_cpp[i].astype(np.float64) / np.maximum(
                arrs_py[i].astype(np.float64), 1e-9
            )
            print(
                f"  cpp/py ratio: mean={ratio.mean():.4f}  std={ratio.std():.4f}  "
                f"p1={_pct(ratio, 1):.4f}  p99={_pct(ratio, 99):.4f}"
            )

    print()

    # ------------------------------------------------------------------ #
    # Aggregate summary
    all_px = np.concatenate([d.ravel() for d in px_diffs])
    bpp_byte_diffs = [round(d / _BYTES_PER_PIXEL) for d in bpp_diffs]
    unique_byte_diffs = sorted(set(bpp_byte_diffs))
    finite_psnrs = [p for p in psnrs if not math.isinf(p)]

    print("=== Summary ===")
    print(
        f"BPP   max|diff|={max(bpp_diffs):.2e}  "
        f"byte_delta dist: { {k: bpp_byte_diffs.count(k) for k in unique_byte_diffs} }  "
        f"{'PASS' if bpp_pass else 'FAIL'} (tol {args.bpp_tol:.0e})"
    )
    if has_zy:
        print(
            f"      Δz_bytes: min={min(z_diffs):+d}  max={max(z_diffs):+d}  "
            f"mean={sum(z_diffs) / len(z_diffs):+.0f}"
        )
        print(
            f"      Δy_bytes: min={min(y_diffs):+d}  max={max(y_diffs):+d}  "
            f"mean={sum(y_diffs) / len(y_diffs):+.0f}"
        )
    print(
        f"Pixel max={all_px.max():.3f}  p99={_pct(all_px, 99):.3f}  "
        f"p95={_pct(all_px, 95):.3f}  mean={all_px.mean():.3f}  "
        f"{'PASS' if pixel_pass else 'FAIL'} (tol {args.pixel_tol:.4f})"
    )
    if finite_psnrs:
        print(
            f"PSNR(cpp‖py): min={min(psnrs):.1f}  "
            f"mean={sum(finite_psnrs) / len(finite_psnrs):.1f}  max={max(psnrs):.1f}  dB"
        )

    # Signed bias — detects systematic scale or offset error
    all_py = np.concatenate([a.ravel().astype(np.float64) for a in arrs_py])
    all_cpp = np.concatenate([a.ravel().astype(np.float64) for a in arrs_cpp])
    bias = (all_cpp - all_py).mean()
    ratio_mean = (all_cpp / np.maximum(all_py, 1e-9)).mean()
    print(f"Pixel bias (cpp-py mean): {bias:+.4f}   cpp/py ratio mean: {ratio_mean:.6f}")
    print(f"py  global: min={all_py.min():.3f}  mean={all_py.mean():.3f}  max={all_py.max():.3f}")
    print(
        f"cpp global: min={all_cpp.min():.3f}  mean={all_cpp.mean():.3f}  max={all_cpp.max():.3f}"
    )
    print()

    # ------------------------------------------------------------------ #
    # PSNR vs MERLIN summary (quality vs ground truth — the metric that actually matters)
    if has_merlin:
        valid_dpm = [d for d in delta_psnr_merlin if d is not None]
        valid_pm_py = [v for v in pm_py if v is not None]
        valid_pm_cpp = [v for v in pm_cpp if v is not None]
        print("=== PSNR vs MERLIN (quality relative to ground truth) ===")
        print(
            f"  Python  : mean={sum(valid_pm_py) / len(valid_pm_py):.3f}  "
            f"min={min(valid_pm_py):.3f}  max={max(valid_pm_py):.3f} dB"
        )
        print(
            f"  C++     : mean={sum(valid_pm_cpp) / len(valid_pm_cpp):.3f}  "
            f"min={min(valid_pm_cpp):.3f}  max={max(valid_pm_cpp):.3f} dB"
        )
        print(
            f"  Δpm(cpp-py): mean={sum(valid_dpm) / len(valid_dpm):+.4f}  "
            f"min={min(valid_dpm):+.4f}  max={max(valid_dpm):+.4f} dB"
        )
        failing = [
            (i, delta_psnr_merlin[i])
            for i in range(N)
            if delta_psnr_merlin[i] is not None and abs(delta_psnr_merlin[i]) > 0.1
        ]
        if failing:
            print(
                "  Patches with |Δpm| > 0.1 dB: "
                + ", ".join(f"{idx}({d:+.3f})" for idx, d in failing)
            )
        else:
            print("  All patches within 0.1 dB of Python PSNR vs MERLIN.")
        print()

    # ------------------------------------------------------------------ #
    # Diagnosis hints
    if bpp_pass and pixel_pass:
        print("ALL PASS — Python and C++ outputs are equivalent.")
    else:
        if not bpp_pass:
            n_nonzero = sum(1 for d in bpp_diffs if d > 0)
            print(
                f"BPP MISMATCH — {n_nonzero}/{N} patches differ by "
                f"{max(bpp_byte_diffs)} bytes max."
            )
            if has_zy:
                z_clean = all(d == 0 for d in z_diffs)
                y_clean = all(d == 0 for d in y_diffs)
                if not z_clean and y_clean:
                    print("  → Δz≠0, Δy=0: bug is in EB (h_a input or EB params).")
                elif z_clean and not y_clean:
                    print("  → Δz=0, Δy≠0: bug is in GC (y/scales layout or GC params).")
                elif not z_clean and not y_clean:
                    print("  → Δz≠0, Δy≠0: bugs in both EB and GC paths.")
        if not pixel_pass:
            if ratio_mean > 1.5 or ratio_mean < 0.5:
                print(
                    f"PIXEL MISMATCH (scale error) — cpp/py ratio={ratio_mean:.4f}. "
                    "Check normalization constants (AMP_MIN/MAX) or fix_point scale."
                )
            elif abs(bias) > 10:
                print(
                    f"PIXEL MISMATCH (offset error) — bias={bias:+.3f}. "
                    "Check denorm formula (AMP_MIN offset) or MERLIN 0.5 factor."
                )
            else:
                print(
                    f"PIXEL MISMATCH — bias={bias:+.3f}, ratio={ratio_mean:.4f}. "
                    "Check DPU quantization, fix_point sign, or denorm path."
                )
        sys.exit(1)


if __name__ == "__main__":
    main()
