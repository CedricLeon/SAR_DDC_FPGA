"""E1 — Symmetrization-granularity study (local GPU, no retraining).

Question
--------
`symmetrize()` (MERLIN zero-Doppler centering) is a **whole-image FFT**, which would break a
per-patch streaming pipeline. But it is an integer spectral roll = spatial phase ramp, so it
**preserves per-image amplitude exactly** and only redistributes energy between the real/imag
channels. Hypothesis: since despeckling is amplitude-dominated, cheaper *local* symmetrization costs
little quality and can live inside the per-patch/per-block pipeline. This script measures it.

For one DDC model, it runs the model on the same patches pre-processed four ways and scores the
reconstructions (linear amplitude) against two references:

  variants     : 1=whole-image  2=none  3=per-256-patch  4=per-1024-block
  references   : vs_whole  (variant-1 output — "how much does cheaper symmetrization degrade OURS")
                 vs_merlin (MERLIN GT — absolute despeckling quality; needs --merlin-gt)

Conventions (single source of truth):
  - x_hat -> linear amplitude follows `scripts/dataset/create_dataset.py::_predict_linA` (the code
    that generated the MERLIN/ADAM ground truths). The model is fed RAW [B,2,H,W] patches and
    normalizes internally, exactly as in create_dataset.
  - metrics are computed in LINEAR AMPLITUDE (src/utils/metrics.py); optional figures are LOG-INTENSITY.

Run one model:
  python scripts/evaluation/symmetrization_study.py --arch ResSHyp --lambda 1000 [--merlin-gt]
Aggregate many runs into a table:
  python scripts/evaluation/symmetrization_study.py --aggregate results/symmetrization_study/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.metrics import enl, epd, get_all_distortion_metrics
from src.utils.reconstruction import predict_linA
from src.utils.sar_utils import load_cosar, symmetrize

PATCH = 256
BLOCK = 1024
COMPILED_MODELS = PROJECT_ROOT / "results" / "fpga" / "compiled_models"
MERLIN_CKPT = (
    PROJECT_ROOT / "data" / "method_ground_truths" / "MERLIN" / "checkpoints" / "last.ckpt"
)
DEFAULT_COS = (
    PROJECT_ROOT / "data/TSX_cos_files/"
    "Hamburg_TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_IMAGE_HH_SRA_strip_004.cos"
)
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "symstudy"
OUT_DIR = PROJECT_ROOT / "results" / "symmetrization_study"

# Canonical arch tokens + a couple of shorthands.
ARCH_ALIASES = {
    "FP": "FP",
    "ResFP": "ResFP",
    "SH": "SHyp",
    "SHyp": "SHyp",
    "ResSH": "ResSHyp",
    "ResSHyp": "ResSHyp",
}
VARIANTS = ["whole", "none", "patch", "block"]
DISTORTION_KEYS = ["mse", "psnr", "ssim", "ms_ssim", "enl", "epd"]


# ---------------------------------------------------------------------------
# Model loading (canonical pattern: create_dataset.py / evaluate.py)
# ---------------------------------------------------------------------------
def _load_module(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Instantiate the LightningModule from its training config and load weights."""
    cfg_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found at {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    assert isinstance(cfg, DictConfig)
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model.to(device)


def _reroot(path: Path) -> Path:
    """Re-anchor an absolute path stored on another machine to this PROJECT_ROOT via the DDC_FPGA
    suffix."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "DDC_FPGA":
            return PROJECT_ROOT / Path(*parts[i + 1 :])
    return path


def resolve_model(arch: str, lmbda: int, seed: int) -> tuple[Path, dict[str, Any], str]:
    """Resolve (arch, lambda, seed) -> (checkpoint, manifest, model_name) via the compiled FPGA
    model.

    Errors explicitly (never silently falls back) if the compiled model, its manifest, or the
    referenced checkpoint is missing.
    """
    model_name = f"{arch}-relu_s{seed}_L{lmbda}_pt"
    model_dir = COMPILED_MODELS / model_name
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No compiled model at {model_dir}.\n"
            f"Expected {manifest_path} (arch/seed/lambda combo may not have been deployed)."
        )
    manifest = json.loads(manifest_path.read_text())
    run_dir = Path(manifest["original_run_dir"])
    for cand in (
        run_dir / "checkpoints" / "last.ckpt",
        _reroot(run_dir) / "checkpoints" / "last.ckpt",
    ):
        if cand.exists():
            return cand, manifest, model_name
    raise FileNotFoundError(
        f"Checkpoint not found for {model_name}. original_run_dir={run_dir} "
        f"(also tried re-rooted to {PROJECT_ROOT})."
    )


# ---------------------------------------------------------------------------
# Reconstruction: predict_linA imported from src.utils.reconstruction (single source of truth).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Region loading (with cache) + patch grid
# ---------------------------------------------------------------------------
def load_region(
    cos_path: Path, region: tuple[int, int, int, int] | None, no_cache: bool
) -> np.ndarray:
    """Return the raw (unsymmetrized) region as float32 [H,W,2]. `region`=None -> full scene.

    Caches the crop under data/cache/symstudy/ so a model sweep over the same region reads the .cos
    once.
    """
    tag = "full" if region is None else "r{}c{}h{}w{}".format(*region)
    cache = CACHE_DIR / f"{cos_path.stem}_{tag}_raw.npy"
    if cache.exists() and not no_cache:
        print(f"[region] cache hit: {cache}")
        return np.load(cache)
    print(f"[region] loading .cos (this reads the whole file): {cos_path}")
    img = load_cosar(cos_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load {cos_path}")
    if region is not None:
        r0, c0, h, w = region
        H, W, _ = img.shape
        if r0 + h > H or c0 + w > W:
            raise ValueError(f"Region {region} out of bounds for image {H}x{W}.")
        img = img[r0 : r0 + h, c0 : c0 + w, :]
    img = np.ascontiguousarray(img, dtype=np.float32)
    if not no_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, img)
    return img


def patch_positions(h: int, w: int) -> list[tuple[int, int]]:
    """Top-left (row, col) of every non-overlapping 256x256 patch (extract_patches order)."""
    return [(r, c) for r in range(0, h - PATCH + 1, PATCH) for c in range(0, w - PATCH + 1, PATCH)]


def whole_sym(region: np.ndarray, cos_stem: str, tag: str, no_cache: bool) -> np.ndarray:
    """Whole-image symmetrization of the region (cached — it is the expensive, model-independent
    step)."""
    cache = CACHE_DIR / f"{cos_stem}_{tag}_sym.npy"
    if cache.exists() and not no_cache:
        return np.load(cache)
    print("[whole] symmetrizing the whole region (FFT)...")
    sym = symmetrize(region).astype(np.float32)
    if not no_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, sym)
    return sym


def variant_patch(
    variant: str,
    r: int,
    c: int,
    region: np.ndarray,
    sym_region: np.ndarray,
    block_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Return the [256,256,2] input patch at (r,c) for a given symmetrization variant."""
    if variant == "none":
        return region[r : r + PATCH, c : c + PATCH, :]
    if variant == "whole":
        return sym_region[r : r + PATCH, c : c + PATCH, :]
    if variant == "patch":
        return symmetrize(region[r : r + PATCH, c : c + PATCH, :]).astype(np.float32)
    if variant == "block":
        br, bc = (r // BLOCK) * BLOCK, (c // BLOCK) * BLOCK
        key = (br, bc)
        if key not in block_cache:
            blk = region[
                br : min(br + BLOCK, region.shape[0]), bc : min(bc + BLOCK, region.shape[1]), :
            ]
            block_cache[key] = symmetrize(blk).astype(np.float32)
        sblk = block_cache[key]
        return sblk[r - br : r - br + PATCH, c - bc : c - bc + PATCH, :]
    raise ValueError(f"Unknown variant {variant}")


def _to_batch(patches: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """List of [256,256,2] -> torch [B,2,256,256] float32 on device."""
    arr = np.stack(patches, axis=0)  # [B,256,256,2]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


# ---------------------------------------------------------------------------
# Core study
# ---------------------------------------------------------------------------
def _metric_row(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """All per-patch distortion metrics for one [H,W] reconstruction vs one [H,W] reference
    (linA)."""
    d = get_all_distortion_metrics(pred, ref)  # mse, psnr, ssim, ms_ssim
    d["enl"] = enl(pred)
    d["epd"] = epd(pred, ref)
    return d


def load_or_compute_merlin_gt(
    merlin: torch.nn.Module,
    sym_region: np.ndarray,
    positions: list[tuple[int, int]],
    device: torch.device,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    """MERLIN despeckling GT per patch (on the whole-image-symmetrized input), cached to disk.

    GT is model-independent (depends only on the region), so a multi-config sweep computes it once
    for the first config and reuses it for the rest — as requested for long runs.
    """
    if cache_path.exists():
        print(f"[merlin] GT cache hit: {cache_path}")
        return np.load(cache_path, mmap_mode="r")
    print(f"[merlin] computing GT for {len(positions)} patches (cached for reuse)...")
    gt = np.empty((len(positions), PATCH, PATCH), dtype=np.float32)
    for s in range(0, len(positions), batch_size):
        chunk = positions[s : s + batch_size]
        wb = _to_batch([sym_region[r : r + PATCH, c : c + PATCH, :] for r, c in chunk], device)
        gt[s : s + len(chunk)] = predict_linA(merlin, wb).cpu().numpy()
        print(
            f"  merlin ...{min(s + batch_size, len(positions))}/{len(positions)}",
            end="\r",
            flush=True,
        )
    print()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, gt)
    print(f"[merlin] saved GT: {cache_path}")
    return gt


def run_study(
    model: torch.nn.Module,
    region: np.ndarray,
    sym_region: np.ndarray,
    positions: list[tuple[int, int]],
    merlin_gt: np.ndarray | None,
    variants: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Stream patches through every variant, accumulate per-patch metrics vs_whole and
    vs_merlin."""
    n = len(positions)
    print(
        f"[study] {n} patches x {len(variants)} variants"
        + (" + MERLIN GT" if merlin_gt is not None else "")
    )

    block_cache: dict[tuple[int, int], np.ndarray] = {}
    acc: dict[str, dict[str, dict[str, list[float]]]] = {
        v: {
            "vs_whole": {k: [] for k in DISTORTION_KEYS},
            "vs_merlin": {k: [] for k in DISTORTION_KEYS},
        }
        for v in variants
    }

    for start in range(0, n, batch_size):
        chunk = positions[start : start + batch_size]
        recon: dict[str, torch.Tensor] = {}
        for v in variants:
            batch = _to_batch(
                [variant_patch(v, r, c, region, sym_region, block_cache) for r, c in chunk], device
            )
            recon[v] = predict_linA(model, batch)  # [b,H,W]
        ref_whole = recon["whole"] if "whole" in recon else None
        ref_merlin = None
        if merlin_gt is not None:
            ref_merlin = torch.from_numpy(
                np.array(merlin_gt[start : start + len(chunk)], dtype=np.float32)
            ).to(device)

        for j in range(len(chunk)):
            for v in variants:
                if ref_whole is not None and v != "whole":  # self-comparison is trivial -> skip
                    for k, val in _metric_row(recon[v][j], ref_whole[j]).items():
                        acc[v]["vs_whole"][k].append(val)
                if ref_merlin is not None:
                    for k, val in _metric_row(recon[v][j], ref_merlin[j]).items():
                        acc[v]["vs_merlin"][k].append(val)
        print(f"  ...{min(start + batch_size, n)}/{n}", end="\r", flush=True)
    print()

    def summarize(vals: list[float]) -> dict[str, float] | None:
        if not vals:
            return None
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            return None
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}

    out: dict[str, Any] = {"n_patches": n, "variants": {}}
    for v in variants:
        out["variants"][v] = {
            "vs_whole": {k: summarize(acc[v]["vs_whole"][k]) for k in DISTORTION_KEYS},
            "vs_merlin": {k: summarize(acc[v]["vs_merlin"][k]) for k in DISTORTION_KEYS},
        }
    return out


# ---------------------------------------------------------------------------
# Aggregation across runs -> table
# ---------------------------------------------------------------------------
def aggregate(results_dir: Path) -> None:
    """Print a compact table of PSNR/SSIM per variant across all JSON runs in `results_dir`."""
    runs = sorted(results_dir.glob("*.json"))
    if not runs:
        print(f"No result JSONs in {results_dir}")
        return
    print(f"Aggregating {len(runs)} run(s) from {results_dir}\n")
    for ref in ("vs_whole", "vs_merlin"):
        print(f"=== reference: {ref} ===")
        header = f"{'model':28s} " + " ".join(f"{v:>18s}" for v in VARIANTS)
        print(header + "   (PSNR dB / SSIM)")
        per_variant: dict[str, list[float]] = {v: [] for v in VARIANTS}
        for rp in runs:
            data = json.loads(rp.read_text())
            label = data.get("model", {}).get("label", rp.stem)
            cells = []
            for v in VARIANTS:
                block = data.get("variants", {}).get(v, {}).get(ref, {})
                psnr_s, ssim_s = block.get("psnr"), block.get("ssim")
                if psnr_s and ssim_s:
                    cells.append(f"{psnr_s['mean']:7.2f}/{ssim_s['mean']:.3f}")
                    per_variant[v].append(psnr_s["mean"])
                else:
                    cells.append(f"{'--':>18s}")
            print(f"{label:28s} " + " ".join(f"{c:>18s}" for c in cells))
        avg = "  ".join(
            f"{v}:{np.mean(per_variant[v]):.2f}±{np.std(per_variant[v]):.2f}"
            for v in VARIANTS
            if per_variant[v]
        )
        print(f"{'MEAN PSNR across models':28s} {avg}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--aggregate", metavar="DIR", help="Aggregate result JSONs in DIR into a table and exit."
    )
    p.add_argument("--arch", choices=sorted(ARCH_ALIASES), help="Model architecture.")
    p.add_argument("--lambda", dest="lmbda", type=int, help="Rate-distortion lambda (e.g. 1000).")
    p.add_argument("--seed", type=int, default=0, help="Training seed (default: 0).")
    p.add_argument("--model-dir", help="Compiled FPGA model dir (alternative to --arch/--lambda).")
    p.add_argument("--cos", default=str(DEFAULT_COS), help="Source .cos tile.")
    p.add_argument(
        "--region",
        default="10000,7500,4096,4096",
        help="'r0,c0,h,w' crop (default: 4096^2 = 256 patches), or 'full' for the whole scene.",
    )
    p.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma list subset of whole,none,patch,block.",
    )
    p.add_argument(
        "--merlin-gt", action="store_true", help="Also score vs MERLIN GT (runs MERLIN)."
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/symmetrization_study/<label>.json).",
    )
    args = p.parse_args()

    if args.aggregate:
        aggregate(Path(args.aggregate))
        return

    # ---- resolve model ----
    if args.model_dir:
        model_dir = Path(args.model_dir)
        manifest = json.loads((model_dir / "manifest.json").read_text())
        model_name = manifest["model_name"]
        run_dir = Path(manifest["original_run_dir"])
        ckpt = next(
            c
            for c in (
                run_dir / "checkpoints/last.ckpt",
                _reroot(run_dir) / "checkpoints/last.ckpt",
            )
            if c.exists()
        )
        arch, lmbda, seed = (
            manifest.get("model_name", "?").split("-")[0],
            manifest.get("lambda"),
            manifest.get("seed"),
        )
    else:
        if not args.arch or args.lmbda is None:
            p.error("provide --arch and --lambda (or --model-dir, or --aggregate).")
        arch = ARCH_ALIASES[args.arch]
        lmbda, seed = args.lmbda, args.seed
        ckpt, manifest, model_name = resolve_model(arch, lmbda, seed)
    label = f"{arch}_s{seed}_L{lmbda}"
    print(f"[model] {model_name}\n[ckpt ] {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available — running on CPU (slow).")
    model = _load_module(ckpt, device)

    # ---- region ----
    region_spec = (
        None
        if args.region.strip().lower() == "full"
        else tuple(int(x) for x in args.region.split(","))
    )
    if region_spec is not None and len(region_spec) != 4:
        p.error("--region must be 'r0,c0,h,w' or 'full'.")
    cos_path = Path(args.cos)
    region = load_region(cos_path, region_spec, no_cache=False)
    tag = "full" if region_spec is None else "r{}c{}h{}w{}".format(*region_spec)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    need_sym = ("whole" in variants) or args.merlin_gt
    sym_region = whole_sym(region, cos_path.stem, tag, no_cache=False) if need_sym else region
    positions = patch_positions(region.shape[0], region.shape[1])

    # ---- MERLIN GT (model-independent, cached so a multi-config sweep computes it once) ----
    merlin_gt = None
    if args.merlin_gt:
        merlin_model = _load_module(MERLIN_CKPT, device)
        merlin_gt = load_or_compute_merlin_gt(
            merlin_model,
            sym_region,
            positions,
            device,
            CACHE_DIR / f"{cos_path.stem}_{tag}_merlin_gt.npy",
            args.batch_size,
        )
        del merlin_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- run ----
    res = run_study(
        model, region, sym_region, positions, merlin_gt, variants, args.batch_size, device
    )
    res["model"] = {
        "label": label,
        "arch": arch,
        "lambda": lmbda,
        "seed": seed,
        "name": model_name,
        "ckpt": str(ckpt),
    }
    res["region"] = {
        "cos": str(cos_path),
        "spec": args.region,
        "H": region.shape[0],
        "W": region.shape[1],
    }

    out_path = Path(args.output) if args.output else OUT_DIR / f"{label}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[done] wrote {out_path}")

    # brief console summary
    for v in variants:
        w = res["variants"][v]["vs_whole"]["psnr"]
        m = res["variants"][v]["vs_merlin"]["psnr"]
        msg = f"  {v:6s}  vs_whole PSNR={w['mean']:.2f}dB" if w else f"  {v:6s}  (reference)"
        if m:
            msg += f"   vs_merlin PSNR={m['mean']:.2f}dB"
        print(msg)


if __name__ == "__main__":
    sys.exit(main())
