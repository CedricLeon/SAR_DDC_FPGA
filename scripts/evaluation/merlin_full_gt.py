#!/usr/bin/env python3
"""merlin_full_gt.py — generate the project's MERLIN full-tile despeckled GT (seam-free) for the
overlap study (pass 3 reference).

Runs the project MERLIN U-net (data/method_ground_truths/MERLIN — the same reference §5 scores
against, so overlap-study absolute numbers stay consistent with the rest of the project) over the
whole-image-symmetrized scene on a heavy-overlap grid, then ramp-blends to a seam-free [H,W]
linear-amplitude tile. Heavy overlap (default 64 px) keeps the *reference* itself seam-free, so it
does not confound the DDC recon's seam measurement. Reuses the symmetrization-study machinery
(load_region / whole_sym / MERLIN loader) and the shared tiling blend. See docs/onboard_pipeline.md §10.

    # validate on a 1024^2 crop that already has a project linA_MERLIN.npy:
    python scripts/evaluation/merlin_full_gt.py --region 11000 8500 1024 1024 --out /tmp/gt.npy
    # full scene (heavy — whole-image FFT + ~8k patches):
    python scripts/evaluation/merlin_full_gt.py --full --out data/method_ground_truths/MERLIN/linA_MERLIN_full_Hamburg.npy
"""

import argparse
from pathlib import Path

import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.evaluation.symmetrization_study import (  # noqa: E402
    DEFAULT_COS,
    MERLIN_CKPT,
    PATCH,
    _load_module,
    load_region,
    whole_sym,
)
from src.utils.reconstruction import predict_linA  # noqa: E402
from src.utils.tiling import blend_patches, make_offsets  # noqa: E402


@torch.no_grad()
def merlin_full_gt(
    sym_region: np.ndarray, model, device, gt_overlap: int, batch_size: int = 16
) -> np.ndarray:
    """MERLIN despeckle the (symmetrized) region on a gt_overlap grid, ramp-blend -> [H,W] linA."""
    H, W = sym_region.shape[:2]
    stride = PATCH - gt_overlap
    offsets = [
        (r, c) for r in make_offsets(H, PATCH, stride) for c in make_offsets(W, PATCH, stride)
    ]
    patches = np.empty((len(offsets), PATCH, PATCH), dtype=np.float32)
    for s in range(0, len(offsets), batch_size):
        chunk = offsets[s : s + batch_size]
        batch = np.stack([sym_region[r : r + PATCH, c : c + PATCH, :] for r, c in chunk])
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)  # [b,2,P,P] raw
        patches[s : s + len(chunk)] = predict_linA(model, t).cpu().numpy()
        print(
            f"  merlin {min(s + batch_size, len(offsets))}/{len(offsets)} patches",
            end="\r",
            flush=True,
        )
    print()
    return blend_patches(patches, H, W, PATCH, gt_overlap)


def main() -> None:
    """CLI: generate the MERLIN full-tile GT for a --region or the --full scene."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--region", nargs=4, type=int, metavar=("R0", "C0", "H", "W"))
    g.add_argument("--full", action="store_true", help="whole scene")
    ap.add_argument("--cos", default=str(DEFAULT_COS))
    ap.add_argument(
        "--sym",
        choices=["whole", "none"],
        default="whole",
        help="whole = §5 reference (whole-image symmetrization); none = raw input",
    )
    ap.add_argument(
        "--gt-overlap", type=int, default=64, help="patch overlap px for the seam-free GT"
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    region = None if args.full else tuple(args.region)
    tag = "full" if region is None else "r{}c{}h{}w{}".format(*region)
    cos = Path(args.cos)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gt] region={tag} sym={args.sym} gt_overlap={args.gt_overlap} device={device}")

    raw = load_region(cos, region, args.no_cache)  # [H,W,2] f32, unsymmetrized
    sym = whole_sym(raw, cos.stem, tag, args.no_cache) if args.sym == "whole" else raw
    model = _load_module(MERLIN_CKPT, device)

    gt = merlin_full_gt(sym, model, device, args.gt_overlap, args.batch_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, gt)
    print(
        f"[gt] wrote {args.out}  shape={gt.shape}  "
        f"range=[{float(gt.min()):.4g}, {float(gt.max()):.4g}]  nan={int(np.isnan(gt).sum())}"
    )


if __name__ == "__main__":
    main()
