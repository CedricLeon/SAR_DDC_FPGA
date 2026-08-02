"""Overlap-aware patch tiling + blend for the onboard-streaming overlap study (host side).

Pure-numpy grid + feathered blend that operates on an already-decoded ``[n, P, P]`` patch stack
(the output of ``stream_pipeline --decode``), unlike ``processing_utils.patch_infer`` which runs a
model on the fly. It mirrors two sources of truth, both pinned by ``_selftest``:

  * ``inference_cpp/src/stream/stream_pipeline.cpp::make_offsets`` — the ``.ddc`` grid rule
    (stride-spaced, last patch snapped flush to the edge for full coverage).
  * ``src/utils/processing_utils.py::patch_infer`` — the sigmoid feathering blend used in eval
    (and itself the counterpart of the C++ ``tile_infer``).

Consumed by the host stitcher (``scripts/evaluation/stitch_ddc.py``) and the MERLIN full-tile GT
generator. See docs/onboard_pipeline.md §10.
"""

from __future__ import annotations

import numpy as np


def make_offsets(dim: int, patch: int, stride: int) -> list[int]:
    """Top-left offsets along one axis: stride-spaced from 0, with the last patch snapped flush to
    ``dim - patch`` so the full extent is covered. Byte-identical rule to the C++ ``make_offsets``
    (the ``.ddc`` grid), so a stitcher reproduces the exact patch positions from the header alone.
    """
    if dim < patch:
        raise ValueError(f"make_offsets: dim {dim} < patch {patch}")
    if stride < 1:
        raise ValueError(f"make_offsets: stride {stride} < 1 (overlap >= patch?)")
    offs = list(range(0, dim - patch + 1, stride))
    if offs[-1] != dim - patch:
        offs.append(dim - patch)  # snap the last patch flush to the edge
    return offs


def sigmoid_ramp(n: int, alpha: float = 6.0) -> np.ndarray:
    """Length-``n`` feather ramp 0->1 (sigmoid, renormalized to [0,1]).

    Matches the "sigmoid"
    profile of ``processing_utils._make_ramp`` and the C++ ``sigmoid_ramp``. ``n<=0`` -> empty.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r = 1.0 / (1.0 + np.exp(-alpha * (t - 0.5)))
    return ((r - r[0]) / (r[-1] - r[0] + 1e-12)).astype(np.float32)


def _edge_weights(
    offset: int, length: int, patch: int, overlap: int, ramp: np.ndarray
) -> np.ndarray:
    """1-D length-``patch`` weight vector: ramp up over the leading ``overlap`` px and down over
    the trailing ``overlap`` px, except where the patch touches the global image border (no feather
    at the scene edge, so those pixels keep full weight).

    ``overlap<=0`` -> all ones.
    """
    w = np.ones(patch, dtype=np.float32)
    if overlap <= 0:
        return w
    if offset != 0:  # not touching the top/left border -> feather the leading edge
        w[:overlap] = ramp
    if (
        offset + patch != length
    ):  # not touching the bottom/right border -> feather the trailing edge
        w[patch - overlap :] = ramp[::-1]
    return w


def blend_patches(
    patches: np.ndarray, H: int, W: int, patch: int, overlap: int, alpha: float = 6.0
) -> np.ndarray:
    """Weighted ramp-blend an ``[n, patch, patch]`` stack (row-major: azimuth row outer, range col
    inner — the ``.ddc`` record order) into a full ``[H, W]`` canvas, using ``make_offsets``
    positions.

    ``overlap=0`` still averages the snap band where the final snapped patch overlaps its neighbour
    (weight 1 each -> mean), so there is no double-counting. Raises if the stack size disagrees with
    the grid implied by (H, W, patch, overlap) — errors over silent fallbacks.
    """
    stride = patch - overlap
    row_offs = make_offsets(H, patch, stride)
    col_offs = make_offsets(W, patch, stride)
    n_expected = len(row_offs) * len(col_offs)
    if patches.shape[0] != n_expected:
        raise ValueError(
            f"blend_patches: {patches.shape[0]} patches but grid implies {n_expected} "
            f"({len(row_offs)} rows x {len(col_offs)} cols) at overlap={overlap}"
        )
    if tuple(patches.shape[1:]) != (patch, patch):
        raise ValueError(f"blend_patches: patch stack {patches.shape} != (n, {patch}, {patch})")

    ramp = sigmoid_ramp(overlap, alpha)
    canvas = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    k = 0
    for ro in row_offs:
        vw = _edge_weights(ro, H, patch, overlap, ramp)
        for co in col_offs:
            hw = _edge_weights(co, W, patch, overlap, ramp)
            wmap = np.outer(vw, hw)  # [patch, patch] separable weight
            canvas[ro : ro + patch, co : co + patch] += patches[k].astype(np.float32) * wmap
            weight[ro : ro + patch, co : co + patch] += wmap
            k += 1
    covered = weight > 1e-12
    canvas[covered] /= weight[covered]
    return canvas


# ---------------------------------------------------------------------------
# Self-test: grid rule vs board-verified counts, partition-of-unity, and blend equivalence to the
# canonical processing_utils.patch_infer. Run:  python src/utils/tiling.py
# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Grid-rule vs board counts, partition-of-unity, and patch_infer equivalence."""
    # 1. make_offsets matches the board-verified grids (see the on-board --overlap validation run).
    assert make_offsets(1024, 256, 256) == [0, 256, 512, 768], "1024/ov0"
    assert make_offsets(1024, 256, 248) == [0, 248, 496, 744, 768], "1024/ov8 (snap adds edge)"
    assert len(make_offsets(14686, 256, 256)) == 58, "full-scene range cols (57 + snap)"
    assert len(make_offsets(32901, 256, 256)) == 129, "full-scene azimuth rows"

    # 2. Partition of unity: constant patches -> constant canvas everywhere (incl. snap/overlap
    #    bands), for both overlap=0 (snap-only overlap) and overlap=8.
    for ov in (0, 8):
        stride = 256 - ov
        H = W = 1024
        n = len(make_offsets(H, 256, stride)) * len(make_offsets(W, 256, stride))
        stack = np.full((n, 256, 256), 7.0, dtype=np.float32)
        out = blend_patches(stack, H, W, 256, ov)
        assert np.allclose(out, 7.0, atol=1e-4), f"partition-of-unity failed at overlap={ov}"

    # 3. Equivalence to the canonical blend (processing_utils.patch_infer). Random per-patch values
    #    blended both ways must match bit-for-bit-close — pins tiling.py against blend drift.
    try:
        import rootutils

        rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
        import torch

        from src.utils.processing_utils import patch_infer
    except Exception as e:  # torch/rootutils not available here -> skip, don't fail
        print(
            f"tiling self-check: grid + partition-of-unity PASS (patch_infer equiv SKIPPED: {e})"
        )
        return

    H = W = 520
    ov = 8
    row_offs = make_offsets(H, 256, 256 - ov)
    col_offs = make_offsets(W, 256, 256 - ov)
    rng = np.random.default_rng(0)
    stack = rng.standard_normal((len(row_offs) * len(col_offs), 256, 256)).astype(np.float32)

    supply = iter([torch.from_numpy(stack[k])[None, None] for k in range(stack.shape[0])])
    recon, _ = patch_infer(
        torch.zeros(1, 1, H, W), lambda _p: (next(supply), {}), patch_size=256, overlap=ov
    )
    mine = blend_patches(stack, H, W, 256, ov)
    max_err = float(np.abs(recon[0, 0].numpy() - mine).max())
    assert max_err < 1e-5, f"blend != patch_infer (max err {max_err:.2e})"
    print(
        f"tiling self-check: grid + partition-of-unity + patch_infer equivalence PASS "
        f"(blend vs patch_infer max err {max_err:.2e})"
    )


if __name__ == "__main__":
    _selftest()
