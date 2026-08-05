from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch


def clip(
    img: np.ndarray,
    mean_std_norm: bool = True,
    clip_factor: int = 3,
    percentiles: tuple[int, int] = (5, 95),
) -> np.ndarray:
    """Clip to either mean +/- clip_factor * std or percentiles[0]th/percentiles[1]th
    percentile."""
    if mean_std_norm:
        img = img.clip(
            img.mean() - clip_factor * img.std(),
            img.mean() + clip_factor * img.std(),
        )
    else:
        p_low = np.percentile(img, percentiles[0])
        p_high = np.percentile(img, percentiles[1])
        img = img.clip(p_low, p_high)
    return img


def extract_short_name_from_TSX_filepath(filepath: Path):
    """Extract short name from the given filepath.

    Assumes the short name is the substring before the first underscore.
    """
    if "_" not in filepath.name:
        raise ValueError(
            f"File {filepath.name} does not contain an underscore '_' to extract the short name."
        )
    return filepath.name.split("_")[0]


def patch_infer(
    image: torch.Tensor,
    infer_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]],
    patch_size: int = 256,
    overlap: int = 16,
    eliminate_border_px: int = 0,
    blend_profile: Literal["sigmoid", "linear", "cosine"] = "sigmoid",
    blend_alpha: float = 6.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run overlap-blended patch inference on a large image using a PyTorch model.

    The model's input buffer has a fixed spatial size (typically 256×256), so large images
    must be split into patches, processed individually, and reassembled. A naive
    non-overlapping split introduces visible seams at patch boundaries because the model has
    no context outside each patch. This function solves that by using overlapping windows and
    blending the results in the overlap zones with smooth feathering ramps.

    The GPU/CPU counterpart of the FPGA-side overlap-blended tiling (now implemented in the
    C++ inference binary, ``inference_cpp/``). The tiling strategy and blending logic are
    identical; the differences below reflect the PyTorch training environment.

    **Input format: NCHW.**  PyTorch models and LightningModules work in
    [B, C, H, W] layout. Patches are extracted and accumulated in NCHW without any
    transposition, unlike the FPGA version which keeps HWC throughout.

    **Callable infer_fn instead of a raw model.**  The function does not call
    ``model.forward()`` directly because inference logic differs between modules and
    evaluation modes:

    - ``MerlinModule`` processes channels independently (real and imag separately).
    - ``SARDDCModule`` forward returns a dict; the reconstruction is under ``"x_hat"``.
    - In test/evaluation mode, ``SARDDCModule`` must additionally call
      ``net.compress()`` / ``net.decompress()`` to produce a real bitstream BPP.

    Encapsulating this in a closure keeps ``patch_infer`` model-agnostic.

    **Criterion accumulation.**  ``infer_fn`` returns a per-patch criterion dict (loss,
    bpp, bpp_bitstream, …). ``patch_infer`` sums these across all patches and returns
    their average alongside the blended reconstruction, so callers receive the same
    structure as a single-patch forward pass.

    **Tiling strategy.**  Sliding window with stride = ``patch_size - overlap`` plus a
    snap-to-border window, guaranteeing full coverage for any image dimension ≥
    ``patch_size``, without padding or post-crop.

    **Blending.**  In overlap zones each patch contributes according to a 1-D feathering
    ramp (sigmoid, linear, or cosine) that rises from 0 at the leading edge to 1 toward
    the interior. The 2-D weight map is the outer product of the horizontal and vertical
    ramps. At the global image borders no ramp is applied. The final pixel value is the
    weighted average of all patches covering it.

    **Canvas allocation.**  Allocated lazily on the first ``infer_fn`` call so that the
    output channel count C_out is inferred from the actual result, avoiding a dummy forward
    pass just to probe the shape.

    Parameters
    ----------
    image : torch.Tensor, shape [B, C, H, W]
        Input image in NCHW layout, already on the intended device. Must not be
        normalized; normalization is the caller’s responsibility inside ``infer_fn``.
    infer_fn : Callable
        Model inference function with signature::

            (patch: Tensor[B, C, patch_size, patch_size])
            -> (recon: Tensor[B, C_out, patch_size, patch_size],
                criterion: Dict[str, Any])

        Typically a closure over the LightningModule. Examples::

            # SARDDCModule — likelihood evaluation
            def infer_fn(patch):
                out = pl_module.forward(patch)
                crit = pl_module.criterion(out, patch)
                return out["x_hat"], crit

            # SARDDCModule — test evaluation with real bitstream
            def infer_fn(patch):
                out = pl_module.forward(patch)
                crit = pl_module.criterion(out, patch)
                patch_norm = (torch.log(torch.square(patch) + EPS) - 2*AMP_MIN) / (2*AMP_MAX - 2*AMP_MIN)
                out_enc = pl_module.net.compress(patch_norm)
                out_dec = pl_module.net.decompress(out_enc["strings"], out_enc["shape"])
                N, _, H, W = patch.shape
                crit["bpp_bitstream"] = compute_bitstream_bpp(out_enc["strings"], H, W, N)
                return out_dec, crit

            # MerlinModule — real/imag channels processed independently
            def infer_fn(patch):
                r = pl_module.forward(patch[:, 0:1])
                i = pl_module.forward(patch[:, 1:2])
                recon = torch.cat([r, i], dim=1)
                return recon, pl_module.criterion(recon, patch)

        ``infer_fn`` is called inside a ``torch.no_grad()`` context.
    patch_size : int, default=256
        Square patch size in pixels. Must match the model’s expected input size.
    overlap : int, default=16
        Number of pixels of overlap between adjacent patches. Must be a positive even
        integer, strictly less than ``patch_size``.
    eliminate_border_px : int, default=0
        Outermost pixels at each patch edge forced to zero weight before the feathering
        ramp begins (hard-discards the strongest edge artefacts). Must be a non-negative
        even integer, strictly less than ``overlap``.
    blend_profile : {"sigmoid", "linear", "cosine"}, default="sigmoid"
        Shape of the 1-D feathering ramp in overlap zones.
    blend_alpha : float, default=6.0
        Steepness of the sigmoid curve. Only used when ``blend_profile="sigmoid"``.

    Returns
    -------
    recon : torch.Tensor, shape [B, C_out, H, W]
        Blended reconstruction in NCHW layout, on the same device as the input.
    criterion : Dict[str, Any]
        Per-metric averages across all patches (e.g., ``loss``, ``bpp``,
        ``bpp_bitstream``). Tensor values remain as Tensors; float values remain as
        floats.

    Raises
    ------
    ValueError
        If overlap / eliminate_border_px constraints are violated, or if the image is
        smaller than ``patch_size`` in either spatial dimension.
    """
    B, C, H, W = image.shape
    device = image.device

    # ------------------------------------------------------------------
    # 1. Validate parameters
    # ------------------------------------------------------------------
    if overlap <= 0 or overlap % 2 != 0:
        raise ValueError(f"`overlap` must be a positive even integer, got {overlap}.")
    if overlap >= patch_size:
        raise ValueError(
            f"`overlap` ({overlap}) must be strictly less than `patch_size` ({patch_size})."
        )
    if eliminate_border_px < 0 or eliminate_border_px % 2 != 0:
        raise ValueError(
            f"`eliminate_border_px` must be a non-negative even integer, got {eliminate_border_px}."
        )
    if eliminate_border_px >= overlap:
        raise ValueError(
            f"`eliminate_border_px` ({eliminate_border_px}) must be strictly less than `overlap` ({overlap})."
        )
    if H < patch_size:
        raise ValueError(f"Image height ({H}) is smaller than `patch_size` ({patch_size}).")
    if W < patch_size:
        raise ValueError(f"Image width ({W}) is smaller than `patch_size` ({patch_size}).")

    # ------------------------------------------------------------------
    # 2. Build the list of overlapping patch windows.
    #    Sliding window with stride = patch_size - overlap, plus a
    #    snap-to-border window to guarantee full coverage for any image
    #    dimension >= patch_size, without padding or post-crop.
    # ------------------------------------------------------------------
    stride = patch_size - overlap

    def _make_offsets(dim_size: int) -> List[int]:
        """Return starting offsets for one spatial dimension."""
        offsets = list(range(0, dim_size - patch_size + 1, stride))
        # Snap-to-border: ensure the last window ends exactly at the image edge
        last = dim_size - patch_size
        if not offsets or offsets[-1] != last:
            offsets.append(last)
        return offsets

    row_offsets = _make_offsets(H)
    col_offsets = _make_offsets(W)
    n_patches = len(row_offsets) * len(col_offsets)
    print(
        f"[patch_infer] {H}x{W} image => {n_patches} patches "
        f"({len(row_offsets)} rows x {len(col_offsets)} cols), "
        f"patch_size={patch_size}, overlap={overlap}, stride={stride}."
    )

    # ------------------------------------------------------------------
    # 3. Allocate accumulators.
    #    The canvas is allocated lazily on the first infer_fn result so that
    #    C_out is inferred from the actual output (avoids a dummy forward pass).
    #    Weight maps are torch.Tensors on the correct device, avoiding any
    #    numpy ↔ torch round-trips during the main loop.
    # ------------------------------------------------------------------
    canvas: Optional[torch.Tensor] = None
    weight_canvas = torch.zeros(H, W, device=device, dtype=torch.float32)
    criterion_sums: Dict[str, Any] = {}
    patch_count = 0

    # ------------------------------------------------------------------
    # 5. Helper: build a 1-D feathering ramp of length n, going 0 → 1
    # ------------------------------------------------------------------
    def _make_ramp(n: int) -> torch.Tensor:
        if n <= 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float32)
        if blend_profile == "linear":
            return t
        elif blend_profile == "sigmoid":
            a = float(blend_alpha)
            r = 1.0 / (1.0 + torch.exp(-a * (t - 0.5)))
            r = (r - r[0]) / (r[-1] - r[0] + 1e-12)  # renormalize to [0, 1]
            return r
        elif blend_profile == "cosine":
            return 0.5 * (1.0 - torch.cos(float(np.pi) * t))
        else:
            raise ValueError(
                f"Unknown blend_profile '{blend_profile}'. Choose 'sigmoid', 'linear', or 'cosine'."
            )

    # ------------------------------------------------------------------
    # 5. Helper: 2-D weight map for one patch [patch_size, patch_size].
    #    Handles eliminate_border_px, feathering ramp, and global-border guard
    #    (no feathering at image edges so the output has full weight there).
    # ------------------------------------------------------------------
    def _patch_weight_map(row_off: int, col_off: int) -> torch.Tensor:
        touch_top = row_off == 0
        touch_bottom = row_off + patch_size == H
        touch_left = col_off == 0
        touch_right = col_off + patch_size == W

        ramp_len = overlap - eliminate_border_px
        u = torch.ones(patch_size, device=device, dtype=torch.float32)  # horizontal (W)
        v = torch.ones(patch_size, device=device, dtype=torch.float32)  # vertical   (H)
        ramp = _make_ramp(ramp_len)

        for vec, touch_start, touch_end in [
            (u, touch_left, touch_right),
            (v, touch_top, touch_bottom),
        ]:
            if not touch_start:
                # Hard-zero the outermost eliminate_border_px pixels
                if eliminate_border_px > 0:
                    vec[:eliminate_border_px] = 0.0
                # Then apply the ramp (0 → 1) over the next ramp_len pixels
                if ramp_len > 0:
                    vec[eliminate_border_px : eliminate_border_px + ramp_len] = ramp

            if not touch_end:
                # Mirror of the above on the trailing edge (1 → 0)
                if eliminate_border_px > 0:
                    vec[-eliminate_border_px:] = 0.0
                if ramp_len > 0:
                    vec[
                        -(eliminate_border_px + ramp_len) : (
                            None if eliminate_border_px == 0 else -eliminate_border_px
                        )
                    ] = ramp.flip(0)

        return v.unsqueeze(1) * u.unsqueeze(0)  # [patch_size, patch_size]

    # ------------------------------------------------------------------
    # 6. Main loop: extract patch (NCHW) → infer_fn → accumulate into canvas.
    #
    #    Patches are extracted as NCHW slices (.contiguous() is required by
    #    some torch operations on non-contiguous views from slicing).
    #    infer_fn handles all model-specific logic: channel splitting for
    #    MerlinModule, dict unpacking for SARDDCModule, and compress /
    #    decompress for real-bitstream evaluation.
    #    Criterion values are detached before accumulation to avoid retaining
    #    unnecessary computation graphs.
    #    The weight map is broadcast over the B and C dimensions via
    #    w2d[None, None, :, :], matching the NCHW canvas shape.
    # ------------------------------------------------------------------
    with torch.no_grad():
        for row_off in row_offsets:
            for col_off in col_offsets:
                patch = image[
                    :,
                    :,
                    row_off : row_off + patch_size,
                    col_off : col_off + patch_size,
                ].contiguous()  # [B, C, ps, ps]

                out_patch, patch_criterion = infer_fn(patch)  # [B, C_out, ps, ps], dict

                # Lazy canvas allocation on the first result
                if canvas is None:
                    C_out = out_patch.shape[1]
                    canvas = torch.zeros(B, C_out, H, W, device=device, dtype=torch.float32)

                # Accumulate criterion (detach to free the computation graph)
                if patch_count == 0:
                    criterion_sums = {
                        k: (v.detach() if isinstance(v, torch.Tensor) else v)
                        for k, v in patch_criterion.items()
                    }
                else:
                    for k, v in patch_criterion.items():
                        if k in criterion_sums:
                            v_det = v.detach() if isinstance(v, torch.Tensor) else v
                            criterion_sums[k] = criterion_sums[k] + v_det

                # Accumulate weighted patch into canvas
                w2d = _patch_weight_map(row_off, col_off)  # [ps, ps]
                canvas[
                    :,
                    :,
                    row_off : row_off + patch_size,
                    col_off : col_off + patch_size,
                ] += (
                    out_patch * w2d[None, None, :, :]
                )
                weight_canvas[
                    row_off : row_off + patch_size,
                    col_off : col_off + patch_size,
                ] += w2d

                patch_count += 1

    if canvas is None:
        raise ValueError("No patches were processed. Verify image dimensions and patch_size.")

    # ------------------------------------------------------------------
    # 7. Normalise canvas and average criterion.
    #    Every pixel is covered by at least one patch so weight_canvas > 0
    #    everywhere; clamp guards against fp precision edge cases.
    # ------------------------------------------------------------------
    weight_canvas = weight_canvas.clamp(min=1e-8)
    output = canvas / weight_canvas[None, None, :, :]  # [B, C_out, H, W]

    criterion_avg: Dict[str, Any] = {k: v / patch_count for k, v in criterion_sums.items()}

    # Bitrates are *rates over the image*, not per-patch quantities to be averaged. With
    # overlap the patches cover more pixels than the image (25 x 256² vs 1024² = 1.5625x at
    # overlap 16), and every one of those bits is actually transmitted — so the honest tile
    # bitrate is total bits / image pixels. Averaging per-patch bpp silently discards the
    # overlap cost and made the GPU tile bitrate incomparable with the board, which already
    # reports total_bytes / (TH*TW) (inference_cpp/src/inference_runner.cpp::_run_tile_eval_impl).
    # criterion_sums[k] = Σ(bits_i / patch_size²), so scaling by patch_size²/(H*W) is exact.
    _rate_scale = (patch_size * patch_size) / float(H * W)
    for _k in ("bpp", "bpp_bitstream"):
        if _k in criterion_sums:
            criterion_avg[_k] = criterion_sums[_k] * _rate_scale

    return output, criterion_avg
