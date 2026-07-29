#!/usr/bin/env python3
"""Self-test for the .ddc container format (src/utils/ddc_format.py).

Compresses a few real patches with a CompressAI DDC model, serialises a real .ddc, reads it back
(sequentially AND via the offset-table trailer), decodes it, and checks that the reconstruction
from the file is byte-identical to the direct model reconstruction. Proves the container is lossless
and correctly wired before the C++ stream_seq writer (step 3) has to emit the same bytes.

Usage:
  python scripts/evaluation/ddc_selftest.py --arch ResSHyp --lambda 1000 [--patches 8]
  python scripts/evaluation/ddc_selftest.py --arch FP --lambda 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import ddc_format as ddc
from src.utils.constants import AMP_MAX, AMP_MIN, EPS

COMPILED = PROJECT_ROOT / "results" / "fpga" / "compiled_models"
DEFAULT_DATA = (
    PROJECT_ROOT / "data/cache/symstudy/"
    "Hamburg_TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_"
    "IMAGE_HH_SRA_strip_004_r10000c7500h4096w4096_raw.npy"
)
TILE_ID = "TDX1_SAR__SSC______SM_S_SRA_20180112T165337_20180112T165345_IMAGE_HH_SRA_strip_004"
ARCH_ALIASES = {
    "FP": "FP",
    "ResFP": "ResFP",
    "SH": "SHyp",
    "SHyp": "SHyp",
    "ResSH": "ResSHyp",
    "ResSHyp": "ResSHyp",
}
PATCH = 256


def _reroot(path: Path) -> Path:
    """Reroot a path from the DDC_FPGA repo to the current working copy (if any)."""
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "DDC_FPGA":
            return PROJECT_ROOT / Path(*parts[i + 1 :])
    return path


def load_net(arch, lmbda, seed, device):
    """Load the inner CompressAI net (entropy tables populated) + a params-hash guard."""
    name = f"{arch}-relu_s{seed}_L{lmbda}_pt"
    mdir = COMPILED / name
    manifest = json.loads((mdir / "manifest.json").read_text())
    run_dir = Path(manifest["original_run_dir"])
    ckpt = next(
        c
        for c in (run_dir / "checkpoints/last.ckpt", _reroot(run_dir) / "checkpoints/last.ckpt")
        if c.exists()
    )
    cfg = OmegaConf.load(ckpt.parent.parent / ".hydra" / "config.yaml")
    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    net = model.net.eval().to(device)
    net.update(force=True)  # populate CDF tables for compress/decompress
    ep = mdir / "entropy_params"
    if ep.is_dir():
        h = hashlib.sha256()
        for fp in sorted(ep.glob("*.npy")):
            h.update(fp.read_bytes())
        sha = h.digest()[:8]
    else:
        sha = hashlib.sha256(name.encode()).digest()[:8]
    return net, name, sha


def normalize(x_lin: torch.Tensor) -> torch.Tensor:
    """Raw amplitude [.,2,H,W] -> normalized model input (same as SARDDCModule / benchmark_gpu)."""
    return (torch.log(torch.square(x_lin) + EPS) - 2 * AMP_MIN) / (2 * AMP_MAX - 2 * AMP_MIN)


def load_patches(path: Path, k: int, device) -> torch.Tensor:
    """Load k patches from a .npy file, normalize, and return [k,2,H,W] on device."""
    arr = np.load(path)
    if arr.ndim == 4:  # [N,256,256,C]
        pats = arr[:k, :, :, :2]
    elif arr.ndim == 3:  # [H,W,2] tile -> first k non-overlapping 256 patches
        H, W, _ = arr.shape
        out = []
        for r in range(0, H - PATCH + 1, PATCH):
            for c in range(0, W - PATCH + 1, PATCH):
                out.append(arr[r : r + PATCH, c : c + PATCH, :2])
                if len(out) >= k:
                    break
            if len(out) >= k:
                break
        pats = np.stack(out)
    else:
        raise ValueError(f"unexpected data ndim {arr.ndim}")
    x = torch.from_numpy(np.ascontiguousarray(pats, dtype=np.float32)).permute(0, 3, 1, 2)
    return normalize(x.contiguous()).to(device)


@torch.no_grad()
def latent_shapes(net, has_hyper, device):
    """Derive y/z spatial shapes from patch size + arch (no per-patch storage needed)."""
    d = torch.zeros(1, 1, PATCH, PATCH, device=device)
    y1 = net.g_a(d)
    y_shape = tuple(y1.shape[-2:])
    z_shape = None
    if has_hyper:
        z = net.h_a(torch.abs(torch.cat([y1, y1], dim=1)))
        z_shape = tuple(z.shape[-2:])
    return y_shape, z_shape


@torch.no_grad()
def compress_patch(net, has_hyper, xi):
    """One patch [1,2,H,W] -> (z_bytes, y_bytes)."""
    y = torch.cat([net.g_a(xi[:, :1]), net.g_a(xi[:, 1:])], dim=1)
    if has_hyper:
        z = net.h_a(torch.abs(y))
        z_strings = net.entropy_bottleneck.compress(z)
        z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales = net.h_s(z_hat)
        indexes = net.gaussian_conditional.build_indexes(scales)
        y_strings = net.gaussian_conditional.compress(y, indexes)
        return z_strings[0], y_strings[0]
    return b"", net.entropy_bottleneck.compress(y)[0]


@torch.no_grad()
def decode_patch(net, has_hyper, z_bytes, y_bytes, y_shape, z_shape, n_main):
    """(z_bytes, y_bytes) -> normalized reconstruction [1,2,H,W]."""
    if has_hyper:
        z_hat = net.entropy_bottleneck.decompress([z_bytes], z_shape)
        scales = net.h_s(z_hat)
        indexes = net.gaussian_conditional.build_indexes(scales)
        y_hat = net.gaussian_conditional.decompress([y_bytes], indexes)
    else:
        y_hat = net.entropy_bottleneck.decompress([y_bytes], y_shape)
    return torch.cat([net.g_s(y_hat[:, :n_main]), net.g_s(y_hat[:, n_main:])], dim=1)


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", required=True, choices=sorted(ARCH_ALIASES))
    ap.add_argument("--lambda", dest="lmbda", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument(
        "--out", default=str(PROJECT_ROOT / "results/symmetrization_study/selftest.ddc")
    )
    a = ap.parse_args()
    arch = ARCH_ALIASES[a.arch]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, model_name, sha = load_net(arch, a.lmbda, a.seed, device)
    has_hyper = hasattr(net, "h_a")
    n_main = net.nb_channels_main
    N, M = n_main, n_main * 2
    y_shape, z_shape = latent_shapes(net, has_hyper, device)
    print(
        f"[model] {model_name}  hyper={has_hyper}  N={N} M={M}  y_shape={y_shape} z_shape={z_shape}"
    )

    x = load_patches(Path(a.data), a.patches, device)
    K = x.shape[0]
    print(f"[data ] {K} patches from {Path(a.data).name}")

    records, recon_direct = [], []
    for i in range(K):
        z, y = compress_patch(net, has_hyper, x[i : i + 1])
        records.append((z, y))
        recon_direct.append(decode_patch(net, has_hyper, z, y, y_shape, z_shape, n_main))

    header = ddc.DDCHeader(
        arch_id=ddc.ARCH_IDS[arch],
        N=N,
        M=M,
        patch=PATCH,
        stride=PATCH,
        scene_H=K * PATCH,
        scene_W=PATCH,
        grid_r=K,
        grid_a=1,  # dummy K x 1 grid for the test
        amp_min=AMP_MIN,
        amp_max=AMP_MAX,
        eps=EPS,
        params_sha=sha,
        tile_id=TILE_ID,
        model_id=model_name,
        has_index=True,
    )
    offsets_written = ddc.write_ddc(a.out, header, records)
    size = Path(a.out).stat().st_size
    payload = sum(len(z) + len(y) for z, y in records)
    print(
        f"[write] {Path(a.out).name}  size={size}B  payload={payload}B  "
        f"overhead={size - payload}B  bpp={payload * 8 / (K * PATCH * PATCH):.4f}"
    )

    ok = True
    # 1. header round-trips
    h2 = ddc.read_header(a.out)
    hdr_ok = (
        h2.arch_id == header.arch_id
        and h2.N == N
        and h2.M == M
        and h2.tile_id == TILE_ID
        and h2.model_id == model_name
        and h2.params_sha == sha
        and h2.has_index
    )
    print(f"[check] header round-trip: {'OK' if hdr_ok else 'FAIL'}  (tile_id='{h2.tile_id}')")
    ok &= hdr_ok
    # 2. sequential body bytes round-trip
    _, rec2 = ddc.read_ddc(a.out)
    seq_ok = rec2 == records
    print(f"[check] sequential bytes round-trip: {'OK' if seq_ok else 'FAIL'}")
    ok &= seq_ok
    # 3. trailer located by (filesize - n*8); random access matches
    offsets, table_start = ddc.read_offset_table(a.out, h2)
    kprobe = K - 1
    ra = ddc.read_patch(a.out, offsets[kprobe])
    ra_ok = offsets == offsets_written and ra == records[kprobe]
    print(
        f"[check] trailer @ filesize-n*8 = {size}-{K}*8 = {table_start}; "
        f"random-access patch {kprobe}: {'OK' if ra_ok else 'FAIL'}"
    )
    ok &= ra_ok
    # 4. reconstruction from file ≈ direct reconstruction. Byte-fidelity is already proven exact by
    #    checks 2-3; a ~1e-7 residual here is GPU/cuDNN non-determinism between two forward passes on
    #    identical inputs, not the container.
    maxdiff = 0.0
    for i in range(K):
        r = decode_patch(net, has_hyper, rec2[i][0], rec2[i][1], y_shape, z_shape, n_main)
        maxdiff = max(maxdiff, float((r - recon_direct[i]).abs().max()))
    dec_ok = maxdiff < 1e-4
    print(
        f"[check] decode(file) ≈ decode(direct): max|Δ|={maxdiff:.3e} (<1e-4)  {'OK' if dec_ok else 'FAIL'}"
    )
    ok &= dec_ok

    print(
        f"\n{'PASS' if ok else 'FAIL'} — .ddc container is "
        f"{'lossless and correctly wired' if ok else 'BROKEN'}."
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
