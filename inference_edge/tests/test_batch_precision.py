"""Tests for the batching (`--batch-size`) and precision (`--precision`) features of
`compress_tile` / `decompress_record`, and the real/imag fusion flag (`fuse_reim`).

What is asserted reflects what is *empirically true* on this stack, not an idealization:

* **x86 CPU is byte-transparent.** On x86 CPU, batching, a short final batch, and real/imag fusion all
  produce records byte-identical to the sequential (batch=1, no-fuse) path — the proof that the batching
  *logic* (grid order, chunking, per-record assembly) is correct. This byte-identity is an x86-CPU
  property, NOT universal: on the Orin's ARM CPU the hyperprior path already reorders conv float
  reductions by batch, so it is gated to x86 (see `IS_X86`).
* **GPU (and ARM-CPU hyperprior) are not byte-transparent** — batched conv uses a different reduction
  than per-item conv, so bytes differ at the float level. We therefore assert the *invariants that hold
  everywhere*: `bpp` is stable across batch size, and a **hyperprior** `.ddc` compressed at batch>1 is
  intentionally NOT decodable by the per-record decoder (the entropy decode desyncs and the non-finite
  guard fires). Batching is a compress/throughput lever; verify/decode must run at batch=1.
* **Precision** fp32 is byte-identical to the pre-precision path; fp16/bf16 run and round-trip finite
  when decoded at the matching precision.

Checkpoints are resolved from env vars so the suite runs on the host and on a board without hardcoding
paths (skips cleanly if unset):
    DDC_EDGE_TEST_MODELDIR_FP / _SHYP   (a model dir with manifest.json), or
    DDC_EDGE_TEST_CKPT_FP     / _SHYP   (a direct last.ckpt path)
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import rootutils

ROOT = rootutils.setup_root(
    str(Path(__file__).resolve()), indicator=".project-root", pythonpath=True
)
sys.path.insert(0, str(ROOT / "inference_edge" / "src"))

import torch
from ddc_edge.model_loading import (
    load_model_from_checkpoint,
    resolve_checkpoint_from_model_dir,
)
from ddc_edge.pipeline import (
    PATCH,
    compress_tile,
    decompress_record,
    latent_shapes,
)
from ddc_edge.timing import StageTimer

ARCHS = ["FP", "SHyp"]
DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
# 800x800 tile at overlap 0 => 4x4 = 16 patches; the batch ladder covers exact divisors and a short
# final batch (7 -> 7,7,2) and an over-full batch (32 -> one batch of 16).
TILE = (np.random.default_rng(0).standard_normal((800, 800, 2)) * 3.0).astype(np.float32)
BATCHES = [2, 3, 4, 7, 8, 16, 32]
IS_X86 = platform.machine() in ("x86_64", "AMD64")

_NET_CACHE: dict = {}


def _resolve_ckpt(arch: str) -> Path | None:
    """Resolve a checkpoint path for the given architecture from env vars, or None if unset."""
    a = arch.upper()  # env vars are uppercase by convention: DDC_EDGE_TEST_CKPT_SHYP, not _SHyp
    md = os.environ.get(f"DDC_EDGE_TEST_MODELDIR_{a}")
    if md:
        ck, _ = resolve_checkpoint_from_model_dir(Path(md).resolve(), ROOT)
        return ck
    ck = os.environ.get(f"DDC_EDGE_TEST_CKPT_{a}")
    return Path(ck) if ck else None


def get_net(arch: str, device: torch.device):
    """Load a network for the given architecture on the given device, caching it."""
    key = (arch, str(device))
    if key not in _NET_CACHE:
        ck = _resolve_ckpt(arch)
        if ck is None or not ck.exists():
            a = arch.upper()
            pytest.skip(f"no checkpoint for {arch} (set DDC_EDGE_TEST_CKPT_{a} / _MODELDIR_{a})")
        _NET_CACHE[key] = load_model_from_checkpoint(ck).to(device)
    return _NET_CACHE[key]


def compress(net, device, batch_size=1, precision="fp32", fuse_reim=False, overlap=0):
    """Compress a single tile with the given parameters, returning the records."""
    return compress_tile(
        net,
        TILE,
        overlap,
        device,
        StageTimer(device),
        batch_size=batch_size,
        precision=precision,
        fuse_reim=fuse_reim,
    )[0]


def bpp(records) -> float:
    """Compute the bits-per-pixel of a list of records."""
    payload = sum(len(z) + len(y) for z, y in records)
    return payload * 8.0 / (len(records) * PATCH * PATCH)


# --------------------------------------------------------------------------------------------------
# CPU bit-transparency: the correctness proof of the batching logic.
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(
    not IS_X86,
    reason="byte-identical batching is an x86-CPU property; ARM CPU reorders conv float reductions by "
    "batch (bpp still stable — see test_bpp_stable_across_batch)",
)
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("overlap", [0, 2])
@pytest.mark.parametrize("batch_size", BATCHES)
def test_batch_matches_sequential_cpu(arch, overlap, batch_size):
    """On x86 CPU, compressing at any batch size (incl.

    a short final batch) is byte-identical to batch=1. Gated to x86 — the guarantee elsewhere is
    bpp-stability, not byte-identity.
    """
    net = get_net(arch, torch.device("cpu"))
    base = compress(net, torch.device("cpu"), batch_size=1, overlap=overlap)
    got = compress(net, torch.device("cpu"), batch_size=batch_size, overlap=overlap)
    assert (
        got == base
    ), f"{arch} ov{overlap} batch={batch_size}: records differ from sequential on CPU"


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("overlap", [0, 2])
def test_fuse_matches_nofuse_cpu(arch, overlap):
    """On CPU, real/imag fusion is byte-identical to two separate g_a calls (math-equivalent)."""
    net = get_net(arch, torch.device("cpu"))
    nofuse = compress(net, torch.device("cpu"), batch_size=4, fuse_reim=False, overlap=overlap)
    fuse = compress(net, torch.device("cpu"), batch_size=4, fuse_reim=True, overlap=overlap)
    assert fuse == nofuse, f"{arch} ov{overlap}: fuse_reim changed the records on CPU"


@pytest.mark.parametrize("arch", ARCHS)
def test_record_count_and_order_stable(arch):
    """Batch size never changes how many records come out or (implicitly) their order."""
    net = get_net(arch, torch.device("cpu"))
    n1 = len(compress(net, torch.device("cpu"), batch_size=1))
    for b in BATCHES:
        assert len(compress(net, torch.device("cpu"), batch_size=b)) == n1


# --------------------------------------------------------------------------------------------------
# Invariants that hold on every device (incl. GPU, where bytes are NOT identical across batch size).
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_bpp_stable_across_batch(arch, device):
    """Compression ratio is batch-independent even where the exact bytes are not (GPU float
    paths)."""
    net = get_net(arch, device)
    b1 = bpp(compress(net, device, batch_size=1))
    b16 = bpp(compress(net, device, batch_size=16))
    assert b1 == pytest.approx(b16, abs=5e-3), f"{arch} {device}: bpp b1={b1} b16={b16}"


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_roundtrip_finite_b1(arch, device):
    """A batch=1 fp32 .ddc decodes to finite reconstructions on the same device."""
    net = get_net(arch, device)
    records = compress(net, device, batch_size=1, precision="fp32")
    y_shape, z_shape = latent_shapes(net, device)
    for z, y in records[:8]:
        out = decompress_record(net, z, y, device, y_shape, z_shape, precision="fp32")
        assert np.isfinite(out).all()


def _precisions_for(device):
    return ["bf16"] if device.type == "cpu" else ["fp16", "bf16"]  # CPU fp16 autocast is limited


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_precision_runs_and_roundtrips(arch, device):
    """Fp16/bf16 compress produce the right number of records and round-trip finite at matching
    precision, batch=1."""
    net = get_net(arch, device)
    y_shape, z_shape = latent_shapes(net, device)
    n_expected = len(compress(net, device, batch_size=1, precision="fp32"))
    for prec in _precisions_for(device):
        records = compress(net, device, batch_size=1, precision=prec)
        assert len(records) == n_expected
        for z, y in records[:8]:
            out = decompress_record(net, z, y, device, y_shape, z_shape, precision=prec)
            assert np.isfinite(out).all(), f"{arch} {device} {prec}: non-finite decode"


@pytest.mark.parametrize("arch", ARCHS)
def test_fp32_precision_is_identity_cpu(arch):
    """Precision='fp32' must be byte-identical to the default path (autocast is a no-op there)."""
    net = get_net(arch, torch.device("cpu"))
    a = compress(net, torch.device("cpu"), batch_size=1, precision="fp32")
    b = compress(net, torch.device("cpu"), batch_size=1)  # default precision
    assert a == b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only contract")
def test_hyper_batch_gt1_not_perrecord_decodable_gpu():
    """Documents the sharp edge: a hyperprior .ddc compressed at batch>1 on GPU is intentionally not
    decodable by the per-record decoder (batched vs per-item h_s desync the entropy decode). The
    non-finite guard must fire. FP (no hyperprior) is exempt — it has no h_s bucketing."""
    device = torch.device("cuda")
    net = get_net("SHyp", device)
    y_shape, z_shape = latent_shapes(net, device)
    records = compress(net, device, batch_size=16, precision="fp32")
    desynced = False
    for z, y in records:
        try:
            out = decompress_record(net, z, y, device, y_shape, z_shape, precision="fp32")
            if not np.isfinite(out).all():
                desynced = True
                break
        except ValueError:
            desynced = True
            break
    assert desynced, "expected a batch>1 hyperprior .ddc to desync the per-record decode on GPU"
