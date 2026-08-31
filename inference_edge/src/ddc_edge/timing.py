"""Per-stage wall-clock accumulation, CUDA-safe (syncs at both ends of each timed span).

Uses the same canonical stage names as scripts/evaluation/benchmark_gpu.py / docs/GPU_benchmark.md's
aligned schema (g_a, h_a, h_s, eb_compress, eb_decompress, gc_compress, normalize, ...) so results from
this pipeline can join the same cross-platform tables without renaming.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch


class StageTimer:
    """Accumulates total seconds spent in each named stage across many calls (e.g. one call per
    patch), plus a call count per stage.

    Not a per-call history — this pipeline processes thousands of patches sequentially, so only
    running totals are kept (a full history would be the point-of-a- warmup-then-N-iterations style
    benchmark that scripts/evaluation/benchmark_gpu.py already does).
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def _sync(self) -> None:
        """Synchronize the device if it's CUDA, so the measured time is pure compute for the timed
        span."""
        if self._device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Time one call of `stage`, syncing the device before and after so the measured span is
        pure compute for `stage` (not queued work from the previous or next stage)."""
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._totals[stage] += time.perf_counter() - t0
            self._counts[stage] += 1

    def add(self, stage: str, seconds: float) -> None:
        """Record an externally-timed span (e.g. file I/O) — no device sync needed."""
        self._totals[stage] += seconds
        self._counts[stage] += 1

    def totals_ms(self) -> dict[str, float]:
        """Return a dict of total time spent in each stage, in milliseconds."""
        return {k: v * 1000.0 for k, v in self._totals.items()}

    def counts(self) -> dict[str, int]:
        """Return a dict of call counts for each stage."""
        return dict(self._counts)

    def total_ms(self, *stages: str) -> float:
        """Sum of named stages in ms — e.g. `total_ms('g_a','h_a','h_s')` for an NN-equivalent
        bucket comparable to the C++ pipeline's single `t_dpu_ms`."""
        return sum(self._totals.get(s, 0.0) for s in stages) * 1000.0
