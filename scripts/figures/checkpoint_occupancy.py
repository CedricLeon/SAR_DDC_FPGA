#!/usr/bin/env python3
"""checkpoint_occupancy.py — DPU + CPU occupancy at ladder checkpoints r0 / r4 / r7.

**Prints a table, draws nothing.** F7 (§4.3) was the conditional "occupancy checkpoint
panel"; the verdict (P1.2) is *no figure* — the r7 column reproduces F4's knee
occupancy panel (to ~1 %, which is the cross-check), and the genuinely new content
(the r0 baseline and the r4→r7 shift) is a couple of sentences for W5.

Two instruments, deliberately kept apart:

* **r0** — the E3 *sequential* per-stage benchmark
  (``s0/<arch>/s0_compress_entoff.json``, 1 worker, 1 DPU core, entropy unoptimised).
  There is no r0 trace (the tracer is fan-out only). "Occupancy" here is a time-share
  of the single active thread: DPU-time / (DPU + CPU) time per patch. Reprojected onto
  the 3-core DPU / 4-core A53 complex (÷3, ÷4) so it sits on the same axis as r4/r7 —
  this reprojection is a construction, flagged in the output.
* **r4 / r7** — the full-scene fan-out traces at each arch's knee
  (``occupancy/<arch>/r{4,7}_full*.csv``), through the SAME ``[t1 - e, t1]``
  reconstruction as the lane-grid occupancy (``fanout_occupancy.occupancy``), with
  ``e_table`` from the arch's uncontended 1-lane trace. r4 = scheduling only (no CPU
  kernels, entropy present but unoptimised); r7 = + neon + dbuf + entropy-opt. DPU
  work per patch is identical across r4/r7, so DPU-busy tracks throughput; the CPU
  side changes with the kernel optimisations.

Raw span unions (no reconstruction) give 99–100 % on every core — that is DPU
queue-wait, not compute (§4.0). This script never does that.

Run:  conda activate DDC_FPGA && python scripts/figures/checkpoint_occupancy.py
"""

import json

import fanout_occupancy as fo
from _figutils import ARCHS, DISPLAY, KNEE, occupancy_full_trace_path

N_DPU, N_A53 = 3, 4
DPU_KERNELS = ("g_a", "h_a", "h_s")


def r0_from_s0(arch: str):
    """(DPU ms/patch, CPU ms/patch) from the sequential per-stage breakdown."""
    stages = json.load(open(f"results/date27/s0/{arch}/s0_compress_entoff.json"))["stages"]
    dpu = sum(v["mean_ms"] for k, v in stages.items() if k in DPU_KERNELS)
    cpu = sum(v["mean_ms"] for k, v in stages.items() if k not in DPU_KERNELS)
    return dpu, cpu


def checkpoint(arch: str, rung: int):
    """(DPU 3-core mean %, per-core %, CPU 4-core %, window_s) for r4 or r7."""
    path = occupancy_full_trace_path(arch, rung)
    nl = KNEE[arch]
    busy, wlen, _ = fo.occupancy(arch, nl, "median", trace_path=path)
    cpu4 = fo._cpu_core_busy(arch, nl, trace_path=path)
    dpu_mean = sum(busy.values()) / N_DPU * 100
    return dpu_mean, {c: busy[c] * 100 for c in busy}, cpu4, wlen / 1000, path.name


def main():
    print(f"{'arch':6} {'ckpt':5} {'DPU %':>7} {'CPU %':>7}   detail")
    print("-" * 76)
    for a in ARCHS:
        dpu_ms, cpu_ms = r0_from_s0(a)
        tot = dpu_ms + cpu_ms
        dpu_duty, cpu_duty = dpu_ms / tot * 100, cpu_ms / tot * 100
        print(
            f"{DISPLAY[a]:6} {'r0':5} {dpu_duty / N_DPU:7.1f} {cpu_duty / N_A53:7.1f}   "
            f"seq 1-thread duty DPU {dpu_duty:.0f}% / CPU {cpu_duty:.0f}%  "
            f"(÷{N_DPU}, ÷{N_A53} → complex); {dpu_ms:.1f}/{cpu_ms:.1f} ms per patch"
        )
        for rung in (4, 7):
            dpu_mean, per_core, cpu4, win_s, name = checkpoint(a, rung)
            pc = "/".join(f"{per_core[c]:.0f}" for c in sorted(per_core))
            print(
                f"{DISPLAY[a]:6} {'r' + str(rung):5} {dpu_mean:7.1f} {cpu4:7.1f}   "
                f"cores {pc}  window {win_s:.0f}s  ({name})"
            )
        print()


if __name__ == "__main__":
    main()
