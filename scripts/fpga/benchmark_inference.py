#!/usr/bin/env python3
"""Benchmark Script for Real FPGA Entropy Coding.

This script measures the throughput (encoding/decoding speed) and verifies the correctness
of the C++ rANS implementation on the FPGA Target. It uses dummy data (random integers/floats)
to simulate the latent representations that would typically come from the DPU.

It is useful for:
1. Validating that the C++ extension (`ans.so`) works correctly.
2. Measuring the pure CPU performance of the entropy coding step (bottleneck analysis).

Usage (On FPGA):
    # Ensure 'ans.so' is compiled and available
    python3 benchmark_inference.py --tables <path_to_entropy_params.npz> --width 512 --height 512 --check
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add current directory to path so we can import entropy_models_real
sys.path.append(str(Path(__file__).parent))

try:
    from entropy_models_inference import EntropyBottleneck, GaussianConditional
except ImportError as e:
    print(f"Error importing entropy models: {e}")
    sys.exit(1)


def benchmark_inference(tables_path, width, height, enable_check=False):
    """Benchmark the DDC Entropy Coding on FPGA.

    Args:
        tables_path (str): Path to the entropy tables (.npz file).
        width (int): Width of the input image.
        height (int): Height of the input image.
        enable_check (bool, optional): Whether to perform a correctness check. Defaults to False.
    """
    print(f"Benchmarking DDC Entropy Coding with shape ({height}, {width})...")

    # 1. Load Tables
    print(f"Loading tables from {tables_path}...")
    try:
        data = np.load(tables_path)
    except FileNotFoundError:
        print("Error: Tables file not found. Run export_static_tables.py first.")
        sys.exit(1)

    # 2. Instantiate Models
    print("Initializing entropy models...")
    eb_channels = data["eb_cdf_length"].shape[0]

    eb = EntropyBottleneck(
        channels=eb_channels,
        quantized_cdf=data["eb_quantized_cdf"],
        cdf_length=data["eb_cdf_length"],
        offset=data["eb_offset"],
        medians=data["eb_medians"] if "eb_medians" in data else None,
    )

    gc = GaussianConditional(
        scale_table=data["gc_scale_table"],
        quantized_cdf=data["gc_quantized_cdf"],
        cdf_length=data["gc_cdf_length"],
        offset=data["gc_offset"],
    )

    # 3. Generate Dummy Data
    # Hyper-latent z (small spatial res)
    z_shape = (1, eb_channels, height // 16, width // 16)  # stride 16 approx
    z_hat = np.random.randint(-5, 5, size=z_shape).astype(np.int32)

    # Latent y (medium spatial res) - scales and means come from hyper-decoder
    y_shape = (
        1,
        height // 4,
        width // 4,
        32,
    )  # Assume 32 channels for y, usually depends on model
    # Wait, GC channels are not fixed in tables, but scale indexes are
    # We simulate y_hat and scales
    y_hat = np.random.randint(-10, 10, size=y_shape).astype(np.float32)
    scales = np.random.uniform(0.1, 5.0, size=y_shape).astype(np.float32)
    means = np.zeros_like(y_hat)

    print("Starting Benchmark loop...")
    iterations = 20
    enc_times = []
    dec_times = []

    for i in range(iterations):
        t0 = time.time()

        # Encode
        z_strings = eb.compress(z_hat)
        y_strings = gc.compress(y_hat, scales, means)

        t1 = time.time()

        # Decode
        z_rec = eb.decompress(z_strings, (z_shape[2], z_shape[3]))
        y_rec = gc.decompress(y_strings, scales, means)

        t2 = time.time()

        enc_times.append(t1 - t0)
        dec_times.append(t2 - t1)

        if enable_check and i == 0:
            # Verify correctness
            # Note: z_hat was int, z_rec is float, check error
            err_z = np.abs(z_hat.transpose(0, 2, 3, 1) - z_rec).max()  # z_rec is NHWC, z_hat NCHW
            err_y = np.abs(y_hat - y_rec).max()
            print(f"  Correctness Check (Iter 0): Max Error Z={err_z}, Y={err_y}")

    avg_enc = np.mean(enc_times)
    avg_dec = np.mean(dec_times)

    print(f"\nResults over {iterations} iterations:")
    print(f"  Avg Encoding Time: {avg_enc * 1000:.2f} ms")
    print(f"  Avg Decoding Time: {avg_dec * 1000:.2f} ms")

    # Total bytes
    total_bytes = sum(len(s) for s in z_strings) + sum(len(s) for s in y_strings)
    bpp = (total_bytes * 8) / (width * height)
    print(f"  Bitrate: {bpp:.4f} bpp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, default="entropy_params.npz")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    benchmark_inference(args.tables, args.width, args.height, args.check)
