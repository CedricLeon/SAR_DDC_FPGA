"""Standalone script to compute statistics over all CoSAR images found in a given folder.

The statistics are computed over the pre-processed data, i.e., each image is loaded symmetrized,
squared, and log-transformed (natural basis).
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import from src
sys.path.append("../..")
from src.utils.sar_utils import load_cosar, symmetrize


class RunningStatistics:
    """Class to efficiently track running statistics without keeping all data in memory."""

    def __init__(self, name="dataset"):
        self.name = name
        self.n = 0
        self.mean = 0
        self.M2 = 0  # For variance calculation
        self.min_val = float("inf")
        self.max_val = float("-inf")

        # Store histogram data
        self.histogram_bins = np.linspace(-20.0, 20.0, 300)
        self.histogram_counts = np.zeros(len(self.histogram_bins) - 1, dtype=np.int64)

        # Track percentiles efficiently by sampling values
        self.sample_values = []
        self.max_samples = 1_000_000  # Store up to 1M samples for percentile calculation
        self.sample_prob = 0.05  # Sample 5% of all pixels

    def update(self, data):
        """Update statistics with new batch of data."""
        batch_size = data.size
        flat_data = data.ravel()

        # Update count
        self.n += batch_size

        # Update min and max
        self.min_val = min(self.min_val, np.min(flat_data))
        self.max_val = max(self.max_val, np.max(flat_data))

        # Update mean and M2 using Welford's online algorithm, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        delta = flat_data - self.mean
        self.mean += np.sum(delta) / self.n
        delta2 = flat_data - self.mean
        self.M2 += np.sum(delta * delta2)

        # Update histogram
        hist, _ = np.histogram(flat_data, bins=self.histogram_bins)
        self.histogram_counts += hist

        # Sample values for percentile calculation
        if self.max_samples > 0:
            mask = np.random.random(flat_data.shape) < self.sample_prob
            sampled = flat_data[mask]

            # Only append if we still have room
            if len(self.sample_values) + len(sampled) <= self.max_samples:
                self.sample_values.extend(sampled.tolist())
            else:
                available = self.max_samples - len(self.sample_values)
                if available > 0:
                    self.sample_values.extend(sampled[:available].tolist())
                self.max_samples = 0  # Stop collecting samples

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0

    @property
    def std(self):
        return np.sqrt(self.variance)

    def percentile(self, q):
        if not self.sample_values:
            return None
        return np.percentile(self.sample_values, q)

    def report(self):
        """Generate a report of the statistics."""
        print(f"Statistics for {self.name} ({self.n:,} values):")
        print(f"  Mean: {self.mean}")
        print(f"  Std dev: {self.std}")
        print(f"  Min: {self.min_val}")
        print(f"  Max: {self.max_val}")

        # Calculate percentiles if we have samples
        if self.sample_values:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            values = [self.percentile(q) for q in percentiles]

            print("  Percentiles:")
            for p, v in zip(percentiles, values):
                print(f"    {p:3d}%: {v}")

        # Return basic stats as a dict for convenience
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
        }


def process_sar_image(filepath, stats_intensity_log, stats_amp_log_sqrt):
    """Process a single SAR image and update statistics."""
    print(f"Processing {filepath.name}...")

    # Load the image
    sar_data = load_cosar(filepath)
    if sar_data is None:
        print(f"Failed to load {filepath}")
        return False

    print(f"  Image shape: {sar_data.shape}, memory: {sar_data.nbytes / (1024**2):.2f} MB")

    # Apply symmetrization
    sar_data = symmetrize(sar_data)

    # Square the components
    sar_data = np.square(sar_data)
    intensity = sar_data[:, :, 0] + sar_data[:, :, 1]

    # Apply log transformation
    inten_log = np.log(intensity + 1e-2)
    amp_log_sqrt = np.log(np.sqrt(intensity) + 1e-2)

    # Update statistics
    stats_intensity_log.update(inten_log)
    stats_amp_log_sqrt.update(amp_log_sqrt)

    # Free memory
    del sar_data, intensity, inten_log, amp_log_sqrt


def plot_histogram(stats, title, filename):
    """Plot histogram from statistics object and save to file."""
    plt.figure(figsize=(10, 6))

    # Calculate bin centers from bin edges
    bin_centers = (stats.histogram_bins[:-1] + stats.histogram_bins[1:]) / 2

    # Plot histogram
    plt.bar(
        bin_centers,
        stats.histogram_counts,
        width=(stats.histogram_bins[1] - stats.histogram_bins[0]),
        alpha=0.7,
        color="steelblue",
    )

    # Add vertical lines for statistics
    plt.axvline(
        stats.mean,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {stats.mean:.4f}",
    )
    plt.axvline(
        stats.mean - stats.std,
        color="g",
        linestyle=":",
        label=f"Mean + Std: {stats.mean - stats.std:.4f}",
    )
    plt.axvline(
        stats.mean + stats.std,
        color="g",
        linestyle=":",
        label=f"Mean - Std: {stats.mean + stats.std:.4f}",
    )

    # Add percentiles if available
    if stats.sample_values:
        p01 = stats.percentile(1)
        p10 = stats.percentile(10)
        p90 = stats.percentile(90)
        p99 = stats.percentile(99)
        plt.axvline(p01, color="yellow", linestyle="-.", label=f"1st percentile: {p01:.4f}")
        plt.axvline(p10, color="orange", linestyle="-.", label=f"10th percentile: {p10:.4f}")
        plt.axvline(p90, color="orange", linestyle="-.", label=f"90th percentile: {p90:.4f}")
        plt.axvline(p99, color="yellow", linestyle="-.", label=f"99th percentile: {p99:.4f}")

    # Add labels and title
    plt.xlabel("Value (log scale)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    # Configuration
    data_dir = Path("../../data/TSX_cos_files")
    output_dir = Path("../../data/analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    # List .cos files
    cos_files = list(data_dir.glob("*.cos"))
    print(f"Found {len(cos_files)} .cos files in {data_dir}")

    # Create statistics trackers
    stats_intensity_log = RunningStatistics("Intensity (log + 1e-2)")
    stats_amp_log_sqrt = RunningStatistics("Amplitude (log(sqrt(intensity) + 1e-2))")

    # Process each image
    for file_path in cos_files:
        process_sar_image(file_path, stats_intensity_log, stats_amp_log_sqrt)

    # Generate reports
    print("\n=== INTENSITY STATISTICS (log + 1e-2) ===")
    stats_intensity_log.report()
    plot_histogram(
        stats_intensity_log,
        "Intensity (log + 1e-2) Distribution",
        output_dir / "intensity_log_1e-2_histogram.png",
    )

    print("\n=== AMPLITUDE STATISTICS (log(sqrt(intensity) + 1e-2)) ===")
    stats_amp_log_sqrt.report()
    plot_histogram(
        stats_amp_log_sqrt,
        "Amplitude (log(sqrt(intensity) + 1e-2)) Distribution",
        output_dir / "amplitude_log-sqrt_1e-2_histogram.png",
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")
