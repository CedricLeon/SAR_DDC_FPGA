#!/usr/bin/env python3
"""
Debugging script for analyzing training runs and identifying issues.
This script helps identify common problems in neural network training by analyzing
logs from previous runs and visualizing key patterns.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm


def plot_loss_components(data, title="Loss Components Analysis"):
    """Plot loss components to check if they're balanced properly."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot main loss
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss Values")
    ax1.plot(data["step"], data["loss"], "b-", label="Total Loss")
    ax1.plot(data["step"], data["mse"], "g-", label="MSE Loss")
    ax1.plot(data["step"], data["bpp"], "r-", label="BPP Loss")
    ax1.tick_params(axis="y")

    # Plot MSE/BPP ratio on second axis to check balance
    ax2 = ax1.twinx()
    ax2.set_ylabel("MSE/BPP Ratio (log scale)")
    ratio = data["mse"] / (data["bpp"] + 1e-8)  # Avoid division by zero
    # Use log scale for the ratio to better visualize imbalances
    ax2.semilogy(data["step"], ratio, "k--", alpha=0.5, label="MSE/BPP Ratio")

    # Add horizontal lines for reference ratios
    ax2.axhline(y=1.0, color="gray", linestyle="-", alpha=0.3)
    ax2.axhline(y=0.1, color="gray", linestyle=":", alpha=0.3)
    ax2.axhline(y=10.0, color="gray", linestyle=":", alpha=0.3)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    return fig


def plot_gradient_analysis(grad_data, title="Gradient Analysis"):
    """Plot gradient norms by network component with filled distributions.

    Creates 4 subplots for g_a (encoder), g_s (decoder), h_a (hyperprior encoder),
    and h_s (hyperprior decoder), showing gradient evolution over training steps.
    """
    # Create a figure with 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Helper function to extract component name and layer index from parameter name
    def parse_layer_name(name):
        # Expected format: "g_a.0.0.weight", "g_s.1.1.bias", etc.
        if not name or "." not in name:
            return None, None, 999  # Unknown component, high index to sort last

        parts = name.split(".")
        if len(parts) < 2:
            return None, None, 999

        component = parts[0]
        try:
            # Extract indices to determine layer depth
            depth_index = int(parts[1]) * 100
            if len(parts) > 2:
                depth_index += int(parts[2]) if parts[2].isdigit() else 0
            return component, name, depth_index
        except (ValueError, IndexError):
            return component, name, 999

    # Group parameters by component (g_a, g_s, h_a, h_s)
    component_layers = {
        "g_a": [],  # Main encoder
        "g_s": [],  # Main decoder
        "h_a": [],  # Hyperprior encoder
        "h_s": [],  # Hyperprior decoder
    }

    # Collect and sort layers by component and depth
    for col in grad_data.columns:
        if col == "step" or col == "overall_norm":
            continue

        component, full_name, depth_index = parse_layer_name(col)
        if component in component_layers:
            component_layers[component].append((full_name, depth_index))

    # Sort each component's layers by depth (earlier layers first)
    for component in component_layers:
        component_layers[component].sort(key=lambda x: x[1])

    # Define titles and positions for each component
    component_info = [
        {"name": "g_a", "title": "Main Encoder (g_a)", "position": 0},
        {"name": "g_s", "title": "Main Decoder (g_s)", "position": 1},
        {"name": "h_a", "title": "Hyperprior Encoder (h_a)", "position": 2},
        {"name": "h_s", "title": "Hyperprior Decoder (h_s)", "position": 3},
    ]

    # Color map for gradient visualization (blue to red)
    cmap = plt.cm.viridis

    # Process each component
    for info in component_info:
        component = info["name"]
        ax = axes[info["position"]]
        component_title = info["title"]

        # Get sorted layers for this component
        layers = component_layers[component]

        if not layers:
            ax.text(
                0.5,
                0.5,
                f"No gradient data for {component_title}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title(component_title)
            continue

        # Extract layer names only
        layer_names = [x[0] for x in layers]

        # Set up the plot
        ax.set_yscale("log")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Gradient Norm (log scale)")
        ax.set_title(component_title)

        # Add reference lines for vanishing/exploding gradients
        ax.axhline(
            y=1e-7, color="r", linestyle="--", alpha=0.5, label="Vanishing Threshold"
        )
        ax.axhline(y=100, color="r", linestyle=":", alpha=0.5, label="Explosion Risk")

        # Number of layers in this component
        n_layers = len(layer_names)
        if n_layers == 0:
            continue

        # Calculate z-order - earlier layers in front (larger z-order)
        z_orders = list(range(n_layers, 0, -1))

        # Plot each layer with filled area
        for i, layer_name in enumerate(layer_names):
            # Skip if no valid data
            if layer_name not in grad_data.columns:
                continue

            color = cmap(i / max(1, n_layers - 1))

            # Get data and create filled plot
            x = grad_data["step"]
            y = grad_data[layer_name]

            # Plot filled area from minimum value up to the gradient norm
            ax.fill_between(
                x,
                1e-10,
                y,  # From a small value to avoid log(0)
                alpha=0.7,
                color=color,
                label=layer_name
                if i % max(1, n_layers // 5) == 0
                else "",  # Only label every few layers
                zorder=z_orders[i],  # Early layers in front
            )

        # Add a legend for this component
        if n_layers > 10:
            # For many layers, show a simplified legend
            ax.legend(loc="upper right", fontsize="small", title="Selected Layers")
        else:
            ax.legend(loc="upper right", fontsize="small")

        ax.grid(True, alpha=0.3)

    # Add overall title and adjust layout
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig


def plot_weight_distribution_evolution(
    weight_data, title="Weight Distribution Over Time"
):
    """Plot how the distribution of weights changes during training."""
    # Create a 4x4 grid to show more layers
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Sample at most 16 layers to plot (4x4 grid)
    selected_layers = []
    for col in weight_data.columns:
        if col != "step" and col not in selected_layers and len(selected_layers) < 16:
            selected_layers.append(col)

    # Create plots for selected layers
    for i, layer in enumerate(selected_layers):
        ax = axes[i]

        # Divide training into 4 segments to see evolution
        # Make a copy of the dataframe to avoid modifying original
        df_copy = weight_data.copy()

        # Make sure we have non-NaN data for this layer
        mask = ~df_copy[layer].isna()
        if mask.sum() == 0:
            continue  # Skip if no valid data

        # Filter out NaN values for this layer
        plot_data = df_copy.loc[mask, ["step", layer]].copy()

        if len(plot_data) == 0:
            continue  # Skip if no data after filtering

        # Ensure plot_data is sorted by step for proper segmentation
        plot_data = plot_data.sort_values("step")

        # Divide into 4 segments
        if len(plot_data) < 4:
            segments = [plot_data]  # Too few points to segment
        else:
            # Calculate segment boundaries
            segment_size = len(plot_data) // 4
            segments = [
                plot_data.iloc[i * segment_size : (i + 1) * segment_size]
                for i in range(3)
            ]
            # Last segment gets the remainder
            segments.append(plot_data.iloc[3 * segment_size :])

        colors = ["blue", "green", "orange", "red"]

        for j, segment in enumerate(segments):
            if len(segment) > 0:
                ax.plot(
                    segment["step"].values,
                    segment[layer].values,
                    color=colors[j % len(colors)],
                    label=f"Segment {j + 1}",
                )

        ax.set_title(f"Layer: {layer}", fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Weight Norm", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Turn off any unused subplots
    for i in range(len(selected_layers), 16):
        axes[i].set_visible(False)

    fig.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)
    return fig


def analyze_wandb_run(run_id=None, project=None):
    """Fetch and analyze data from a W&B run using scan_history() for better performance."""

    print(f"Analyzing W&B run: {run_id} in project {project}")

    # Initialize W&B
    api = wandb.Api()

    # Find the run
    if run_id:
        run = api.run(f"{project}/{run_id}")
    else:
        # Get most recent run
        runs = api.runs(project)
        if not runs:
            print("No runs found in the project")
            return
        run = runs[0]
        run_id = run.id

    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")

    # Print minimal configuration information
    print("\n===== RUN CONFIGURATION =====")
    config = run.config
    print(f"Run Name: {run.name}")
    num_steps = sum(1 for _ in run.scan_history())
    print(f"Training Steps: ~{num_steps}")

    # Extract key configuration values
    lambda_val = config["model"]["criterion"]["lmbda"]
    lr = config["model"]["net_optimizer"]["lr"]
    print(f"Lambda: {lambda_val}")
    print(f"Learning Rate: {lr}")

    # Get final test metrics if available
    print("\n===== TEST METRICS =====")
    summary = run.summary
    test_metrics = {k: v for k, v in summary.items() if k.startswith("test/")}
    if test_metrics:
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")
    else:
        print("No test metrics found")

    print("\nDownloading metrics...")

    # Use scan_history which is more efficient for large runs
    # Each item in scan_history is a single step
    train_data = {
        "step": [],
        "loss": [],
        "mse": [],
        "bpp": [],
        "psnr": [],
        "aux": [],
        "loss_ratio": [],
        "epoch": [],
    }
    grad_data = {"step": []}
    weight_data = {"step": []}

    # First scan: Get all basic training metrics
    print("Scanning training metrics...")

    raw_metrics = []
    for item in tqdm(run.scan_history()):
        raw_metrics.append(item)

    print(f"Downloaded {len(raw_metrics)} records")

    # Get a list of all keys to identify metrics
    all_keys = set()
    for item in raw_metrics:
        all_keys.update(item.keys())
    print(f"Found {len(all_keys)} unique metric keys")

    # Identify relevant metrics
    train_metrics = [k for k in all_keys if k.startswith("train/")]
    valid_metrics = [k for k in all_keys if k.startswith("valid/")]
    grad_metrics = [k for k in all_keys if "grad_norm" in k]
    weight_metrics = [k for k in all_keys if "weight_norm" in k]

    print(f"Training metrics: {len(train_metrics)}")
    print(f"Validation metrics: {len(valid_metrics)}")
    print(f"Gradient metrics: {len(grad_metrics)}")
    print(f"Weight metrics: {len(weight_metrics)}")

    # Extract metrics into dataframes
    for item in tqdm(raw_metrics, desc="Processing metrics"):
        step = item.get("_step")
        if not step:
            continue

        # Basic training metrics
        loss = item.get("train/loss")
        mse = item.get("train/mse")
        bpp = item.get("train/bpp")
        psnr = item.get("train/psnr")
        aux = item.get("train/aux")
        epoch = item.get("_runtime")  # Use runtime if epoch not available

        if loss is not None:
            # Clip loss to max 100 for better visualization
            if loss > 100:
                print(f"Clipping loss at step {step}: {loss:.2f} -> 100.0")
                loss = 100.0

            train_data["step"].append(step)
            train_data["loss"].append(loss)
            train_data["mse"].append(mse if mse is not None else np.nan)
            train_data["bpp"].append(bpp if bpp is not None else np.nan)
            train_data["psnr"].append(psnr if psnr is not None else np.nan)
            train_data["aux"].append(aux if aux is not None else np.nan)
            train_data["epoch"].append(epoch if epoch is not None else np.nan)

            if mse is not None and bpp is not None and bpp != 0:
                train_data["loss_ratio"].append(mse / bpp)
            else:
                train_data["loss_ratio"].append(np.nan)

        # Gradient metrics
        grad_norm = item.get("train/grad_norm")
        if grad_norm is not None:
            if step not in grad_data["step"]:
                grad_data["step"].append(step)
                if "overall_norm" not in grad_data:
                    grad_data["overall_norm"] = []
                grad_data["overall_norm"].append(grad_norm)

            # Layer-specific gradient norms
            for key in grad_metrics:
                if key != "train/grad_norm" and key in item:
                    layer_name = key.replace("train/grad_norm/", "")
                    if layer_name not in grad_data:
                        grad_data[layer_name] = [np.nan] * (
                            len(grad_data["step"]) - 1
                        )  # Fill with NaN for previous steps
                    grad_data[layer_name].append(item[key])

        # Weight metrics
        for key in weight_metrics:
            if key in item:
                if step not in weight_data["step"]:
                    weight_data["step"].append(step)

                layer_name = key.replace("train/weight_norm/", "")
                if layer_name not in weight_data:
                    weight_data[layer_name] = [np.nan] * (
                        len(weight_data["step"]) - 1
                    )  # Fill with NaN for previous steps
                weight_data[layer_name].append(item[key])

    # Fill missing values in gradient and weight data
    for layer in grad_data:
        if layer != "step" and len(grad_data[layer]) < len(grad_data["step"]):
            grad_data[layer].extend(
                [np.nan] * (len(grad_data["step"]) - len(grad_data[layer]))
            )

    for layer in weight_data:
        if layer != "step" and len(weight_data[layer]) < len(weight_data["step"]):
            weight_data[layer].extend(
                [np.nan] * (len(weight_data["step"]) - len(weight_data[layer]))
            )

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    grad_df = pd.DataFrame(grad_data)
    weight_df = pd.DataFrame(weight_data)

    # Create output directory
    output_dir = f"logs/debug_anomalies/debug_analysis_{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("Generating plots...")

    # Basic metrics plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["step"], train_df["loss"], label="Loss (clipped at 100)")
    if not train_df["mse"].isna().all():
        plt.plot(train_df["step"], train_df["mse"], label="MSE")
    if not train_df["bpp"].isna().all():
        plt.plot(train_df["step"], train_df["bpp"], label="BPP")
    if not train_df["psnr"].isna().all():
        plt.plot(train_df["step"], train_df["psnr"], label="PSNR")

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/training_metrics.png")
    print(f"Saved training metrics plot to {output_dir}/training_metrics.png")

    # Loss components plot
    if not train_df["mse"].isna().all() and not train_df["bpp"].isna().all():
        fig = plot_loss_components(train_df)
        plt.savefig(f"{output_dir}/loss_components.png")
        print(f"Saved loss components plot to {output_dir}/loss_components.png")
    else:
        print("Missing MSE or BPP data, skipping loss components plot")

    # Gradient analysis plot
    if len(grad_df) > 1 and len(grad_df.columns) > 1:
        fig = plot_gradient_analysis(grad_df)
        plt.savefig(f"{output_dir}/gradient_analysis.png")
        print(f"Saved gradient analysis plot to {output_dir}/gradient_analysis.png")
    else:
        print("Not enough gradient data to generate gradient analysis plot")

    # Weight distribution plot
    if len(weight_df) > 1 and len(weight_df.columns) > 1:
        fig = plot_weight_distribution_evolution(weight_df)
        plt.savefig(f"{output_dir}/weight_distribution.png")
        print(f"Saved weight distribution plot to {output_dir}/weight_distribution.png")
    else:
        print("Not enough weight data to generate weight distribution plot")

    # Save raw data for reference
    train_df.to_csv(f"{output_dir}/training_metrics.csv", index=False)
    if len(grad_df.columns) > 1:
        grad_df.to_csv(f"{output_dir}/gradient_data.csv", index=False)
    if len(weight_df.columns) > 1:
        weight_df.to_csv(f"{output_dir}/weight_data.csv", index=False)

    # Print summary analysis
    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Analyzed {len(train_df)} training steps")

    if len(train_df) > 0:
        # Calculate improvement ratio
        if len(train_df) > 50:
            early_loss_avg = train_df["loss"].iloc[:50].mean()
            late_loss_avg = train_df["loss"].iloc[-50:].mean()
            improvement = (early_loss_avg - late_loss_avg) / early_loss_avg * 100

            print(f"Loss improvement: {improvement:.2f}%")

            if improvement < 5:
                print("WARNING: Very little improvement detected. Possible stagnation.")
                print("\nSuggested early stopping settings:")
                print("  patience: 5  # epochs with no improvement")
                print("  min_delta: 0.01  # minimum change to count as improvement")
                print("  mode: min  # minimize the validation loss")
                print("  Example YAML configuration:")
                print("    early_stopping:")
                print("      monitor: 'valid/loss'")
                print("      patience: 5")
                print("      min_delta: 0.01")
                print("      mode: 'min'")
            elif improvement < 0:
                print("WARNING: Loss getting worse over time! Training is diverging.")

        # Check for NaN values
        nan_count = train_df["loss"].isna().sum()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN values in loss!")

        # Calculate loss component ratio
        if not train_df["mse"].isna().all() and not train_df["bpp"].isna().all():
            mse_bpp_ratio = (
                train_df["mse"].mean() / train_df["bpp"].replace(0, np.nan).mean()
            )
            print(f"Average MSE/BPP ratio: {mse_bpp_ratio:.4f}")

            if mse_bpp_ratio > 100:
                print("WARNING: MSE dominates BPP by >100x. Check lambda value.")
            elif mse_bpp_ratio < 0.01:
                print("WARNING: BPP dominates MSE by >100x. Check lambda value.")

    if len(grad_df) > 0 and "overall_norm" in grad_df:
        # Check for gradient issues - using 1e-7 threshold as in sar_ddc_module.py
        if grad_df["overall_norm"].min() < 1e-7:
            print(
                "WARNING: Very small gradients detected (< 1e-7). Possible vanishing gradient problem."
            )
        if grad_df["overall_norm"].max() > 100:
            print(
                "WARNING: Very large gradients detected (> 100). Possible exploding gradient problem."
            )

    print("\nAnalysis complete. Results saved to:", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Training Debug Analysis Tool")
    parser.add_argument("--run_id", type=str, help="W&B run ID to analyze")
    parser.add_argument(
        "--project", type=str, default="sar-ddc-test", help="W&B project name"
    )
    args = parser.parse_args()

    output_dir = analyze_wandb_run(args.run_id, args.project)

    print("\nSuggestions for resolving training issues:")
    print("1. Check loss ratio - if MSE/BPP is extremely high or low, adjust lambda")
    print(
        "2. For vanishing gradients - try different initialization or activation functions"
    )
    print(
        "3. For exploding gradients - reduce learning rate or increase gradient clipping"
    )
    print(
        "4. For stagnating loss - check early stopping patience or adjust learning rate"
    )
    print("5. For NaN values - check for division by zero or log(0) operations")
    print(
        "6. For diverging networks - check normalization, initialization, and lambda balancen\n\n\n\n"
    )


if __name__ == "__main__":
    main()
