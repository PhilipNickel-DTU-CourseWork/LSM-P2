#!/usr/bin/env python3
"""
Plot results from MPI sliced Poisson solver.

This script automatically finds and plots the most recent data from compute_mpi_sliced.py.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import datatools

# Set seaborn style
sns.set_theme(style="whitegrid")

# Get directories (automatically mirrors Experiments/ structure)
data_dir = datatools.get_data_dir()
figures_dir = datatools.get_figures_dir()


def plot_slice(filename: str, axis: str, pos, u: np.ndarray) -> None:
    """Create a slice of the grid and store it to a matplotlib file"""
    # get axis specification
    axis_idx: int = "z2y1x0".index(axis.lower()) // 2

    pos = float(pos)
    N = u.shape[axis_idx]
    idx: int = int(math.floor(N * (pos + 1) / 2))

    uslice = np.take(u, idx, axis=axis_idx)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(uslice.T, vmin=-1, vmax=1, extent=(-1, 1, -1, 1), cmap="RdBu_r")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Slice at {axis}={pos}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Slice plot saved to: {filename}")


def plot_per_rank_performance(df_perrank: pd.DataFrame, output_dir: Path):
    """Plot per-rank performance metrics."""

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Wall time per rank
    sns.barplot(data=df_perrank, x="mpi_rank", y="wall_time", ax=axes[0], color="steelblue")
    axes[0].set_title("Wall Time per Rank")
    axes[0].set_xlabel("MPI Rank")
    axes[0].set_ylabel("Wall Time (s)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Compute time per rank
    sns.barplot(data=df_perrank, x="mpi_rank", y="compute_time", ax=axes[1], color="coral")
    axes[1].set_title("Compute Time per Rank")
    axes[1].set_xlabel("MPI Rank")
    axes[1].set_ylabel("Compute Time (s)")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: MPI communication time per rank
    sns.barplot(data=df_perrank, x="mpi_rank", y="mpi_comm_time", ax=axes[2], color="forestgreen")
    axes[2].set_title("MPI Communication Time per Rank")
    axes[2].set_xlabel("MPI Rank")
    axes[2].set_ylabel("MPI Comm Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "performance_per_rank.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"Per-rank performance plot saved to: {output_file}")


def plot_timing_breakdown(df_global: pd.DataFrame, df_perrank: pd.DataFrame, output_dir: Path):
    """Plot timing breakdown comparing global and per-rank totals."""

    # Create timing breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Global timing breakdown
    global_times = {
        "Compute": df_global["compute_time"].iloc[0],
        "MPI Comm": df_global["mpi_comm_time"].iloc[0],
    }
    colors = ["coral", "forestgreen"]
    axes[0].pie(
        global_times.values(),
        labels=global_times.keys(),
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    axes[0].set_title("Global Timing Breakdown\n(Sum across ranks)")

    # Plot 2: Per-rank timing stacked bar
    df_plot = df_perrank[["mpi_rank", "compute_time", "mpi_comm_time"]].set_index("mpi_rank")
    df_plot.plot(kind="bar", stacked=True, ax=axes[1], color=colors)
    axes[1].set_title("Per-Rank Timing Breakdown")
    axes[1].set_xlabel("MPI Rank")
    axes[1].set_ylabel("Time (s)")
    axes[1].legend(["Compute", "MPI Comm"])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "timing_breakdown.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"Timing breakdown plot saved to: {output_file}")


def print_summary(df_config: pd.DataFrame, df_global: pd.DataFrame, df_perrank: pd.DataFrame):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Configuration
    N = df_config["N"].iloc[0]
    h = df_config["h"].iloc[0]
    method = df_config["method"].iloc[0]
    mpi_size = df_config["mpi_size"].iloc[0]
    omega = df_config["omega"].iloc[0]
    tolerance = df_config["tolerance"].iloc[0]

    print(f"\nConfiguration:")
    print(f"  Grid size: N = {N}")
    print(f"  Grid spacing: h = {h:.6f}")
    print(f"  Method: {method}")
    print(f"  MPI size: {mpi_size} ranks")
    print(f"  Omega: {omega}")
    print(f"  Tolerance: {tolerance:.2e}")

    # Global results
    iterations = df_global["iterations"].iloc[0]
    converged = df_global["converged"].iloc[0]
    final_residual = df_global["final_residual"].iloc[0]
    wall_time = df_global["wall_time"].iloc[0]
    compute_time = df_global["compute_time"].iloc[0]
    mpi_comm_time = df_global["mpi_comm_time"].iloc[0]

    print(f"\nGlobal Results:")
    print(f"  Iterations: {iterations}")
    print(f"  Converged: {converged}")
    print(f"  Final residual: {final_residual:.6e}")
    if "final_error" in df_global.columns:
        final_error = df_global["final_error"].iloc[0]
        print(f"  Final error: {final_error:.6e}")

    print(f"\nGlobal Timing (aggregated across ranks):")
    print(f"  Wall time (max): {wall_time:.6f} s")
    print(f"  Compute time (sum): {compute_time:.6f} s")
    print(f"  MPI comm time (sum): {mpi_comm_time:.6f} s")

    # Per-rank statistics
    print(f"\nPer-Rank Statistics:")
    print(f"  Wall time: min={df_perrank['wall_time'].min():.6f} s, "
          f"max={df_perrank['wall_time'].max():.6f} s, "
          f"mean={df_perrank['wall_time'].mean():.6f} s")
    print(f"  Compute time: min={df_perrank['compute_time'].min():.6f} s, "
          f"max={df_perrank['compute_time'].max():.6f} s, "
          f"mean={df_perrank['compute_time'].mean():.6f} s")
    print(f"  MPI comm time: min={df_perrank['mpi_comm_time'].min():.6f} s, "
          f"max={df_perrank['mpi_comm_time'].max():.6f} s, "
          f"mean={df_perrank['mpi_comm_time'].mean():.6f} s")

    print("\n" + "=" * 60)


def main():
    """Main function to load data and create plots."""

    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Run compute_mpi_sliced.py first to generate data")
        return

    # Find most recent config file
    config_files = sorted(
        data_dir.glob("*_config.parquet"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not config_files:
        print(f"Error: No config files found in {data_dir}")
        print("Run compute_mpi_sliced.py first to generate data")
        return

    config_file = config_files[0]
    base_name = config_file.name.replace("_config.parquet", "")
    print(f"Loading most recent data: {base_name}")

    # Load all data files
    global_file = data_dir / f"{base_name}_global.parquet"
    perrank_file = data_dir / f"{base_name}_perrank.parquet"
    grid_file = data_dir / f"{base_name}_grid.npy"

    if not global_file.exists() or not perrank_file.exists():
        print(f"Error: Missing data files for {base_name}")
        return

    df_config = pd.read_parquet(config_file)
    df_global = pd.read_parquet(global_file)
    df_perrank = pd.read_parquet(perrank_file)

    # Print summary
    print_summary(df_config, df_global, df_perrank)

    # Create plots
    plot_per_rank_performance(df_perrank, figures_dir)
    plot_timing_breakdown(df_global, df_perrank, figures_dir)

    # Generate slice plot
    if grid_file.exists():
        u = np.load(grid_file)
        slice_file = figures_dir / "slice_x_0.0.pdf"
        plot_slice(slice_file, "x", "0.0", u)
    else:
        print(f"Warning: Grid file {grid_file.name} not found. Skipping slice plot.")

    print(f"\nAll plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
