#!/usr/bin/env python3
"""
Plot results from sequential Poisson solver.

This script automatically finds and plots the most recent data from compute_sequential.py.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_data_dir, get_figures_dir

# Set seaborn style
sns.set_theme(style="whitegrid")

# Get directories (automatically mirrors Experiments/ structure)
data_dir = get_data_dir()
figures_dir = get_figures_dir()


def plot_slice(filename: str, axis: str, pos, u: np.ndarray) -> None:
    """Create a slice of the grid and store it to a matplotlib file"""
    # get axis specification
    axis_idx: int = "z2y1x0".index(axis.lower()) // 2

    pos = float(pos)
    N = u.shape[axis_idx]
    idx: int = int(math.floor(N * (pos + 1) / 2))

    uslice = np.take(u, idx, axis=axis_idx)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(uslice.T, vmin=-1, vmax=1, extent=(-1, 1, -1, 1), cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Slice at {axis}={pos}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Slice plot saved to: {filename}")


def plot_convergence(df: pd.DataFrame, output_dir: Path):
    """Plot the convergence of the residual over iterations using seaborn."""

    # Create a long-form DataFrame for seaborn
    df_plot = df[['iteration', 'diff_step', 'diff_true']].copy()
    df_long = pd.melt(
        df_plot,
        id_vars=['iteration'],
        value_vars=['diff_step', 'diff_true'],
        var_name='metric',
        value_name='residual'
    )

    # Map metric names to readable labels
    df_long['metric'] = df_long['metric'].map({
        'diff_step': 'Step difference',
        'diff_true': 'True solution difference'
    })

    # Create convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_long,
        x='iteration',
        y='residual',
        hue='metric',
        marker='o',
        ax=ax
    )
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Convergence of Poisson Solver')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / "convergence_combined.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"Convergence plot saved to: {output_file}")

    # Also create individual plots
    # Plot 1: True solution convergence
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x='iteration',
        y='diff_true',
        marker='o',
        color='steelblue',
        ax=ax
    )
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Convergence to True Solution')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / "convergence_true.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"Convergence plot saved to: {output_file}")

    # Plot 2: Step difference convergence
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x='iteration',
        y='diff_step',
        marker='o',
        color='coral',
        ax=ax
    )
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Convergence of Step Difference')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / "convergence_step.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"Convergence plot saved to: {output_file}")


def main():
    """Main function to load data and create plots."""

    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Run compute_sequential.py first to generate data")
        return

    # Find most recent parquet file
    parquet_files = sorted(data_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not parquet_files:
        print(f"Error: No .parquet files found in {data_dir}")
        print("Run compute_sequential.py first to generate data")
        return

    input_file = parquet_files[0]
    print(f"Loading most recent data: {input_file.name}")

    # Load data
    df = pd.read_parquet(input_file)

    # Print loaded data info
    N = df['N'].iloc[0]
    iter_run = df['iter_run'].iloc[0]
    method = df['method'].iloc[0]
    elapsed_time = df['elapsed_time'].iloc[0]

    print(f"Data loaded: N={N}, iterations={iter_run}, method={method}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # Generate convergence plots
    plot_convergence(df, figures_dir)

    # Generate slice plot
    grid_file = input_file.parent / input_file.name.replace('.parquet', '_grid.npy')
    if grid_file.exists():
        u = np.load(grid_file)
        slice_file = figures_dir / "slice_x_0.0.pdf"
        plot_slice(slice_file, "x", "0.0", u)
    else:
        print(f"Warning: Grid file {grid_file.name} not found. Skipping slice plot.")

    print(f"\nAll plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
