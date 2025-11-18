import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

def plot_slice(filename: str, axis: str, pos, u: np.ndarray) -> None:
    """Create a slice of the grid and store it to a matplotlib file"""
    from matplotlib import pyplot as plt

    # get axis specification
    axis: int = "z2y1x0".index(axis.lower()) // 2

    pos = float(pos)
    N = u.shape[axis]
    idx: int = int(math.floor(N * (pos + 1) / 2))

    # For debugging purposes:
    #print(axis, pos, idx)
    uslice = np.take(u, idx, axis=axis)
    plt.imshow(uslice.T, vmin=-1, vmax=1, extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def plot_result(u: np.ndarray, output_dir: str):
    """Plot the final result slices in the output directory."""
    plot_slice(f"{output_dir}/slice_x0.0.png", 'x', 0.0, u)
    plot_slice(f"{output_dir}/slice_y0.0.png", 'y', 0.0, u)
    plot_slice(f"{output_dir}/slice_z0.0.png", 'z', 0.0, u)
    print("mean: ", np.mean(u))

plot_result(np.load("Experiments/cubic/output/result.npy"), "Experiments/cubic/output")
