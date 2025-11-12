from argparse import ArgumentParser
import math
from time import perf_counter as time
import numpy as np
from pathlib import Path
import pandas as pd

from utils import get_data_dir

# from numba import njit


# Create the argument-parser for easier arguments!
parser = ArgumentParser(description="Poisson problem")

parser.add_argument(
    "-N",
    type=int,
    default=100,
    help="Number of divisions along each of the 3 dimensions",
)
parser.add_argument("--iter", type=int, default=20, help="Number of (max) iterations.")
parser.add_argument(
    "-v0", "--value0", type=float, default=0.0, help="The initial value of the grid u"
)
parser.add_argument(
    "--tolerance",
    type=float,
    default=1e-8,
    help="The tolerance of the normalized Frobenius norm of the residual for the convergence.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output filename for saving results (default: data/results_N{N}_iter{iter}_{method}.npz)",
)

# Add your methods
# For each method the corresponding "step_{}" should exist.
methods = ["view"]
parser.add_argument(
    "--method",
    choices=methods,
    default=methods[0],
    help="The chosen method to solve the Poisson problem.",
)


# Grab options!
options = parser.parse_args()
N: int = options.N
method: str = options.method
N_iter: int = options.iter
tolerance: float = options.tolerance


"""
In the below code, we'll have the axes aligned as z, y, x.
"""


# Define the different methods
def step_view(
    uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float, omega: float = 0.75
) -> float:
    """Run a single Poisson step using the *view* method"""

    # The stencil weight
    c: float = 1.0 / 6.0
    h2: float = h * h
    N: int = u.shape[0] - 2

    u[1:-1, 1:-1, 1:-1] = (
        omega
        * c
        * (
            uold[0:-2, 1:-1, 1:-1]
            + uold[2:, 1:-1, 1:-1]
            + uold[1:-1, 0:-2, 1:-1]
            + uold[1:-1, 2:, 1:-1]
            + uold[1:-1, 1:-1, 0:-2]
            + uold[1:-1, 1:-1, 2:]
            + h2 * f[1:-1, 1:-1, 1:-1]
        )
    )

    return math.sqrt(np.sum((u - uold) ** 2)) / N**3


# Allocate the matrices
h: float = 2.0 / (N - 1)
u1: np.ndarray = np.full([N, N, N], options.value0, dtype=np.float64)
u1[[0, -1], :, :] = 0
u1[:, [0, -1], :] = 0
u1[:, :, [0, -1]] = 0
f: np.ndarray = np.zeros_like(u1)

# The boundary conditions are 0 on all edges.
u2: np.ndarray = u1.copy()

# Create f using broadcasting
xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
f[:] = 2 * np.pi**2 * np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)


def validate(u: np.ndarray, u_true: np.ndarray) -> float:
    """
    Validate the grid result using simple bool check.
    Relative tolerance: 1e-6
    Absolute tolerance: 1e-8
    """
    N = u.shape[0]
    diff_true = math.sqrt(np.sum((u - u_true) ** 2)) / N**3

    return diff_true


# Retrieve the method that we'll use for solving the Poisson problem
step = globals()[f"step_{method}"]

# Preset the *result* array!
# Just in case the first run is not used...
u = u2.view()
t0 = time()
diff = i = -1

# Initialize the true solution for validation
xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
u_true = np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)
diff_step = []
diff_true = []

# Run the Poisson problem until either of these are true:
# 1. Maximum number of iterations is met.
# 2. The residual falls below the tolerance.
for i in range(N_iter):
    if i == 1:
        t0 = time()

    # Swapping pointers
    if i % 2 == 0:
        uold = u1.view()
        u = u2.view()
    else:
        u = u1.view()
        uold = u2.view()

    diff_step.append(step(uold, u, f, h))
    diff_true.append(validate(u, u_true))

    # if diff < tolerance:
    # after this point, we know that `u` contains
    # the result, in any case
    # so we can safely break
    #    print("ended by tolerance check")
    #    break


t1 = time()

# Determine the actual number of iterations run
iter_run = i + 1
elapsed_time = t1 - t0
print("time = ", elapsed_time)
print(f"iterations = {iter_run}")

# Create DataFrame with convergence data
df = pd.DataFrame({
    'iteration': range(iter_run),
    'diff_step': diff_step,
    'diff_true': diff_true,
})

# Add metadata columns
df['N'] = N
df['method'] = method
df['tolerance'] = tolerance
df['elapsed_time'] = elapsed_time
df['iter_run'] = iter_run

# Save results to data directory (automatically mirrors Experiments/ structure)
data_dir = get_data_dir()

if options.output:
    base_name = options.output.replace('.npz', '').replace('.parquet', '')
    output_file = data_dir / f"{base_name}.parquet"
    grid_file = data_dir / f"{base_name}_grid.npy"
else:
    output_file = data_dir / f"results_N{N}_iter{iter_run}_{method}.parquet"
    grid_file = data_dir / f"results_N{N}_iter{iter_run}_{method}_grid.npy"

# Save convergence data as parquet
df.to_parquet(output_file, index=False)
print(f"Results saved to: {output_file}")

# Save the 3D grid separately for slice plotting
np.save(grid_file, u)
print(f"Grid saved to: {grid_file}")
