from argparse import ArgumentParser
import math
from time import perf_counter as time
import numpy as np
from pathlib import Path
import pandas as pd

from utils import datatools
from LSM import JacobiSolver

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

# Available solver methods
methods = ["jacobi", "view"]  # "view" is alias for "jacobi"
parser.add_argument(
    "--method",
    choices=methods,
    default=methods[0],
    help="The chosen method to solve the Poisson problem (default: jacobi).",
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


# Initialize the true solution for validation
xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
u_true = np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)

# Create solver instance
solver = JacobiSolver(omega=0.75, use_numba=False, verbose=False)

# Optional: warmup for numba (if enabled)
# solver.warmup(N=10)

# Run the solver
t0 = time()
result = solver.solve(u1, u2, f, h, N_iter, tolerance, u_true=u_true)
t1 = time()

# Extract results
u = result["u"]
iter_run = result["iterations"]
diff_step = result["diff_step"]
diff_true = result["diff_true"]
converged = result["converged"]
elapsed_time = t1 - t0

print("time = ", elapsed_time)
print(f"iterations = {iter_run}")
if converged:
    print(f"Converged within tolerance {tolerance}")

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
df['converged'] = converged

# Save results to data directory (automatically mirrors Experiments/ structure)
data_dir = datatools.get_data_dir()

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
