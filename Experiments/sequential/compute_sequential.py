import math
from time import perf_counter as time
import numpy as np
from pathlib import Path
import pandas as pd

from utils import datatools, cli
from LSM import (
    SequentialJacobi,
    setup_sinusoidal_problem,
    sinusoidal_exact_solution,
)

# Available solvers for future experiments:
# - SequentialJacobi(use_numba=False): Pure numpy (baseline)
# - SequentialJacobi(use_numba=True): JIT-compiled with numba
# - MPIJacobiCubic: MPI with 3D cubic decomposition (TODO)
# - MPIJacobiSliced: MPI with 1D sliced decomposition (TODO)

# Create the argument parser using shared utility
parser = cli.create_parser(
    methods=["jacobi", "view"],  # "view" is alias for "jacobi"
    default_method="jacobi",
    description="Sequential Poisson problem solver",
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

# Set up the test problem using LSM
u1, u2, f, h = setup_sinusoidal_problem(N, initial_value=options.value0)

# Get the exact solution for validation
u_true = sinusoidal_exact_solution(N)

# Create solver instance (using pure numpy version)
# For numba acceleration, use: solver = SequentialJacobi(omega=0.75, verbose=False, use_numba=True)
solver = SequentialJacobi(omega=0.75, verbose=False)

# Optional: warmup for numba (if use_numba=True)
# solver.warmup(N=10)

# Run the solver
u, runtime_config, global_results, per_rank_results = solver.solve(u1, u2, f, h, N_iter, tolerance, u_true=u_true)

print(f"Wall time = {per_rank_results['wall_time']:.6f} s")
print(f"Compute time = {per_rank_results['compute_time']:.6f} s")
print(f"MPI comm time = {per_rank_results['mpi_comm_time']:.6f} s")
print(f"Iterations = {global_results['iterations']}")
if global_results["converged"]:
    print(f"Converged within tolerance {runtime_config['tolerance']}")
if "final_error" in global_results:
    print(f"Final error = {global_results['final_error']:.6e}")

# Create DataFrames (single row each, MLflow-compatible)
df_runtime_config = pd.DataFrame([runtime_config])
df_global_results = pd.DataFrame([global_results])
df_per_rank_results = pd.DataFrame([per_rank_results])

# Save results to data directory (automatically mirrors Experiments/ structure)
data_dir = datatools.get_data_dir()

if options.output:
    base_name = options.output.replace('.npz', '').replace('.parquet', '')
    config_file = data_dir / f"{base_name}_config.parquet"
    global_file = data_dir / f"{base_name}_global.parquet"
    perrank_file = data_dir / f"{base_name}_perrank.parquet"
    grid_file = data_dir / f"{base_name}_grid.npy"
else:
    iter_run = global_results["iterations"]
    config_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_config.parquet"
    global_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_global.parquet"
    perrank_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_perrank.parquet"
    grid_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_grid.npy"

# Save global runtime configuration (same for all ranks)
df_runtime_config.to_parquet(config_file, index=False)
print(f"Config saved to: {config_file}")

# Save global solver results (convergence, quality metrics)
df_global_results.to_parquet(global_file, index=False)
print(f"Global results saved to: {global_file}")

# Save per-rank results (performance data)
df_per_rank_results.to_parquet(perrank_file, index=False)
print(f"Per-rank results saved to: {perrank_file}")

# Save the 3D grid separately for slice plotting
np.save(grid_file, u)
print(f"Grid saved to: {grid_file}")
