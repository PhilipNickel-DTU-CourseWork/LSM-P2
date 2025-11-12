import math
from time import perf_counter as time
import numpy as np
from pathlib import Path
import pandas as pd

from utils import datatools, cli
from LSM import (
    MPIJacobiSliced,
    setup_sinusoidal_problem,
    sinusoidal_exact_solution,
)

# MPI imports
from mpi4py import MPI

# Available solvers for this experiment:
# - MPIJacobiSliced: MPI with 1D sliced decomposition (Z-axis)

# Create the argument parser using shared utility
parser = cli.create_parser(
    methods=["sliced"],
    default_method="sliced",
    description="MPI sliced Poisson problem solver",
)

# Grab options!
options = parser.parse_args()
N: int = options.N
method: str = options.method
N_iter: int = options.iter
tolerance: float = options.tolerance

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
In the below code, we'll have the axes aligned as z, y, x.
The MPIJacobiSliced solver decomposes along the Z-axis.
"""

# Set up the test problem using LSM
u1, u2, f, h = setup_sinusoidal_problem(N, initial_value=options.value0)

# Get the exact solution for validation
u_true = sinusoidal_exact_solution(N)

# Create solver instance (MPI sliced decomposition)
# For numba acceleration, use: solver = MPIJacobiSliced(omega=0.75, verbose=(rank==0), use_numba=True)
solver = MPIJacobiSliced(omega=0.75, verbose=(rank == 0))

# Optional: warmup for numba (if use_numba=True)
# if rank == 0:
#     solver.warmup(N=10)

# Run the solver
u, runtime_config, global_results, per_rank_results = solver.solve(
    u1, u2, f, h, N_iter, tolerance, u_true=u_true
)

# Only rank 0 prints summary
if rank == 0:
    print(f"Wall time = {global_results['wall_time']:.6f} s")
    print(f"Compute time = {global_results['compute_time']:.6f} s")
    print(f"MPI comm time = {global_results['mpi_comm_time']:.6f} s")
    print(f"Iterations = {global_results['iterations']}")
    if global_results["converged"]:
        print(f"Converged within tolerance {runtime_config['tolerance']}")
    if "final_error" in global_results:
        print(f"Final error = {global_results['final_error']:.6e}")

# Gather all per-rank results to rank 0
all_per_rank_results = comm.gather(per_rank_results, root=0)

# Save results to data directory (only rank 0)
if rank == 0:
    # Create DataFrames (MLflow-compatible)
    df_runtime_config = pd.DataFrame([runtime_config])
    df_global_results = pd.DataFrame([global_results])

    # Combine all per-rank results into single DataFrame
    df_per_rank_results = pd.DataFrame(all_per_rank_results)

    # Get data directory (automatically mirrors Experiments/ structure)
    data_dir = datatools.get_data_dir()

    if options.output:
        base_name = options.output.replace(".npz", "").replace(".parquet", "")
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

    # Save per-rank results (performance data for all ranks)
    df_per_rank_results.to_parquet(perrank_file, index=False)
    print(f"Per-rank results saved to: {perrank_file}")

    # Save the 3D grid separately for slice plotting
    np.save(grid_file, u)
    print(f"Grid saved to: {grid_file}")
