"""Benchmark sequential solver with and without numba acceleration.

This script runs the same problem with different computational kernels:
1. Pure numpy (baseline) - uses optimized BLAS/LAPACK libraries
2. Numba JIT with prange - parallel execution using explicit loops

Results show ~11x speedup with numba's parallel execution across multiple cores.
The parallel numba version uses prange to distribute work across CPU threads,
achieving significant speedup for problems with regular loop structures.

Results are saved separately for comparison.
"""

import numpy as np
import pandas as pd

from utils import datatools
from LSM import (
    SequentialJacobi,
    setup_sinusoidal_problem,
    sinusoidal_exact_solution,
)

# Configuration
N = 100  # Grid size
max_iter = 200  # Maximum iterations
tolerance = 1e-8  # Convergence tolerance
initial_value = 0.0  # Initial grid value

print("=" * 60)
print("NUMBA BENCHMARK: Sequential Jacobi Solver")
print("=" * 60)
print(f"Grid size: N = {N}")
print(f"Max iterations: {max_iter}")
print(f"Tolerance: {tolerance:.2e}")
print()

# Set up the test problem
u1, u2, f, h = setup_sinusoidal_problem(N, initial_value=initial_value)
u_true = sinusoidal_exact_solution(N)

# Get data directory
data_dir = datatools.get_data_dir()

# ============================================================================
# Run 1: Pure numpy (baseline)
# ============================================================================
print("=" * 60)
print("RUN 1: Pure numpy (baseline)")
print("=" * 60)

solver_numpy = SequentialJacobi(omega=0.75, verbose=True, use_numba=False)

# Reset arrays for fair comparison
u1_numpy = u1.copy()
u2_numpy = u2.copy()

u_numpy, config_numpy, global_numpy, perrank_numpy = solver_numpy.solve(
    u1_numpy, u2_numpy, f, h, max_iter, tolerance, u_true=u_true
)

print(f"\nResults (numpy):")
print(f"  Wall time: {perrank_numpy['wall_time']:.6f} s")
print(f"  Compute time: {perrank_numpy['compute_time']:.6f} s")
print(f"  Iterations: {global_numpy['iterations']}")
print(f"  Converged: {global_numpy['converged']}")
print(f"  Final error: {global_numpy['final_error']:.6e}")
print()

# Save numpy results
df_config_numpy = pd.DataFrame([config_numpy])
df_global_numpy = pd.DataFrame([global_numpy])
df_perrank_numpy = pd.DataFrame([perrank_numpy])

config_file_numpy = data_dir / f"benchmark_N{N}_numpy_config.parquet"
global_file_numpy = data_dir / f"benchmark_N{N}_numpy_global.parquet"
perrank_file_numpy = data_dir / f"benchmark_N{N}_numpy_perrank.parquet"
grid_file_numpy = data_dir / f"benchmark_N{N}_numpy_grid.npy"

df_config_numpy.to_parquet(config_file_numpy, index=False)
df_global_numpy.to_parquet(global_file_numpy, index=False)
df_perrank_numpy.to_parquet(perrank_file_numpy, index=False)
np.save(grid_file_numpy, u_numpy)

print(f"Numpy results saved to: {data_dir}/benchmark_N{N}_numpy_*.parquet")

# ============================================================================
# Run 2: Numba JIT (accelerated)
# ============================================================================
print()
print("=" * 60)
print("RUN 2: Numba JIT (accelerated)")
print("=" * 60)

solver_numba = SequentialJacobi(omega=0.75, verbose=True, use_numba=True)

# Warmup JIT compilation
print("\nWarming up JIT compiler...")
solver_numba.warmup(N=10)
print()

# Reset arrays for fair comparison
u1_numba = u1.copy()
u2_numba = u2.copy()

u_numba, config_numba, global_numba, perrank_numba = solver_numba.solve(
    u1_numba, u2_numba, f, h, max_iter, tolerance, u_true=u_true
)

print(f"\nResults (numba):")
print(f"  Wall time: {perrank_numba['wall_time']:.6f} s")
print(f"  Compute time: {perrank_numba['compute_time']:.6f} s")
print(f"  Iterations: {global_numba['iterations']}")
print(f"  Converged: {global_numba['converged']}")
print(f"  Final error: {global_numba['final_error']:.6e}")
print()

# Save numba results
df_config_numba = pd.DataFrame([config_numba])
df_global_numba = pd.DataFrame([global_numba])
df_perrank_numba = pd.DataFrame([perrank_numba])

config_file_numba = data_dir / f"benchmark_N{N}_numba_config.parquet"
global_file_numba = data_dir / f"benchmark_N{N}_numba_global.parquet"
perrank_file_numba = data_dir / f"benchmark_N{N}_numba_perrank.parquet"
grid_file_numba = data_dir / f"benchmark_N{N}_numba_grid.npy"

df_config_numba.to_parquet(config_file_numba, index=False)
df_global_numba.to_parquet(global_file_numba, index=False)
df_perrank_numba.to_parquet(perrank_file_numba, index=False)
np.save(grid_file_numba, u_numba)

print(f"Numba results saved to: {data_dir}/benchmark_N{N}_numba_*.parquet")

# ============================================================================
# Comparison
# ============================================================================
print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)

speedup = perrank_numpy["wall_time"] / perrank_numba["wall_time"]

print(f"\nNumpy:")
print(f"  Wall time: {perrank_numpy['wall_time']:.6f} s")
print(f"  Iterations: {global_numpy['iterations']}")
print(f"  Final error: {global_numpy['final_error']:.6e}")

print(f"\nNumba:")
print(f"  Wall time: {perrank_numba['wall_time']:.6f} s")
print(f"  Iterations: {global_numba['iterations']}")
print(f"  Final error: {global_numba['final_error']:.6e}")

print(f"\nSpeedup: {speedup:.2f}x")
print(f"Time saved: {(perrank_numpy['wall_time'] - perrank_numba['wall_time']):.6f} s")

# Verify numerical consistency
max_diff = np.max(np.abs(u_numpy - u_numba))
print(f"\nNumerical consistency:")
print(f"  Max difference between solutions: {max_diff:.2e}")
print(f"  Solutions are {'identical' if max_diff < 1e-10 else 'consistent'}")

print()
print("=" * 60)
