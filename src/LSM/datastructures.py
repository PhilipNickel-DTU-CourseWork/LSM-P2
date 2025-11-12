"""Data structures for solver configuration and results.

These structures are designed to be MLflow-compatible and easily convertible
to pandas DataFrames for analysis and visualization.

Key design:
- RuntimeConfig: Global configuration (same for all ranks)
- GlobalResults: Global solver results (convergence, quality metrics)
- PerRankResults: Per-rank performance data (timings, hostname)

Sequential runs: 1 RuntimeConfig, 1 GlobalResults, 1 PerRankResults
MPI runs: 1 RuntimeConfig, 1 GlobalResults, N PerRankResults (one per rank)
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class RuntimeConfig(TypedDict, total=False):
    """Global runtime configuration (same for all ranks).

    Contains problem configuration, solver settings, and MPI configuration.
    Single row per run, regardless of number of MPI ranks.

    All fields are optional to allow partial construction.

    Attributes
    ----------
    # Problem configuration
    N : int
        Grid size (number of points along each dimension)
    h : float
        Grid spacing

    # Solver configuration
    method : str
        Solver method name (e.g., "sequential_jacobi", "mpi_cubic")
    omega : float
        Relaxation parameter for weighted Jacobi
    tolerance : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    use_numba : bool
        Whether numba JIT compilation was used
    num_threads : int
        Number of threads configured for parallel execution (numba/OpenMP)

    # MPI configuration
    mpi_size : int
        Total number of MPI processes (1 for sequential)

    # Run metadata
    timestamp : str
        ISO format timestamp of run start
    """

    # Problem configuration
    N: int
    h: float

    # Solver configuration
    method: str
    omega: float
    tolerance: float
    max_iter: int
    use_numba: bool
    num_threads: int

    # MPI configuration
    mpi_size: int

    # Run metadata
    timestamp: str


class GlobalResults(TypedDict, total=False):
    """Global solver results (same for all ranks).

    Contains convergence status, solution quality metrics, and aggregated timings.
    Single row per run, regardless of number of MPI ranks.

    All fields are optional to allow partial construction.

    Attributes
    ----------
    iterations : int
        Actual number of iterations performed
    converged : bool
        Whether solver converged within tolerance
    final_residual : float
        Final global residual value
    final_error : float
        Final error vs true solution (if available)

    # Aggregated performance metrics
    wall_time : float
        Maximum wall time across all ranks (bottleneck time)
    compute_time : float
        Sum of compute time across all ranks (total compute effort)
    mpi_comm_time : float
        Sum of MPI communication time across all ranks (total comm overhead)
    """

    iterations: int
    converged: bool
    final_residual: float
    final_error: float

    # Aggregated performance metrics
    wall_time: float
    compute_time: float
    mpi_comm_time: float


class PerRankResults(TypedDict, total=False):
    """Per-rank performance results.

    Contains rank-specific performance data and system information.
    For MPI runs with N ranks, you'll have N rows (one per rank).
    For sequential runs, you'll have 1 row (rank 0).

    All fields are optional to allow partial construction.

    Attributes
    ----------
    # Rank identification
    mpi_rank : int
        MPI rank of this process (0 for sequential)

    # System information
    hostname : str
        Host machine name where this rank ran

    # Performance metrics
    wall_time : float
        Total wall clock time in seconds for this rank
    compute_time : float
        Time spent in computation for this rank
    mpi_comm_time : float
        Time spent in MPI communication for this rank
    """

    # Rank identification
    mpi_rank: int

    # System information
    hostname: str

    # Performance metrics
    wall_time: float
    compute_time: float
    mpi_comm_time: float
