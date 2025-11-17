"""Data structures for solver configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """Global runtime configuration (same for all ranks)."""
    # Problem
    N: int = 0

    # Specs 
    mpi_ranks: int = 1
    method: str = ""

    # Jacobi Solver
    omega: float = 0.75
    use_numba: bool = True
    num_threads: int = 1
    max_iter: int = 0
    tolerance: float = 0.0

@dataclass
class GlobalResults:
    """Global solver results (same for all ranks)."""
    # Convergence info
    iterations: int = 0
    residual_history: list[float] = field(default_factory=list)
    converged: bool = False
    final_error: float = 0.0
    # Global timings
    wall_time: float = 0.0
    compute_time: float = 0.0
    mpi_comm_time: float = 0.0




@dataclass
class PerRankResults:
    """Per-rank performance results."""
    mpi_rank: int = 0
    hostname: str = ""
    wall_time: float = 0.0
    compute_time: float = 0.0
    mpi_comm_time: float = 0.0
