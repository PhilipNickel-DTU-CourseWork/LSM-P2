"""Sequential Jacobi solver implementation.

This module provides a single-node Jacobi solver that can use different
computational kernels (numpy or numba-accelerated).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from mpi4py import MPI

from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class SequentialJacobi(PoissonSolver):
    """Sequential Jacobi solver with pluggable computational kernel.

    Single-node Jacobi solver that can use either pure numpy or numba-accelerated
    kernels. Implemented as an MPI-aware solver with size=1 (no domain decomposition).

    Parameters
    ----------
    omega : float, default 0.75
        Relaxation parameter
    use_numba : bool, default True
        Use numba JIT-compiled kernel for better performance
    mpi_strategy : str, default "numpy_buffer"
        MPI communication strategy (not used for sequential, but kept for interface consistency)
    verbose : bool, default False
        Print convergence information

    Examples
    --------
    >>> # Pure numpy version
    >>> solver = SequentialJacobi(omega=0.75, use_numba=False)
    >>> result = solver.solve(u1, u2, f, h, max_iter=100, tolerance=1e-6)

    >>> # Numba-accelerated version
    >>> solver = SequentialJacobi(omega=0.75, use_numba=True)
    >>> solver.warmup(N=10)  # Trigger JIT compilation
    >>> result = solver.solve(u1, u2, f, h, max_iter=100, tolerance=1e-6)

    """

    def __init__(
        self,
        omega: float = 0.75,
        use_numba: bool = True,
        mpi_strategy: str = "numpy_buffer",
        verbose: bool = False,
    ):
        """Initialize sequential Jacobi solver with chosen kernel."""
        # Use MPI.COMM_SELF to create a single-rank "MPI" communicator
        super().__init__(
            comm=MPI.COMM_SELF,
            omega=omega,
            use_numba=use_numba,
            mpi_strategy=mpi_strategy,
            verbose=verbose,
        )

    def _decompose_domain(self, N: int):
        """No domain decomposition for sequential solver (full domain on single rank)."""
        return None  # Not used in sequential solver

    def _exchange_boundaries(self, u: np.ndarray, compute_time_list: list, comm_time_list: list) -> None:
        """No boundary exchange needed for sequential solver."""
        pass  # No-op for single rank

    def _gather_solution(self, u_local: np.ndarray, N: int) -> np.ndarray:
        """No gathering needed for sequential solver (already have full solution)."""
        return u_local  # Just return the input

    def _setup_local_arrays(
        self, u1: np.ndarray, u2: np.ndarray, f: np.ndarray, N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sequential solver uses arrays as-is (no decomposition or ghost layers)."""
        return u1, u2, f

    def warmup(self, N: int = 10) -> None:
        """Warmup JIT compilation (only needed for numba kernel)."""
        h = 2.0 / (N - 1)
        self._warmup_kernel(self._step, (N, N, N), h)

    def solve(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        f: np.ndarray,
        h: float,
        max_iter: int,
        tolerance: float = 1e-8,
        u_true: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, RuntimeConfig, GlobalResults, PerRankResults]:
        """Solve using sequential Jacobi iteration.

        Parameters
        ----------
        u1, u2 : np.ndarray
            Solution arrays for swapping, shape (N, N, N)
        f : np.ndarray
            Source term, shape (N, N, N)
        h : float
            Grid spacing
        max_iter : int
            Maximum iterations
        tolerance : float, default 1e-8
            Convergence tolerance
        u_true : np.ndarray, optional
            True solution for final error computation

        Returns
        -------
        u : np.ndarray
            Final solution array
        runtime_config : RuntimeConfig
            Global runtime configuration
        global_results : GlobalResults
            Global solver results
        per_rank_results : PerRankResults
            Per-rank performance data
        """
        return self._solve_common(
            u1, u2, f, h, max_iter, tolerance, u_true,
            method_name="sequential_jacobi"
        )
