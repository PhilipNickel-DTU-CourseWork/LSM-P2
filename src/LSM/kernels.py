"""Computational kernels for Poisson solvers.

This module contains the core computational step functions that are used
by the solver implementations. These are pure functions with no class
dependencies, making them ideal for JIT compilation with numba.
"""

from __future__ import annotations

import math

import numpy as np


def jacobi_step_numpy(
    uold: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
    h: float,
    omega: float,
) -> float:
    """Perform a single Jacobi iteration step using pure numpy.

    This is the computational kernel. It's a pure function with no class
    references, making it ideal for JIT compilation with numba.

    Parameters
    ----------
    uold : np.ndarray
        Previous solution array, shape (N, N, N)
    u : np.ndarray
        Current solution array to update, shape (N, N, N)
    f : np.ndarray
        Source term (right-hand side), shape (N, N, N)
    h : float
        Grid spacing
    omega : float
        Relaxation parameter (0 < omega <= 1)

    Returns
    -------
    float
        Normalized residual: sqrt(sum((u - uold)²)) / N³

    """
    # The stencil weight (1/6 for 7-point stencil in 3D)
    c: float = 1.0 / 6.0
    h2: float = h * h
    N: int = u.shape[0] - 2  # Number of interior points

    # Weighted Jacobi update on interior points
    u[1:-1, 1:-1, 1:-1] = (
        omega
        * c
        * (
            uold[0:-2, 1:-1, 1:-1]  # z-1
            + uold[2:, 1:-1, 1:-1]  # z+1
            + uold[1:-1, 0:-2, 1:-1]  # y-1
            + uold[1:-1, 2:, 1:-1]  # y+1
            + uold[1:-1, 1:-1, 0:-2]  # x-1
            + uold[1:-1, 1:-1, 2:]  # x+1
            + h2 * f[1:-1, 1:-1, 1:-1]
        )
        + (1.0 - omega) * uold[1:-1, 1:-1, 1:-1]
    )

    # Compute and return normalized residual
    return math.sqrt(np.sum((u - uold) ** 2)) / N**3


# Try to create numba-compiled versions
try:
    from numba import njit, prange

    # Non-parallel numba JIT (just compilation, no parallelization)
    jacobi_step_numba = njit(jacobi_step_numpy, cache=True)

    # Parallel numba JIT with explicit loops
    @njit(parallel=True, cache=True)
    def jacobi_step_numba_parallel(
        uold: np.ndarray,
        u: np.ndarray,
        f: np.ndarray,
        h: float,
        omega: float,
    ) -> float:
        """Perform a single Jacobi iteration step using numba with parallel execution.

        This version uses explicit loops with prange for parallelization.

        Parameters
        ----------
        uold : np.ndarray
            Previous solution array, shape (N, N, N)
        u : np.ndarray
            Current solution array to update, shape (N, N, N)
        f : np.ndarray
            Source term (right-hand side), shape (N, N, N)
        h : float
            Grid spacing
        omega : float
            Relaxation parameter (0 < omega <= 1)

        Returns
        -------
        float
            Normalized residual: sqrt(sum((u - uold)²)) / N³
        """
        c = 1.0 / 6.0
        h2 = h * h
        N = u.shape[0] - 2

        # Parallel loop over z-dimension
        for i in prange(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                for k in range(1, u.shape[2] - 1):
                    u[i, j, k] = (
                        omega
                        * c
                        * (
                            uold[i - 1, j, k]  # z-1
                            + uold[i + 1, j, k]  # z+1
                            + uold[i, j - 1, k]  # y-1
                            + uold[i, j + 1, k]  # y+1
                            + uold[i, j, k - 1]  # x-1
                            + uold[i, j, k + 1]  # x+1
                            + h2 * f[i, j, k]
                        )
                        + (1.0 - omega) * uold[i, j, k]
                    )

        # Compute residual
        diff_sum = 0.0
        for i in prange(u.shape[0]):
            for j in range(u.shape[1]):
                for k in range(u.shape[2]):
                    diff = u[i, j, k] - uold[i, j, k]
                    diff_sum += diff * diff

        return math.sqrt(diff_sum) / N**3

    NUMBA_AVAILABLE = True
    NUMBA_PARALLEL_AVAILABLE = True
except ImportError:
    jacobi_step_numba = jacobi_step_numpy
    jacobi_step_numba_parallel = jacobi_step_numpy
    NUMBA_AVAILABLE = False
    NUMBA_PARALLEL_AVAILABLE = False


# Alias for backward compatibility
jacobi_step = jacobi_step_numpy
