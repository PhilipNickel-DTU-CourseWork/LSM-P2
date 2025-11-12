"""Iterative solvers for the 3D Poisson problem.

This module provides various iterative methods for solving the discrete
3D Poisson equation with Dirichlet boundary conditions.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np


class SolverResult(TypedDict):
    """Result from iterative solver.

    Attributes
    ----------
    u : np.ndarray
        Final solution array
    iterations : int
        Number of iterations performed
    diff_step : list[float]
        Residual at each iteration (change between steps)
    diff_true : list[float]
        Error compared to true solution at each iteration (if provided)
    converged : bool
        Whether the solver converged within tolerance
    """

    u: np.ndarray
    iterations: int
    diff_step: list[float]
    diff_true: list[float]
    converged: bool


def jacobi_step(
    uold: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
    h: float,
    omega: float = 0.75,
) -> float:
    """Perform a single Jacobi iteration step for the 3D Poisson problem.

    Solves -∇²u = f on a 3D grid using the 7-point stencil with
    weighted (relaxed) Jacobi method.

    The update formula is:
        u^{n+1}_ijk = ω * (1/6) * (sum of 6 neighbors + h² * f_ijk)
                      + (1-ω) * u^n_ijk

    When ω=1, this is the standard Jacobi method.
    When ω<1, this is a damped/relaxed Jacobi method.

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
    omega : float, default 0.75
        Relaxation parameter. Must satisfy 0 < omega <= 1.
        omega = 1: standard Jacobi
        omega < 1: damped/relaxed Jacobi (often improves convergence)

    Returns
    -------
    float
        Normalized residual (change between iterations):
        sqrt(sum((u - uold)²)) / N³

    Notes
    -----
    - Assumes Dirichlet boundary conditions (u = 0 on all faces)
    - Grid axes are aligned as (z, y, x)
    - Updates only interior points [1:-1, 1:-1, 1:-1]
    - Boundary values in u remain unchanged

    Examples
    --------
    >>> N = 10
    >>> h = 2.0 / (N - 1)
    >>> u1 = np.zeros((N, N, N))
    >>> u2 = np.zeros((N, N, N))
    >>> f = np.random.randn(N, N, N)
    >>> residual = jacobi_step(u1, u2, f, h, omega=0.75)

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


class JacobiSolver:
    """Jacobi iterative solver for 3D Poisson problem.

    This class provides a high-level interface for solving the 3D Poisson
    equation using the Jacobi method, with optional numba acceleration.

    Parameters
    ----------
    omega : float, default 0.75
        Relaxation parameter for weighted Jacobi (0 < omega <= 1)
    use_numba : bool, default False
        Whether to use numba-accelerated step function
    verbose : bool, default False
        Whether to print convergence information

    Attributes
    ----------
    omega : float
        Relaxation parameter
    use_numba : bool
        Whether numba is enabled
    verbose : bool
        Verbosity flag
    _step_function : callable
        The step function to use (regular or numba version)

    Examples
    --------
    >>> solver = JacobiSolver(omega=0.75)
    >>> solver.warmup(N=10)  # Optional: warmup for numba
    >>> result = solver.solve(u1, u2, f, h, max_iter=100, tolerance=1e-6)

    """

    def __init__(self, omega: float = 0.75, use_numba: bool = False, verbose: bool = False):
        """Initialize Jacobi solver.

        Parameters
        ----------
        omega : float, default 0.75
            Relaxation parameter
        use_numba : bool, default False
            Whether to use numba acceleration (requires numba installed)
        verbose : bool, default False
            Whether to print progress information

        """
        self.omega = omega
        self.use_numba = use_numba
        self.verbose = verbose

        # Select step function based on numba flag
        if use_numba:
            try:
                # Try to import numba version (to be implemented)
                # from .solvers_numba import jacobi_step_numba
                # self._step_function = jacobi_step_numba
                raise NotImplementedError("Numba version not yet implemented")
            except ImportError:
                if verbose:
                    print("Warning: numba not available, using standard version")
                self._step_function = jacobi_step
        else:
            self._step_function = jacobi_step

    def warmup(self, N: int = 10) -> None:
        """Warmup numba JIT compilation with a small test problem.

        This method runs the solver on a small problem to trigger JIT
        compilation, which avoids compilation overhead during actual runs.

        Parameters
        ----------
        N : int, default 10
            Grid size for warmup problem

        """
        if self.verbose:
            print(f"Warming up solver with N={N}...")

        # Create small test problem
        u1 = np.zeros((N, N, N))
        u2 = np.zeros((N, N, N))
        f = np.random.randn(N, N, N)
        h = 2.0 / (N - 1)

        # Run a few iterations to trigger compilation
        for _ in range(5):
            self._step_function(u1, u2, f, h, self.omega)
            u1, u2 = u2, u1  # Swap

        if self.verbose:
            print("Warmup complete!")

    def solve(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        f: np.ndarray,
        h: float,
        max_iter: int,
        tolerance: float = 1e-8,
        u_true: np.ndarray | None = None,
    ) -> SolverResult:
        """Solve 3D Poisson problem using Jacobi iteration.

        Iteratively solves -∇²u = f on a 3D grid until convergence or
        maximum iterations reached.

        Parameters
        ----------
        u1 : np.ndarray
            First solution array with boundary conditions, shape (N, N, N)
        u2 : np.ndarray
            Second solution array for swapping, shape (N, N, N)
        f : np.ndarray
            Source term (right-hand side), shape (N, N, N)
        h : float
            Grid spacing
        max_iter : int
            Maximum number of iterations
        tolerance : float, default 1e-8
            Convergence tolerance for normalized residual
        u_true : np.ndarray, optional
            True solution for error tracking (if known)

        Returns
        -------
        SolverResult
            Dictionary containing:
            - u: final solution
            - iterations: number of iterations performed
            - diff_step: list of residuals at each iteration
            - diff_true: list of errors vs true solution (if u_true provided)
            - converged: whether solver converged within tolerance

        """
        N = u1.shape[0]
        diff_step: list[float] = []
        diff_true: list[float] = []
        converged = False

        # Validation function
        def compute_error(u: np.ndarray, u_ref: np.ndarray) -> float:
            """Compute normalized L2 error."""
            return math.sqrt(np.sum((u - u_ref) ** 2)) / N**3

        # Main iteration loop
        u = u2  # Will point to the result array
        for i in range(max_iter):
            # Swap grids
            if i % 2 == 0:
                uold = u1
                u = u2
            else:
                u = u1
                uold = u2

            # Perform one Jacobi step
            residual = self._step_function(uold, u, f, h, self.omega)
            diff_step.append(residual)

            # Track error vs true solution if provided
            if u_true is not None:
                error = compute_error(u, u_true)
                diff_true.append(error)

            # Check convergence
            if residual < tolerance:
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {i + 1} (residual: {residual:.2e})")
                break

        if self.verbose and not converged:
            print(f"Did not converge after {max_iter} iterations (residual: {residual:.2e})")

        return SolverResult(
            u=u,
            iterations=i + 1,
            diff_step=diff_step,
            diff_true=diff_true,
            converged=converged,
        )
