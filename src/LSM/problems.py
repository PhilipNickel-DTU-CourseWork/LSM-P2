"""Test problems and exact solutions for Poisson equation.

This module provides various test problems for the 3D Poisson equation,
including exact solutions and source terms for verification.
"""

from __future__ import annotations

import numpy as np


def create_grid_3d(N: int, value: float = 0.0, boundary_value: float = 0.0) -> np.ndarray:
    """Create a 3D grid with specified interior and boundary values.

    Parameters
    ----------
    N : int
        Grid size (N × N × N)
    value : float, default 0.0
        Initial value for interior points
    boundary_value : float, default 0.0
        Value for boundary points (Dirichlet BC)

    Returns
    -------
    np.ndarray
        Grid array of shape (N, N, N) with boundary conditions applied

    """
    u = np.full((N, N, N), value, dtype=np.float64)
    # Set all six faces to boundary value
    u[[0, -1], :, :] = boundary_value
    u[:, [0, -1], :] = boundary_value
    u[:, :, [0, -1]] = boundary_value
    return u


def sinusoidal_exact_solution(N: int) -> np.ndarray:
    """Generate exact solution for sin(π x)sin(π y)sin(π z) test problem.

    This is the exact solution to:
        -∇²u = 2π² sin(π x) sin(π y) sin(π z)
    on the domain [-1, 1]³ with zero Dirichlet boundary conditions.

    Parameters
    ----------
    N : int
        Grid size (N × N × N)

    Returns
    -------
    np.ndarray
        Exact solution array of shape (N, N, N)

    Examples
    --------
    >>> u_true = sinusoidal_exact_solution(N=50)

    """
    xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
    return np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)


def sinusoidal_source_term(N: int) -> np.ndarray:
    """Generate source term for sin(π x)sin(π y)sin(π z) test problem.

    Computes the right-hand side:
        f = 2π² sin(π x) sin(π y) sin(π z)
    for the Poisson equation -∇²u = f.

    Parameters
    ----------
    N : int
        Grid size (N × N × N)

    Returns
    -------
    np.ndarray
        Source term array of shape (N, N, N)

    Examples
    --------
    >>> f = sinusoidal_source_term(N=50)

    """
    xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
    return 2 * np.pi**2 * np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)


def setup_sinusoidal_problem(N: int, initial_value: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Set up complete sinusoidal test problem for Poisson equation.

    Creates grids, source term, and exact solution for the test problem:
        -∇²u = 2π² sin(π x) sin(π y) sin(π z)
    on [-1, 1]³ with zero Dirichlet BCs.

    Parameters
    ----------
    N : int
        Grid size (N × N × N)
    initial_value : float, default 0.0
        Initial guess for interior points

    Returns
    -------
    u1 : np.ndarray
        First solution grid with BCs, shape (N, N, N)
    u2 : np.ndarray
        Second solution grid with BCs, shape (N, N, N)
    f : np.ndarray
        Source term, shape (N, N, N)
    h : float
        Grid spacing

    Examples
    --------
    >>> u1, u2, f, h = setup_sinusoidal_problem(N=50)
    >>> # Now ready to solve

    """
    # Grid spacing
    h = 2.0 / (N - 1)

    # Create solution grids with boundary conditions
    u1 = create_grid_3d(N, value=initial_value, boundary_value=0.0)
    u2 = u1.copy()

    # Generate source term
    f = sinusoidal_source_term(N)

    return u1, u2, f, h
