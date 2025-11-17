"""Test problems and exact solutions for Poisson equation."""

from __future__ import annotations

import numpy as np


def create_grid_3d(N: int, value: float = 0.0, boundary_value: float = 0.0) -> np.ndarray:
    """Create 3D grid with specified interior and boundary values."""
    u = np.full((N, N, N), value, dtype=np.float64)
    u[[0, -1], :, :] = boundary_value
    u[:, [0, -1], :] = boundary_value
    u[:, :, [0, -1]] = boundary_value
    return u


def sinusoidal_exact_solution(N: int) -> np.ndarray:
    """Exact solution: sin(π x)sin(π y)sin(π z) on [-1,1]³."""
    xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
    return np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)


def sinusoidal_source_term(N: int) -> np.ndarray:
    """Source term: f = 2π² sin(π x)sin(π y)sin(π z)."""
    xs, ys, zs = np.ogrid[-1 : 1 : complex(N), -1 : 1 : complex(N), -1 : 1 : complex(N)]
    return 2 * np.pi**2 * np.sin(np.pi * xs) * np.sin(np.pi * ys) * np.sin(np.pi * zs)


def setup_sinusoidal_problem(N: int, initial_value: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Set up sinusoidal test problem: -∇²u = 2π² sin(π x)sin(π y)sin(π z)."""
    h = 2.0 / (N - 1)
    u1 = create_grid_3d(N, value=initial_value, boundary_value=0.0)
    u2 = u1.copy()
    f = sinusoidal_source_term(N)
    return u1, u2, f, h
