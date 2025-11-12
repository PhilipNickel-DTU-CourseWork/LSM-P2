"""Large Scale Modeling (LSM) package.

Contains solvers and utilities for numerical methods in scientific computing.
"""

from .solvers import JacobiSolver, jacobi_step

__all__ = ["JacobiSolver", "jacobi_step"]
