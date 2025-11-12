"""Large Scale Modeling (LSM) package.

Contains solvers and utilities for numerical methods in scientific computing.
"""

from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults
from .kernels import (
    jacobi_step,
    jacobi_step_numpy,
    jacobi_step_numba,
    NUMBA_AVAILABLE,
)
from .mpi_strategies import (
    MPICommunicationStrategy,
    NumpyBufferStrategy,
    MPIDatatypeStrategy,
    create_mpi_strategy,
)
from .sequential import SequentialJacobi
from .mpi_cubic import MPIJacobiCubic
from .mpi_sliced import MPIJacobiSliced
from .problems import (
    create_grid_3d,
    sinusoidal_exact_solution,
    sinusoidal_source_term,
    setup_sinusoidal_problem,
)

__all__ = [
    # Abstract base
    "PoissonSolver",
    # Data structures
    "RuntimeConfig",
    "GlobalResults",
    "PerRankResults",
    # Step functions
    "jacobi_step",
    "jacobi_step_numpy",
    "jacobi_step_numba",
    "NUMBA_AVAILABLE",
    # MPI communication strategies
    "MPICommunicationStrategy",
    "NumpyBufferStrategy",
    "MPIDatatypeStrategy",
    "create_mpi_strategy",
    # Sequential solver
    "SequentialJacobi",
    # MPI solvers
    "MPIJacobiCubic",
    "MPIJacobiSliced",
    # Test problems
    "create_grid_3d",
    "sinusoidal_exact_solution",
    "sinusoidal_source_term",
    "setup_sinusoidal_problem",
]
