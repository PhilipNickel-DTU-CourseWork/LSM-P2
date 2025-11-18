"""Large Scale Modeling package."""

from .base import PoissonSolver
from .datastructures import GlobalConfig, GlobalFields, LocalFields, GlobalResults, LocalResults, TimeSeriesLocal, TimeSeriesGlobal
from .kernels import jacobi_step_numpy, jacobi_step_numba
from .sequential import SequentialJacobi
from .mpi_sliced import MPIJacobiSliced
from .problems import create_grid_3d, sinusoidal_exact_solution, sinusoidal_source_term, setup_sinusoidal_problem

__all__ = [
    "PoissonSolver",
    "GlobalConfig",
    "GlobalFields",
    "LocalFields",
    "GlobalResults",
    "LocalResults",
    "TimeSeriesLocal",
    "TimeSeriesGlobal",
    "jacobi_step_numpy",
    "jacobi_step_numba",
    "SequentialJacobi",
    "MPIJacobiSliced",
    "create_grid_3d",
    "sinusoidal_exact_solution",
    "sinusoidal_source_term",
    "setup_sinusoidal_problem",
]
