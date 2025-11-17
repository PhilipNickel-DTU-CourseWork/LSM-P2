"""Base class for Poisson solvers."""

from dataclasses import asdict
import os
import numpy as np
from mpi4py import MPI
from .kernels import jacobi_step_numpy, jacobi_step_numba
from numba import get_num_threads
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults
import mlflow
import pandas as pd

class PoissonSolver:
    """Base class for all Poisson solvers.

    Provides shared bookkeeping and utility methods.
    Subclasses override solve() to implement specific strategies.

    Parameters
    ----------
    omega : float, default 0.75
        Relaxation parameter
    use_numba : bool, default True
        Use numba JIT compilation
    """

    def __init__(self, **kwargs):
        self.config = RuntimeConfig(**kwargs)
        self.config.num_threads = get_num_threads()
        self.config.mpi_size = MPI.COMM_WORLD.Get_size()
        self.global_results = GlobalResults()
        self.per_rank_results = PerRankResults()
        self.all_per_rank_results = []

        # Runtime accumulation lists
        self.compute_times = []
        self.comm_times = []
        self.halo_times = []
        self.residual_history = []

        if self.config.use_numba:
            self._step = jacobi_step_numba
        else:
            self._step = jacobi_step_numpy

    def solve(self, u1, u2, f, h, max_iter, tolerance=1e-8, u_true=None):
        """Solve the Poisson problem. Subclasses must override this."""
        raise NotImplementedError("Subclass must implement solve()")

    def _aggregate_timing_results(self, all_per_rank_results):
        """Aggregate per-rank timing results into global timings."""
        return {
            'wall_time': max(pr.wall_time for pr in all_per_rank_results),
            'compute_time': sum(pr.compute_time for pr in all_per_rank_results),
            'mpi_comm_time': sum(pr.mpi_comm_time for pr in all_per_rank_results),
            'halo_exchange_time': sum(pr.halo_exchange_time for pr in all_per_rank_results),
        }

    def warmup(self, N=10):
        """Warmup the solver (trigger JIT compilation)."""
        h = 2.0 / (N - 1)
        u1 = np.zeros((N, N, N))
        u2 = np.zeros((N, N, N))
        f = np.random.randn(N, N, N)

        for _ in range(5):
            self._step(u1, u2, f, h, self.config.omega)
            u1, u2 = u2, u1

    def mlflow_start_log(self, experiment_name, N, max_iter, tolerance):
        mlflow.login(backend="databricks", interactive=False)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

        # Update config with runtime values and log immediately
        self.config.N = N
        self.config.max_iter = max_iter
        self.config.tolerance = tolerance
        mlflow.log_params(asdict(self.config))

    def mlflow_end_log(self):

        # Log global results (excluding lists which can't be logged as metrics)
        global_dict = asdict(self.global_results)
        residual_history = global_dict.pop('residual_history', [])

        mlflow.log_metrics(global_dict)

        # Log residual history as step-by-step metrics for convergence graph
        for step, residual in enumerate(residual_history):
            mlflow.log_metric("residual", residual, step=step)

        # Log per-rank results as a table
        per_rank_dicts = [asdict(pr) for pr in self.all_per_rank_results]
        mlflow.log_table(pd.DataFrame(per_rank_dicts), "per_rank_results.json")
        mlflow.end_run()

    



