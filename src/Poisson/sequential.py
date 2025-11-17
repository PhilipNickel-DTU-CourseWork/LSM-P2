"""Sequential Jacobi solver."""

import time
import socket
import numpy as np
from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class SequentialJacobi(PoissonSolver):
    """Sequential Jacobi solver (single-node, no domain decomposition)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def solve(self, u1, u2, f, h, max_iter, tolerance=1e-8, u_true=None):
        """Solve using sequential Jacobi iteration."""
        N = u1.shape[0]
        converged = False

        # Clear runtime accumulation lists
        self.compute_times.clear()
        self.residual_history.clear()

        t_start = time.perf_counter()

        # Main iteration loop
        for i in range(max_iter):
            if i % 2 == 0:
                uold, u = u1, u2
            else:
                u, uold = u1, u2

            # Jacobi step
            t_comp_start = time.perf_counter()
            residual = self._step(uold, u, f, h, self.config.omega)
            t_comp_end = time.perf_counter()
            self.compute_times.append(t_comp_end - t_comp_start)
            self.residual_history.append(float(residual))

            # Check convergence
            if residual < tolerance:
                converged = True
                break

        elapsed_time = time.perf_counter() - t_start

        # Compute error
        final_error = 0.0
        if u_true is not None:
            final_error = float(np.linalg.norm(u - u_true))

        # Update config with method
        self.config.method = "sequential_jacobi"

        # Build results
        runtime_config = RuntimeConfig(
            N=N,
            mpi_size=self.config.mpi_size,
            method="sequential_jacobi",
            omega=self.config.omega,
            tolerance=tolerance,
            max_iter=max_iter,
            use_numba=self.config.use_numba,
            num_threads=self.config.num_threads,
        )

        self.per_rank_results = PerRankResults(
            mpi_rank=0, hostname=socket.gethostname(), wall_time=elapsed_time,
            compute_time=sum(self.compute_times), mpi_comm_time=0.0, halo_exchange_time=0.0,
        )

        # Store all per-rank results and aggregate timings
        self.all_per_rank_results = [self.per_rank_results]
        timings = self._aggregate_timing_results(self.all_per_rank_results)

        self.global_results = GlobalResults(
            iterations=i + 1, residual_history=self.residual_history,
            converged=converged, final_error=final_error, **timings
        )

        return u, runtime_config, self.global_results, self.per_rank_results
