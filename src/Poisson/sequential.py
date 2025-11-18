"""Sequential Jacobi solver."""

import time
import socket
import numpy as np
from mpi4py import MPI
from .base import PoissonSolver
from .datastructures import LocalResults


class SequentialJacobi(PoissonSolver):
    """Sequential Jacobi solver (single-node, no domain decomposition)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.method = "sequential"

    def method_solve(self):
        """Solve using sequential Jacobi iteration.

        Results are stored in solver instance variables and nothing is returned.
        """
        # Get references to fields
        u1 = self.global_fields.u1
        u2 = self.global_fields.u2
        f = self.global_fields.f
        h = self.global_fields.h
        u_exact = self.global_fields.u_exact

        # Main iteration loop
        for i in range(self.config.max_iter):
            if i % 2 == 0:
                uold, u = u1, u2
            else:
                u, uold = u1, u2

            # Jacobi step
            t_comp_start = MPI.Wtime()
            residual = self._step(uold, u, f, h, self.config.omega)
            t_comp_end = MPI.Wtime()

            self.global_timeseries.compute_times.append(t_comp_end - t_comp_start)
            self.global_timeseries.residual_history.append(float(residual))

            # Check convergence
            if residual < self.config.tolerance:
                self.global_results.converged = True
                self.global_results.iterations = i + 1
                break
        else:
            # Max iterations reached
            self.global_results.iterations = self.config.max_iter

        # Compute error
        self.global_results.final_error = float(np.linalg.norm(u - u_exact))

        # Store final solution in u1
        if self.global_results.iterations % 2 == 0:
            self.global_fields.u1[:] = u2
        # else: solution is already in u1
