"""Base class for Poisson solvers."""

import os
import numpy as np
from mpi4py import MPI

from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


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
    verbose : bool, default False
        Print convergence info (rank 0 only)
    """

    def __init__(self, **kwargs):
        # Extract verbose before passing to RuntimeConfig
        self.verbose = kwargs.pop('verbose', False)
        self.config = RuntimeConfig(**kwargs)
        self._step = self._select_kernel(self.config.use_numba)

    def solve(self, u1, u2, f, h, max_iter, tolerance=1e-8, u_true=None):
        """Solve the Poisson problem. Subclasses must override this."""
        raise NotImplementedError("Subclass must implement solve()")

    def warmup(self, N=10):
        """Warmup the solver (trigger JIT compilation)."""
        h = 2.0 / (N - 1)
        u1 = np.zeros((N, N, N))
        u2 = np.zeros((N, N, N))
        f = np.random.randn(N, N, N)

        for _ in range(5):
            self._step(u1, u2, f, h, self.config.omega)
            u1, u2 = u2, u1

    def get_num_threads(self, use_numba=False):
        """Get number of threads available for parallel execution."""
        if use_numba:
            import numba
            return numba.get_num_threads()
        return os.cpu_count() or 1

    def _select_kernel(self, use_numba):
        from .kernels import jacobi_step_numpy, jacobi_step_numba

        if use_numba:
            return jacobi_step_numba
        else:
            return jacobi_step_numpy

    def _smart_sendrecv(self, comm, sendbuf, dest, sendtag, recvbuf, source, recvtag):
        """Smart sendrecv: zero-copy for contiguous arrays, copy fallback otherwise."""
        if sendbuf.flags['C_CONTIGUOUS'] and recvbuf.flags['C_CONTIGUOUS']:
            # Zero-copy communication
            comm.Sendrecv(
                sendbuf,
                dest=dest,
                sendtag=sendtag,
                recvbuf=recvbuf,
                source=source,
                recvtag=recvtag,
            )
        else:
            # Use temporary buffers
            temp = np.empty_like(recvbuf)
            comm.Sendrecv(
                sendbuf.copy() if not sendbuf.flags['C_CONTIGUOUS'] else sendbuf,
                dest=dest,
                sendtag=sendtag,
                recvbuf=temp,
                source=source,
                recvtag=recvtag,
            )
            recvbuf[:] = temp
