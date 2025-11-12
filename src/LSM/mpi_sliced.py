"""MPI Jacobi solver with sliced (1D) domain decomposition.

This module implements parallel Jacobi solver using MPI with 1D sliced
domain decomposition. Each MPI rank owns a horizontal slice of the domain.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from mpi4py import MPI

from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class MPIJacobiSliced(PoissonSolver):
    """MPI Jacobi solver with sliced (1D) domain decomposition.

    Decomposes the 3D domain along the z-axis, with each MPI rank owning
    a horizontal slice. Communication occurs between adjacent ranks only
    (2-way halo exchange).

    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI communicator (default: MPI.COMM_WORLD)
    omega : float, default 0.75
        Relaxation parameter
    use_numba : bool, default True
        Use numba for local computation
    mpi_strategy : str, default "numpy_buffer"
        MPI communication strategy: "numpy_buffer" or "mpi_datatype"
    verbose : bool, default False
        Print convergence information (rank 0 only)

    Examples
    --------
    >>> from mpi4py import MPI
    >>> comm = MPI.COMM_WORLD
    >>> solver = MPIJacobiSliced(comm, omega=0.75, use_numba=True)
    >>> u, config, global_res, perrank = solver.solve(u1, u2, f, h, max_iter=100)

    """

    def __init__(
        self,
        comm=None,
        omega: float = 0.75,
        use_numba: bool = True,
        mpi_strategy: str = "numpy_buffer",
        verbose: bool = False,
    ):
        """Initialize MPI sliced decomposition solver."""
        super().__init__(comm, omega, use_numba, mpi_strategy, verbose)

        if self.verbose:
            print(
                f"Using {'numba' if self.use_numba else 'numpy'} kernel "
                f"with {self.size} MPI ranks (comm strategy: {self.mpi_strategy_name})"
            )

    def warmup(self, N: int = 10) -> None:
        """Warmup JIT compilation (only needed for numba kernel)."""
        if self.rank != 0:
            return  # Only rank 0 prints and warms up

        local_N = N // self.size + 2  # Add ghost layers for sliced decomposition
        h = 2.0 / (N - 1)
        self._warmup_kernel(self._step, (local_N, N, N), h)

    def _decompose_domain(self, N: int) -> Tuple[int, int, int]:
        """Compute local domain decomposition.

        Parameters
        ----------
        N : int
            Global grid size

        Returns
        -------
        local_N : int
            Number of z-planes owned by this rank (without ghost layers)
        z_start : int
            Global starting z-index for this rank
        z_end : int
            Global ending z-index for this rank (exclusive)
        """
        # Divide interior points among ranks
        interior_N = N - 2
        base_size = interior_N // self.size
        remainder = interior_N % self.size

        # Distribute remainder to first ranks
        if self.rank < remainder:
            local_N = base_size + 1
            z_start = self.rank * local_N + 1
        else:
            local_N = base_size
            z_start = remainder * (base_size + 1) + (self.rank - remainder) * base_size + 1

        z_end = z_start + local_N

        return local_N, z_start, z_end

    def _exchange_boundaries(
        self, u: np.ndarray, compute_time_list: list, comm_time_list: list
    ) -> None:
        """Exchange ghost/halo layers with neighboring ranks.

        Uses the configured MPI communication strategy for sending/receiving.

        Parameters
        ----------
        u : np.ndarray
            Local array with ghost layers
        compute_time_list : list
            List to accumulate compute times (not used here, but for consistency)
        comm_time_list : list
            List to accumulate communication times
        """
        t0 = time.perf_counter()

        # Exchange with rank below (send down, receive from down)
        if self.rank > 0:
            self.mpi_strategy.sendrecv_slice(
                self.comm,
                sendbuf=u[1, :, :],      # Send bottom interior layer
                dest=self.rank - 1,
                sendtag=0,
                recvbuf=u[0, :, :],      # Receive into bottom ghost layer
                source=self.rank - 1,
                recvtag=1,
            )

        # Exchange with rank above (send up, receive from up)
        if self.rank < self.size - 1:
            self.mpi_strategy.sendrecv_slice(
                self.comm,
                sendbuf=u[-2, :, :],     # Send top interior layer
                dest=self.rank + 1,
                sendtag=1,
                recvbuf=u[-1, :, :],     # Receive into top ghost layer
                source=self.rank + 1,
                recvtag=0,
            )

        # Barrier to ensure all exchanges complete
        self.comm.Barrier()

        t1 = time.perf_counter()
        comm_time_list.append(t1 - t0)

    def _gather_solution(self, u_local: np.ndarray, N: int) -> np.ndarray:
        """Gather local solutions to rank 0.

        Parameters
        ----------
        u_local : np.ndarray
            Local array with ghost layers
        N : int
            Global grid size

        Returns
        -------
        np.ndarray
            Global solution (only valid on rank 0, None on other ranks)
        """
        # Initialize global solution on rank 0
        u_global = np.zeros((N, N, N)) if self.rank == 0 else None

        # Copy interior points (exclude ghost layers)
        local_interior = u_local[1:-1, :, :].copy()

        # Gather all local interiors to rank 0
        all_locals = self.comm.gather(local_interior, root=0)

        if self.rank == 0:
            # Reconstruct global solution
            current_z = 1  # Start after bottom boundary
            for rank_data in all_locals:
                rank_local_N = rank_data.shape[0]
                u_global[current_z : current_z + rank_local_N, :, :] = rank_data
                current_z += rank_local_N

        return u_global

    def _setup_local_arrays(
        self, u1: np.ndarray, u2: np.ndarray, f: np.ndarray, N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set up local arrays with ghost layers for sliced decomposition.

        Parameters
        ----------
        u1, u2, f : np.ndarray
            Global arrays (shape: N x N x N)
        N : int
            Global grid size

        Returns
        -------
        u1_local, u2_local, f_local : np.ndarray
            Local arrays with ghost layers
        """
        # Decompose domain
        local_N, z_start, z_end = self._decompose_domain(N)

        # Create local arrays with ghost layers
        local_shape = (local_N + 2, N, N)  # +2 for top and bottom ghost layers
        u1_local = np.zeros(local_shape)
        u2_local = np.zeros(local_shape)
        f_local = np.zeros(local_shape)

        # Copy interior data to local arrays
        u1_local[1:-1, :, :] = u1[z_start:z_end, :, :]
        u2_local[1:-1, :, :] = u2[z_start:z_end, :, :]
        f_local[1:-1, :, :] = f[z_start:z_end, :, :]

        # Copy boundary conditions (if at domain edges)
        if self.rank == 0:
            u1_local[0, :, :] = u1[z_start - 1, :, :]
            u2_local[0, :, :] = u2[z_start - 1, :, :]
            f_local[0, :, :] = f[z_start - 1, :, :]
        if self.rank == self.size - 1:
            u1_local[-1, :, :] = u1[z_end, :, :]
            u2_local[-1, :, :] = u2[z_end, :, :]
            f_local[-1, :, :] = f[z_end, :, :]

        return u1_local, u2_local, f_local

    def solve(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        f: np.ndarray,
        h: float,
        max_iter: int,
        tolerance: float = 1e-8,
        u_true: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, RuntimeConfig, GlobalResults, PerRankResults]:
        """Solve using MPI sliced Jacobi iteration.

        Parameters
        ----------
        u1, u2 : np.ndarray
            Global solution arrays for swapping, shape (N, N, N)
        f : np.ndarray
            Global source term, shape (N, N, N)
        h : float
            Grid spacing
        max_iter : int
            Maximum iterations
        tolerance : float, default 1e-8
            Convergence tolerance
        u_true : np.ndarray, optional
            Global true solution for final error computation

        Returns
        -------
        u_global : np.ndarray
            Global solution array (only valid on rank 0)
        runtime_config : RuntimeConfig
            Global runtime configuration
        global_results : GlobalResults
            Global solver results
        per_rank_results : PerRankResults
            Per-rank performance data for this rank
        """
        return self._solve_common(
            u1, u2, f, h, max_iter, tolerance, u_true,
            method_name="mpi_sliced_jacobi"
        )
