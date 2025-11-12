"""MPI Jacobi solver with cubic (3D) domain decomposition.

This module implements parallel Jacobi solver using MPI with 3D cubic
domain decomposition. Each MPI rank owns a cubic sub-block of the domain.
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
from mpi4py import MPI

from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class MPIJacobiCubic(PoissonSolver):
    """MPI Jacobi solver with cubic (3D) domain decomposition.

    Decomposes the 3D domain into a 3D grid of subdomains, with each MPI
    rank owning a cubic sub-block. Communication happens in all 6 directions
    for ghost cell exchange.

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
    >>> solver = MPIJacobiCubic(comm, omega=0.75, use_numba=True)
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
        """Initialize MPI cubic decomposition solver."""
        super().__init__(comm, omega, use_numba, mpi_strategy, verbose)

        # Create 3D Cartesian topology
        dims = self._compute_topology_dims(self.size)
        self.cart_comm = self.comm.Create_cart(dims, periods=[False, False, False])
        self.coords = self.cart_comm.Get_coords(self.rank)
        self.dims = dims

        # Use Cartesian communicator for all MPI operations
        # This allows base class helpers to work with the correct communicator
        self.comm = self.cart_comm

        # Get neighbors
        self.neighbors = self._get_neighbors()

        if self.verbose:
            print(
                f"Using {'numba' if self.use_numba else 'numpy'} kernel with {self.size} MPI ranks "
                f"(topology: {dims[0]}x{dims[1]}x{dims[2]}, comm strategy: {self.mpi_strategy_name})"
            )

    def _compute_topology_dims(self, nprocs: int) -> Tuple[int, int, int]:
        """Compute 3D processor grid dimensions."""
        # Try to find a factorization close to cubic
        # Start with cube root and adjust
        n = int(math.ceil(nprocs ** (1.0 / 3.0)))

        # Try to find factors
        for nz in range(n, 0, -1):
            if nprocs % nz == 0:
                remaining = nprocs // nz
                for ny in range(int(math.sqrt(remaining)), 0, -1):
                    if remaining % ny == 0:
                        nx = remaining // ny
                        return (nz, ny, nx)

        # Fallback: use MPI_Dims_create
        dims = MPI.Compute_dims(nprocs, [0, 0, 0])
        return tuple(dims)

    def _get_neighbors(self) -> dict:
        """Get neighbor ranks in all 6 directions."""
        # Shift returns (source, dest) for each direction
        # Direction 0: z-axis, Direction 1: y-axis, Direction 2: x-axis
        neighbors = {}

        # Z direction (up/down)
        neighbors["z_down"], neighbors["z_up"] = self.cart_comm.Shift(0, 1)

        # Y direction (south/north)
        neighbors["y_down"], neighbors["y_up"] = self.cart_comm.Shift(1, 1)

        # X direction (west/east)
        neighbors["x_down"], neighbors["x_up"] = self.cart_comm.Shift(2, 1)

        return neighbors

    def warmup(self, N: int = 10) -> None:
        """Warmup JIT compilation (only needed for numba kernel)."""
        if self.rank != 0:
            return  # Only rank 0 prints and warms up

        local_size = N // self.dims[0] + 2  # Add ghost layers for cubic decomposition
        h = 2.0 / (N - 1)
        self._warmup_kernel(self._step, (local_size, local_size, local_size), h)

    def _decompose_domain(self, N: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        """Compute local domain decomposition.

        Returns
        -------
        local_dims : tuple
            Number of points owned by this rank in each dimension (without ghosts)
        start_indices : tuple
            Global starting indices for this rank
        end_indices : tuple
            Global ending indices for this rank (exclusive)
        """
        interior_N = N - 2
        local_dims = []
        start_indices = []
        end_indices = []

        for dim in range(3):
            base_size = interior_N // self.dims[dim]
            remainder = interior_N % self.dims[dim]

            if self.coords[dim] < remainder:
                local_size = base_size + 1
                start = self.coords[dim] * local_size + 1
            else:
                local_size = base_size
                start = remainder * (base_size + 1) + (self.coords[dim] - remainder) * base_size + 1

            end = start + local_size

            local_dims.append(local_size)
            start_indices.append(start)
            end_indices.append(end)

        return tuple(local_dims), tuple(start_indices), tuple(end_indices)

    def _exchange_boundaries(self, u: np.ndarray, compute_time_list: list, comm_time_list: list) -> None:
        """Exchange ghost/halo layers with all 6 neighbors.

        Uses the configured MPI communication strategy for sending/receiving.
        """
        t0 = time.perf_counter()

        # Z direction exchanges (contiguous in memory)
        if self.neighbors["z_down"] != MPI.PROC_NULL:
            self.mpi_strategy.sendrecv_slice(
                self.cart_comm,
                sendbuf=u[1, :, :],
                dest=self.neighbors["z_down"],
                sendtag=0,
                recvbuf=u[0, :, :],
                source=self.neighbors["z_down"],
                recvtag=1,
            )
        if self.neighbors["z_up"] != MPI.PROC_NULL:
            self.mpi_strategy.sendrecv_slice(
                self.cart_comm,
                sendbuf=u[-2, :, :],
                dest=self.neighbors["z_up"],
                sendtag=1,
                recvbuf=u[-1, :, :],
                source=self.neighbors["z_up"],
                recvtag=0,
            )

        # Y direction exchanges (non-contiguous)
        if self.neighbors["y_down"] != MPI.PROC_NULL:
            self.mpi_strategy.sendrecv_slice(
                self.cart_comm,
                sendbuf=u[:, 1, :],
                dest=self.neighbors["y_down"],
                sendtag=2,
                recvbuf=u[:, 0, :],
                source=self.neighbors["y_down"],
                recvtag=3,
            )
        if self.neighbors["y_up"] != MPI.PROC_NULL:
            self.mpi_strategy.sendrecv_slice(
                self.cart_comm,
                sendbuf=u[:, -2, :],
                dest=self.neighbors["y_up"],
                sendtag=3,
                recvbuf=u[:, -1, :],
                source=self.neighbors["y_up"],
                recvtag=2,
            )

        # X direction exchanges (non-contiguous)
        if self.neighbors["x_down"] != MPI.PROC_NULL:
            self.mpi_strategy.sendrecv_slice(
                self.cart_comm,
                sendbuf=u[:, :, 1],
                dest=self.neighbors["x_down"],
                sendtag=4,
                recvbuf=u[:, :, 0],
                source=self.neighbors["x_down"],
                recvtag=5,
            )
        if self.neighbors["x_up"] != MPI.PROC_NULL:
            self.mpi_strategy.sendrecv_slice(
                self.cart_comm,
                sendbuf=u[:, :, -2],
                dest=self.neighbors["x_up"],
                sendtag=5,
                recvbuf=u[:, :, -1],
                source=self.neighbors["x_up"],
                recvtag=4,
            )

        self.cart_comm.Barrier()

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

        # Get local decomposition for reconstructing global indices
        local_dims, start_indices, end_indices = self._decompose_domain(N)

        # Copy interior points (exclude ghost layers)
        local_interior = u_local[1:-1, 1:-1, 1:-1].copy()

        # Gather all local data with their indices to rank 0
        all_locals = self.cart_comm.gather((local_interior, start_indices, end_indices), root=0)

        if self.rank == 0:
            # Reconstruct global solution
            for local_data, start, end in all_locals:
                u_global[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = local_data

        return u_global

    def _setup_local_arrays(
        self, u1: np.ndarray, u2: np.ndarray, f: np.ndarray, N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set up local arrays with ghost layers for cubic decomposition.

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
        local_dims, start_indices, end_indices = self._decompose_domain(N)

        # Create local arrays with ghost layers
        local_shape = tuple(d + 2 for d in local_dims)
        u1_local = np.zeros(local_shape)
        u2_local = np.zeros(local_shape)
        f_local = np.zeros(local_shape)

        # Copy interior data
        u1_local[1:-1, 1:-1, 1:-1] = u1[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ]
        u2_local[1:-1, 1:-1, 1:-1] = u2[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ]
        f_local[1:-1, 1:-1, 1:-1] = f[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ]

        # Copy boundary conditions for edges (if at domain boundaries)
        for dim in range(3):
            if self.coords[dim] == 0:
                # At the low edge of this dimension
                slices_from_global = [
                    slice(start_indices[i], end_indices[i]) if i != dim
                    else slice(start_indices[i] - 1, start_indices[i])
                    for i in range(3)
                ]
                slices_to_local = [
                    slice(1, -1) if i != dim else slice(0, 1)
                    for i in range(3)
                ]
                u1_local[tuple(slices_to_local)] = u1[tuple(slices_from_global)]
                u2_local[tuple(slices_to_local)] = u2[tuple(slices_from_global)]
                f_local[tuple(slices_to_local)] = f[tuple(slices_from_global)]

            if self.coords[dim] == self.dims[dim] - 1:
                # At the high edge of this dimension
                slices_from_global = [
                    slice(start_indices[i], end_indices[i]) if i != dim
                    else slice(end_indices[i], end_indices[i] + 1)
                    for i in range(3)
                ]
                slices_to_local = [
                    slice(1, -1) if i != dim else slice(-1, None)
                    for i in range(3)
                ]
                u1_local[tuple(slices_to_local)] = u1[tuple(slices_from_global)]
                u2_local[tuple(slices_to_local)] = u2[tuple(slices_from_global)]
                f_local[tuple(slices_to_local)] = f[tuple(slices_from_global)]

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
        """Solve using MPI cubic Jacobi iteration.

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
            method_name="mpi_cubic_jacobi"
        )
