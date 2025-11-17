"""MPI Jacobi solver with 1D sliced domain decomposition (z-axis)."""

import time
import socket
import numpy as np
from mpi4py import MPI
from .base import PoissonSolver
from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class MPIJacobiSliced(PoissonSolver):
    """MPI sliced decomposition: each rank owns a horizontal slice (z-axis split)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.plane_type = None  # MPI datatype for 2D plane

    def _create_plane_datatype(self, N):
        """Create MPI datatype for a 2D plane."""
        if self.plane_type is None:
            # Create subarray type for a single YZ plane
            self.plane_type = MPI.DOUBLE.Create_contiguous(N * N)
            self.plane_type.Commit()
        return self.plane_type

    def solve(self, u1, u2, f, h, max_iter, tolerance=1e-8, u_true=None):
        """Solve using MPI sliced Jacobi iteration."""
        N = u1.shape[0]

        # Create MPI datatype for plane communication
        self._create_plane_datatype(N)

        # Setup local arrays
        u1_local, u2_local, f_local = self._setup_local_arrays(u1, u2, f, N)

        # Clear runtime accumulation lists
        self.compute_times.clear()
        self.comm_times.clear()
        self.halo_times.clear()
        self.residual_history.clear()

        converged = False
        t_start = time.perf_counter()

        # Main iteration loop
        for i in range(max_iter):
            if i % 2 == 0:
                uold_local, u_local = u1_local, u2_local
            else:
                u_local, uold_local = u1_local, u2_local

            # Exchange boundaries
            self._exchange_boundaries(uold_local)

            # Jacobi step
            t_comp_start = time.perf_counter()
            local_residual = self._step(uold_local, u_local, f_local, h, self.config.omega)
            t_comp_end = time.perf_counter()
            self.compute_times.append(t_comp_end - t_comp_start)

            # Global residual
            t_comm_start = time.perf_counter()
            global_residual = self.comm.allreduce(local_residual**2, op=MPI.SUM)
            global_residual = np.sqrt(global_residual)
            t_comm_end = time.perf_counter()
            self.comm_times.append(t_comm_end - t_comm_start)
            self.residual_history.append(float(global_residual))

            # Check convergence
            if global_residual < tolerance:
                converged = True
                break

        elapsed_time = time.perf_counter() - t_start

        # Gather solution
        u_global = self._gather_solution(u_local, N)

        # Compute error on rank 0 and broadcast
        final_error = 0.0
        if self.rank == 0 and u_true is not None:
            final_error = float(np.linalg.norm(u_global - u_true))
        final_error = self.comm.bcast(final_error, root=0)

        # Build per-rank results
        per_rank_results = PerRankResults(
            mpi_rank=self.rank, hostname=socket.gethostname(),
            wall_time=elapsed_time, compute_time=sum(self.compute_times),
            mpi_comm_time=sum(self.comm_times), halo_exchange_time=sum(self.halo_times),
        )

        # Gather all per-rank results
        all_perrank = self.comm.gather(per_rank_results, root=0)

        # Store all per-rank results for MLflow logging
        if self.rank == 0:
            self.all_per_rank_results = all_perrank

        # Update config with method
        self.config.method = "mpi_sliced_jacobi"

        # Build config and global results on rank 0
        if self.rank == 0:
            runtime_config = RuntimeConfig(
                N=N, method="mpi_sliced_jacobi", omega=self.config.omega,
                tolerance=tolerance, max_iter=max_iter, use_numba=self.config.use_numba,
                num_threads=self.config.num_threads, mpi_size=self.config.mpi_size,
            )

            timings = self._aggregate_timing_results(all_perrank)
            self.global_results = GlobalResults(
                iterations=i + 1, residual_history=self.residual_history,
                converged=converged, final_error=final_error, **timings
            )
        else:
            runtime_config, self.global_results = RuntimeConfig(), GlobalResults()

        # Broadcast to all ranks
        runtime_config = self.comm.bcast(runtime_config, root=0)
        self.global_results = self.comm.bcast(self.global_results, root=0)

        return u_global, runtime_config, self.global_results, per_rank_results

    def _decompose_domain(self, N):
        interior_N = N - 2
        base_size, remainder = divmod(interior_N, self.size)
        local_N = base_size + (1 if self.rank < remainder else 0)
        z_start = self.rank * local_N + 1 if self.rank < remainder else \
                  remainder * (base_size + 1) + (self.rank - remainder) * base_size + 1
        return local_N, z_start, z_start + local_N

    def _exchange_boundaries(self, u):
        t0 = time.perf_counter()

        # Exchange with lower neighbor
        if self.rank > 0:
            self.comm.Sendrecv(
                [u[1, :, :], 1, self.plane_type],
                dest=self.rank - 1,
                sendtag=0,
                recvbuf=[u[0, :, :], 1, self.plane_type],
                source=self.rank - 1,
                recvtag=1,
            )

        # Exchange with upper neighbor
        if self.rank < self.size - 1:
            self.comm.Sendrecv(
                [u[-2, :, :], 1, self.plane_type],
                dest=self.rank + 1,
                sendtag=1,
                recvbuf=[u[-1, :, :], 1, self.plane_type],
                source=self.rank + 1,
                recvtag=0,
            )

        t1 = time.perf_counter()
        self.halo_times.append(t1 - t0)

    def _gather_solution(self, u_local, N):
        u_global = np.zeros((N, N, N)) if self.rank == 0 else None
        local_interior = u_local[1:-1, :, :].copy()
        all_locals = self.comm.gather(local_interior, root=0)

        if self.rank == 0:
            current_z = 1
            for rank_data in all_locals:
                rank_local_N = rank_data.shape[0]
                u_global[current_z : current_z + rank_local_N, :, :] = rank_data
                current_z += rank_local_N

        return u_global

    def _setup_local_arrays(self, u1, u2, f, N):
        local_N, z_start, z_end = self._decompose_domain(N)
        local_shape = (local_N + 2, N, N)

        # Initialize and populate arrays
        locals_arrays = []
        for arr in [u1, u2, f]:
            local_arr = np.zeros(local_shape)
            local_arr[1:-1, :, :] = arr[z_start:z_end, :, :]
            if self.rank == 0:
                local_arr[0, :, :] = arr[z_start - 1, :, :]
            if self.rank == self.size - 1:
                local_arr[-1, :, :] = arr[z_end, :, :]
            locals_arrays.append(local_arr)

        return tuple(locals_arrays)
