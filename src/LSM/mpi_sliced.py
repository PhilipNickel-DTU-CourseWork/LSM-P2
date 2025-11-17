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
        if self.verbose and self.rank == 0:
            print(f"Using {'numba' if self.config.use_numba else 'numpy'} kernel with {self.size} MPI ranks")

    def solve(self, u1, u2, f, h, max_iter, tolerance=1e-8, u_true=None):
        """Solve using MPI sliced Jacobi iteration."""
        N = u1.shape[0]

        # Setup local arrays
        u1_local, u2_local, f_local = self._setup_local_arrays(u1, u2, f, N)

        converged = False
        compute_times = []
        comm_times = []
        residual_history = []
        t_start = time.perf_counter()

        # Main iteration loop
        for i in range(max_iter):
            if i % 2 == 0:
                uold_local, u_local = u1_local, u2_local
            else:
                u_local, uold_local = u1_local, u2_local

            # Exchange boundaries
            self._exchange_boundaries(uold_local, compute_times, comm_times)

            # Jacobi step
            t_comp_start = time.perf_counter()
            local_residual = self._step(uold_local, u_local, f_local, h, self.config.omega)
            t_comp_end = time.perf_counter()
            compute_times.append(t_comp_end - t_comp_start)

            # Global residual
            t_comm_start = time.perf_counter()
            global_residual = self.comm.allreduce(local_residual**2, op=MPI.SUM)
            global_residual = np.sqrt(global_residual)
            t_comm_end = time.perf_counter()
            comm_times.append(t_comm_end - t_comm_start)
            residual_history.append(global_residual)

            # Check convergence
            if global_residual < tolerance:
                converged = True
                if self.verbose and self.rank == 0:
                    print(f"Converged at iteration {i + 1} (residual: {global_residual:.2e})")
                break

        elapsed_time = time.perf_counter() - t_start

        if not converged and self.verbose and self.rank == 0:
            print(f"Did not converge after {max_iter} iterations (residual: {global_residual:.2e})")

        # Gather solution
        u_global = self._gather_solution(u_local, N)

        # Compute error on rank 0 and broadcast
        final_error = 0.0
        if self.rank == 0 and u_true is not None:
            final_error = np.linalg.norm(u_global - u_true)
            if self.verbose:
                print(f"Final error vs true solution: {final_error:.2e})")
        final_error = self.comm.bcast(final_error, root=0)

        # Build per-rank results
        per_rank_results = PerRankResults(
            mpi_rank=self.rank,
            hostname=socket.gethostname(),
            wall_time=elapsed_time,
            compute_time=sum(compute_times),
            mpi_comm_time=sum(comm_times),
        )

        # Gather all per-rank results
        all_perrank = self.comm.gather(per_rank_results, root=0)

        # Build config and global results on rank 0
        if self.rank == 0:
            runtime_config = RuntimeConfig(
                N=N,
                method="mpi_sliced_jacobi",
                omega=self.config.omega,
                tolerance=tolerance,
                max_iter=max_iter,
                use_numba=self.config.use_numba,
                num_threads=self.get_num_threads(self.config.use_numba),
                mpi_ranks=self.size,
            )

            max_wall_time = max(pr.wall_time for pr in all_perrank)
            total_compute_time = sum(pr.compute_time for pr in all_perrank)
            total_comm_time = sum(pr.mpi_comm_time for pr in all_perrank)

            global_results = GlobalResults(
                iterations=i + 1,
                residual_history=residual_history,
                converged=converged,
                final_error=final_error,
                wall_time=max_wall_time,
                compute_time=total_compute_time,
                mpi_comm_time=total_comm_time,
            )
        else:
            runtime_config = RuntimeConfig()
            global_results = GlobalResults()

        # Broadcast to all ranks
        runtime_config = self.comm.bcast(runtime_config, root=0)
        global_results = self.comm.bcast(global_results, root=0)

        return u_global, runtime_config, global_results, per_rank_results

    def _decompose_domain(self, N):
        interior_N = N - 2
        base_size = interior_N // self.size
        remainder = interior_N % self.size

        if self.rank < remainder:
            local_N = base_size + 1
            z_start = self.rank * local_N + 1
        else:
            local_N = base_size
            z_start = remainder * (base_size + 1) + (self.rank - remainder) * base_size + 1

        z_end = z_start + local_N
        return local_N, z_start, z_end

    def _exchange_boundaries(self, u, compute_time_list, comm_time_list):
        t0 = time.perf_counter()

        if self.rank > 0:
            self._smart_sendrecv(
                self.comm,
                sendbuf=u[1, :, :],
                dest=self.rank - 1,
                sendtag=0,
                recvbuf=u[0, :, :],
                source=self.rank - 1,
                recvtag=1,
            )

        if self.rank < self.size - 1:
            self._smart_sendrecv(
                self.comm,
                sendbuf=u[-2, :, :],
                dest=self.rank + 1,
                sendtag=1,
                recvbuf=u[-1, :, :],
                source=self.rank + 1,
                recvtag=0,
            )

        self.comm.Barrier()
        t1 = time.perf_counter()
        comm_time_list.append(t1 - t0)

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
        u1_local = np.zeros(local_shape)
        u2_local = np.zeros(local_shape)
        f_local = np.zeros(local_shape)

        u1_local[1:-1, :, :] = u1[z_start:z_end, :, :]
        u2_local[1:-1, :, :] = u2[z_start:z_end, :, :]
        f_local[1:-1, :, :] = f[z_start:z_end, :, :]

        if self.rank == 0:
            u1_local[0, :, :] = u1[z_start - 1, :, :]
            u2_local[0, :, :] = u2[z_start - 1, :, :]
            f_local[0, :, :] = f[z_start - 1, :, :]
        if self.rank == self.size - 1:
            u1_local[-1, :, :] = u1[z_end, :, :]
            u2_local[-1, :, :] = u2[z_end, :, :]
            f_local[-1, :, :] = f[z_end, :, :]

        return u1_local, u2_local, f_local
