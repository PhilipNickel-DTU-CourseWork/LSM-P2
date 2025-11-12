"""Abstract base class for Poisson solvers.

This module defines the common interface that all solver implementations must follow.
"""

from __future__ import annotations

from datetime import datetime
import math
import os
import socket
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from mpi4py import MPI

from .datastructures import RuntimeConfig, GlobalResults, PerRankResults


class PoissonSolver(ABC):
    """Abstract base class for all Poisson solvers.

    This defines the common interface that all solver implementations must follow.
    All solvers are MPI-aware, with sequential solvers being a special case where
    size=1 (single rank, no domain decomposition).

    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI communicator (default: MPI.COMM_WORLD for parallel, MPI.COMM_SELF for sequential)
    omega : float, default 0.75
        Relaxation parameter for weighted Jacobi (0 < omega <= 1)
    use_numba : bool, default True
        Use numba JIT compilation for computational kernels
    mpi_strategy : str, default "numpy_buffer"
        MPI communication strategy: "numpy_buffer" (copy-based) or "mpi_datatype" (zero-copy)
    verbose : bool, default False
        Whether to print convergence information (rank 0 only)

    """

    def __init__(
        self,
        comm=None,
        omega: float = 0.75,
        use_numba: bool = True,
        mpi_strategy: str = "numpy_buffer",
        verbose: bool = False,
    ):
        """Initialize solver with MPI setup."""
        self.omega = omega

        # MPI setup
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # MPI communication strategy
        from .mpi_strategies import create_mpi_strategy
        self.mpi_strategy = create_mpi_strategy(mpi_strategy)
        self.mpi_strategy_name = mpi_strategy

        # Only rank 0 prints
        self.verbose = verbose and (self.rank == 0)

        # Kernel selection (using helper method)
        self._step, self.use_numba = self._select_kernel(use_numba)

    @abstractmethod
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
        """Solve the Poisson problem.

        All subclasses must implement this method.

        Parameters
        ----------
        u1, u2 : np.ndarray
            Solution arrays for swapping, shape (N, N, N)
        f : np.ndarray
            Source term, shape (N, N, N)
        h : float
            Grid spacing
        max_iter : int
            Maximum iterations
        tolerance : float, default 1e-8
            Convergence tolerance
        u_true : np.ndarray, optional
            True solution for error tracking

        Returns
        -------
        u : np.ndarray
            Final solution array
        runtime_config : RuntimeConfig
            Global runtime configuration (same for all ranks)
        global_results : GlobalResults
            Global solver results (convergence, quality metrics)
        per_rank_results : PerRankResults
            Per-rank performance data for this rank

        """
        pass

    @abstractmethod
    def warmup(self, N: int = 10) -> None:
        """Warmup the solver (e.g., trigger JIT compilation).

        Parameters
        ----------
        N : int, default 10
            Grid size for warmup problem

        """
        pass

    @abstractmethod
    def _decompose_domain(self, N: int):
        """Compute local domain decomposition.

        Subclasses must implement this based on their decomposition strategy.
        For sequential solvers, this can be a no-op returning the full domain.
        """
        pass

    @abstractmethod
    def _exchange_boundaries(self, u: np.ndarray, compute_time_list: list, comm_time_list: list) -> None:
        """Exchange ghost/halo layers with neighbors.

        Subclasses must implement this based on their communication pattern.
        For sequential solvers, this can be a no-op.
        """
        pass

    @abstractmethod
    def _gather_solution(self, u_local: np.ndarray, N: int) -> np.ndarray:
        """Gather local solutions to rank 0.

        Subclasses must implement this based on their decomposition.
        For sequential solvers, this just returns the input array.

        Returns
        -------
        np.ndarray
            Global solution (only valid on rank 0, None on other ranks)
        """
        pass

    @abstractmethod
    def _setup_local_arrays(
        self, u1: np.ndarray, u2: np.ndarray, f: np.ndarray, N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set up local arrays from global arrays.

        Subclasses implement domain-specific setup (decomposition, ghost layers, etc.).

        Parameters
        ----------
        u1, u2 : np.ndarray
            Global solution arrays
        f : np.ndarray
            Global source term
        N : int
            Global grid size

        Returns
        -------
        u1_local : np.ndarray
            Local u1 array with ghost layers
        u2_local : np.ndarray
            Local u2 array with ghost layers
        f_local : np.ndarray
            Local f array with ghost layers
        """
        pass

    def _solve_common(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        f: np.ndarray,
        h: float,
        max_iter: int,
        tolerance: float,
        u_true: np.ndarray | None,
        method_name: str,
    ) -> Tuple[np.ndarray, RuntimeConfig, GlobalResults, PerRankResults]:
        """Common solve implementation for all solvers.

        This template method implements the main iteration loop.
        Subclasses provide domain-specific setup via _setup_local_arrays.

        Parameters
        ----------
        u1, u2 : np.ndarray
            Solution arrays
        f : np.ndarray
            Source term
        h : float
            Grid spacing
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
        u_true : np.ndarray, optional
            True solution
        method_name : str
            Solver method name for config

        Returns
        -------
        u_global : np.ndarray
            Final solution
        runtime_config : RuntimeConfig
            Runtime configuration
        global_results : GlobalResults
            Global solver results
        per_rank_results : PerRankResults
            Per-rank performance data
        """
        import time

        N = u1.shape[0]

        # Setup local arrays (subclass-specific)
        u1_local, u2_local, f_local = self._setup_local_arrays(u1, u2, f, N)

        converged = False
        compute_times = []
        comm_times = []

        # Start timing
        t_start = time.perf_counter()

        # Main iteration loop
        u_local = u2_local
        for i in range(max_iter):
            # Swap grids
            if i % 2 == 0:
                uold_local, u_local = u1_local, u2_local
            else:
                u_local, uold_local = u1_local, u2_local

            # Exchange boundaries (subclass-specific)
            self._exchange_boundaries(uold_local, compute_times, comm_times)

            # Perform one Jacobi step
            t_comp_start = time.perf_counter()
            local_residual = self._step(uold_local, u_local, f_local, h, self.omega)
            t_comp_end = time.perf_counter()
            compute_times.append(t_comp_end - t_comp_start)

            # Global residual reduction
            t_comm_start = time.perf_counter()
            global_residual = self.comm.allreduce(local_residual**2, op=MPI.SUM)
            global_residual = np.sqrt(global_residual)
            t_comm_end = time.perf_counter()
            comm_times.append(t_comm_end - t_comm_start)

            # Check convergence
            if global_residual < tolerance:
                converged = True
                self.log_convergence(converged, i, global_residual, max_iter)
                break

        # End timing
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start

        if not converged:
            self.log_convergence(converged, i, global_residual, max_iter)

        # Gather solution to rank 0
        u_global = self._gather_solution(u_local, N)

        # Compute final error and broadcast
        final_error = self._compute_and_broadcast_error(u_global, u_true)

        # Build per-rank results
        per_rank_results = self._build_per_rank_results(
            rank=self.rank,
            hostname=self.get_hostname(),
            wall_time=elapsed_time,
            compute_time=sum(compute_times),
            mpi_comm_time=sum(comm_times),
        )

        # Gather all per-rank results
        all_perrank = self._gather_all_per_rank(per_rank_results)

        # Build config and global results on rank 0
        runtime_config, global_results = self._build_results_on_rank0(
            N=N,
            h=h,
            method=method_name,
            tolerance=tolerance,
            max_iter=max_iter,
            iterations=i + 1,
            converged=converged,
            final_residual=global_residual,
            final_error=final_error,
            all_perrank=all_perrank,
        )

        # Broadcast results to all ranks
        runtime_config, global_results = self._broadcast_results(runtime_config, global_results)

        return u_global, runtime_config, global_results, per_rank_results

    def compute_error(self, u: np.ndarray, u_ref: np.ndarray) -> float:
        """Compute normalized L2 error between two arrays.

        This is a shared utility method available to all subclasses.

        Parameters
        ----------
        u : np.ndarray
            Computed solution
        u_ref : np.ndarray
            Reference solution

        Returns
        -------
        float
            Normalized L2 error

        """
        N = u.shape[0]
        return math.sqrt(np.sum((u - u_ref) ** 2)) / N**3

    def get_num_threads(self, use_numba: bool = False) -> int:
        """Get the number of threads available for parallel execution.

        This is a shared utility method available to all subclasses.

        Parameters
        ----------
        use_numba : bool, default False
            Whether numba is being used

        Returns
        -------
        int
            Number of threads available for parallel execution

        Notes
        -----
        - If numba is used and available, returns numba's thread count
        - Otherwise returns the CPU count (used by numpy's BLAS/LAPACK)
        - Falls back to 1 if thread count cannot be determined
        """
        if use_numba:
            try:
                from .kernels import NUMBA_AVAILABLE
                if NUMBA_AVAILABLE:
                    import numba
                    return numba.get_num_threads()
            except Exception:
                pass

        # For numpy operations or if numba not available, return CPU count
        return os.cpu_count() or 1

    def compute_final_error(self, u: np.ndarray, u_true: np.ndarray | None) -> float | None:
        """Compute final error against true solution with optional logging.

        This is a shared utility method available to all subclasses.

        Parameters
        ----------
        u : np.ndarray
            Computed solution
        u_true : np.ndarray, optional
            True solution for comparison

        Returns
        -------
        float or None
            Final error if u_true is provided, otherwise None
        """
        if u_true is None:
            return None

        final_error = self.compute_error(u, u_true)
        if self.verbose:
            print(f"Final error vs true solution: {final_error:.2e}")

        return final_error

    def log_convergence(self, converged: bool, iteration: int, residual: float, max_iter: int) -> None:
        """Log convergence status with consistent formatting.

        This is a shared utility method available to all subclasses.

        Parameters
        ----------
        converged : bool
            Whether the solver converged
        iteration : int
            Current iteration number (0-indexed)
        residual : float
            Final residual value
        max_iter : int
            Maximum number of iterations
        """
        if not self.verbose:
            return

        if converged:
            print(f"Converged at iteration {iteration + 1} (residual: {residual:.2e})")
        else:
            print(f"Did not converge after {max_iter} iterations (residual: {residual:.2e})")

    def get_hostname(self) -> str:
        """Get the hostname of the current machine.

        This is a shared utility method available to all subclasses.

        Returns
        -------
        str
            Hostname of the current machine
        """
        return socket.gethostname()

    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        This is a shared utility method available to all subclasses.

        Returns
        -------
        str
            Current timestamp in ISO format
        """
        return datetime.now().isoformat()

    def _select_kernel(self, use_numba: bool):
        """Select computational kernel (numpy or numba).

        This is a shared utility method for kernel selection.

        Parameters
        ----------
        use_numba : bool
            Whether to use numba JIT compilation

        Returns
        -------
        callable
            The selected kernel function
        bool
            Actual use_numba value (may be False if numba unavailable)
        """
        from .kernels import jacobi_step_numpy, jacobi_step_numba_parallel, NUMBA_AVAILABLE

        if use_numba:
            if NUMBA_AVAILABLE:
                if self.verbose:
                    print("Using numba-accelerated kernel with parallel execution")
                return jacobi_step_numba_parallel, True
            else:
                if self.verbose:
                    print("Warning: numba requested but not available, falling back to numpy")
                return jacobi_step_numpy, False
        else:
            return jacobi_step_numpy, False

    def _warmup_kernel(self, kernel, array_shape: tuple, h: float) -> None:
        """Warmup JIT compilation for the kernel.

        Parameters
        ----------
        kernel : callable
            The kernel function to warm up
        array_shape : tuple
            Shape of test arrays
        h : float
            Grid spacing for test
        """
        if self.verbose:
            print(f"Warming up kernel with shape {array_shape}...")

        u1 = np.zeros(array_shape)
        u2 = np.zeros(array_shape)
        f = np.random.randn(*array_shape)

        # Run a few iterations to trigger compilation
        for _ in range(5):
            kernel(u1, u2, f, h, self.omega)
            u1, u2 = u2, u1

        if self.verbose:
            print("Warmup complete!")

    def _build_per_rank_results(
        self,
        rank: int,
        hostname: str,
        wall_time: float,
        compute_time: float,
        mpi_comm_time: float,
    ) -> PerRankResults:
        """Build per-rank results dictionary.

        Parameters
        ----------
        rank : int
            MPI rank
        hostname : str
            Hostname
        wall_time : float
            Wall clock time
        compute_time : float
            Computation time
        mpi_comm_time : float
            MPI communication time

        Returns
        -------
        PerRankResults
            Per-rank performance data
        """
        return PerRankResults(
            mpi_rank=rank,
            hostname=hostname,
            wall_time=wall_time,
            compute_time=compute_time,
            mpi_comm_time=mpi_comm_time,
        )

    def _build_runtime_config(
        self,
        N: int,
        h: float,
        method: str,
        tolerance: float,
        max_iter: int,
        use_numba: bool,
        mpi_size: int,
    ) -> RuntimeConfig:
        """Build runtime configuration dictionary.

        Parameters
        ----------
        N : int
            Grid size
        h : float
            Grid spacing
        method : str
            Solver method name
        tolerance : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
        use_numba : bool
            Whether numba was used
        mpi_size : int
            Number of MPI ranks

        Returns
        -------
        RuntimeConfig
            Runtime configuration
        """
        return RuntimeConfig(
            N=N,
            h=h,
            method=method,
            omega=self.omega,
            tolerance=tolerance,
            max_iter=max_iter,
            use_numba=use_numba,
            num_threads=self.get_num_threads(use_numba=use_numba),
            mpi_size=mpi_size,
            timestamp=self.get_timestamp(),
        )

    def _gather_all_per_rank(self, per_rank_results: PerRankResults) -> list:
        """Gather all per-rank results to rank 0.

        Parameters
        ----------
        per_rank_results : PerRankResults
            This rank's performance data

        Returns
        -------
        list or None
            List of all per-rank results (only on rank 0, None on others)
        """
        return self.comm.gather(per_rank_results, root=0)

    def _aggregate_global_results(
        self,
        all_perrank: list,
        iterations: int,
        converged: bool,
        final_residual: float,
        final_error: float | None,
    ) -> GlobalResults:
        """Aggregate global results from all ranks.

        Parameters
        ----------
        all_perrank : list
            List of PerRankResults from all ranks
        iterations : int
            Number of iterations performed
        converged : bool
            Whether solver converged
        final_residual : float
            Final global residual
        final_error : float or None
            Final error vs true solution

        Returns
        -------
        GlobalResults
            Aggregated global results
        """
        if self.rank != 0:
            return GlobalResults()

        # Aggregate timing data
        max_wall_time = max(pr["wall_time"] for pr in all_perrank)
        total_compute_time = sum(pr["compute_time"] for pr in all_perrank)
        total_comm_time = sum(pr["mpi_comm_time"] for pr in all_perrank)

        global_results = GlobalResults(
            iterations=iterations,
            converged=converged,
            final_residual=final_residual,
            wall_time=max_wall_time,
            compute_time=total_compute_time,
            mpi_comm_time=total_comm_time,
        )

        if final_error is not None:
            global_results["final_error"] = final_error

        return global_results

    def _compute_and_broadcast_error(
        self, u_global: np.ndarray | None, u_true: np.ndarray | None
    ) -> float | None:
        """Compute error on rank 0 and broadcast to all ranks.

        Parameters
        ----------
        u_global : np.ndarray or None
            Global solution (valid on rank 0)
        u_true : np.ndarray or None
            True solution

        Returns
        -------
        float or None
            Final error (same value on all ranks)
        """
        final_error = None
        if self.rank == 0 and u_true is not None:
            final_error = self.compute_final_error(u_global, u_true)

        # Broadcast to all ranks
        final_error = self.comm.bcast(final_error, root=0)
        return final_error

    def _broadcast_results(
        self, runtime_config: RuntimeConfig, global_results: GlobalResults
    ) -> Tuple[RuntimeConfig, GlobalResults]:
        """Broadcast config and results from rank 0 to all ranks.

        Parameters
        ----------
        runtime_config : RuntimeConfig
            Runtime configuration (only valid on rank 0)
        global_results : GlobalResults
            Global results (only valid on rank 0)

        Returns
        -------
        RuntimeConfig
            Runtime configuration (valid on all ranks after broadcast)
        GlobalResults
            Global results (valid on all ranks after broadcast)
        """
        runtime_config = self.comm.bcast(runtime_config, root=0)
        global_results = self.comm.bcast(global_results, root=0)
        return runtime_config, global_results

    def _build_results_on_rank0(
        self,
        N: int,
        h: float,
        method: str,
        tolerance: float,
        max_iter: int,
        iterations: int,
        converged: bool,
        final_residual: float,
        final_error: float | None,
        all_perrank: list,
    ) -> Tuple[RuntimeConfig, GlobalResults]:
        """Build config and global results on rank 0.

        Parameters
        ----------
        N : int
            Grid size
        h : float
            Grid spacing
        method : str
            Solver method name
        tolerance : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
        iterations : int
            Actual iterations performed
        converged : bool
            Whether converged
        final_residual : float
            Final residual
        final_error : float or None
            Final error
        all_perrank : list
            All per-rank results

        Returns
        -------
        RuntimeConfig
            Runtime configuration
        GlobalResults
            Global solver results
        """
        if self.rank == 0:
            runtime_config = self._build_runtime_config(
                N=N,
                h=h,
                method=method,
                tolerance=tolerance,
                max_iter=max_iter,
                use_numba=self.use_numba,
                mpi_size=self.size,
            )

            global_results = self._aggregate_global_results(
                all_perrank=all_perrank,
                iterations=iterations,
                converged=converged,
                final_residual=final_residual,
                final_error=final_error,
            )
        else:
            runtime_config = RuntimeConfig()
            global_results = GlobalResults()

        return runtime_config, global_results
