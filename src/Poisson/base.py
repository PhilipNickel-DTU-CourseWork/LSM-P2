"""Base class for Poisson solvers."""

from dataclasses import asdict
import os
from numba.core.ir_utils import raise_on_unsupported_feature
import numpy as np
from pathlib import Path
from mpi4py import MPI
from .kernels import jacobi_step_numpy, jacobi_step_numba
from numba import get_num_threads
from .datastructures import GlobalConfig, GlobalFields, LocalFields, GlobalResults, LocalResults, TimeSeriesLocal, TimeSeriesGlobal
import mlflow
import pandas as pd

class PoissonSolver:

    def __init__(self, **kwargs):
        # MPI setup
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.rank = rank 

        # global Configuration
        self.config = GlobalConfig(**kwargs)
        self.config.num_threads = get_num_threads()
        self.config.mpi_size = comm.Get_size()

        # Problem --- contains solution arrays and so on 

        # Local Results and fields
        self.local_results = LocalResults()
        self.local_fields = LocalFields()
        self.global_timeseries = TimeSeriesLocal()
        
        # Global Results and fields
        if rank == 0: 
            self.global_results = GlobalResults()
            self.global_fields = GlobalFields(N=self.config.N)
            self.global_timeseries = TimeSeriesGlobal()

        # Kernel selection
        self._step = jacobi_step_numba if self.config.use_numba else jacobi_step_numpy
    
    def method_solve(self):
        pass 
   
    def solve(self):
        # start wall timings 
        if self.rank == 0:
            time_start = MPI.Wtime()
        
        # run method specific solver
        self.method_solve()

        # perform _post_solve
        self._post_solve(time_start)

   
    # ============================================================================
    # Internal methods
    # ============================================================================

    def _post_solve(self, start_time):
        """Aggregate timing results after solve."""
        # Calculate wall time
        if self.rank == 0:
            wall_time = MPI.Wtime() - start_time

            # Aggregate local timings to global results
            self.global_results.wall_time = wall_time
            self.global_results.compute_time = sum(self.global_timeseries.compute_times)
            self.global_results.mpi_comm_time = sum(self.global_timeseries.mpi_comm_times)
            self.global_results.halo_exchange_time = sum(self.global_timeseries.halo_exchange_times) 
        



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

    def print_summary(self):
        """Print a summary of the solver results."""
        print(f"Wall time = {self.global_results.wall_time:.6f} s")
        print(f"Compute time = {self.global_results.compute_time:.6f} s")
        print(f"MPI comm time = {self.global_results.mpi_comm_time:.6f} s")
        print(f"Iterations = {self.global_results.iterations}")
        if self.global_results.converged:
            print(f"Converged within tolerance {self.config.tolerance}")
        if self.global_results.final_error > 0:
            print(f"Final error = {self.global_results.final_error:.6e}")

    def save_results(self, data_dir, N, method, output_name=None):
        """Save all results to parquet files and grid to npy.

        Parameters
        ----------
        data_dir : Path
            Directory to save results
        N : int
            Grid size
        method : str
            Method name for file naming
        output_name : str, optional
            Custom base name for output files
        """
        # Create DataFrames
        df_runtime_config = pd.DataFrame([asdict(self.config)])
        df_global_results = pd.DataFrame([asdict(self.global_results)])
        df_per_rank_results = pd.DataFrame([asdict(pr) for pr in self.all_per_rank_results])

        # Generate file names
        if output_name:
            base_name = output_name.replace('.npz', '').replace('.parquet', '')
            config_file = data_dir / f"{base_name}_config.parquet"
            global_file = data_dir / f"{base_name}_global.parquet"
            perrank_file = data_dir / f"{base_name}_perrank.parquet"
            grid_file = data_dir / f"{base_name}_grid.npy"
        else:
            iter_run = self.global_results.iterations
            config_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_config.parquet"
            global_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_global.parquet"
            perrank_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_perrank.parquet"
            grid_file = data_dir / f"run_N{N}_iter{iter_run}_{method}_grid.npy"

        # Save files
        df_runtime_config.to_parquet(config_file, index=False)
        print(f"Config saved to: {config_file}")

        df_global_results.to_parquet(global_file, index=False)
        print(f"Global results saved to: {global_file}")

        df_per_rank_results.to_parquet(perrank_file, index=False)
        print(f"Per-rank results saved to: {perrank_file}")

        np.save(grid_file, self.u)
        print(f"Grid saved to: {grid_file}")





