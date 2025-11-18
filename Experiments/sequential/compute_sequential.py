import math
from time import perf_counter as time
import numpy as np
from pathlib import Path
import pandas as pd

from utils import datatools, cli
from Poisson import SequentialJacobi


# Create the argument parser using shared utility
parser = cli.create_parser(
    methods=["jacobi", "view"],  # "view" is alias for "jacobi"
    default_method="jacobi",
    description="Sequential Poisson problem solver",
)

# Grab options!
options = parser.parse_args()
N: int = options.N
method: str = options.method
N_iter: int = options.iter
tolerance: float = options.tolerance


"""
In the below code, we'll have the axes aligned as z, y, x.
"""

# Create solver instance with all configuration
# For numba acceleration, use: use_numba=True
solver = SequentialJacobi(N=N, omega=0.75, max_iter=N_iter, tolerance=tolerance)

# Optional: warmup for numba (if use_numba=True)
# solver.warmup(N=10)

# Start MLflow logging
#solver.mlflow_start_log("/Shared/sequential_poisson_solver")

# Run the solver
solver.solve()

# End MLflow logging
#solver.mlflow_end_log()

# Print summary
solver.print_summary()

# Save results
#data_dir = datatools.get_data_dir()
#solver.save_results(data_dir, method, output_name=options.output)
