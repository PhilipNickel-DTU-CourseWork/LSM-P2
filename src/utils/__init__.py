"""Utility modules for I/O, formatting, plotting, and numerical norms."""

from .io import (
    ensure_output_dir,
    load_simulation_data,
    save_simulation_data,
    get_repo_root,
    get_experiment_name,
    get_data_dir,
    get_figures_dir,
)
from .formatting import (
    extract_metadata,
    format_dt_latex,
    format_parameter_range,
    build_parameter_string,
)
from .norms import discrete_l2_error, discrete_l2_norm, discrete_linf_error

__all__ = [
    # I/O
    "ensure_output_dir",
    "load_simulation_data",
    "save_simulation_data",
    "get_repo_root",
    "get_experiment_name",
    "get_data_dir",
    "get_figures_dir",
    # Formatting
    "extract_metadata",
    "format_dt_latex",
    "format_parameter_range",
    "build_parameter_string",
    # Norms
    "discrete_l2_error",
    "discrete_l2_norm",
    "discrete_linf_error",
]
