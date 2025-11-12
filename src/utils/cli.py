"""Command-line interface utilities for Poisson solver experiments.

This module provides shared argument parsing functionality for all solver implementations.
"""

from argparse import ArgumentParser
from typing import List


def create_parser(
    methods: List[str],
    default_method: str | None = None,
    description: str = "Poisson problem solver",
) -> ArgumentParser:
    """Create argument parser for Poisson solver experiments.

    Parameters
    ----------
    methods : List[str]
        List of available solver methods
    default_method : str, optional
        Default method to use (defaults to first method in list)
    description : str
        Parser description

    Returns
    -------
    ArgumentParser
        Configured argument parser

    Examples
    --------
    >>> # For sequential solver
    >>> parser = create_parser(["jacobi", "view"], description="Sequential Poisson solver")
    >>> options = parser.parse_args()

    >>> # For MPI sliced solver
    >>> parser = create_parser(["sliced"], description="MPI sliced Poisson solver")
    >>> options = parser.parse_args()
    """
    if default_method is None:
        default_method = methods[0]

    parser = ArgumentParser(description=description)

    # Grid size
    parser.add_argument(
        "-N",
        type=int,
        default=100,
        help="Number of divisions along each of the 3 dimensions",
    )

    # Iteration control
    parser.add_argument(
        "--iter",
        type=int,
        default=200,
        help="Number of (max) iterations.",
    )

    # Initial value
    parser.add_argument(
        "-v0",
        "--value0",
        type=float,
        default=0.0,
        help="The initial value of the grid u",
    )

    # Convergence tolerance
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="The tolerance of the normalized Frobenius norm of the residual for convergence.",
    )

    # Output file
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for saving results (default: auto-generated based on N, iterations, and method)",
    )

    # Solver method
    parser.add_argument(
        "--method",
        choices=methods,
        default=default_method,
        help=f"The chosen method to solve the Poisson problem (default: {default_method}).",
    )

    return parser
