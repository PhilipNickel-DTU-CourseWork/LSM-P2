"""Plotting utilities for spectral method visualizations.

Automatically applies seaborn style and custom utils.mplstyle on import.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# Auto-apply plotting styles on import
def _apply_styles():
    """Apply seaborn style and custom utils.mplstyle."""
    # Apply seaborn style first
    plt.style.use("seaborn-v0_8")

    # Then apply custom style on top
    style_path = Path(__file__).parent / "utils.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


# Apply styles when module is imported
_apply_styles()


def get_repo_root() -> Path:
    """Get repository root directory, handling both local and sphinx-gallery execution.

    Returns the repository root by detecting the presence of pyproject.toml.
    Works in both local execution (via main.py) and sphinx-gallery contexts.

    Returns
    -------
    Path
        Absolute path to the repository root

    """
    try:
        # Try to get caller's __file__ if available (local execution)
        import inspect

        frame = inspect.currentframe().f_back
        caller_file = frame.f_globals.get("__file__")
        if caller_file:
            current = Path(caller_file).resolve().parent
        else:
            # __file__ not available (sphinx-gallery)
            current = Path.cwd()
    except (AttributeError, KeyError):
        # Fallback to cwd
        current = Path.cwd()

    # Walk up until we find pyproject.toml (marks repo root)
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume 2 levels up from script directory
    # Works for Exercises/exercise_X/script.py structure
    return current.parent.parent if caller_file else current.parent.parent
