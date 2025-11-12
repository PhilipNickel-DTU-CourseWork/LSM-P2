#!/usr/bin/env python3
"""Main script to run all examples."""

import argparse
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get repository root directory.

    Returns the repository root by detecting the presence of pyproject.toml.
    Works from any subdirectory of the repository.

    Returns
    -------
    Path
        Absolute path to the repository root

    """
    # Start from this script's location
    current = Path(__file__).resolve().parent

    # Walk up until we find pyproject.toml (marks repo root)
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume script is in repo root
    return current


# Get repo root once at module level
REPO_ROOT = get_repo_root()


def discover_scripts():
    """Find all Python scripts in Experiments/ directory."""
    experiments_dir = REPO_ROOT / "Experiments"

    if not experiments_dir.exists():
        return [], []

    scripts = [
        p
        for p in experiments_dir.rglob("*.py")
        if p.is_file() and p.name != "__init__.py"
    ]

    compute_scripts = sorted([s for s in scripts if "compute" in s.name])
    plot_scripts = sorted([s for s in scripts if "plot" in s.name])

    return compute_scripts, plot_scripts


def run_scripts(scripts):
    """Run scripts sequentially and report results."""
    if not scripts:
        print("  No scripts to run")
        return

    print(f"\nRunning {len(scripts)} scripts...\n")

    success_count = 0
    fail_count = 0

    for script in scripts:
        # Display path relative to repo root
        display_path = script.relative_to(REPO_ROOT)

        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(REPO_ROOT),  # Run from repo root
            )

            if result.returncode == 0:
                print(f"  ✓ {display_path}")
                success_count += 1
            else:
                print(f"  ✗ {display_path} (exit {result.returncode})")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print(f"  ✗ {display_path} (timeout)")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ {display_path} ({e})")
            fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")


def build_docs():
    """Build Sphinx documentation."""
    docs_dir = REPO_ROOT / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    print("\nBuilding Sphinx documentation...")

    if not source_dir.exists():
        print(f"  Error: Documentation source directory not found: {source_dir}")
        return False

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "sphinx-build",
                "-M",
                "html",
                str(source_dir),
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            print("  ✓ Documentation built successfully")
            print(f"  → Open: {build_dir / 'html' / 'index.html'}\n")
            return True
        else:
            print(f"  ✗ Documentation build failed (exit {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ✗ Documentation build timed out")
        return False
    except FileNotFoundError:
        print("  ✗ sphinx-build not found. Install with: uv sync")
        return False
    except Exception as e:
        print(f"  ✗ Documentation build failed: {e}")
        return False


def clean_docs():
    """Clean built Sphinx documentation."""
    import shutil

    build_dir = REPO_ROOT / "docs" / "build"

    print("\nCleaning Sphinx documentation...")

    if not build_dir.exists():
        print(f"  No build directory found at {build_dir}")
        return

    try:
        shutil.rmtree(build_dir)
        print(f"  ✓ Cleaned {build_dir.relative_to(REPO_ROOT)}\n")
    except Exception as e:
        print(f"  ✗ Failed to clean documentation: {e}\n")


def clean_all():
    """Clean all generated files and caches."""
    import shutil

    print("\nCleaning all generated files and caches...")

    cleaned = []
    failed = []

    # List of paths to clean (relative to repo root)
    clean_targets = [
        REPO_ROOT / "docs" / "build",
        REPO_ROOT / "docs" / "source" / "example_gallery",
        REPO_ROOT / "docs" / "source" / "generated",
        REPO_ROOT / "build",
        REPO_ROOT / "dist",
        REPO_ROOT / ".pytest_cache",
        REPO_ROOT / ".ruff_cache",
        REPO_ROOT / ".mypy_cache",
    ]

    # Clean directories
    for target_path in clean_targets:
        if target_path.exists():
            try:
                shutil.rmtree(target_path)
                cleaned.append(str(target_path.relative_to(REPO_ROOT)))
            except Exception as e:
                failed.append(f"{target_path.relative_to(REPO_ROOT)}: {e}")

    # Clean __pycache__ directories
    for pycache in REPO_ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            cleaned.append(str(pycache.relative_to(REPO_ROOT)))
        except Exception as e:
            failed.append(f"{pycache.relative_to(REPO_ROOT)}: {e}")

    # Clean .pyc files
    for pyc in REPO_ROOT.rglob("*.pyc"):
        try:
            pyc.unlink()
            cleaned.append(str(pyc.relative_to(REPO_ROOT)))
        except Exception as e:
            failed.append(f"{pyc.relative_to(REPO_ROOT)}: {e}")

    # Clean data directory (but keep README.md)
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.name != "README.md" and item.name != ".gitkeep":
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    cleaned.append(str(item.relative_to(REPO_ROOT)))
                except Exception as e:
                    failed.append(f"{item.relative_to(REPO_ROOT)}: {e}")

    # Print results
    if cleaned:
        print(f"  ✓ Cleaned {len(cleaned)} items")
    if failed:
        print(f"  ✗ Failed to clean {len(failed)} items:")
        for fail in failed[:5]:  # Show first 5 failures
            print(f"    - {fail}")
    if not cleaned and not failed:
        print("  Nothing to clean")
    print()


def ruff_check():
    """Run ruff linter."""
    print("\nRunning ruff check...")

    try:
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "."],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print("  ✓ No issues found\n")
            return True
        else:
            print(f"  ✗ Found issues (exit code {result.returncode})\n")
            return False

    except FileNotFoundError:
        print("  ✗ ruff not found. Install with: uv sync\n")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ ruff check timed out\n")
        return False
    except Exception as e:
        print(f"  ✗ ruff check failed: {e}\n")
        return False


def ruff_format():
    """Run ruff formatter."""
    print("\nRunning ruff format...")

    try:
        result = subprocess.run(
            ["uv", "run", "ruff", "format", "."],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print("  ✓ Code formatted successfully\n")
            return True
        else:
            print(f"  ✗ Formatting failed (exit code {result.returncode})\n")
            return False

    except FileNotFoundError:
        print("  ✗ ruff not found. Install with: uv sync\n")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ ruff format timed out\n")
        return False
    except Exception as e:
        print(f"  ✗ ruff format failed: {e}\n")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run example scripts and manage documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --compute                     Run data generation scripts
  python main.py --plot                        Run plotting scripts
  python main.py --build-docs                  Build Sphinx HTML documentation
  python main.py --clean-docs                  Clean built documentation
  python main.py --clean-all                   Clean all generated files and caches
  python main.py --lint                        Check code with ruff
  python main.py --format                      Format code with ruff
  python main.py --compute --plot              Run all example scripts
        """,
    )

    parser.add_argument(
        "--compute", action="store_true", help="Run data generation (compute) scripts"
    )
    parser.add_argument("--plot", action="store_true", help="Run plotting scripts")
    parser.add_argument(
        "--build-docs", action="store_true", help="Build Sphinx HTML documentation"
    )
    parser.add_argument(
        "--clean-docs", action="store_true", help="Clean built Sphinx documentation"
    )
    parser.add_argument(
        "--clean-all", action="store_true", help="Clean all generated files and caches"
    )
    parser.add_argument("--lint", action="store_true", help="Run ruff linter")
    parser.add_argument("--format", action="store_true", help="Run ruff formatter")

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n Error: No arguments provided. Please specify at least one option.\n")
        sys.exit(1)

    args = parser.parse_args()

    # Handle cleaning commands
    if args.clean_all:
        clean_all()

    if args.clean_docs:
        clean_docs()

    # Handle code quality commands
    if args.lint:
        ruff_check()

    if args.format:
        ruff_format()

    # Handle documentation commands
    if args.build_docs:
        build_docs()

    # Handle example scripts
    if args.compute or args.plot:
        compute_scripts, plot_scripts = discover_scripts()
        print(
            f"\nFound {len(compute_scripts)} compute scripts and {len(plot_scripts)} plot scripts"
        )

        if args.compute:
            run_scripts(compute_scripts)
        if args.plot:
            run_scripts(plot_scripts)


if __name__ == "__main__":
    main()
