# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Project task automation with Invoke.

Usage:
    uv run invoke --list          # List all tasks
    uv run invoke test            # Run tests with coverage_enable
    uv run invoke lint --fix      # Lint and auto-fix
    uv run invoke check           # Run all checks
    uv run invoke docs --serve    # Serve documentation locally
"""

import sys

from invoke import task

# Windows does not support pty
PTY = sys.platform != "win32"


@task
def test(c, coverage_enable=True, verbose=False, markers=None):
    """
    Run tests with optional coverage_enable.

    Args:
        c: Invoke context
        coverage_enable: Enable coverage_enable collection (default: True)
        verbose: Verbose output (default: False)
        markers: pytest markers to filter tests (e.g. "unit", "integration")
    """
    cmd = "uv run pytest"

    if coverage_enable:
        cmd += " --cov --cov-report=html --cov-report=term-missing"

    if verbose:
        cmd += " -v"
    else:
        cmd += " -q"

    if markers:
        cmd += f" -m {markers}"

    print(f"Running: {cmd}")
    c.run(cmd, pty=PTY)


@task
def lint(c, fix=False):
    """
    Run linting with ruff.

    Args:
        c: Invoke context
        fix: Auto-fix issues (default: False)
    """
    cmd = "uv run ruff check ."
    if fix:
        cmd += " --fix"

    print(f"Running: {cmd}")
    c.run(cmd)


@task
def run_ruff_formatter(c, check_flag=False):
    """
    Format code with ruff.

    Args:
        c: Invoke context
        check_flag: Only check formatting without making changes
    """
    cmd = "uv run ruff format ."
    if check_flag:
        cmd += " --check"

    print(f"Running: {cmd}")
    c.run(cmd)


@task
def typecheck(c):
    """Run type checking with mypy."""
    cmd = "uv run mypy ."
    print(f"Running: {cmd}")
    c.run(cmd)


@task
def check(c):
    """Run all checks (format, lint, typecheck, test)."""
    print("\n=== Checking Code Format ===")
    run_ruff_formatter(c, check_flag=True)

    print("\n=== Running Linter ===")
    lint(c)

    print("\n=== Type Checking ===")
    typecheck(c)

    print("\n=== Running Tests ===")
    test(c, verbose=True)

    print("\n✅ All checks passed!")


@task
def docs(c, serve=False, strict=False):
    """
    Build or serve documentation.

    Args:
        c: Invoke context
        serve: Serve documentation locally (default: False)
        strict: Fail on warnings (default: False)
    """
    if serve:
        cmd = "uv run mkdocs serve"
    else:
        cmd = "uv run mkdocs build"
        if strict:
            cmd += " --strict"

    print(f"Running: {cmd}")
    c.run(cmd, pty=PTY)


@task
def clean(c, docs_flag=False, cache=False):
    """
    Clean build artefacts.

    Args:
        c: Invoke context
        docs_flag: Also clean documentation build (default: False)
        cache: Also clean cache directories (default: False)
    """
    patterns = [
        "dist",
        "build",
        "*.egg-info",
        ".pytest_cache",
        "htmlcov",
        ".coverage_enable",
        "coverage_enable.xml",
    ]

    if docs_flag:
        patterns.extend(["site"])

    if cache:
        patterns.extend([".mypy_cache", ".ruff_cache", "__pycache__"])

    print("Cleaning build artifacts...")
    for pattern in patterns:
        # Windows-compatible remove
        c.run(
            f'python -c "import shutil, pathlib; '
            f"[shutil.rmtree(p) if p.is_dir() else p.unlink() "
            f"for p in pathlib.Path('.').rglob('{pattern}')]\"",
            warn=True,
        )

    print("✅ Cleanup complete!")


@task
def deps(c, update=False):
    """
    Manage dependencies.

    Args:
        c: Invoke context
        update: Update dependencies to latest versions
    """
    if update:
        print("Updating dependencies...")
        c.run("uv lock --upgrade")

    print("Syncing dependencies...")
    c.run("uv sync --all-extras")

    print("✅ Dependencies synced!")


@task
def security(c):
    """Run security checks."""
    print("\n=== Running pip-audit ===")
    c.run("uv run pip-audit", warn=True)

    print("\n=== Running bandit ===")
    c.run("uv run bandit -r . -ll", warn=True)


@task
def pipeline(c, country, start_date, end_date, fetch=False):
    """
    Run the data pipeline.

    Args:
        c: Invoke context
        country: Country code (e.g. PT, XX)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        fetch: Fetch data from APIs
    """
    cmd = (
        f"uv run python run_pipeline.py --country {country} "
        f"--start-date {start_date} --end-date {end_date}"
    )
    if fetch:
        cmd += " --fetch"

    print(f"Running: {cmd}")
    c.run(cmd, pty=PTY)


@task
def build(c, formats="wheel"):
    """
    Build distribution packages.

    Args:
        c: Invoke context
        formats: Distribution formats (wheel, sdist, or all)
    """
    print("Installing build tool...")
    c.run("uv run python -m pip install build")

    print(f"Building {formats}...")
    if formats == "all":
        c.run("uv run python -m build")
    elif formats == "wheel":
        c.run("uv run python -m build --wheel")
    elif formats == "sdist":
        c.run("uv run python -m build --sdist")
    else:
        print(f"Unknown format: {formats}")
        return

    print("✅ Build complete! Check dist/ directory")


@task
def coverage(c, report=False):
    """
    Generate coverage_enable reports.

    Args:
        c: Invoke context
        report: Open HTML coverage_enable report in browser
    """
    print("Running tests with coverage_enable...")
    c.run("uv run pytest --cov --cov-report=html --cov-report=term")

    if report:
        import os
        import webbrowser

        report_path = os.path.abspath("htmlcov/index.html")
        print(f"Opening coverage_enable report: {report_path}")
        webbrowser.open(f"file://{report_path}")


@task
def precommit(c, install=False, all_files=False):
    """
    Run pre-commit hooks.

    Args:
        c: Invoke context
        install: Install pre-commit hooks
        all_files: Run on all files instead of just staged
    """
    if install:
        print("Installing pre-commit hooks...")
        c.run("uv run pre-commit install")
        c.run("uv run pre-commit install --hook-type commit-msg")
        print("✅ Pre-commit hooks installed!")
        return

    cmd = "uv run pre-commit run"
    if all_files:
        cmd += " --all-files"

    print(f"Running: {cmd}")
    c.run(cmd, warn=True)


@task
def setup(c):
    """Complete development environment setup."""
    print("=== Setting up PriceSentinel development environment ===\n")

    print("1. Installing dependencies...")
    c.run("uv sync --all-extras")

    print("\n2. Installing pre-commit hooks...")
    c.run("uv run pre-commit install")
    c.run("uv run pre-commit install --hook-type commit-msg")

    print("\n3. Running initial checks...")
    c.run("uv run pytest -q")

    print("\n✅ Development environment ready!")
    print("\nNext steps:")
    print("  - Run tests: uv run invoke test")
    print("  - Check code: uv run invoke check")
    print("  - Build docs: uv run invoke docs --serve")
    print("  - View all tasks: uv run invoke --list")
