# Contributing to PriceSentinel

Thank you for your interest in contributing to PriceSentinel!

This document gives a concise overview of how to work with the repository. The goal is to make contributions predictable, reviewable, and maintainable.

If anything here is unclear, feel free to open an issue or draft PR and ask.

---

## How to Propose Changes

1. **Check existing issues**
   Look through open issues to see if your bug or feature request is already tracked.

2. **Open an issue (preferred)**
   - For bugs, use the *Bug report* template.
   - For features or refactors, use the *Feature request* template.

3. **Fork and branch**
   - Fork the repo (if working from GitHub) and create a feature branch:
     - `feat/...` for new features
     - `fix/...` for bug fixes
     - `chore/...` for maintenance work

4. **Implement the change**
   - Keep changes focused and small.
   - Follow the existing code style and structure (see below).
   - Update or add tests where appropriate.

5. **Open a Pull Request**
   - Use the PR template and fill in the checklist.
   - Link relevant issues (`Fixes #123`).
   - Be ready to iterate based on review feedback.

---

## Development Setup

From the repository root:

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install the project and dev extras
pip install ".[dev,test]"
```

### Quick Training Demo (Mock Country)

To verify the training pipeline end-to-end using only synthetic data:

```bash
# 1. Ensure the virtualenv is active and dependencies installed

# 2. Run a short mock-country training run (no API keys needed)
python run_pipeline.py --country XX --all \
  --start-date 2024-01-01 --end-date 2024-01-07

# This will:
# - Fetch synthetic electricity, weather, gas, and event data
# - Clean and aggregate the data
# - Build basic features (lags + calendar features)
# - Train a baseline scikit-learn model and store it under models/XX/baseline/<run_id>/
```

This is the fastest way to check that your changes did not break the core pipeline.

### Useful Tasks

Using `invoke` tasks (see `tasks.py`):

```bash
# Run tests with coverage
uv run invoke test

# Run linting and type checking
uv run invoke check

# Format code
uv run invoke format

# Build docs (if configured)
uv run invoke docs --serve
```

---

## Code Style and Guidelines

- **Language & version**: Python 3.13+
- **Formatting & linting**: `ruff` and `ruff format`
- **Types**: Prefer explicit type hints; keep mypy happy where practical
- **Testing**: Use `pytest`; aim to cover new logic with tests

When making changes:

- Reuse existing abstractions and patterns (`core/abstractions.py`, `config/country_registry.py`, `core/pipeline.py`, etc.).
- Keep responsibilities small and composable (follow SOLID principles already used in the project).
- Avoid global state; prefer dependency injection via constructors where possible.

---

## Adding a New Country (High‑Level)

At a high level (see docs for full details):

1. Create `config/countries/{CODE}.yaml`.
2. Implement fetchers under `data_fetchers/{country}/` or reuse shared ones.
3. Register the country in `data_fetchers/__init__.py`.
4. Add or update tests for the new fetchers and registry behavior.

---

## Testing and Quality Gates

Before opening a PR, please run:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

If you have pre‑commit installed:

```bash
uv run pre-commit run --all-files
```

The CI pipeline will run these checks as well; keeping them green locally speeds up reviews.

---

## Versioning and Releases

At the moment the project is in **active development** and does not follow a formal release cycle. The general plan is:

- Use semantic versioning (**SemVer**) once the first public release is cut (e.g. `0.1.0`, `0.2.0`, `1.0.0`).
- Treat the default branch as the source of truth for the next release.
- Use tags to mark release points when they are created.

When contributing:

- Keep changes scoped so they can be reviewed and, if needed, cherry-picked into a release.
- Avoid mixing unrelated refactors and features in a single PR.
- If a change is potentially breaking for users (API, CLI flags, directory structure), call it out clearly in the PR summary so it can be scheduled for an appropriate version bump.

---

## Security and Responsible Disclosure

If you find a security issue, **please do not open a public issue**.
Instead, follow the instructions in `SECURITY.md` for reporting vulnerabilities responsibly.

---

## Code of Conduct

By participating in this project, you agree to follow the guidelines outlined in `CODE_OF_CONDUCT.md`.

We want a respectful and constructive environment for all contributors.
