# Build Configuration Implementation Guide

This guide provides step-by-step instructions to implement the recommendations from the Build Assessment Report.

## Phase 1: Critical Fixes (30 minutes)

### Step 1: Fix Missing Dependencies

**Action**: Update D:\Coding\01_Projects\pricesentinel\pyproject.toml

**Changes**:
1. Add missing runtime dependency
2. Move pytest to optional dependencies
3. Add development tool dependencies
4. Add build system configuration
5. Loosen pandas-stubs constraint

**Execute**:
```bash
# Backup current file
cp pyproject.toml pyproject.toml.backup

# Replace with improved version
cp pyproject.toml.improved pyproject.toml

# Sync dependencies
uv sync --all-extras

# Verify installation
uv run python -c "import xmltodict; print('xmltodict OK')"
uv run ruff --version
uv run mypy --version
uv run pytest --version
```

**Expected Output**:
```
xmltodict OK
ruff 0.8.4 (or higher)
mypy 1.14.1 (or higher)
pytest 9.0.1 (or higher)
```

**Verification**:
```bash
# Test Portugal electricity fetcher (uses xmltodict)
uv run python -c "from data_fetchers.portugal.electricity import PortugalElectricityFetcher; print('Import successful')"
```

### Step 2: Update CI Workflow

**Action**: Update D:\Coding\01_Projects\pricesentinel\.github\workflows\ci.yml

**Changes**:
1. Use setup-uv action with caching
2. Simplify to cross-platform commands
3. Enable coverage collection
4. Add coverage upload
5. Build on all platforms

**Execute**:
```bash
# Backup current file
cp .github/workflows/ci.yml .github/workflows/ci.yml.backup

# Replace with improved version
cp .github/workflows/ci.yml.improved .github/workflows/ci.yml
```

**Note**: You'll need to sign up for Codecov (free for open source) or remove the codecov upload step.

### Step 3: Enable Coverage Collection

**Action**: Update D:\Coding\01_Projects\pricesentinel\pytest.ini

**Changes**:
1. Uncomment coverage options
2. Add coverage report formats

**Execute**:
```bash
# The improved pyproject.toml includes [tool.pytest.ini_options]
# You can either:
# Option A: Keep pytest.ini and update manually
# Option B: Delete pytest.ini (config now in pyproject.toml)

# Recommended: Option B
mv pytest.ini pytest.ini.backup

# Test coverage
uv run pytest --cov --cov-report=term
```

**Expected Output**:
```
---------- coverage: platform win32, python 3.13.7-final-0 -----------
Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
config/__init__.py                            8      0   100%
config/country_registry.py                  120     15    88%
...
-------------------------------------------------------------
TOTAL                                      XXXX    XXX    XX%
```

### Step 4: Update Renovate Configuration (Optional)

**Action**: Update D:\Coding\01_Projects\pricesentinel\renovate.json

**Execute**:
```bash
cp renovate.json renovate.json.backup
cp renovate.json.improved renovate.json
```

---

## Phase 2: Quality Improvements (2 hours)

### Step 5: Complete Package Metadata

**Status**: Already included in improved pyproject.toml

**Verification**:
```bash
uv run python -m build
ls dist/

# Should show:
# pricesentinel-0.1.0-py3-none-any.whl
# pricesentinel-0.1.0.tar.gz
```

### Step 6: Add CLI Entry Point

**Status**: Already included in improved pyproject.toml

**Action Required**: Update run_pipeline.py to have a proper main() function

**Current** (run_pipeline.py, bottom):
```python
if __name__ == "__main__":
    # ... existing code ...
```

**Required**: Wrap in a function
```python
def main():
    """Main entry point for pricesentinel CLI."""
    # ... move all existing code here ...

if __name__ == "__main__":
    main()
```

**Verification**:
```bash
uv sync
uv run pricesentinel --help
```

### Step 7: Consolidate Tool Configurations

**Status**: Completed in improved pyproject.toml

**Optional Cleanup**:
```bash
# After verifying everything works:
rm pytest.ini.backup
rm mypy.ini  # Config now in pyproject.toml
rm ruff.toml  # Config now in pyproject.toml

# Note: This consolidation is OPTIONAL
# You can keep separate files if preferred
```

### Step 8: Run Full Test Suite

**Execute**:
```bash
# Run tests with coverage
uv run pytest -v --cov --cov-report=html --cov-report=term

# View HTML coverage report
# Open in browser: htmlcov/index.html

# Run type checking
uv run mypy .

# Run linting
uv run ruff check .
```

---

## Phase 3: Documentation Setup (2-3 hours)

### Step 9: Install Documentation Tools

**Execute**:
```bash
# Install docs dependencies
uv sync --group docs

# Or install individually
uv add --group docs mkdocs mkdocs-material mkdocstrings[python]
```

### Step 10: Set Up MkDocs Structure

**Execute**:
```bash
# Copy template
cp mkdocs.yml.template mkdocs.yml

# Create docs structure
mkdir -p docs/getting-started
mkdir -p docs/user-guide
mkdir -p docs/architecture
mkdir -p docs/api/core
mkdir -p docs/api/config
mkdir -p docs/api/data-fetchers
mkdir -p docs/development

# Create index from README
cp README.md docs/index.md

# Create initial pages (examples below)
```

**Create docs/getting-started/installation.md**:
```markdown
# Installation

## Prerequisites

- Python 3.13 or higher
- Git

## Install from source

\`\`\`bash
git clone https://github.com/JDPS/pricesentinel.git
cd pricesentinel
uv sync --all-extras
\`\`\`

## Install with pip (future)

\`\`\`bash
pip install pricesentinel
\`\`\`
```

**Create docs/api/core/abstractions.md**:
```markdown
# Core Abstractions

::: core.abstractions
    options:
      show_source: true
      members:
        - ElectricityDataFetcher
        - WeatherDataFetcher
        - GasDataFetcher
        - EventDataProvider
```

### Step 11: Build and Serve Documentation

**Execute**:
```bash
# Build documentation
uv run mkdocs build

# Serve locally
uv run mkdocs serve

# Open browser to http://127.0.0.1:8000
```

### Step 12: Add Documentation Deployment Workflow

**Create** D:\Coding\01_Projects\pricesentinel\.github\workflows\docs.yml:

```yaml
# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"

      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --group docs

      - name: Build documentation
        run: uv run mkdocs build

      - name: Deploy to GitHub Pages
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: uv run mkdocs gh-deploy --force
```

---

## Phase 4: Advanced Enhancements (Optional, 1-2 days)

### Step 13: Add Task Runner (Optional)

**Install Invoke**:
```bash
uv add --group dev invoke
```

**Create tasks.py**:
```python
# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Project task automation with Invoke."""

from invoke import task


@task
def test(c, coverage=True, verbose=False):
    """Run tests with optional coverage."""
    cmd = "uv run pytest"
    if coverage:
        cmd += " --cov --cov-report=html --cov-report=term"
    if verbose:
        cmd += " -v"
    c.run(cmd)


@task
def lint(c, fix=False):
    """Run linting with ruff."""
    cmd = "uv run ruff check ."
    if fix:
        cmd += " --fix"
    c.run(cmd)


@task
def format(c):
    """Format code with ruff."""
    c.run("uv run ruff format .")


@task
def typecheck(c):
    """Run type checking with mypy."""
    c.run("uv run mypy .")


@task
def check(c):
    """Run all checks (lint, typecheck, test)."""
    lint(c)
    typecheck(c)
    test(c)


@task
def docs(c, serve=False):
    """Build or serve documentation."""
    if serve:
        c.run("uv run mkdocs serve")
    else:
        c.run("uv run mkdocs build")


@task
def clean(c):
    """Clean build artifacts."""
    patterns = [
        "dist",
        "build",
        "*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        ".coverage",
        "coverage.xml",
        "__pycache__",
    ]
    for pattern in patterns:
        c.run(f"rm -rf {pattern}", warn=True)
```

**Usage**:
```bash
# Run tasks
uv run invoke test
uv run invoke lint --fix
uv run invoke format
uv run invoke check
uv run invoke docs --serve

# Or with alias
alias inv="uv run invoke"
inv test
inv check
```

### Step 14: Add Pre-commit Hook Installation

**Update** D:\Coding\01_Projects\pricesentinel\README.md or create CONTRIBUTING.md:

```markdown
## Development Setup

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/JDPS/pricesentinel.git
   cd pricesentinel
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   uv sync --all-extras
   \`\`\`

3. Install pre-commit hooks:
   \`\`\`bash
   uv run pre-commit install
   \`\`\`

4. Run tests to verify setup:
   \`\`\`bash
   uv run pytest
   \`\`\`
```

### Step 15: Configure Coverage Badges (Optional)

**If using Codecov**:
1. Sign up at https://codecov.io
2. Link your GitHub repository
3. Get your badge markdown from Codecov dashboard
4. Add to README.md:

```markdown
[![codecov](https://codecov.io/gh/JDPS/pricesentinel/branch/main/graph/badge.svg)](https://codecov.io/gh/JDPS/pricesentinel)
```

**Alternative: Local badge generation**:
```bash
uv add --group dev coverage-badge
uv run coverage-badge -o coverage.svg -f
```

Add to README.md:
```markdown
![Coverage](./coverage.svg)
```

---

## Verification Checklist

After completing the implementation, verify:

### Critical Fixes
- [ ] xmltodict imported successfully
- [ ] All dev tools installed (ruff, mypy, pytest-cov)
- [ ] Tests run with coverage
- [ ] CI workflow updated
- [ ] Build produces artifacts

### Quality Improvements
- [ ] Package metadata complete
- [ ] CLI entry point works (`pricesentinel --help`)
- [ ] All tests pass
- [ ] Coverage > 75%
- [ ] No linting errors
- [ ] No type errors

### Documentation
- [ ] MkDocs builds successfully
- [ ] Documentation site accessible locally
- [ ] API reference generated
- [ ] GitHub Pages deployed (if configured)

### CI/CD
- [ ] CI passes on all platforms
- [ ] Coverage uploaded (if configured)
- [ ] Security scan runs
- [ ] Build artifacts created

---

## Rollback Plan

If anything goes wrong, rollback is simple:

```bash
# Restore original files
cp pyproject.toml.backup pyproject.toml
cp .github/workflows/ci.yml.backup .github/workflows/ci.yml
cp pytest.ini.backup pytest.ini
cp renovate.json.backup renovate.json

# Re-sync dependencies
uv sync

# Verify
uv run pytest
```

---

## Next Steps

After completing this implementation:

1. **Commit Changes**:
   ```bash
   git add .
   git commit -m "build: improve dependency management and CI configuration

   - Add missing xmltodict dependency
   - Move dev tools to optional dependencies
   - Add build system configuration
   - Enable coverage collection
   - Optimize CI with caching
   - Set up documentation tooling"
   ```

2. **Push and Verify CI**:
   ```bash
   git push
   # Check GitHub Actions for CI status
   ```

3. **Monitor Coverage**:
   - Check coverage reports
   - Aim for >80% coverage
   - Add tests for uncovered code

4. **Set Up Documentation**:
   - Complete documentation pages
   - Deploy to GitHub Pages
   - Link from README

5. **Plan Future Improvements**:
   - Performance benchmarking
   - Integration tests
   - Release automation
   - Multi-version testing

---

## Estimated Time

- **Phase 1 (Critical)**: 30-60 minutes
- **Phase 2 (Quality)**: 1-2 hours
- **Phase 3 (Documentation)**: 2-3 hours
- **Phase 4 (Advanced)**: 1-2 days (optional)

**Total for Essential Changes**: 4-5 hours
**Total with All Enhancements**: 2-3 days

---

## Support

If you encounter issues:

1. Check the BUILD_ASSESSMENT_REPORT.md for detailed explanations
2. Verify Python version: `python --version` (should be 3.13+)
3. Verify uv version: `uv --version` (should be latest)
4. Check GitHub Actions logs for CI failures
5. Review individual tool documentation:
   - uv: https://github.com/astral-sh/uv
   - ruff: https://docs.astral.sh/ruff/
   - mypy: https://mypy.readthedocs.io/
   - pytest: https://docs.pytest.org/
   - MkDocs: https://www.mkdocs.org/

---

## File Locations Reference

**Configuration Files**:
- D:\Coding\01_Projects\pricesentinel\pyproject.toml - Main project config
- D:\Coding\01_Projects\pricesentinel\.github\workflows\ci.yml - CI workflow
- D:\Coding\01_Projects\pricesentinel\mkdocs.yml - Documentation config
- D:\Coding\01_Projects\pricesentinel\renovate.json - Dependency updates

**Improved Templates**:
- D:\Coding\01_Projects\pricesentinel\pyproject.toml.improved
- D:\Coding\01_Projects\pricesentinel\.github\workflows\ci.yml.improved
- D:\Coding\01_Projects\pricesentinel\mkdocs.yml.template
- D:\Coding\01_Projects\pricesentinel\renovate.json.improved

**Reports**:
- D:\Coding\01_Projects\pricesentinel\BUILD_ASSESSMENT_REPORT.md
- D:\Coding\01_Projects\pricesentinel\IMPLEMENTATION_GUIDE.md (this file)
