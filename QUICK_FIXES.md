# Quick Fixes for Critical Issues

**Last Updated**: 2025-11-15
**Priority**: CRITICAL - Implement immediately

---

## Issue 1: Missing xmltodict Dependency

**Impact**: Portugal electricity fetcher will crash at runtime

**Fix** (5 minutes):

1. Open `pyproject.toml`
2. Add to dependencies array:
   ```toml
   "xmltodict>=0.14.2",
   ```
3. Run:
   ```bash
   uv sync
   ```

**Verify**:
```bash
uv run python -c "import xmltodict; print('OK')"
```

---

## Issue 2: Missing Development Tool Dependencies

**Impact**: CI will fail, inconsistent dev environments

**Fix** (10 minutes):

1. Open `pyproject.toml`
2. Add after dependencies section:
   ```toml
   [project.optional-dependencies]
   dev = [
       "ruff>=0.8.4",
       "mypy>=1.14.1",
       "pre-commit>=4.0.0",
       "types-PyYAML>=6.0.0",
       "types-requests>=2.32.0",
   ]
   test = [
       "pytest>=9.0.1",
       "pytest-cov>=6.0.0",
   ]
   build = [
       "build>=1.2.1",
   ]
   all = ["pricesentinel[dev,test,build]"]
   ```

3. Move pytest from dependencies to optional dependencies
4. Run:
   ```bash
   uv sync --all-extras
   ```

**Verify**:
```bash
uv run ruff --version
uv run mypy --version
uv run pytest --version
```

---

## Issue 3: Missing Build System Configuration

**Impact**: Cannot build distribution packages

**Fix** (2 minutes):

1. Open `pyproject.toml`
2. Add at the very top:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
   ```

**Verify**:
```bash
uv run python -m build
ls dist/
```

---

## Issue 4: CI Missing Dependencies

**Impact**: All CI checks will fail

**Fix** (5 minutes):

1. Open `.github/workflows/ci.yml`
2. Replace the sync steps (lines 48-56) with:
   ```yaml
   - name: Install dependencies
     run: uv sync --all-extras
   ```

**Better Fix** (10 minutes):

Replace entire file with improved version:
```bash
cp .github/workflows/ci.yml.improved .github/workflows/ci.yml
```

---

## Issue 5: pytest in Runtime Dependencies

**Impact**: Bloated production installs

**Fix** (included in Issue 2 above):

1. Remove `pytest>=9.0.1` from dependencies
2. Add to [project.optional-dependencies.test]

---

## Complete Quick Fix Script

**Run this to fix all critical issues** (15 minutes total):

```bash
# Navigate to project root
cd D:\Coding\01_Projects\pricesentinel

# Backup current configuration
cp pyproject.toml pyproject.toml.backup

# Apply improved configuration
cp pyproject.toml.improved pyproject.toml

# Sync all dependencies
uv sync --all-extras

# Verify installations
echo "=== Verifying installations ==="
uv run python -c "import xmltodict; print('✓ xmltodict')"
uv run ruff --version | head -1 | sed 's/^/✓ /'
uv run mypy --version | sed 's/^/✓ /'
uv run pytest --version | sed 's/^/✓ /'

# Update CI workflow
cp .github/workflows/ci.yml .github/workflows/ci.yml.backup
cp .github/workflows/ci.yml.improved .github/workflows/ci.yml

# Run tests to verify everything works
echo "=== Running tests ==="
uv run pytest -q

echo ""
echo "✅ All critical fixes applied!"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff pyproject.toml"
echo "  2. Test locally: uv run pytest"
echo "  3. Commit changes: git add . && git commit -m 'fix: add missing dependencies and build config'"
echo "  4. Push and verify CI: git push"
```

---

## Verification Checklist

After applying fixes, verify:

- [ ] xmltodict imports successfully
- [ ] ruff installed and working
- [ ] mypy installed and working
- [ ] pytest-cov installed
- [ ] All tests pass: `uv run pytest`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Type checking passes: `uv run mypy .`
- [ ] Package builds: `uv run python -m build`
- [ ] Coverage works: `uv run pytest --cov`

---

## Rollback (if needed)

```bash
cp pyproject.toml.backup pyproject.toml
cp .github/workflows/ci.yml.backup .github/workflows/ci.yml
uv sync
```

---

## Next Steps After Quick Fixes

1. **Enable Coverage in pytest.ini** (or delete it, config is in pyproject.toml)
2. **Complete package metadata** (see pyproject.toml.improved)
3. **Update Renovate config** (see renovate.json.improved)
4. **Set up documentation** (see IMPLEMENTATION_GUIDE.md Phase 3)

---

## Support

- Full details: BUILD_ASSESSMENT_REPORT.md
- Step-by-step guide: IMPLEMENTATION_GUIDE.md
- Configuration examples: *.improved files
