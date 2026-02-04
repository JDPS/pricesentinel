# Sensitive Information Audit Report

**Repository:** JDPS/pricesentinel
**Date:** 2026-02-04
**Scope:** Full repository content including source code, configuration, CI/CD,
test data, git history, and documentation.

---

## Overall Assessment: PASS

No hardcoded secrets, leaked credentials, or critical sensitive information
exposures were found. The repository follows security best practices for
credential management.

---

## Detailed Findings

### 1. Hardcoded Credentials & API Keys

| Check | Result |
|-------|--------|
| Hardcoded API keys (sk_*, pk_*, AKIA*, etc.) | None found |
| Hardcoded passwords or passphrases | None found |
| Database connection strings with credentials | None found |
| Private keys (RSA, EC, DSA, PEM) | None found |
| Bearer tokens, JWTs, OAuth secrets | None found |
| Webhook URLs with secrets | None found |
| Base64-encoded secrets | None found |
| URLs with embedded user:pass | None found |

**Status:** CLEAN

### 2. Environment Variable Handling

API keys are correctly loaded from environment variables, never hardcoded:

- `data_fetchers/portugal/electricity.py:48` — `os.getenv("ENTSOE_API_KEY")`
- `core/logging_config.py:39` — `os.getenv("LOG_LEVEL", level)`
- `core/features.py:297` — `os.getenv("PRICESENTINEL_SKIP_SAVE_ON_BAD_METRICS", "0")`

The `.env.example` file contains only placeholder values (`your_entsoe_api_key_here`),
not real credentials.

**Status:** CLEAN

### 3. .gitignore Coverage

The `.gitignore` properly excludes:

- `.env` (line 37)
- Data files: `*.csv`, `*.parquet`, `*.pkl`, `*.h5`, `*.hdf5` (lines 44-48)
- Model files: `models/*/` (line 51)
- Logs: `logs/`, `*.log` (lines 56-57)
- Jupyter notebooks: `*.ipynb` (line 80)
- IDE and OS artifacts

**Status:** CLEAN

### 4. CI/CD Workflows

All GitHub Actions workflows use GitHub Secrets properly:

- `.github/workflows/commitlint.yml` — `${{ secrets.GITHUB_TOKEN }}`
- No hardcoded tokens or credentials in any workflow file
- Security scanning (pip-audit, bandit) is configured in CI

**Status:** CLEAN

### 5. Git History

Checked deleted files across all branches. No `.env`, `.pem`, `.key`,
`credentials.*`, `secret*`, or `token*` files were ever committed and
subsequently deleted.

**Status:** CLEAN

### 6. Test Data

Test files use dummy credentials appropriately:

- `tests/test_portugal_and_shared_fetchers.py` — `monkeypatch.setenv("ENTSOE_API_KEY", "dummy-key")`
- `tests/test_portugal_pipeline_smoke.py` — same pattern
- Test metric files (`tests/models/XX/baseline/*/metrics.json`) contain only
  numerical model performance metrics (MAE, RMSE), no sensitive data.

**Status:** CLEAN

### 7. Logging — No Secret Leakage

No instances of API keys, tokens, passwords, or secrets being logged via
`logger.*()` or `print()` calls. The debug log at `electricity.py:56` only logs
the ENTSO-E domain identifier, not the API key.

**Status:** CLEAN

### 8. Personal Information Exposure

The maintainer email `joaosoarex@gmail.com` appears in 5 files:

| File | Context |
|------|---------|
| `pyproject.toml` | Package author/maintainer metadata |
| `SECURITY.md` | Security vulnerability reporting contact |
| `CODE_OF_CONDUCT.md` | Code of conduct enforcement contact |
| `README.md` | Technical contact |
| `REUSE.toml` | SPDX package supplier |

**Assessment:** This is a **low-severity informational note**, not a vulnerability.
Including maintainer contact information in these files is standard open-source
practice (required by PyPI, REUSE/SPDX, and recommended for security policies).
The maintainer has intentionally published this email for project communication.

**Status:** ACCEPTABLE — standard open-source practice

### 9. Country Configuration Files

`config/countries/PT.yaml` and `config/countries/XX.yaml` contain only
non-sensitive operational parameters: country codes, ENTSO-E domain identifiers
(public EIC codes), geographic coordinates of cities, and feature flags.

**Status:** CLEAN

---

## Security Tooling Already in Place

The repository has strong security infrastructure:

- **Pre-commit hooks** (`.pre-commit-config.yaml`): ruff, codespell, YAML validation
- **CI security job** (`.github/workflows/ci.yml`): pip-audit + bandit on every PR
- **Weekly security audit** (`.github/workflows/security.yml`): scheduled pip-audit
- **Dependency management**: Renovate + Dependabot for automated updates
- **Security policy** (`SECURITY.md`): documented vulnerability reporting process

---

## Recommendations

1. **Consider adding `detect-secrets`** as a pre-commit hook for automated secret
   scanning on every commit. This provides defense-in-depth beyond the current
   linting hooks.

2. **Consider GitHub secret scanning** — enable GitHub's built-in secret scanning
   alerts for the repository if not already enabled.

3. No remediation actions are required for the current repository state.

---

## Conclusion

The PriceSentinel repository demonstrates mature security practices. All
credentials are managed through environment variables, `.gitignore` excludes
sensitive file types, CI/CD uses GitHub Secrets exclusively, and no secrets
exist in git history. No action is required.
