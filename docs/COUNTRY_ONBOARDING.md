# Country Onboarding

This document defines the standard onboarding flow for new countries and the
qualification gates required before production rollout.

## Scope

Use this checklist when adding `config/countries/{CODE}.yaml`, country-specific
fetchers, and operational automation for a new market.

## Checklist

- Create country data directories:
  - `uv run python setup_country.py {CODE}`
- Scaffold optional starter files:
  - `uv run python setup_country.py {CODE} --scaffold-config --scaffold-fetchers --scaffold-tests`
- Validate config schema:
  - `config/countries/{CODE}.yaml` must pass strict schema validation.
- Create selection policy:
  - `config/selection_policies/{CODE}.yaml`
- Create monitoring thresholds:
  - `config/monitoring/{CODE}.yaml`
  - include uncertainty thresholds:
    - `quantile_coverage_warn`
    - `pinball_warn`
    - `interval_width_warn`
- Implement and register adapters:
  - `data_fetchers/{code}/`
  - registration in `data_fetchers/__init__.py`
- Add tests:
  - unit tests for fetchers/events
  - qualification tests (`tests/qualification/`)

## Qualification Gates

Run:

```bash
uv run python experiments/qualify_country.py --country {CODE} --start YYYY-MM-DD --end YYYY-MM-DD
```

Qualification gates:

- Fetch/Clean schema compliance:
  - required columns present for cleaned electricity/load/weather/gas/holidays.
- Feature completeness:
  - non-empty feature set
  - required baseline columns present
  - low missingness on required core features.
- End-to-end smoke forecast:
  - train + forecast pipeline creates a non-empty forecast artifact.

Generated artifacts:

- `outputs/reports/qualification_{CODE}_{YYYYMMDD}.json`
- `outputs/reports/qualification_{CODE}_{YYYYMMDD}.md`

## Rollout Policy

### Phase A: Fetch + Clean

Entry criteria:

- schema checks pass
- data freshness checks configured and operational.

Allowed usage:

- raw/clean ingestion validation only.

### Phase B: Features + Offline Validation

Entry criteria:

- Phase A complete
- feature completeness pass
- model comparison/selection runs successfully.

Allowed usage:

- offline evaluation and champion selection; no production forecast commitments.

### Phase C: Daily Ops Loop

Entry criteria:

- Phase B complete
- smoke forecast pass
- `experiments/daily_ops.py forecast/evaluate/health` operational
- monitoring alerts and runbooks in place.

Allowed usage:

- production daily forecast and end-of-day evaluation.

## Licensing and Data Constraints

Before promotion to Phase C:

- verify API/data licensing permits production usage
- document retention and redistribution restrictions
- confirm fallback strategy for each source (electricity/weather/gas/events).
