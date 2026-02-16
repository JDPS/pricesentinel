# Operations Runbooks

This runbook describes actions for common operational alerts produced by:

- `experiments/daily_ops.py evaluate`
- `experiments/daily_ops.py health`
- `core.monitoring.generate_health_summary`

## Artifacts

- Daily scorecard:
  - `data/{CODE}/processed/scorecards/daily_scorecard.csv`
  - `data/{CODE}/processed/scorecards/daily_scorecard.jsonl`
- Health summary:
  - `data/{CODE}/processed/monitoring/health_summary_{YYYYMMDD}.json`
  - `data/{CODE}/processed/monitoring/health_summary_{YYYYMMDD}.md`
- Alert signal file:
  - `data/{CODE}/processed/monitoring/alerts_{YYYYMMDD}.log`

## Alert: `forecast_missing`

Symptoms:

- daily evaluate record is deferred with `reason=forecast_missing`
- health summary includes critical/latest-eval condition.

Actions:

1. Run forecast manually:
   - `uv run python experiments/daily_ops.py forecast --country {CODE}`
2. Verify forecast artifact exists under `processed/forecasts/`.
3. Re-run evaluation:
   - `uv run python experiments/daily_ops.py evaluate --country {CODE}`
4. If forecast generation failed, verify champion/model availability:
   - `models/{CODE}/champion.json`
   - `models/{CODE}/{model}/{run_id}/model.pkl`

## Alert: `actuals_incomplete`

Symptoms:

- daily evaluate deferred with `reason=actuals_incomplete`.

Actions:

1. Wait for source completion window.
2. Validate latest electricity raw/clean files for target date.
3. Re-run evaluate command after data arrival.

## Alert: stale or missing raw data (`freshness`)

Symptoms:

- health summary freshness status `warn` or `critical`.

Actions:

1. Trigger fetch:
   - `uv run python run_pipeline.py --country {CODE} --fetch --start-date ... --end-date ...`
2. Check upstream source/API status and credentials.
3. Confirm raw files update under `data/{CODE}/raw/{source}/`.
4. Regenerate health summary.

## Alert: MAE spike / drift

Symptoms:

- health summary status warn/critical due `mae_spike` or `drift`.

Actions:

1. Inspect recent scorecard metrics (7d vs 30d).
2. Validate feature completeness and data freshness.
3. Re-run selection/champion update:
   - `uv run python experiments/select_champion.py --country {CODE} --start ... --end ...`
4. If needed retrain directly:
   - `uv run python experiments/run_training.py --country {CODE} --train-start ... --train-end ...`

## Alert: uncertainty caution (`uncertainty_coverage` / `pinball_loss`)

Symptoms:

- health summary has warn conditions related to quantile coverage or pinball loss.

Actions:

1. Inspect interval outputs in forecast CSV:
   - `forecast_p10_eur_mwh`, `forecast_p50_eur_mwh`, `forecast_p90_eur_mwh`.
2. Check daily scorecard uncertainty columns:
   - `quantile_coverage_10_90`, `pinball_loss_avg`, `interval_width_avg`.
3. Re-run champion selection and retraining for a fresher window.
4. If intervals are consistently too wide/narrow, tune model/hyperparameters and re-evaluate.

## Alert: coverage degradation

Symptoms:

- low successful evaluation coverage in 7d/30d windows.

Actions:

1. Count deferred rows by `reason` in scorecard.
2. Address dominant cause (forecast missing, no overlap, actuals incomplete).
3. Backfill missing days with forecast/evaluate reruns where possible.

## Escalation

Escalate to engineering when:

- two consecutive days remain `critical`
- critical freshness persists >24h
- MAE critical threshold persists >3 evaluated days.
