# Training & Feature Pipeline

This document describes how the feature engineering and training pipeline
works in PriceSentinel, and how to control it from configuration and the CLI.

It complements `docs/ARCHITECTURE.md`, which focuses on the overall code
architecture.

---

## 1. Data Stages Recap

For a given country and date range:

1. **Fetch** (`Pipeline.fetch_data`)
   - Writes raw CSVs under `data/{country}/raw/...`.
2. **Clean** (`DataCleaner.clean_*`)
   - Filters, normalises timestamps, de-duplicates.
   - Writes `*_clean` CSVs under `data/{country}/processed/`.
3. **Features** (`FeatureEngineer.build_electricity_features`)
   - Builds a supervised learning dataset for hourly electricity prices.
   - Writes `electricity_features` CSV under `processed/`.
4. **Train** (`FeatureEngineer.train_with_trainer`)
   - Loads features, splits train/validation, trains a model, and saves
     artefacts under `models/`.

---

## 2. Features Produced

`FeatureEngineer.build_electricity_features` currently creates:

- **Target**
  - `target_price`: next-hour `price_eur_mwh` (shift -1).
- **Price history**
  - `price_lag_1`, `price_lag_2`, `price_lag_24`.
- **Calendar**
  - `hour` (0–23), `day_of_week` (0–6).
- **Load (optional)**
  - `load_mw` if `electricity_load_clean` exists.
- **Weather (optional, feature-gated)**
  - From `weather_clean`, aggregated per `timestamp`:
    - `temperature_c` (mean across locations),
    - `wind_speed_ms` (mean),
    - `solar_radiation_wm2` (mean),
    - `precipitation_mm` (sum).
- **Gas (optional, feature-gated)**
  - From `gas_prices_clean`, aggregated per day:
    - `gas_price_eur_mwh` (latest value per date).
- **Events (feature-gated)**
  - Always-present numeric flags:
    - `is_holiday`:
      - 1 if the date appears in `holidays_clean.timestamp`, else 0.
    - `is_event`:
      - 1 for timestamps within any `[date_start, date_end]` interval
        listed in `manual_events_clean`, else 0.

Rows with missing target or critical lags are dropped. If the resulting
DataFrame becomes empty, features are not written and downstream training
must handle this.

---

## 3. Controlling Features via Config

Country YAML (`config/countries/{CODE}.yaml`) includes a `features` block:

```yaml
features:
  use_cross_border_flows: false
  neighbors: []
  custom_feature_plugins: []
  use_weather_features: true
  use_gas_features: true
  use_event_features: true
```

`FeatureEngineer` uses these booleans to decide which blocks to apply:

- `use_weather_features`:
  - When `true`, aggregates and joins weather-derived columns.
  - When `false`, weather files are ignored; the columns above are omitted.
- `use_gas_features`:
  - When `true`, adds `gas_price_eur_mwh`.
  - When `false`, gas data is ignored and the column is omitted.
- `use_event_features`:
  - When `true`, `is_holiday` and `is_event` incorporate holiday/events data.
  - When `false`, the flags remain present but stay at 0.

This allows you to keep the core feature schema consistent while toggling
costly or unavailable inputs per country.

---

## 4. Trainers and Models

Trainers live in `models/`:

- `BaseTrainer` (`models/base.py`):
  - Abstract interface: `train(...) -> metrics`, `save(...)`.
- `SklearnRegressorTrainer` (`models/sklearn_trainer.py`):
  - Baseline `RandomForestRegressor` for hourly price forecasting.
  - Computes basic metrics:
    - `train_mae`, `train_rmse`, and optional validation metrics.

The factory `models.get_trainer(country_code, model_name, models_root)`
returns a `SklearnRegressorTrainer` instance for now.

### Fast mode

`SklearnRegressorTrainer` treats model names ending with `"_fast"` as a
lighter configuration:

- `n_estimators` reduced (e.g. 50 instead of 100),
- `max_depth` reduced (e.g. 5 instead of 10),
- same data and metrics, but quicker runs.

---

## 5. CLI Usage for Training

The main CLI is `run_pipeline.py`. Relevant flags:

- `--features` – run feature engineering only.
- `--train` – run training only (uses existing features).
- `--all` – run full pipeline (fetch → clean → features → train → forecast stub).
- `--model-name` – trainer/model identifier (default: `baseline`).
- `--fast-train` – append `"_fast"` to `--model-name` (if not already present)
  and use the fast trainer configuration.

Example flows:

```bash
# Standard full pipeline for mock country
python run_pipeline.py --country XX --all \
  --start-date 2024-01-01 --end-date 2024-01-07

# Use a custom model name
python run_pipeline.py --country XX --all \
  --model-name my_experiment \
  --start-date 2024-01-01 --end-date 2024-01-07

# Fast demo training (smaller RF, shorter runs)
python run_pipeline.py --country XX --all \
  --fast-train \
  --start-date 2024-01-01 --end-date 2024-01-07
```

You can also combine `--features` and `--train` manually if you only want
to retrain on existing cleaned data without refetching.

---

## 6. Artefact Locations

- **Cleaned data**:
  - `data/{CODE}/processed/{CODE}_electricity_prices_clean_*.csv`
  - `data/{CODE}/processed/{CODE}_electricity_load_clean_*.csv`
  - `data/{CODE}/processed/{CODE}_weather_clean_*.csv`
  - `data/{CODE}/processed/{CODE}_gas_prices_clean_*.csv`
  - `data/{CODE}/processed/{CODE}_holidays_clean_*.csv`
  - `data/{CODE}/processed/{CODE}_manual_events_clean_*.csv`
- **Features**:
  - `data/{CODE}/processed/{CODE}_electricity_features_{start}_{end}.csv`
- **Models + metrics**:
  - `models/{CODE}/{model_name}/{run_id}/model.pkl`
  - `models/{CODE}/{model_name}/{run_id}/metrics.json`

These paths are managed by `CountryDataManager` and `SklearnRegressorTrainer`.

---

## 7. Forecast Uncertainty and Calibration

Forecast generation (`Pipeline.generate_forecast`) now writes a stable interval
schema alongside point forecasts:

- `forecast_p10_eur_mwh`
- `forecast_p50_eur_mwh`
- `forecast_p90_eur_mwh`
- `forecast_interval_width_eur_mwh`
- `uncertainty_source`

Calibration metadata is also stored in each forecast row:

- `interval_calibration_factor`
- `interval_calibration_samples`
- `interval_calibration_window_days`
- `interval_observed_coverage_10_90`
- `interval_nominal_coverage_10_90`

### How interval calibration works

1. A base residual scale (`sigma`) is inferred from model metrics metadata:
   - prefer `val_rmse`, then `train_rmse`/`rmse`, then MAE-derived fallback.
2. Recent empirical interval coverage is read from:
   - `data/{CODE}/processed/scorecards/daily_scorecard.csv`
   - uses `status == "ok"` rows with `quantile_coverage_10_90`
   - rolling window: last 30 days, minimum 7 samples.
3. A calibration factor rescales `sigma` so observed 10-90 coverage trends
   toward nominal 0.80 coverage.
4. Safety constraints apply:
   - calibration factor is clipped to `[0.5, 3.0]`
   - factor trust is ramped as sample size grows.

If insufficient scorecard history exists, intervals fall back to
`uncertainty_source = residual_proxy_normal`.
