# Feature Toggles (Country Config)

This document explains the feature-related flags in the `features` block of
each country YAML configuration and how they affect the pipeline.

It focuses on configuration; see `docs/TRAINING.md` for how these features
are used during feature engineering and training.

---

## 1. `features` Block Overview

Every country config in `config/countries/{CODE}.yaml` includes a `features`
section, validated by `config/validation.py::FeaturesConfig`.

Example (Portugal, PT):

```yaml
features:
  use_cross_border_flows: false
  neighbors: []
  custom_feature_plugins: []
  use_weather_features: true
  use_gas_features: true
  use_event_features: true
```

The first three keys are reserved for future cross-border and plugin-based
feature extensions. The last three booleans directly control which core
feature blocks are active.

---

## 2. Weather Features

**Flag:** `use_weather_features: bool`

- **On (`true`)**:
  - `FeatureEngineer` loads `weather_clean` for the requested date range and
    aggregates per `timestamp`:
    - `temperature_c` – mean across configured locations.
    - `wind_speed_ms` – mean across locations.
    - `solar_radiation_wm2` – mean across locations.
    - `precipitation_mm` – sum across locations.
  - These columns are merged into the feature set.
- **Off (`false`)**:
  - The weather file (if present) is ignored for feature building.
  - The columns above are not created.

**When to disable:**

- No reliable weather data yet.
- You want a purely electricity+calendar baseline.

---

## 3. Gas Features

**Flag:** `use_gas_features: bool`

- **On (`true`)**:
  - `FeatureEngineer` reads `gas_prices_clean`, converts `timestamp` to
    daily dates, and constructs a daily series:
    - `gas_price_eur_mwh` – latest price per day.
  - The daily gas price is joined to hourly rows via the date and used as an
    additional numeric feature.
- **Off (`false`)**:
  - Gas data is ignored.
  - No `gas_price_eur_mwh` column is added.

**When to disable:**

- No TTF (or equivalent) gas price series available for the country.
- You want to compare performance with and without gas sensitivity.

---

## 4. Event Features (Holidays & Manual Events)

**Flag:** `use_event_features: bool`

- `FeatureEngineer` always creates two numeric columns:
  - `is_holiday` – 0 or 1.
  - `is_event` – 0 or 1.

Their behaviour depends on the flag:

- **On (`true`)**:
  - `holidays_clean` (if present) is used to mark dates whose timestamps are
    holidays: `is_holiday = 1` for those days.
  - `manual_events_clean` (if present) is used to mark any timestamp that
    falls in a `[date_start, date_end]` interval: `is_event = 1` for those
    rows.
- **Off (`false`)**:
  - The columns remain in the feature set but stay at `0` for all rows.
  - This preserves the schema while removing any influence from event data.

**When to disable:**

- Manual events are not curated yet and may be noisy.
- You want to isolate the effect of calendar-only or price-only features.

---

## 5. Cross-Border & Plugins (Placeholders)

The following keys are present for future extensibility but are not yet
actively used in the core pipeline:

- `use_cross_border_flows: bool`
- `neighbors: list[str]`
- `custom_feature_plugins: list[str]`

These are intended for:

- Injecting cross-border flow / import-export features based on neighbouring
  countries and interconnector data.
- Dynamically loading custom feature modules (e.g. from a plugin registry).

The current implementation safely ignores them but keeps them part of the
validated configuration to make future rollout smoother.

---

## 6. Recommended Defaults

For most countries at the current stage of the project:

- **PT (Portugal):**
  - `use_weather_features: true` (Open-Meteo available)
  - `use_gas_features: true` (TTF gas prices available)
  - `use_event_features: true` (holidays + manual events planned)
- **XX (Mock):**
  - All three `true`, so tests and demos exercise the full feature pipeline.

For new countries without certain data sources, start with:

```yaml
features:
  use_cross_border_flows: false
  neighbors: []
  custom_feature_plugins: []
  use_weather_features: false    # until weather fetchers are ready
  use_gas_features: false        # until gas data is wired
  use_event_features: false      # until holiday/manual events are curated
```

As data sources become available, flip the relevant flags to `true` and
rebuild features to include the new signals.
