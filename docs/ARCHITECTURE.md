# PriceSentinel Architecture

## Overview

PriceSentinel is designed with a country-agnostic core and country-specific adapters, enabling seamless addition of new countries without modifying core pipeline code.

## Design Principles

1. **Country Abstraction**: Core pipeline has no country-specific logic
2. **Adapter Pattern**: Data sources implement standard interfaces
3. **Configuration-Driven**: Country differences handled via YAML configs
4. **Fail-Fast Validation**: Invalid configs rejected at startup
5. **Separation of Concerns**: Clear boundaries between modules

## Component Overview

### Core Abstractions (`core/abstractions.py`)

Defines interfaces that all country-specific fetchers must implement:

- `ElectricityDataFetcher`: Day-ahead prices, load, generation
- `WeatherDataFetcher`: Temperature, wind, solar, precipitation
- `GasDataFetcher`: Gas hub prices
- `EventDataProvider`: Holidays and manual events

All fetchers return DataFrames with standardized schemas.

### Country Registry (`config/country_registry.py`)

Central registry mapping country codes to adapter classes:

```python
CountryRegistry.register('PT', {
    'electricity': PortugalElectricityFetcher,
    'weather': OpenMeteoWeatherFetcher,
    'gas': TTFGasFetcher,
    'events': PortugalEventProvider
})
```

**Key Classes**:
- `CountryRegistry`: Stores country-to-adapter mappings
- `FetcherFactory`: Creates fetcher instances for a country
- `CountryConfig`: Parsed country configuration
- `ConfigLoader`: Loads and validates country configs

### Data Management (`core/data_manager.py`)

Handles country-specific data directories and file naming:

```
data/
├── PT/
│   ├── raw/electricity/
│   ├── raw/weather/
│   ├── raw/gas/
│   ├── processed/
│   ├── events/
│   └── metadata/
└── ES/  (future)
```

**File Naming**: `{country}_{source}_{start}_{end}.{ext}`
Example: `PT_electricity_20240101_20240131.csv`

### Pipeline Orchestration (`core/pipeline.py`)

Country-agnostic pipeline that coordinates all stages:

1. **Data Fetching**: Calls appropriate fetchers
2. **Data Cleaning**: Verification and normalization (Phase 3)
3. **Feature Engineering**: Generate model features (Phase 4)
4. **Model Training**: Train forecasting models (Phase 6)
5. **Forecast Generation**: Produce predictions (Phase 7)

The pipeline never contains country-specific logic - it only calls registered adapters.

### Data Fetchers

#### Country-Specific Fetchers

**Portugal** (`data_fetchers/portugal/`):
- `PortugalElectricityFetcher`: ENTSO-E API integration
- `PortugalEventProvider`: Portuguese holidays

#### Reusable Fetchers

**Shared** (`data_fetchers/shared/`):
- `OpenMeteoWeatherFetcher`: Works for any coordinates
- `TTFGasFetcher`: Works for EU countries using TTF

#### Mock Fetchers

**Mock Country** (`data_fetchers/mock/`):
- Synthetic data generators for testing
- No API access required
- Reproducible outputs

## Data Flow

```
User Command (run_pipeline.py)
    ↓
CLI Argument Parsing
    ↓
Auto-register Countries
    ↓
CountryRegistry → ConfigLoader
    ↓
FetcherFactory creates adapters
    ↓
Pipeline orchestrates stages
    ↓
Fetchers retrieve data
    ↓
CountryDataManager saves outputs
```

## Configuration System

### Country Configuration Structure

```yaml
country_code: PT
country_name: Portugal
timezone: Europe/Lisbon

electricity:
  api_type: entsoe
  entsoe_domain: PT

weather:
  api_type: open_meteo
  coordinates:
    - name: Lisbon
      lat: 38.7223
      lon: -9.1393

gas:
  api_type: ttf
  hub_name: TTF

events:
  holiday_library: portugal
  manual_events_path: data/PT/events/manual_events.csv

features:
  use_cross_border_flows: false
```

### Validation

Pydantic models (`config/validation.py`) validate configs:
- Required fields present
- Valid coordinate ranges
- Valid timezone strings
- Enum values for API types

## Extensibility

### Adding a New Country

**Minimal changes required**:

1. **Create config** (`config/countries/ES.yaml`)
2. **Reuse or implement fetchers**:
   - Electricity: Reuse if ENTSO-E, else implement new
   - Weather: Reuse Open-Meteo
   - Gas: Reuse if using TTF/PEG
   - Events: Implement country-specific
3. **Register** in `data_fetchers/__init__.py`
4. **Test** with CLI

**No changes needed to**:
- Core pipeline code
- Data management
- CLI interface
- Testing infrastructure

### Adding a New Data Source

To add a new data source (e.g., solar forecasts):

1. Create abstract interface in `core/abstractions.py`
2. Update country configs to include new source
3. Implement country-specific fetchers
4. Update pipeline to call new fetchers

## Testing Strategy

### Unit Tests
- Abstract base classes enforce implementation
- Registry properly stores/retrieves adapters
- Individual fetchers return correct schemas

### Integration Tests
- Mock country validates full pipeline
- Country data isolation verified
- Multi-country scenarios tested

### Test Fixtures
- `clean_registry`: Clears registry before/after tests
- `mock_country_setup`: Registers mock country

## Error Handling

### Graceful Degradation
- Missing data sources logged but don't crash
- Empty DataFrames returned with correct schema
- Pipeline continues with available data

### Validation
- Configs validated at startup (fail-fast)
- Missing API keys raise clear errors
- Invalid dates rejected early

## Future Enhancements

### Phase 2-3: Data Quality
- Country-specific data quality limits
- Timezone-aware normalization
- Missing value strategies

### Phase 4-5: Features & Guards
- Country-specific feature plugins
- Runtime data quality guards
- Alert system

### Phase 6-7: Modeling
- Country-specific model registry
- Multi-horizon forecasting
- Confidence intervals

### Phase 8-10: Production
- Monitoring dashboards
- Automated retraining
- Deployment packaging

## Performance Considerations

### Current (MVP)
- Sequential data fetching
- CSV storage
- Single-threaded pipeline

### Future Optimizations
- Parallel fetching for multiple countries
- Parquet/database storage
- Distributed processing
- Caching layer

## Security

### API Keys
- Stored in `.env` (gitignored)
- Loaded via `python-dotenv`
- Never hardcoded

### Data Privacy
- Raw data gitignored
- No PII collected
- Logs exclude sensitive info

## Deployment

### Current
- Run locally via CLI
- Manual data fetching

### Planned (Phase 10)
- Docker containerization
- Scheduled runs (cron/scheduler)
- Cloud deployment options

---

This architecture ensures PriceSentinel can scale from Portugal to dozens of countries without refactoring core logic.
