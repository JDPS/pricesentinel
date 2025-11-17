![Coverage](coverage.svg)

# PriceSentinel: Event-Aware Energy Price Forecasting

Multi-country energy price forecasting system with event awareness and data quality guards.

## Features

- **Multi-country support**: Extensible architecture for any country/market
- **Event-aware forecasting**: Incorporates holidays, DST transitions, and manual events
- **Data quality guards**: Cleaning and basic validation for electricity, weather, and gas data
- **Feature engineering & training**: Baseline feature set and model training (scikit-learn) for mock country
- **Comprehensive monitoring**: Planned monitoring of data quality, model performance, and alerts
- **Country abstraction**: Add new countries with minimal code changes

## Current Status

- **Phase 0 Complete**: Architecture and abstraction layer
- **Phase 1 Complete**: Project setup and Portugal implementation
- **Initial Cleaning & Features**: Basic cleaning and feature engineering implemented for electricity, weather, and gas
- **Baseline Training**: End-to-end training pipeline implemented for mock country (XX) using scikit-learn
- **Phases 2–10**: Advanced guards, forecasting, monitoring, and deployment in development

### Implemented Countries

- **Portugal (PT)**: Full implementation with ENTSO-E, Open-Meteo, and TTF data
- **Mock Country (XX)**: Synthetic data for testing and fast training demos

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/JDPS/pricesentinel.git
cd pricesentinel

# Create and activate a virtual environment (example with venv)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install runtime dependencies
pip install .

# (Optional) Install development extras (tests, linting, docs)
pip install ".[dev,test,docs]"

# Configure environment
copy .env.example .env
# Edit .env with your API keys
```

### Configuration

1. Add your ENTSO-E API key to `.env`:
   ```bash
   ENTSOE_API_KEY=your_key_here
   ```

2. (Optional) Download TTF gas prices and place in:
   ```text
   data/manual_imports/ttf_gas_prices.csv
   ```

### Basic Usage

```bash
# Test with mock country (no API keys required) – fetch only
python run_pipeline.py --country XX --fetch --start-date 2024-01-01 --end-date 2024-01-07

# Run full pipeline for mock country (fetch → clean → features → train)
python run_pipeline.py --country XX --all --start-date 2024-01-01 --end-date 2024-01-07

# Fetch data for Portugal
python run_pipeline.py --country PT --fetch --start-date 2024-01-01 --end-date 2024-01-31

# Run full pipeline for Portugal (training assumes sufficient data and configuration)
python run_pipeline.py --country PT --all --start-date 2023-01-01 --end-date 2024-12-31

# Run individual stages in a single command (mock country example)
python run_pipeline.py --country XX --fetch --clean --features --train --start-date 2024-01-01 --end-date 2024-01-07

# Get pipeline information
python run_pipeline.py --country PT --info
```

## Project Structure

```text
pricesentinel/
  config/                  # Configuration files
    countries/             # Country-specific configs
      PT.yaml              # Portugal configuration
      XX.yaml              # Mock country configuration
    country_registry.py    # Country registry and factory
    validation.py          # Pydantic validation schemas

  core/                    # Core pipeline logic
    abstractions.py        # Abstract base classes
    data_manager.py        # Data directory management
    logging_config.py      # Logging setup
    cleaning.py            # Data cleaning and verification
    features.py            # Feature engineering
    pipeline.py            # Main pipeline orchestration

  data_fetchers/           # Data source adapters
    mock/                  # Mock country (synthetic data)
    portugal/              # Portugal-specific fetchers
    shared/                # Reusable fetchers (Open-Meteo, TTF)

  models/                  # Model trainers and saved artefacts

  data/                    # Data storage (gitignored)
    PT/                    # Portugal data
    XX/                    # Mock country data

  tests/                   # Test suite
    test_abstractions.py
    test_registry.py
    test_mock_country.py
    test_pipeline_mock_training.py

  run_pipeline.py          # Main CLI entry point
  setup_country.py         # Country setup utility
  tasks.py                 # Invoke-based automation
  pyproject.toml           # Project and dependency metadata
```

## Architecture

PriceSentinel uses an adapter pattern to remain country-agnostic while supporting country-specific data sources:

1. **Abstract Base Classes**: Define interfaces for all data fetchers
2. **Country Registry**: Maps country codes to specific implementations
3. **Factory Pattern**: Creates appropriate fetchers for each country
4. **Country Configuration**: YAML files define country-specific parameters

### Adding a New Country

See `dev_ws/RevisedPhase0_and_Phase1.md` (country extension guide) for detailed instructions.

Quick summary:

1. Create `config/countries/{CODE}.yaml`
2. Implement country-specific fetchers (if needed)
3. Register in `data_fetchers/__init__.py`
4. Test with:
   ```bash
   python run_pipeline.py --country {CODE} --info
   ```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage (HTML and terminal)
pytest --cov=. --cov-report=html --cov-report=term-missing
```

The coverage badge at the top of this README (`coverage.svg`) can be regenerated from the coverage tools.

### Setting Up a New Country

```bash
# Create directory structure
python setup_country.py ES

# This creates:
# - data/ES/ directories
# - Prompts for next steps
```

## Documentation

- **Architecture Overview**: `docs/ARCHITECTURE.md`
- **Implementation Phases 0–1**: `.dev_ws/RevisedPhase0_and_Phase1.md`
- **Extended Roadmap Phases 2–10**: `.dev_ws/RevisedPhases2-10_CountryAbstraction.md`
- **Consolidated Assessment & Refactoring Plan**: `.dev_ws/CONSOLIDATED_ASSESSMENT.md`

## Roadmap

### Phase 0 (Complete)
- Core abstractions
- Country registry
- Mock country implementation

### Phase 1 (Complete)
- Project setup
- Portugal implementation
- CLI interface
- Initial tests

### Phase 2–3 (In progress)
- Data verification and cleaning (basic cleaning implemented for electricity, weather, gas)
- Timestamp normalization
- Advanced quality checks and guards

### Phase 4–5
- Feature engineering (initial lags and calendar features implemented)
- Runtime guards and additional feature plugins

### Phase 6–7
- Model training (baseline scikit-learn regressor implemented for mock country)
- Inference engine and production-ready model registry

### Phase 8–10
- Reporting and monitoring
- Extended testing and QA
- Packaging and deployment

## Requirements

- Python 3.13+
- See `pyproject.toml` for dependencies and optional extras

### External APIs

- **ENTSO-E Transparency Platform** (for EU electricity data)
  - Register at: <https://transparency.entsoe.eu/>
  - Free API key required

- **Open-Meteo** (for weather data)
  - Free tier: <https://open-meteo.com/>
  - No API key required

- **TTF Gas Prices** (manual download for MVP)
  - Future: API integration planned

## Contributing

This is currently a development project. Contribution guidelines will be added in Phase 10.

## License

This project is licensed under the Apache Licence 2.0 – see the `LICENSE` file for details.

## Contact

For questions or issues:
- Technical issues: joaosoarex@gmail.com
- API access: Check respective API provider documentation

---

**Note**: This project is in active development. Many features are planned but not yet fully implemented.
See the roadmap above for current status.
