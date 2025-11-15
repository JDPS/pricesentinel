# PriceSentinel: Event-Aware Energy Price Forecasting

Multi-country energy price forecasting system with event awareness and data quality guards.

## Features

- **Multi-country support**: Extensible architecture for any country/market
- **Event-aware forecasting**: Incorporates holidays, DST transitions, and manual events
- **Data quality guards**: Runtime validation and anomaly handling
- **Comprehensive monitoring**: Track data quality, model performance, and alerts
- **Country abstraction**: Add new countries with minimal code changes

## Current Status

- ✅ **Phase 0 Complete**: Architecture and abstraction layer
- ✅ **Phase 1 Complete**: Project setup and Portugal implementation
- ⏳ **Phases 2–10**: In development

### Implemented Countries

- **Portugal (PT)**: Full implementation with ENTSO-E, Open-Meteo, and TTF data
- **Mock Country (XX)**: Synthetic data for testing

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/JDPS/pricesentinel.git
cd pricesentinel

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your API keys
```

### Configuration

1. Add your ENTSO-E API key to `.env`:
   ```
   ENTSOE_API_KEY=your_key_here
   ```

2. (Optional) Download TTF gas prices and place in `data/manual_imports/ttf_gas_prices.csv`

### Basic Usage

```bash
# Test with mock country (no API keys required)
python run_pipeline.py --country XX --fetch --start-date 2024-01-01 --end-date 2024-01-31

# Fetch data for Portugal
python run_pipeline.py --country PT --fetch --start-date 2024-01-01 --end-date 2024-01-31

# Get pipeline information
python run_pipeline.py --country PT --info

# Run full pipeline (when all phases are implemented)
python run_pipeline.py --country PT --all --start-date 2023-01-01 --end-date 2024-12-31
```

## Project Structure

```
pricesentinel/
├── config/                  # Configuration files
│   ├── countries/           # Country-specific configs
│   │   ├── PT.yaml          # Portugal configuration
│   │   └── XX.yaml          # Mock country configuration
│   ├── country_registry.py  # Country registry and factory
│   └── validation.py        # Pydantic validation schemas
│
├── core/                    # Core pipeline logic
│   ├── abstractions.py      # Abstract base classes
│   ├── data_manager.py      # Data directory management
│   ├── logging_config.py    # Logging setup
│   └── pipeline.py          # Main pipeline orchestration
│
├── data_fetchers/           # Data source adapters
│   ├── mock/                # Mock country (synthetic data)
│   ├── portugal/            # Portugal-specific fetchers
│   └── shared/              # Reusable fetchers
│
├── data/                    # Data storage (gitignored)
│   ├── PT/                  # Portugal data
│   └── XX/                  # Mock country data
│
├── tests/                   # Test suite
│   ├── test_abstractions.py
│   ├── test_registry.py
│   └── test_mock_country.py
│
├── run_pipeline.py          # Main CLI entry point
├── setup_country.py         # Country setup utility
└── requirements.txt         # Python dependencies
```

## Architecture

PriceSentinel uses an adapter pattern to remain country-agnostic while supporting country-specific data sources:

1. **Abstract Base Classes**: Define interfaces for all data fetchers
2. **Country Registry**: Maps country codes to specific implementations
3. **Factory Pattern**: Creates appropriate fetchers for each country
4. **Country Configuration**: YAML files define country-specific parameters

### Adding a New Country

See [docs/COUNTRY_EXTENSION_GUIDE.md](dev_ws/RevisedPhase0_and_Phase1.md#country-extension-guide) for detailed instructions.

Quick summary:

1. Create `config/countries/{CODE}.yaml`
2. Implement country-specific fetchers (if needed)
3. Register in `data_fetchers/__init__.py`
4. Test with `python run_pipeline.py --country {CODE} --info`

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_registry.py

# Run with verbose output
pytest -v
```

### Setting Up a New Country

```bash
# Create directory structure
python setup_country.py ES

# This creates:
# - data/ES/ directories
# - Prompts for next steps
```

## Documentation

- **Architecture Overview**: See `dev_ws/RevisedPhase0_and_Phase1.md`
- **Detailed Work Plan**: See `dev_ws/RevisedPhases2-10_CountryAbstraction.md`
- **Original Assessment**: See `dev_ws/DetailedWorkPlan.md`

## Roadmap

### Phase 0 ✅ (Complete)
- Core abstractions
- Country registry
- Mock country implementation

### Phase 1 ✅ (Complete)
- Project setup
- Portugal implementation
- CLI interface
- Tests

### Phase 2–3 (Next)
- Data verification and cleaning
- Timestamp normalization
- Quality checks

### Phase 4-5
- Feature engineering
- Runtime guards

### Phase 6-7
- Model training
- Inference engine

### Phase 8-10
- Reporting and monitoring
- Testing
- Deployment

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

### External APIs

- **ENTSO-E Transparency Platform** (for EU electricity data)
  - Register at: https://transparency.entsoe.eu/
  - Free API key required

- **Open-Meteo** (for weather data)
  - Free tier: https://open-meteo.com/
  - No API key required

- **TTF Gas Prices** (manual download for MVP)
  - Future: API integration planned

## Contributing

This is currently a development project. Contribution guidelines will be added in Phase 10.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues:
- Technical issues: [Your Email]
- API access: Check respective API provider documentation

---

**Note**: This project is in active development. Many features are planned but not yet implemented.
See the roadmap above for current status.
