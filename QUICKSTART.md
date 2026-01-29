# Quick Start Guide

This guide will help you test the PriceSentinel implementation immediately.

## Phase 0 & 1: What's Been Implemented

✅ **Complete**:

- Core abstraction layer
- Country registry system
- Mock country with synthetic data
- Portugal data fetchers (ENTSO-E, Open-Meteo, TTF gas)
- CLI interface
- Data management system
- **Data cleaning and verification**
- **Feature engineering**
- **Model training (Baseline)**
- **Forecasting (Baseline)**
- Basic tests & CI/CD

⏳ **Not Yet Implemented** (Future Phases):

- Advanced forecasting models
- Production monitoring
- Deployment infrastructure

## Testing Without API Keys (Mock Country)

The easiest way to test is using the mock country (XX) which generates synthetic data:

```bash
# 1. Ensure you're in the project directory
cd D:\Coding\01_Projects\pricesentinel

# 2. Test with mock country (no API keys needed)
python run_pipeline.py --country XX --fetch --start-date 2024-01-01 --end-date 2024-01-07

# 3. Check the generated data
python run_pipeline.py --country XX --info
```

**Expected Output**:

- Synthetic electricity prices (hourly, 7 days = 168 records)
- Synthetic weather data
- Synthetic gas prices (daily, 7 records)
- Mock holidays and events

**Data Location**: `data/XX/raw/`

## Testing With Real Data (Portugal)

### Prerequisites

1. **Get ENTSO-E API Key**:
   - Register at: <https://transparency.entsoe.eu/>
   - Go to "Account Settings" → "Web API Security Token"
   - Copy your API key

2. **Configure Environment**:

   ```bash
   # Copy environment template
   copy .env.example .env

   # Edit .env and add your key:
   # ENTSOE_API_KEY=your_actual_key_here
   ```

3. **(Optional) Add TTF Gas Data**:
   - Create directory: `data/manual_imports/`
   - Add CSV file: `ttf_gas_prices.csv` with columns: `date, ttf_price`

### Fetch Real Data

```bash
# Fetch Portugal data for one week
python run_pipeline.py --country PT --fetch --start-date 2024-01-01 --end-date 2024-01-07

# View what was fetched
python run_pipeline.py --country PT --info
```

**Expected Output**:

- Electricity prices from ENTSO-E
- Electricity load from ENTSO-E
- Weather data from Open-Meteo
- Portuguese holidays
- (Empty or manual) TTF gas prices

**Data Location**: `data/PT/raw/`

## Running Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_mock_country.py -v
```

**Expected Results**: All tests should pass ✅

## Verify Installation

### 1. Check Directory Structure

```bash
python run_pipeline.py --country PT --info
```

Should show:

- Country: Portugal
- Timezone: Europe/Lisbon
- Data directory structure

### 2. List Available Countries

```bash
python run_pipeline.py --help
```

The help text should display without errors.

### 3. Test Registry

Create a simple test script ``:

```python
from data_fetchers import auto_register_countries
from config.country_registry import CountryRegistry

auto_register_countries()
print("Registered countries:", CountryRegistry.list_countries())
```

Run it:

```bash
python test_registry.py
```

Expected output: `Registered countries: ['PT', 'XX']`

## Common Issues

### Issue: "ENTSOE_API_KEY not found"

**Solution**: Make sure `.env` file exists in the project root with your API key.

### Issue: "Country 'PT' not registered"

**Solution**: The auto-registration might have failed. Check that `data_fetchers/portugal/__init__.py` exists.

### Issue: "No module named 'core'"

**Solution**: Make sure you're running from the project root directory.

### Issue: "TTF gas CSV not found"

**Solution**: This is expected for MVP. You can:

- Ignore it (a pipeline continues without gas data)
- Create a template: `data/manual_imports/ttf_gas_prices.csv`

### Issue: Import errors

**Solution**: Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## What to Expect

### Current Functionality ✅

- Data fetching from APIs
- Synthetic data generation (mock)
- **Data cleaning and verification**
- **Feature engineering**
- **Model training and forecasting**
- CLI interface fully functional

### Not Yet Working ⏳

- Advanced Deep Learning models
- Real-time monitoring dashboard

When you run `--all`, it will:

1. ✅ Fetch data successfully
2. ✅ Clean and verify data
3. ✅ Generate features
4. ✅ Train baseline model
5. ✅ Generate forecast

## Next Steps

After verifying Phase 0 & 1 work:

1. **Phase 2-3**: Implement data cleaning and verification
2. **Phase 4–5**: Add feature engineering and runtime guards
3. **Phase 6-7**: Build model training and inference
4. **Phase 8-10**: Add monitoring and deployment

## Success Indicators

You'll know everything is working if:

✅ Mock country generates data without errors
✅ Portugal fetches real data from ENTSO-E (with the API key)
✅ Data is saved in correct directories with proper naming
✅ Tests pass
✅ `--info` command shows correct directory structure
✅ Logs show clear progress and no critical errors

## Need Help?

Check:

- `README.md` - Full project documentation
- `docs/ARCHITECTURE.md` - Architecture details
- `dev_ws/RevisedPhase0_and_Phase1.md` - Implementation details
- Test files in `tests/` - Example usage

---

**Congratulations!** You've completed Phase 0 & 1 of PriceSentinel. The architecture is in place and ready for the remaining phases.
