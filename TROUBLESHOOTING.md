# Troubleshooting Guide

## Issue 1: Tests Pass in PowerShell but Fail in IDE

### Problem
When running `pytest -v` from PowerShell, all tests pass. But when running from PyCharm or other IDEs, tests fail with:
```
FileNotFoundError: Configuration file not found: config\countries\XX.yaml
```

### Root Cause
IDEs often run tests from the `tests/` directory as the working directory, while PowerShell runs from the project root. This caused relative paths to fail.

### Solution ✅ FIXED
Updated `config/country_registry.py` to use absolute paths based on the project root:

```python
# Before (failed in IDE)
config_path = Path(config_dir) / f'{country_code}.yaml'

# After (works everywhere)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_path = project_root / config_dir / f'{country_code}.yaml'
```

**What this does**: Finds the project root dynamically by going up from the `country_registry.py` file location, ensuring the path works regardless of where tests are run from.

### Verification
Run tests from your IDE - they should now pass:
```
✅ 24 passed in X.XXs
```

---

## Issue 2: ENTSO-E API Returns 400 Bad Request

### Problem
```
ERROR - Failed to fetch electricity prices: 400 Client Error: Bad Request
```

### Root Cause (FIXED!)
**Configuration error**: The Portugal configuration was using country code `PT` instead of the ENTSO-E EIC code `10YPT-REN------W`.

ENTSO-E requires **EIC (Energy Identification Code)** codes for domain identification, not ISO country codes.

### Solution ✅ FIXED

The Portugal configuration has been corrected:

```yaml
# BEFORE (wrong - caused 400 errors)
electricity:
  api_type: entsoe
  entsoe_domain: PT

# AFTER (correct - now working!)
electricity:
  api_type: entsoe
  entsoe_domain: 10YPT-REN------W  # Portugal EIC code
```

### Testing the Fix

Now you can fetch real Portugal data with recent dates:

```bash
python run_pipeline.py --country PT --fetch --start-date 2025-01-09 --end-date 2025-01-10
```

**Expected results:**
- ✅ 72 electricity price records
- ✅ 48 electricity load records
- ✅ 96 weather records
- ✅ No API errors

### Important Notes

**For older dates**: ENTSO-E still has historical data limitations. Day-ahead prices are typically only available for recent periods (last 1-3 months). If you need older data, use the mock country for testing:

```bash
python run_pipeline.py --country XX --fetch --start-date 2024-01-01 --end-date 2024-12-31
```

### Adding New Countries with ENTSO-E

When adding a new country, you **must** use the correct EIC code. Refer to `dev_ws/ENTSOE_EIC_CODES.md` for a complete list.

**Common EIC codes:**
- Portugal: `10YPT-REN------W` ✅
- Spain: `10YES-REE------0`
- France: `10YFR-RTE------C`
- Germany: `10Y1001A1001A83F`
- Italy: Multiple zones (see reference document)

**Testing EIC codes:**
```bash
# Replace YOUR_KEY and EIC_CODE with actual values
curl "https://web-api.tp.entsoe.eu/api?securityToken=YOUR_KEY&documentType=A44&in_Domain=10YPT-REN------W&out_Domain=10YPT-REN------W&periodStart=202501090000&periodEnd=202501100000"
```

### Enhanced Error Messages

The code provides helpful guidance when API calls fail:

```python
# For 400 errors
logger.warning(
    "ENTSO-E API returned 400 Bad Request. Common causes:\n"
    "  1. Historical data too old (try more recent dates)\n"
    "  2. Data not available for this domain/period\n"
    "  3. API rate limiting\n"
    f"  Requested period: {start_date} to {end_date}\n"
    "  Suggestion: Try dates from the last 30 days"
)

# For authentication errors
logger.error(
    "Authentication error. Please check:\n"
    "  1. ENTSOE_API_KEY is set correctly in .env\n"
    "  2. API key is valid and not expired"
)
```

### Alternative: Use Mock Country for Testing

The mock country generates synthetic data without API calls:

```bash
python run_pipeline.py --country XX --fetch --start-date 2024-01-01 --end-date 2024-01-07
```

**Expected output**:
- ✅ 168 electricity price records
- ✅ 168 load records
- ✅ 336 weather records (2 locations × 168 hours)
- ✅ Holidays and events

---

## Issue 3: Gas Price CSV Error

### Problem
```
ERROR - Failed to load TTF gas prices from CSV: "['price_eur_mwh'] not in index"
```

### Root Cause
After filtering the CSV for the requested date range, the DataFrame was empty (no data in that range), which caused column selection to fail.

### Solution ✅ FIXED
Updated `data_fetchers/shared/ttf_gas.py` to handle empty results gracefully:

```python
# Filter data
df_filtered = df[mask].reset_index(drop=True)

# Handle empty results
if len(df_filtered) == 0:
    return pd.DataFrame(columns=['timestamp', 'price_eur_mwh', 'hub_name', 'quality_flag'])

return df_filtered[result_columns]
```

### Creating Sample Gas Data

If you want to test with gas prices, create `data/manual_imports/ttf_gas_prices.csv`:

```csv
date,ttf_price
2024-01-01,30.5
2024-01-02,31.2
2024-01-03,29.8
2024-01-04,30.1
2024-01-05,32.3
2024-01-06,31.7
2024-01-07,30.9
2024-01-08,29.5
2024-01-09,31.0
```

---

## Summary of All Fixes

| Issue | Status | Files Modified |
|-------|--------|----------------|
| IDE path resolution | ✅ FIXED | `config/country_registry.py` |
| Empty gas CSV handling | ✅ FIXED | `data_fetchers/shared/ttf_gas.py` |
| ENTSO-E API 400 errors | ✅ FIXED | `config/countries/PT.yaml` (EIC code) |
| ENTSO-E error messages | ✅ IMPROVED | `data_fetchers/portugal/electricity.py` |
| Mock data date range | ✅ FIXED | `data_fetchers/mock/electricity.py` |
| Timezone comparisons | ✅ FIXED | `data_fetchers/mock/events.py` |
| Unicode encoding | ✅ FIXED | Multiple files (logger cleanup) |

---

## Testing the Fixes

### 1. Test in IDE
Open PyCharm and run tests:
```
Right-click on tests/ → Run 'pytest in tests'
```

Expected: ✅ 24 passed

### 2. Test with Mock Country
```bash
python run_pipeline.py --country XX --fetch --start-date 2024-01-01 --end-date 2024-01-07
python run_pipeline.py --country XX --info
```

Expected: All data fetched successfully

### 3. Test with Portugal (Recent Dates)
```bash
python run_pipeline.py --country PT --fetch --start-date 2025-01-09 --end-date 2025-01-10
```

Expected:
- ✅ 72 electricity price records
- ✅ 48 electricity load records
- ✅ 96 weather records
- ✅ No API errors

---

## Best Practices for Development

### For Testing Without API Access
Use the mock country:
```bash
python run_pipeline.py --country XX --all --start-date 2024-01-01 --end-date 2024-12-31
```

### For Real Data
1. **Always use recent dates** (last 7-30 days)
2. **Check ENTSO-E platform** for data availability
3. **Respect rate limits** (wait between requests)
4. **Have fallback data** for development (manual CSV files)

### For Production
1. **Schedule daily fetches** for yesterday's data
2. **Implement retry logic** for temporary failures
3. **Cache results** to reduce API calls
4. **Monitor API status** and adjust date ranges dynamically

---

## Quick Reference

### ✅ Working Commands

```bash
# Run tests (works in PowerShell and IDE)
pytest -v

# Fetch mock data (always works)
python run_pipeline.py --country XX --fetch --start-date 2024-01-01 --end-date 2024-01-07

# Fetch Portugal recent data (should work)
python run_pipeline.py --country PT --fetch --start-date 2025-01-09 --end-date 2025-01-09

# Get pipeline info
python run_pipeline.py --country PT --info
python run_pipeline.py --country XX --info
```

### ❌ Commands That Won't Work

```bash
# Old historical data (ENTSO-E limits)
python run_pipeline.py --country PT --fetch --start-date 2023-01-01 --end-date 2023-12-31

# Future dates (no data available)
python run_pipeline.py --country PT --fetch --start-date 2026-01-01 --end-date 2026-01-07
```

---

## Getting Help

If issues persist:

1. **Check logs**: `logs/pricesentinel_YYYYMMDD.log`
2. **Run verification script**: `python verify_fixes.py`
3. **Test API directly**: Use curl to test ENTSO-E
4. **Check ENTSO-E status**: https://transparency.entsoe.eu/

## Related Documentation

- `QUICKSTART.md` - Getting started guide
- `dev_ws/ENTSOE_API_NOTES.md` - Detailed ENTSO-E API notes
- `dev_ws/FIXES_APPLIED.md` - Technical details of fixes
- `README.md` - Full project documentation
