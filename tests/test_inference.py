import pandas as pd
import pytest

from core.guards import RuntimeGuard
from models.run_forecast import main


def test_runtime_guard_clamping():
    """Test that RuntimeGuard clamps values correctly."""
    limits = {
        "price_max": 100.0,
        "price_min": -10.0,
        "load_max": 5000.0,
        "load_min": 1000.0,
    }
    guard = RuntimeGuard(limits)

    df = pd.DataFrame(
        {
            "price_eur_mwh": [150.0, -50.0, 50.0],
            "load_mw": [6000.0, 500.0, 3000.0],
            "other": [1, 2, 3],
        }
    )

    clamped = guard.validate_and_clamp(df)

    # Check Price clamping
    assert clamped["price_eur_mwh"].max() == 100.0
    assert clamped["price_eur_mwh"].min() == -10.0
    assert clamped["price_eur_mwh"].iloc[2] == 50.0  # Unchanged

    # Check Load clamping
    assert clamped["load_mw"].max() == 5000.0
    assert clamped["load_mw"].min() == 1000.0


def test_runtime_guard_lags():
    """Test that lag columns are also clamped."""
    limits = {"price_max": 100.0}
    guard = RuntimeGuard(limits)

    df = pd.DataFrame({"price_lag_1": [200.0, 50.0], "price_rolling_mean_24": [150.0, 50.0]})

    clamped = guard.validate_and_clamp(df)
    assert clamped["price_lag_1"].max() == 100.0
    assert clamped["price_rolling_mean_24"].max() == 100.0


@pytest.mark.asyncio
async def test_run_forecast_cli_help(monkeypatch):
    """Test that CLI help works (simple smoke test)."""
    monkeypatch.setattr("sys.argv", ["run_forecast.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        await main()
    assert exc.value.code == 0
