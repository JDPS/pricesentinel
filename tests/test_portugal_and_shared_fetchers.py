# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Portugal-specific and shared data fetchers (with mocked HTTP).
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from config.country_registry import CountryConfig
from data_fetchers.portugal.electricity import PortugalElectricityFetcher
from data_fetchers.portugal.events import HOLIDAYS_AVAILABLE, PortugalEventProvider
from data_fetchers.shared.open_meteo import OpenMeteoWeatherFetcher
from data_fetchers.shared.ttf_gas import TTFGasFetcher


class DummyConfig:
    """Lightweight config object mimicking the CountryConfig interface."""

    def __init__(self):
        self.country_code = "PT"
        self.timezone = "Europe/Lisbon"
        self.electricity_config = {"entsoe_domain": "PT"}
        self.weather_config = {
            "coordinates": [
                {"name": "Lisbon", "lat": 38.7223, "lon": -9.1393},
            ]
        }
        self.events_config = {"manual_events_path": "data/PT/events/manual_events.csv"}
        self.gas_config = {"hub_name": "TTF", "currency": "EUR"}


def test_portugal_electricity_fetcher_requires_api_key(monkeypatch):
    """PortugalElectricityFetcher should fail fast without ENTSOE_API_KEY."""
    dummy_config = DummyConfig()
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)

    with pytest.raises(ValueError):
        PortugalElectricityFetcher(dummy_config)


def _make_price_xml():
    """Build minimal ENTSO-E-like XML dict for prices."""
    return {
        "Publication_MarketDocument": {
            "TimeSeries": {
                "Period": {
                    "timeInterval": {"start": "2024-01-01T00:00Z"},
                    "resolution": "PT60M",
                    "Point": [
                        {"position": "1", "price.amount": "10.0"},
                        {"position": "2", "price.amount": "20.0"},
                    ],
                }
            }
        }
    }


def _make_load_xml():
    """Build minimal ENTSO-E-like XML dict for load."""
    return {
        "GL_MarketDocument": {
            "TimeSeries": {
                "Period": {
                    "timeInterval": {"start": "2024-01-01T00:00Z"},
                    "resolution": "PT60M",
                    "Point": [
                        {"position": "1", "quantity": "1000.0"},
                        {"position": "2", "quantity": "1100.0"},
                    ],
                }
            }
        }
    }


def test_portugal_electricity_parse_helpers(monkeypatch):
    """_parse_price_response and _parse_load_response should build DataFrames."""
    dummy_config = DummyConfig()
    monkeypatch.setenv("ENTSOE_API_KEY", "dummy-key")
    fetcher = PortugalElectricityFetcher(dummy_config)

    price_xml = _make_price_xml()
    load_xml = _make_load_xml()

    price_df = fetcher._parse_price_response(price_xml)
    load_df = fetcher._parse_load_response(load_xml)

    # Price dataframe checks
    assert isinstance(price_df, pd.DataFrame)
    assert len(price_df) == 2
    assert list(price_df.columns) == ["timestamp", "price_eur_mwh", "market", "quality_flag"]
    assert price_df["price_eur_mwh"].iloc[0] == 10.0

    # Load dataframe checks
    assert isinstance(load_df, pd.DataFrame)
    assert len(load_df) == 2
    assert list(load_df.columns) == ["timestamp", "load_mw", "quality_flag"]
    assert load_df["load_mw"].iloc[0] == 1000.0


def test_open_meteo_fetcher_empty_coordinates_raises():
    """OpenMeteoWeatherFetcher requires at least one coordinate."""
    cfg_dict = {
        "country_code": "PT",
        "country_name": "Portugal",
        "timezone": "Europe/Lisbon",
        "electricity": {"api_type": "entsoe", "entsoe_domain": "PT", "market_type": "day_ahead"},
        "weather": {"api_type": "open_meteo", "coordinates": []},
        "gas": {"api_type": "ttf", "hub_name": "TTF", "currency": "EUR"},
        "events": {
            "holiday_library": "portugal",
            "manual_events_path": "data/PT/events/manual_events.csv",
            "persistence_required": 1,
        },
        "features": {
            "use_cross_border_flows": False,
            "neighbors": [],
            "custom_feature_plugins": [],
        },
    }

    cfg = CountryConfig(cfg_dict)

    with pytest.raises(ValueError):
        OpenMeteoWeatherFetcher(cfg)


def test_open_meteo_fetcher_parses_single_location(monkeypatch):
    """
    Tests the OpenMeteoWeatherFetcher to ensure it correctly parses weather data
    for a single location using a mocked HTTP response.

    Raises
    ------
    AssertionError
        If the resulting DataFrame does not meet the expected structure, contents,
        or column names during validation, an AssertionError is raised.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        A pytest-provided fixture used for dynamically modifying or replacing
        parts of the system under test, in this case, monkey-patching the
        HTTP GET request used by the weather fetcher.
    """
    cfg = DummyConfig()
    fetcher = OpenMeteoWeatherFetcher(cfg)  # type: ignore[arg-type]

    # Minimal synthetic JSON response
    start = datetime(2024, 1, 1, 0, tzinfo=UTC)
    times = [start.isoformat(), (start.replace(hour=1)).isoformat()]
    response_json = {
        "hourly": {
            "time": times,
            "temperature_2m": [10.0, 11.0],
            "windspeed_10m": [5.0, 6.0],
            "shortwave_radiation": [0.0, 100.0],
            "precipitation": [0.0, 0.1],
        }
    }

    class DummyResponse:
        """
        Dummy Response
        """

        @staticmethod
        def raise_for_status():
            return None

        def json(self):
            return response_json

    def dummy_get(url, params=None, timeout=None):  # noqa: ARG001
        """
        Performs a simple GET request to the specified URL with optional query
        parameters and timeout.

        This function is a placeholder for performing HTTP GET requests and normally
        would send a request to the specified URL. However, in this implementation,
        it simply returns a dummy response object.

        Parameters:
            url: str
                The URL to which the GET request should be sent.
            params: dict, optional
                A dictionary of query parameters to be included in the GET request.
            timeout: float, optional
                The maximum time (in seconds) to wait for a response.

        Returns:
            DummyResponse
                A dummy response object that simulates the result of an HTTP GET
                request.
        """

        return DummyResponse()

    monkeypatch.setattr("data_fetchers.shared.open_meteo.requests.get", dummy_get)

    df = fetcher.fetch_weather("2024-01-01", "2024-01-01")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["location_name"].unique()) == {"Lisbon"}
    assert "temperature_c" in df.columns
    assert "wind_speed_ms" in df.columns
    assert "solar_radiation_wm2" in df.columns
    assert "precipitation_mm" in df.columns


@pytest.mark.skipif(not HOLIDAYS_AVAILABLE, reason="holidays package not installed")
def test_portugal_event_provider_holidays(monkeypatch, tmp_path):
    """PortugalEventProvider should return holidays within the range."""
    cfg = DummyConfig()
    # Use a temporary manual events path
    cfg.events_config["manual_events_path"] = str(tmp_path / "manual_events.csv")

    provider = PortugalEventProvider(cfg)  # type: ignore[arg-type]

    df = provider.get_holidays("2024-01-01", "2024-12-31")

    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "event_type" in df.columns
    assert "description" in df.columns


def test_ttf_gas_fetcher_returns_empty_when_csv_missing(tmp_path):
    """TTFGasFetcher should return an empty DataFrame if CSV is missing."""
    cfg = DummyConfig()
    fetcher = TTFGasFetcher(cfg)  # type: ignore[arg-type]

    # Point to a non-existent CSV inside tmp_path
    fetcher.manual_csv_path = tmp_path / "missing_ttf.csv"  # type: ignore[assignment]

    df = fetcher.fetch_prices("2024-01-01", "2024-01-31")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["timestamp", "price_eur_mwh", "hub_name", "quality_flag"]


def test_ttf_gas_fetcher_template_and_fetch(tmp_path, monkeypatch):
    """TTFGasFetcher should create a template CSV and load filtered prices."""
    monkeypatch.chdir(tmp_path)
    cfg = DummyConfig()
    fetcher = TTFGasFetcher(cfg)  # type: ignore[arg-type]

    # Create a template CSV at the default location
    fetcher.create_template_csv()
    assert fetcher.manual_csv_path.exists()

    df = fetcher.fetch_prices("2023-01-01", "2023-01-07")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "price_eur_mwh" in df.columns
    assert "hub_name" in df.columns
    # All rows should have the configured hub name
    assert set(df["hub_name"].unique()) == {cfg.gas_config.get("hub_name", "TTF")}
