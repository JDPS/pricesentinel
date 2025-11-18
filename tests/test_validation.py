"""
Tests for configuration validation schemas and helpers.
"""

import pytest
from pydantic import ValidationError

from config.validation import (
    CountryConfigSchema,
    ElectricityConfig,
    GasConfig,
    WeatherConfig,
    generate_config_template,
    validate_country_config,
)


def test_generate_config_template_and_validate_roundtrip():
    """Template produced by a helper should validate as a full schema."""
    template = generate_config_template("pt", "Portugal")

    schema = validate_country_config(template)

    assert isinstance(schema, CountryConfigSchema)
    assert schema.country_code == "PT"
    assert schema.country_name == "Portugal"
    # Basic structural checks
    assert schema.electricity.api_type == "entsoe"
    assert schema.weather.coordinates[0].name == "Capital"
    assert schema.gas.currency == "EUR"


def test_electricity_config_requires_valid_api_type():
    """ElectricityConfig rejects unknown api_type values."""
    with pytest.raises(ValidationError):
        ElectricityConfig(api_type="invalid", entsoe_domain=None, market_type="day_ahead")


def test_electricity_config_requires_entsoe_domain_when_entsoe():
    """ENTSO-E configuration must include entsoe_domain."""
    with pytest.raises(ValidationError):
        ElectricityConfig(api_type="entsoe", entsoe_domain=None, market_type="day_ahead")


def test_weather_config_requires_supported_api_type():
    """WeatherConfig rejects unsupported API types."""
    with pytest.raises(ValidationError):
        WeatherConfig(api_type="unsupported", coordinates=[])


def test_gas_config_rejects_invalid_currency_code():
    """GasConfig enforces 3-letter uppercase currency codes."""
    with pytest.raises(ValidationError):
        GasConfig(api_type="ttf", hub_name="TTF", currency="eur1")


def test_country_config_timezone_format_validation():
    """CountryConfigSchema enforces a simple IANA-style timezone format."""
    template = generate_config_template("DE", "Germany")
    template["timezone"] = "InvalidTimezone"

    try:
        validate_country_config(template)
    except ValidationError as exc:
        # Ensure the timezone validator is the source of the failure
        messages = str(exc)
        assert "timezone must be in IANA format" in messages
    else:
        raise AssertionError("ValidationError expected for invalid timezone")
