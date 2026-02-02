# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration validation using Pydantic.

This module provides Pydantic models for validating country configuration files,
ensuring all required fields are present and have valid values.
"""

import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def validate_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """
    Validate that start_date and end_date are valid dates and start_date <= end_date.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        tuple[str, str]: Validated start and end date strings

    Raises:
        ValueError: If dates are invalid or start_date > end_date
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD: {e}") from e

    if start_dt > end_dt:
        raise ValueError(
            f"start_date ({start_date}) must be before or equal to end_date ({end_date})"
        )

    return start_date, end_date


class CoordinateConfig(BaseModel):
    """Weather station coordinates."""

    name: str = Field(..., description="Location name")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class ElectricityConfig(BaseModel):
    """Electricity market configuration."""

    api_type: str = Field(..., description="API type (e.g., 'entsoe', 'eia', 'elexon')")
    entsoe_domain: str | None = Field(None, description="ENTSO-E domain code")
    market_type: str = Field("day_ahead", description="Market type")

    @field_validator("api_type")
    @classmethod
    def validate_api_type(cls, v: str) -> str:
        """Validate API type is recognised."""
        allowed = ["entsoe", "eia", "elexon", "aemo", "custom"]
        if v not in allowed:
            raise ValueError(f"api_type must be one of {allowed}, got: {v}")
        return v

    @field_validator("entsoe_domain")
    @classmethod
    def validate_entsoe_domain(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Ensure ENTSO-E domain is provided if using ENTSO-E API."""
        api_type = info.data.get("api_type")
        if api_type == "entsoe" and not v:
            raise ValueError("entsoe_domain is required when api_type is 'entsoe'")
        return v


class WeatherConfig(BaseModel):
    """Weather data configuration."""

    api_type: str = Field(..., description="Weather API type")
    coordinates: list[CoordinateConfig] = Field(
        ..., min_length=1, description="list of weather station coordinates"
    )

    @field_validator("api_type")
    @classmethod
    def validate_api_type(cls, v: str) -> str:
        """Validate weather API type."""
        allowed = ["open_meteo", "openweather", "darksky", "custom"]
        if v not in allowed:
            raise ValueError(f"api_type must be one of {allowed}, got: {v}")
        return v


class GasConfig(BaseModel):
    """Gas market configuration."""

    api_type: str = Field(..., description="Gas API type")
    hub_name: str = Field(..., description="Gas hub name (e.g., TTF, PVB, NBP)")
    currency: str = Field("EUR", description="Currency code")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        # Simple 3-letter currency code validation
        if not re.match(r"^[A-Z]{3}$", v):
            raise ValueError(f"currency must be 3-letter code, got: {v}")
        return v


class EventsConfig(BaseModel):
    """Events configuration."""

    holiday_library: str = Field(..., description="Holiday library country code")
    manual_events_path: str = Field(..., description="Path to manual events CSV")
    persistence_required: int = Field(1, ge=1, description="Minimum event persistence")


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    use_cross_border_flows: bool = Field(False, description="Include cross-border flow features")
    neighbors: list[str] = Field([], description="Neighboring countries for cross-border features")
    custom_feature_plugins: list[str] = Field([], description="Custom feature plugin names")
    use_weather_features: bool = Field(True, description="Include weather-derived features")
    use_gas_features: bool = Field(True, description="Include gas price features")
    use_event_features: bool = Field(True, description="Include holiday and event flags")


class CountryConfigSchema(BaseModel):
    """
    Complete country configuration schema.

    This model validates the entire country configuration file,
    ensuring all required fields are present and have valid values.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        json_schema_extra={
            "example": {
                "country_code": "PT",
                "country_name": "Portugal",
                "timezone": "Europe/Lisbon",
                "electricity": {
                    "api_type": "entsoe",
                    "entsoe_domain": "PT",
                    "market_type": "day_ahead",
                },
                "weather": {
                    "api_type": "open_meteo",
                    "coordinates": [{"name": "Lisbon", "lat": 38.7223, "lon": -9.1393}],
                },
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
        },
    )

    country_code: str = Field(..., pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 code")
    country_name: str = Field(..., description="Full country name")
    timezone: str = Field(..., description="IANA timezone (e.g., Europe/Lisbon)")

    electricity: ElectricityConfig
    weather: WeatherConfig
    gas: GasConfig
    events: EventsConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone string."""
        # Basic validation - check format
        if "/" not in v:
            raise ValueError(f"timezone must be in IANA format (e.g., Europe/Lisbon) got: {v}")
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()


def validate_country_config(config_dict: dict) -> CountryConfigSchema:
    """
    Validate a country configuration dictionary.

    Args:
        config_dict: Dictionary containing country configuration

    Returns:
        Validated CountryConfigSchema instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return CountryConfigSchema(**config_dict)


def generate_config_template(country_code: str, country_name: str) -> dict:
    """
    Generate a template configuration dictionary for a new country.

    Args:
        country_code: ISO 3166-1 alpha-2 code
        country_name: Full country name

    Returns:
        Dictionary with template configuration
    """
    return {
        "country_code": country_code.upper(),
        "country_name": country_name,
        # Use a valid IANA-style timezone; users may override this
        "timezone": "Etc/UTC",
        "electricity": {
            "api_type": "entsoe",  # Most common for EU
            "entsoe_domain": country_code.upper(),
            "market_type": "day_ahead",
        },
        "weather": {
            "api_type": "open_meteo",
            "coordinates": [
                {
                    "name": "Capital",
                    "lat": 0.0,  # User should change this
                    "lon": 0.0,  # User should change this
                }
            ],
        },
        "gas": {"api_type": "ttf", "hub_name": "TTF", "currency": "EUR"},
        "events": {
            "holiday_library": country_code.lower(),
            "manual_events_path": f"data/{country_code.upper()}/events/manual_events.csv",
            "persistence_required": 1,
        },
        "features": {
            "use_cross_border_flows": False,
            "neighbors": [],
            "custom_feature_plugins": [],
        },
    }
