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
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


class StrictBaseModel(BaseModel):
    """Base model with strict unknown-key handling."""

    model_config = ConfigDict(extra="forbid")


class CoordinateConfig(StrictBaseModel):
    """Weather station coordinates."""

    name: str = Field(..., description="Location name")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class ElectricityConfig(StrictBaseModel):
    """Electricity market configuration."""

    api_type: str = Field(..., description="API type (e.g., 'entsoe', 'eia', 'elexon')")
    entsoe_domain: str | None = Field(None, description="ENTSO-E domain code")
    market_type: str = Field("day_ahead", description="Market type")

    @field_validator("api_type")
    @classmethod
    def validate_api_type(cls, v: str) -> str:
        """Validate API type is recognised."""
        allowed = ["entsoe", "eia", "elexon", "aemo", "custom", "mock"]
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


class WeatherConfig(StrictBaseModel):
    """Weather data configuration."""

    api_type: str = Field(..., description="Weather API type")
    coordinates: list[CoordinateConfig] = Field(
        ..., min_length=1, description="list of weather station coordinates"
    )

    @field_validator("api_type")
    @classmethod
    def validate_api_type(cls, v: str) -> str:
        """Validate weather API type."""
        allowed = ["open_meteo", "openweather", "darksky", "custom", "mock"]
        if v not in allowed:
            raise ValueError(f"api_type must be one of {allowed}, got: {v}")
        return v


class GasConfig(StrictBaseModel):
    """Gas market configuration."""

    api_type: str = Field(..., description="Gas API type")
    hub_name: str = Field(..., description="Gas hub name (e.g., TTF, PVB, NBP)")
    currency: str = Field("EUR", description="Currency code")

    @field_validator("api_type")
    @classmethod
    def validate_api_type(cls, v: str) -> str:
        """Validate gas API type."""
        allowed = ["ttf", "custom", "mock"]
        if v not in allowed:
            raise ValueError(f"api_type must be one of {allowed}, got: {v}")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        if not re.match(r"^[A-Z]{3}$", v):
            raise ValueError(f"currency must be 3-letter code, got: {v}")
        return v


class EventsConfig(StrictBaseModel):
    """Events configuration."""

    holiday_library: str = Field(..., description="Holiday library country code")
    manual_events_path: str = Field(..., description="Path to manual events CSV")
    persistence_required: int = Field(1, ge=1, description="Minimum event persistence")


class FeatureWindowConfig(StrictBaseModel):
    """Configuration for lag and rolling feature windows."""

    lags: list[int] = Field(default_factory=lambda: [1, 2, 24], description="Lag hours")
    rolling_windows: list[int] = Field(default_factory=list, description="Rolling window sizes")
    rolling_stats: list[str] = Field(
        default_factory=lambda: ["mean"], description="Rolling statistics"
    )

    @field_validator("lags", "rolling_windows")
    @classmethod
    def validate_positive_ints(cls, values: list[int]) -> list[int]:
        """Ensure all window values are positive integers."""
        if any(v <= 0 for v in values):
            raise ValueError("window values must be positive integers")
        return values

    @field_validator("rolling_stats")
    @classmethod
    def validate_rolling_stats(cls, values: list[str]) -> list[str]:
        """Validate rolling statistics enum."""
        allowed = {"mean", "std", "min", "max"}
        invalid = sorted(set(values) - allowed)
        if invalid:
            raise ValueError(f"rolling_stats contains unsupported values: {invalid}")
        return values


class FeaturesConfig(StrictBaseModel):
    """Feature engineering configuration."""

    use_cross_border_flows: bool = Field(False, description="Include cross-border flow features")
    feature_windows: FeatureWindowConfig = Field(default_factory=FeatureWindowConfig)
    use_weather_features: bool = Field(True, description="Include weather-derived features")
    use_gas_features: bool = Field(True, description="Include gas price features")
    use_event_features: bool = Field(True, description="Include holiday and event flags")
    neighbors: list[str] = Field(
        default_factory=list, description="Neighboring countries for cross-border features"
    )
    custom_feature_plugins: list[str] = Field(
        default_factory=list, description="Custom feature plugin names"
    )
    use_fourier_features: bool = Field(False, description="Include Fourier cyclic features")
    fourier_periods: list[int] = Field(
        default_factory=lambda: [24, 168], description="Fourier periods in hours"
    )
    use_price_volatility: bool = Field(False, description="Include volatility-derived features")
    use_price_momentum: bool = Field(False, description="Include momentum-derived features")

    @field_validator("fourier_periods")
    @classmethod
    def validate_fourier_periods(cls, values: list[int]) -> list[int]:
        """Ensure all Fourier periods are positive."""
        if any(v <= 0 for v in values):
            raise ValueError("fourier_periods must contain only positive integers")
        return values


class RuntimeLimitsConfig(StrictBaseModel):
    """Optional runtime clamping limits."""

    price_max: float | None = Field(None, description="Maximum allowed price during inference")
    price_min: float | None = Field(None, description="Minimum allowed price during inference")
    load_max: float | None = Field(None, description="Maximum allowed load during inference")
    load_min: float | None = Field(None, description="Minimum allowed load during inference")


class ValidationLimitsConfig(StrictBaseModel):
    """Data validation guardrails."""

    price_min: float = Field(..., description="Minimum accepted historical price")
    price_max: float = Field(..., description="Maximum accepted historical price")
    load_min: float = Field(..., description="Minimum accepted historical load")
    load_max: float = Field(..., description="Maximum accepted historical load")


class CountryConfigSchema(BaseModel):
    """
    Complete country configuration schema.

    This model validates the entire country configuration file,
    ensuring all required fields are present and have valid values.
    """

    model_config = ConfigDict(
        extra="forbid",
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
                    "feature_windows": {
                        "lags": [1, 2, 24],
                        "rolling_windows": [24],
                        "rolling_stats": ["mean"],
                    },
                    "use_weather_features": True,
                    "use_gas_features": True,
                    "use_event_features": True,
                    "neighbors": [],
                    "custom_feature_plugins": [],
                },
                "runtime_limits": {
                    "price_max": 500.0,
                    "price_min": -100.0,
                    "load_max": 20000.0,
                    "load_min": 2000.0,
                },
                "validation": {
                    "price_min": -150.0,
                    "price_max": 3000.0,
                    "load_min": 2000.0,
                    "load_max": 15000.0,
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
    features: FeaturesConfig
    runtime_limits: RuntimeLimitsConfig = Field(default_factory=RuntimeLimitsConfig)
    validation: ValidationLimitsConfig

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone string."""
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"timezone must be a valid IANA timezone, got: {v}") from exc
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="python")


def validate_country_config(config_dict: dict[str, Any]) -> CountryConfigSchema:
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


def generate_config_template(country_code: str, country_name: str) -> dict[str, Any]:
    """
    Generate a template configuration dictionary for a new country.

    Args:
        country_code: ISO 3166-1 alpha-2 code
        country_name: Full country name

    Returns:
        Dictionary with template configuration
    """
    code = country_code.upper()
    return {
        "country_code": code,
        "country_name": country_name,
        "timezone": "Etc/UTC",
        "electricity": {
            "api_type": "entsoe",
            "entsoe_domain": code,
            "market_type": "day_ahead",
        },
        "weather": {
            "api_type": "open_meteo",
            "coordinates": [
                {
                    "name": "Capital",
                    "lat": 0.0,
                    "lon": 0.0,
                }
            ],
        },
        "gas": {"api_type": "ttf", "hub_name": "TTF", "currency": "EUR"},
        "events": {
            "holiday_library": code.lower(),
            "manual_events_path": f"data/{code}/events/manual_events.csv",
            "persistence_required": 1,
        },
        "features": {
            "use_cross_border_flows": False,
            "feature_windows": {
                "lags": [1, 2, 24],
                "rolling_windows": [24],
                "rolling_stats": ["mean"],
            },
            "neighbors": [],
            "custom_feature_plugins": [],
            "use_weather_features": True,
            "use_gas_features": True,
            "use_event_features": True,
            "use_fourier_features": False,
            "fourier_periods": [24, 168],
            "use_price_volatility": False,
            "use_price_momentum": False,
        },
        "runtime_limits": {
            "price_max": 500.0,
            "price_min": -100.0,
            "load_max": 20000.0,
            "load_min": 2000.0,
        },
        "validation": {
            "price_min": -150.0,
            "price_max": 3000.0,
            "load_min": 2000.0,
            "load_max": 15000.0,
        },
    }
