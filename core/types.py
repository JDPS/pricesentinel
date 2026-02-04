# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for PriceSentinel."""

from typing import Literal, NewType, TypedDict

# Semantic types
CountryCode = NewType("CountryCode", str)
ISODate = NewType("ISODate", str)  # YYYY-MM-DD format
Timezone = NewType("Timezone", str)  # IANA timezone identifier
APIKey = NewType("APIKey", str)


# TypedDict for configuration
class ElectricityConfig(TypedDict):
    """Electricity data source configuration."""

    api_type: Literal["entsoe", "mock"]
    entsoe_domain: str
    market_type: str


class WeatherLocation(TypedDict):
    """Weather station coordinates."""

    name: str
    lat: float
    lon: float


class WeatherConfig(TypedDict):
    """Weather data source configuration."""

    api_type: Literal["open_meteo", "mock"]
    coordinates: list[WeatherLocation]


class GasConfig(TypedDict):
    """Gas price data source configuration."""

    api_type: Literal["ttf", "mock"]
    hub_name: str


class EventsConfig(TypedDict):
    """Events data source configuration."""

    holiday_library: str
    manual_events_path: str


class FeatureWindowConfig(TypedDict):
    """Configuration for time-series feature windows."""

    lags: list[int]
    rolling_windows: list[int]
    rolling_stats: list[Literal["mean", "std", "min", "max"]]


class FeaturesConfig(TypedDict):
    """Feature engineering configuration."""

    use_cross_border_flows: bool
    feature_windows: FeatureWindowConfig
    use_weather_features: bool
    use_gas_features: bool
    use_event_features: bool
    neighbors: list[str]
    custom_feature_plugins: list[str]


class ValidationLimits(TypedDict):
    """Data validation limits."""

    price_min: float
    price_max: float
    load_min: float
    load_max: float


class CountryConfigDict(TypedDict):
    """Complete country configuration."""

    country_code: CountryCode
    country_name: str
    timezone: Timezone
    electricity: ElectricityConfig
    weather: WeatherConfig
    gas: GasConfig
    events: EventsConfig
    features: FeaturesConfig
    validation: ValidationLimits


class DirectoryInfo(TypedDict):
    """Data directory information."""

    country_code: str
    base_path: str
    total_size_bytes: int
    total_size_mb: str
    sources: dict[str, int]
    file_count: int
    created: str
    exists: bool


class PipelineInfo(TypedDict):
    """Pipeline information dictionary."""

    country_code: str
    country_name: str
    timezone: str
    run_id: str
    data_directory: str
    data_info: DirectoryInfo


class DatasetMetadata(TypedDict):
    """Metadata for a single dataset."""

    filename: str
    source: str
    start_date: str
    end_date: str
    fetch_timestamp: str
    checksum: str
    row_count: int
    column_count: int
    file_size_bytes: int
