# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for the Portugal (PT) integration.
"""

from config.country_registry import CountryRegistry, FetcherFactory
from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries
from data_fetchers.portugal import PortugalElectricityFetcher, PortugalEventProvider
from data_fetchers.shared.open_meteo import OpenMeteoWeatherFetcher
from data_fetchers.shared.ttf_gas import TTFGasFetcher


def test_pt_fetchers_created_via_factory(monkeypatch):
    """FetcherFactory should create PT fetchers wired to ENTSO-E, Open-Meteo, and TTF."""
    monkeypatch.setenv("ENTSOE_API_KEY", "dummy-key")
    CountryRegistry.clear()
    auto_register_countries()

    assert CountryRegistry.is_registered("PT")

    fetchers = FetcherFactory.create_fetchers("PT")

    assert isinstance(fetchers["electricity"], PortugalElectricityFetcher)
    assert isinstance(fetchers["weather"], OpenMeteoWeatherFetcher)
    assert isinstance(fetchers["gas"], TTFGasFetcher)
    assert isinstance(fetchers["events"], PortugalEventProvider)


def test_pt_pipeline_initialisation_uses_portugal_config(tmp_path, monkeypatch):
    """Pipeline for PT should load PT.yaml config and register PT fetchers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENTSOE_API_KEY", "dummy-key")
    CountryRegistry.clear()
    auto_register_countries()

    pipeline = PipelineBuilder.create_pipeline("PT")

    assert pipeline.config.country_code == "PT"
    assert pipeline.config.country_name == "Portugal"
    assert pipeline.config.timezone == "Europe/Lisbon"
    assert "PT" in pipeline.data_manager.base_path.parts

    # Ensure fetchers mapping is PT-specific
    assert isinstance(pipeline.fetch_stage.fetchers["electricity"], PortugalElectricityFetcher)
    assert isinstance(pipeline.fetch_stage.fetchers["weather"], OpenMeteoWeatherFetcher)
    assert isinstance(pipeline.fetch_stage.fetchers["gas"], TTFGasFetcher)
    assert isinstance(pipeline.fetch_stage.fetchers["events"], PortugalEventProvider)
