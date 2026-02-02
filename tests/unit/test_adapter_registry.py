"""Unit tests for AdapterRegistry."""

import pytest

from config.adapter_registry import create_registry, get_default_registry
from config.country_registry import CountryRegistry
from core.exceptions import CountryNotRegisteredError


# Mocks for adapters
class MockElectricity:
    pass


class MockWeather:
    pass


class MockGas:
    pass


class MockEvents:
    pass


@pytest.fixture
def clean_registry():
    """Clear default registry before and after test."""
    get_default_registry().clear()
    yield
    get_default_registry().clear()


@pytest.fixture
def mock_adapters():
    return {
        "electricity": MockElectricity,
        "weather": MockWeather,
        "gas": MockGas,
        "events": MockEvents,
    }


def test_registry_registration(mock_adapters):
    """Test standard registration flow."""
    registry = create_registry()
    registry.register("PT", **mock_adapters)

    assert registry.is_registered("PT")
    assert not registry.is_registered("XX")

    adapters = registry.get_adapters("PT")
    assert adapters["electricity"] == MockElectricity


def test_registry_unregistered_error():
    """Test error on missing country."""
    registry = create_registry()
    with pytest.raises(CountryNotRegisteredError):
        registry.get_adapters("XX")


def test_backward_compatibility(clean_registry, mock_adapters):
    """Test that static CountryRegistry delegates to default AdapterRegistry."""
    # Register via new instance-based global registry
    get_default_registry().register("PT", **mock_adapters)

    # Access via old static class
    assert CountryRegistry.is_registered("PT")
    fetched = CountryRegistry.get_adapters("PT")
    assert fetched["electricity"] == MockElectricity

    # Register via old static class
    CountryRegistry.register("XX", mock_adapters)

    # Check via new global instance
    assert get_default_registry().is_registered("XX")
    assert CountryRegistry.is_registered("XX")

    # Check listing
    countries = CountryRegistry.list_countries()
    assert "PT" in countries
    assert "XX" in countries


def test_registry_isolation():
    """Test that separate instances are isolated."""
    reg1 = create_registry()
    reg2 = create_registry()

    reg1.register(
        "PT", electricity=MockElectricity, weather=MockWeather, gas=MockGas, events=MockEvents
    )

    assert reg1.is_registered("PT")
    assert not reg2.is_registered("PT")
