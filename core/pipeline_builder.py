# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Builder pattern for constructing Pipeline instances.

This module isolates the complexity of creating a Pipeline and its dependencies
(Configuration, Fetchers, DataManager, etc.) from the Pipeline class itself.
"""

import logging

from config.country_registry import ConfigLoader, FetcherFactory
from core.cleaning import DataCleaner
from core.data_manager import CountryDataManager
from core.features import FeatureEngineer
from core.pipeline import Pipeline
from core.stages.fetch_stage import DataFetchStage
from core.verification import DataVerifier

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Builder for creating fully configured Pipeline instances."""

    @staticmethod
    def create_pipeline(country_code: str) -> Pipeline:
        """
        Create a fully configured Pipeline instance for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 code

        Returns:
            Configured Pipeline instance
        """
        country_code = country_code.upper()

        # 1. Load Configuration
        config = ConfigLoader.load_country_config(country_code)

        # 2. Create Infrastructure
        data_manager = CountryDataManager(country_code)
        data_manager.create_directories()  # Ensure directories exist

        # 3. Create Components
        fetchers = FetcherFactory.create_fetchers(country_code)
        cleaner = DataCleaner(data_manager, country_code)
        feature_engineer = FeatureEngineer(country_code, features_config=config.features_config)
        verifier = DataVerifier(country_code)

        # 4. Create Stages
        fetch_stage = DataFetchStage(country_code, fetchers, data_manager)

        # 5. Construct Pipeline
        pipeline = Pipeline(
            country_code=country_code,
            config=config,
            data_manager=data_manager,
            cleaner=cleaner,
            feature_engineer=feature_engineer,
            fetch_stage=fetch_stage,
            verifier=verifier,
        )

        logger.info(f"Built pipeline for {country_code}")
        return pipeline
