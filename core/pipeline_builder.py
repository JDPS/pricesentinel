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
from core.metadata_manager import MetadataManager
from core.pipeline import Pipeline
from core.repository import CsvDataRepository
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

        # Instantiate MetadataManager
        metadata_manager = MetadataManager(data_manager.get_metadata_path())

        # Instantiate ModelRegistry
        from models.model_registry import ModelRegistry

        model_registry = ModelRegistry(models_root="models")

        repository = CsvDataRepository(data_manager, metadata_manager=metadata_manager)
        cleaner = DataCleaner(repository, country_code, timezone=config.timezone)
        feature_engineer = FeatureEngineer(
            country_code, repository, features_config=config.features_config
        )
        verifier = DataVerifier(country_code, validation_config=config.validation_config)

        # 4. Create Stages
        fetch_stage = DataFetchStage(country_code, fetchers, repository)

        # 5. Construct Pipeline
        pipeline = Pipeline(
            country_code=country_code,
            config=config,
            data_manager=data_manager,
            cleaner=cleaner,
            feature_engineer=feature_engineer,
            fetch_stage=fetch_stage,
            verifier=verifier,
            repository=repository,
            model_registry=model_registry,
        )

        logger.info(f"Built pipeline for {country_code}")
        return pipeline
