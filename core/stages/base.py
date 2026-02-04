# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for pipeline stages.

Defines the contract for all processing stages in the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Return type


class BaseStage(ABC):
    """Abstract base class for all pipeline stages."""

    def __init__(self, country_code: str):
        self.country_code = country_code

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the stage logic."""
        pass
