"""Base abstraction for decision providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Sequence
from latent_earth.models import Coordinate, Prediction


class DecisionProvider(ABC):
    """Abstract base class for land/water decision providers."""

    def __init__(
        self,
        model: str,
        prompt_version: str = "land-water-v1",
        concurrency: int = 16,
    ) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'jev', 'mock', 'openai_compatible')."""
        ...

    @abstractmethod
    async def predict_coordinate(self, coordinate: Coordinate) -> Prediction:
        """Predict whether a single coordinate is on land.

        Must adhere to the scientific constraint:
        Treat the coordinate as an independent decision with no neighboring context.
        """
        ...

    async def predict_coordinates(
        self,
        coordinates: Sequence[Coordinate],
    ) -> list[Prediction]:
        """Predict land/water for a sequence of coordinates concurrently.

        Coordinates are processed respecting concurrency limits and preserving order.
        """
        async def _bounded_predict(coord: Coordinate) -> Prediction:
            async with self._semaphore:
                return await self.predict_coordinate(coord)

        tasks = [_bounded_predict(coord) for coord in coordinates]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Release any underlying resources, connections, or sessions."""
        pass

    async def __aenter__(self) -> DecisionProvider:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
