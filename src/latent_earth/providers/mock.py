"""Mock decision provider for testing and offline development."""

from __future__ import annotations

import asyncio
import math
from typing import Literal
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.base import DecisionProvider


class MockProvider(DecisionProvider):
    """Deterministic mock provider that requires no external API or credentials."""

    def __init__(
        self,
        model: str = "mock-earth-v1",
        prompt_version: str = "mock-v1",
        mode: Literal["simple", "stylized"] = "stylized",
        concurrency: int = 64,
        simulate_delay: float = 0.0,
        fail_first_n_requests: int = 0,
    ) -> None:
        super().__init__(model=model, prompt_version=prompt_version, concurrency=concurrency)
        self.mode = mode
        self.simulate_delay = simulate_delay
        self.fail_counter = fail_first_n_requests

    @property
    def provider_name(self) -> str:
        return "mock"

    async def predict_coordinate(self, coordinate: Coordinate) -> Prediction:
        if self.fail_counter > 0:
            self.fail_counter -= 1
            raise ConnectionError("Simulated transient network failure in MockProvider")

        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)

        lat = coordinate.latitude
        lon = coordinate.longitude

        if self.mode == "simple":
            # Simple hemisphere split
            prob = 0.95 if lat > 0 else 0.05
        else:
            # Stylized continuous function resembling rough continental landmasses
            # Combination of spherical waves
            rad_lat = math.radians(lat)
            rad_lon = math.radians(lon)

            val = (
                0.6 * math.sin(rad_lat) * math.cos(rad_lon)
                + 0.4 * math.cos(2 * rad_lat) * math.sin(2 * rad_lon)
                + 0.3 * math.sin(3 * rad_lat)
                - 0.2 * math.cos(3 * rad_lon)
            )
            # Sigmoid mapping to [0.01, 0.99]
            sigmoid = 1.0 / (1.0 + math.exp(-3.5 * val))
            prob = round(float(sigmoid), 4)

        is_land = prob >= 0.5
        return Prediction(
            coordinate=coordinate,
            probability_land=prob,
            is_land=is_land,
            provider=self.provider_name,
            model=self.model,
            prompt_version=self.prompt_version,
            raw_response={"mock_val": prob, "mode": self.mode},
        )
