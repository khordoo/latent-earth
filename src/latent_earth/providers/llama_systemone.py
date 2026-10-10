"""Local llama.cpp System One Decision Provider.

Connects to a local llama.cpp server running with native /v1/systemone support
(e.g., llama-server -hf ggml-org/Kev-4B-GGUF or ggml-org/Julia-1-GGUF).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from typing import Any
import httpx

from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.base import DecisionProvider

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_URL = "http://localhost:8080/v1/systemone"
DEFAULT_LLAMA_MODEL = "ggml-org/Julia-1-GGUF"


class LlamaSystemOneProvider(DecisionProvider):
    """Decision provider integrating with a local llama.cpp instance running System One decision models."""

    def __init__(
        self,
        model: str = DEFAULT_LLAMA_MODEL,
        prompt_version: str = "land-water-v1",
        base_url: str = DEFAULT_LLAMA_URL,
        concurrency: int = 32,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(model=model, prompt_version=prompt_version, concurrency=concurrency)
        self.base_url = base_url or DEFAULT_LLAMA_URL
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "llama_cpp"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
                limits=httpx.Limits(
                    max_keepalive_connections=self.concurrency * 2,
                    max_connections=self.concurrency * 4,
                ),
            )
        return self._client

    def build_payload(self, coordinate: Coordinate) -> dict[str, Any]:
        """Construct the System One choice decision payload for a single coordinate."""
        cardinal = coordinate.format_cardinal()
        state = (
            f"Geographic coordinate: {cardinal} "
            f"(Latitude: {coordinate.latitude:+.4f}, Longitude: {coordinate.longitude:+.4f})"
        )
        return {
            "model": self.model,
            "state": {
                "scenario": (
                    "You are an expert Earth scientist and geospatial intelligence system. "
                    "Determine whether the given geographic coordinate on Earth is located on land or water. "
                    "Consider physical Earth geography, continental landmasses, islands, and open water bodies."
                ),
                "facts": [state],
            },
            "questions": {
                "is_land": {
                    "type": "choice",
                    "instructions": (
                        f"Is the geographic coordinate {cardinal} on land or water? "
                        "Choose 'land' if the coordinate falls on continental land, an island, or inland water enclosed by land. "
                        "Choose 'water' if it falls in the ocean, seas, or major coastal water bodies."
                    ),
                    "criteria": {
                        "land": "The coordinate is located on land.",
                        "water": "The coordinate is located on open water or ocean.",
                    },
                }
            },
        }

    async def predict_coordinate(self, coordinate: Coordinate) -> Prediction:
        """Query the local llama.cpp /v1/systemone endpoint for a single coordinate."""
        payload = self.build_payload(coordinate)
        headers = {"Content-Type": "application/json"}
        client = self._get_client()

        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                )
                req_latency = time.monotonic() - t0

                if response.status_code == 200:
                    data = response.json()
                    data["latency_seconds"] = round(req_latency, 4)
                    data["attempt_count"] = attempt

                    answers = data.get("answers", {})
                    is_land_ans = answers.get("is_land", {})
                    probs = is_land_ans.get("probabilities", {})
                    prob_land = probs.get("land")

                    if prob_land is None:
                        choice = is_land_ans.get("choice", "").lower()
                        conf = is_land_ans.get("confidence", 0.5)
                        prob_land = conf if choice == "land" else (1.0 - conf)

                    prob_float = float(prob_land)
                    prob_float = max(0.0, min(1.0, prob_float))
                    is_land = prob_float >= 0.5
                    actual_model = data.get("model", self.model)

                    return Prediction(
                        coordinate=coordinate,
                        probability_land=prob_float,
                        is_land=is_land,
                        provider=self.provider_name,
                        model=actual_model,
                        prompt_version=self.prompt_version,
                        raw_response=data,
                    )

                logger.warning(
                    "Local llama.cpp HTTP %d for %s (attempt %d/%d): %s",
                    response.status_code,
                    coordinate.id,
                    attempt,
                    self.max_retries,
                    response.text[:200],
                )
                await asyncio.sleep(0.5 * attempt)

            except httpx.RequestError as exc:
                logger.warning(
                    "Local llama.cpp network error for %s (attempt %d/%d): %s",
                    coordinate.id,
                    attempt,
                    self.max_retries,
                    exc,
                )
                await asyncio.sleep(0.5 * attempt)

        raise RuntimeError(f"Failed to query local llama.cpp for coordinate {coordinate.id} after {self.max_retries} attempts.")

    async def predict_coordinates(self, coordinates: list[Coordinate]) -> list[Prediction]:
        """Query multiple coordinates concurrently."""
        sem = asyncio.Semaphore(self.concurrency)

        async def _bounded_predict(c: Coordinate) -> Prediction:
            async with sem:
                return await self.predict_coordinate(c)

        tasks = [_bounded_predict(c) for c in coordinates]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
