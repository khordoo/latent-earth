"""Fastino GLiDE System One decision provider."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any
import httpx
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.base import DecisionProvider

logger = logging.getLogger(__name__)

DEFAULT_GLIDE_URL = "https://api.fastino.ai/v1/systemone"
DEFAULT_GLIDE_MODEL = "fastino/glide"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class FastinoGlideProvider(DecisionProvider):
    """Decision provider integrating with Fastino's GLiDE model via the System One API."""

    def __init__(
        self,
        model: str = DEFAULT_GLIDE_MODEL,
        prompt_version: str = "land-water-v1",
        base_url: str = DEFAULT_GLIDE_URL,
        api_key: str | None = None,
        concurrency: int = 12,
        max_retries: int = 5,
        timeout_seconds: float = 90.0,
        rate_limit_per_second: float = 30.0,
    ) -> None:
        super().__init__(model=model, prompt_version=prompt_version, concurrency=concurrency)
        self.base_url = base_url or DEFAULT_GLIDE_URL
        self.api_key = api_key or os.getenv("FASTINO_API_KEY")
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.rate_limit_per_second = rate_limit_per_second
        self._min_interval = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0.0
        self._last_request_time = 0.0
        self._rate_lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "fastino_glide"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=self.concurrency * 2,
                    max_connections=self.concurrency * 4,
                ),
            )
        return self._client

    async def _throttle(self) -> None:
        """Cooperative pacing to avoid exceeding rate limits."""
        if self._min_interval <= 0:
            return
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

    def build_payload(self, coordinate: Coordinate) -> dict[str, Any]:
        """Construct the System One choice decision payload for a single coordinate."""
        cardinal = coordinate.format_cardinal()
        state = (
            f"Geographic coordinate: {cardinal} "
            f"(Latitude: {coordinate.latitude:+.4f}, Longitude: {coordinate.longitude:+.4f})"
        )
        return {
            "model": self.model,
            "state": state,
            "questions": {
                "is_land": {
                    "type": "choice",
                    "instructions": "Is this geographic coordinate located on land rather than water?",
                    "criteria": {
                        "land": "The coordinate is located on land (continents, islands, landmasses).",
                        "water": "The coordinate is located on water (oceans, seas, lakes).",
                    },
                }
            },
        }

    async def predict_coordinate(self, coordinate: Coordinate) -> Prediction:
        """Query Fastino GLiDE with native choice probabilities and exponential backoff."""
        if not self.api_key:
            raise ValueError(
                "Fastino API key not found. Please set FASTINO_API_KEY environment variable."
            )

        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "LatentEarth/0.1.0",
        }
        payload = self.build_payload(coordinate)

        attempt = 0
        backoff_base = 1.0

        while True:
            attempt += 1
            await self._throttle()
            try:
                t0 = time.monotonic()
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
                        prob_land = is_land_ans.get("noul")

                    if prob_land is None:
                        raise ValueError(f"Malformed GLiDE response: missing probability: {data}")

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

                if response.status_code in TRANSIENT_STATUS_CODES:
                    if attempt > self.max_retries:
                        response.raise_for_status()

                    retry_after_str = response.headers.get("retry-after")
                    if retry_after_str:
                        try:
                            delay = float(retry_after_str)
                        except ValueError:
                            delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    else:
                        delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)

                    logger.warning(
                        "Fastino GLiDE HTTP %d for %s. Retrying attempt %d/%d after %.2fs...",
                        response.status_code,
                        coordinate.id,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(
                    "Permanent Fastino GLiDE error HTTP %d for %s: %s",
                    response.status_code,
                    coordinate.id,
                    response.text,
                )
                response.raise_for_status()

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                if attempt > self.max_retries:
                    logger.error("Fastino GLiDE request failed after %d attempts for %s: %s", attempt, coordinate.id, exc)
                    raise
                delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                logger.warning(
                    "Network error %s for %s. Retrying attempt %d/%d in %.2fs...",
                    exc,
                    coordinate.id,
                    attempt,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

    async def close(self) -> None:
        """Close underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
