"""OpenRouter Liquid D1 System One decision provider."""

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

DEFAULT_OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_OPENROUTER_MODEL = "liquid/d1"
TRANSIENT_STATUS_CODES = {429, 529, 500, 502, 503, 504}


class OpenRouterD1Provider(DecisionProvider):
    """Decision provider integrating with Liquid AI's d1 model via the OpenRouter Decisions API."""

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_MODEL,
        prompt_version: str = "land-water-v1",
        base_url: str = DEFAULT_OPENROUTER_DECISIONS_URL,
        api_key: str | None = None,
        concurrency: int = 20,
        max_retries: int = 5,
        timeout_seconds: float = 30.0,
        rate_limit_per_second: float = 40.0,
    ) -> None:
        super().__init__(model=model, prompt_version=prompt_version, concurrency=concurrency)
        self.base_url = base_url or DEFAULT_OPENROUTER_DECISIONS_URL
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.rate_limit_per_second = rate_limit_per_second
        self._min_interval = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0.0
        self._last_request_time = 0.0
        self._rate_lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "openrouter"

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
        """Construct the System One decision request payload for a single coordinate."""
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
                    "type": "noul",
                    "instructions": "Is this geographic coordinate located on land rather than water?",
                    "criteria": {
                        "true": "The coordinate is located on land (continents, islands, landmasses).",
                        "false": "The coordinate is located on water (oceans, seas, lakes).",
                    },
                }
            },
        }

    async def predict_coordinate(self, coordinate: Coordinate) -> Prediction:
        """Query Liquid D1 on OpenRouter with native Noul probability decision."""
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable."
            )

        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/latent-earth",
            "X-Title": "Latent Earth",
            "User-Agent": "LatentEarth/0.1.0",
        }
        payload = self.build_payload(coordinate)

        attempt = 0
        backoff_base = 1.0

        while True:
            attempt += 1
            await self._throttle()
            try:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    answers = data.get("answers", {})
                    is_land_data = answers.get("is_land", {})
                    prob = is_land_data.get("noul")
                    if prob is None:
                        prob = is_land_data.get("probability")

                    if prob is None:
                        raise ValueError(
                            f"Malformed OpenRouter D1 response: missing 'noul' in answers: {data}"
                        )

                    prob_float = float(prob)
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
                        "OpenRouter HTTP %d for %s. Retrying attempt %d/%d after %.2fs...",
                        response.status_code,
                        coordinate.id,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(
                    "Permanent OpenRouter error HTTP %d for %s: %s",
                    response.status_code,
                    coordinate.id,
                    response.text,
                )
                response.raise_for_status()

            except httpx.RequestError as exc:
                if attempt > self.max_retries:
                    logger.error("OpenRouter request failed permanently after %d attempts for %s: %s", attempt, coordinate.id, exc)
                    raise
                delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                logger.warning(
                    "OpenRouter network error (%s) on %s. Retrying in %.2fs (attempt %d/%d)...",
                    type(exc).__name__,
                    coordinate.id,
                    delay,
                    attempt,
                    self.max_retries,
                )
                await asyncio.sleep(delay)

    async def close(self) -> None:
        """Close persistent HTTP connection pools."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
