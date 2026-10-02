"""OpenAI-compatible decision provider for vLLM, Ollama, LM Studio, or OpenAI."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import re
from typing import Any
import httpx
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.base import DecisionProvider

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider(DecisionProvider):
    """Decision provider for any OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_version: str = "openai-land-water-v1",
        base_url: str = DEFAULT_OPENAI_URL,
        api_key: str | None = None,
        concurrency: int = 8,
        max_retries: int = 4,
        timeout_seconds: float = 30.0,
        request_logprobs: bool = False,
    ) -> None:
        super().__init__(model=model, prompt_version=prompt_version, concurrency=concurrency)
        # Ensure endpoint ends with /chat/completions if only base URL provided
        if not base_url.endswith("/chat/completions"):
            self.endpoint = base_url.rstrip("/") + "/chat/completions"
        else:
            self.endpoint = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.request_logprobs = request_logprobs
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

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

    def build_prompt(self, coordinate: Coordinate) -> str:
        cardinal = coordinate.format_cardinal()
        return (
            f"Determine whether the geographic coordinate below is located on land.\n\n"
            f"Coordinate: {cardinal}\n"
            f"Latitude: {coordinate.latitude:+.4f}\n"
            f"Longitude: {coordinate.longitude:+.4f}\n\n"
            f"Answer with exactly one token:\n\n"
            f"LAND\n\n"
            f"or\n\n"
            f"WATER"
        )

    async def predict_coordinate(self, coordinate: Coordinate) -> Prediction:
        client = self._get_client()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LatentEarth/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        prompt = self.build_prompt(coordinate)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }
        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        attempt = 0
        backoff_base = 1.0

        while True:
            attempt += 1
            try:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError(f"No choices returned from model: {data}")

                    first_choice = choices[0]
                    content = first_choice.get("message", {}).get("content", "").strip()

                    # Extract binary classification
                    is_land = False
                    if re.search(r"\bLAND\b", content, re.IGNORECASE):
                        is_land = True
                    elif re.search(r"\bWATER\b", content, re.IGNORECASE):
                        is_land = False
                    else:
                        # Fallback parsing
                        is_land = "land" in content.lower()

                    # Attempt to extract probability from logprobs if available
                    prob_land: float | None = None
                    logprobs_content = first_choice.get("logprobs", {}).get("content")
                    if logprobs_content and len(logprobs_content) > 0:
                        top_logprobs = logprobs_content[0].get("top_logprobs", [])
                        land_logprob: float | None = None
                        water_logprob: float | None = None
                        for item in top_logprobs:
                            token = item.get("token", "").strip().upper()
                            lp = item.get("logprob")
                            if lp is not None:
                                if "LAND" in token and land_logprob is None:
                                    land_logprob = float(lp)
                                elif "WATER" in token and water_logprob is None:
                                    water_logprob = float(lp)

                        if land_logprob is not None and water_logprob is not None:
                            # Softmax over land vs water
                            max_lp = max(land_logprob, water_logprob)
                            e_land = math.exp(land_logprob - max_lp)
                            e_water = math.exp(water_logprob - max_lp)
                            prob_land = round(e_land / (e_land + e_water), 4)
                        elif land_logprob is not None:
                            prob_land = round(math.exp(land_logprob), 4)

                    if prob_land is None:
                        prob_land = 1.0 if is_land else 0.0

                    return Prediction(
                        coordinate=coordinate,
                        probability_land=prob_land,
                        is_land=is_land,
                        provider=self.provider_name,
                        model=data.get("model", self.model),
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
                        "OpenAI HTTP %d for %s. Retrying attempt %d/%d after %.2fs...",
                        response.status_code,
                        coordinate.id,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                if attempt > self.max_retries:
                    raise exc
                delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                logger.warning(
                    "Network error %s for %s. Retrying in %.2fs...",
                    exc.__class__.__name__,
                    coordinate.id,
                    delay,
                )
                await asyncio.sleep(delay)

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
