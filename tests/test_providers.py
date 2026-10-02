"""Tests for decision providers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.mock import MockProvider
from latent_earth.providers.jev import JevProvider
from latent_earth.providers.openai_compatible import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_mock_provider_batching_and_order():
    provider = MockProvider(mode="stylized")
    coords = [
        Coordinate(latitude=45.0, longitude=-73.0),
        Coordinate(latitude=-20.0, longitude=130.0),
        Coordinate(latitude=0.0, longitude=0.0),
    ]

    preds = await provider.predict_coordinates(coords)
    assert len(preds) == 3
    for c, p in zip(coords, preds):
        assert p.coordinate.id == c.id
        assert 0.0 <= p.probability_land <= 1.0
        assert p.is_land == (p.probability_land >= 0.5)


@pytest.mark.asyncio
async def test_mock_provider_simple_mode():
    provider = MockProvider(mode="simple")
    north_coord = Coordinate(latitude=30.0, longitude=0.0)
    south_coord = Coordinate(latitude=-30.0, longitude=0.0)

    p_north = await provider.predict_coordinate(north_coord)
    p_south = await provider.predict_coordinate(south_coord)

    assert p_north.is_land is True
    assert p_north.probability_land == 0.95
    assert p_south.is_land is False
    assert p_south.probability_land == 0.05


@pytest.mark.asyncio
async def test_jev_provider_successful_prediction():
    coord = Coordinate(latitude=45.0, longitude=-73.0)
    mock_response_data = {
        "model": "jev-1.13.0",
        "answers": {
            "is_land": {
                "type": "noul",
                "noul": 0.992,
            }
        },
        "usage": {"input_tokens": 58, "output_tokens": 1},
    }

    async def mock_post(*args, **kwargs):
        payload = kwargs.get("json", {})
        assert payload["model"] == "jev-latest"
        assert "45.000° N" in payload["state"]
        assert "is_land" in payload["questions"]
        assert payload["questions"]["is_land"]["type"] == "noul"
        return httpx.Response(200, json=mock_response_data)

    provider = JevProvider(api_key="mock_test_key", model="jev-latest")
    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        pred = await provider.predict_coordinate(coord)

    assert pred.coordinate.id == coord.id
    assert pred.probability_land == 0.992
    assert pred.is_land is True
    assert pred.model == "jev-1.13.0"
    await provider.close()


@pytest.mark.asyncio
async def test_jev_provider_retry_on_rate_limit():
    coord = Coordinate(latitude=10.0, longitude=20.0)
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt returns 429
            return httpx.Response(429, headers={"retry-after": "0.01"})
        # Second attempt succeeds
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"is_land": {"type": "noul", "noul": 0.85}},
            },
        )

    provider = JevProvider(api_key="mock_test_key", max_retries=2)
    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        pred = await provider.predict_coordinate(coord)

    assert call_count == 2
    assert pred.probability_land == 0.85
    assert pred.is_land is True
    await provider.close()


@pytest.mark.asyncio
async def test_jev_provider_fails_fast_on_client_error():
    coord = Coordinate(latitude=10.0, longitude=20.0)

    async def mock_post(*args, **kwargs):
        req = httpx.Request("POST", "https://api.typesafe.ai/v1/systemone")
        return httpx.Response(401, text="Unauthorized: Invalid API key", request=req)

    provider = JevProvider(api_key="invalid_key", max_retries=3)
    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        with pytest.raises(httpx.HTTPStatusError):
            await provider.predict_coordinate(coord)
    await provider.close()


@pytest.mark.asyncio
async def test_openai_compatible_provider():
    coord = Coordinate(latitude=45.0, longitude=-73.0)

    async def mock_post(*args, **kwargs):
        payload = kwargs.get("json", {})
        assert "messages" in payload
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {"content": "LAND"},
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "LAND", "logprob": -0.05},
                                        {"token": "WATER", "logprob": -3.5},
                                    ]
                                }
                            ]
                        },
                    }
                ],
            },
        )

    provider = OpenAICompatibleProvider(api_key="mock_key", request_logprobs=True)
    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        pred = await provider.predict_coordinate(coord)

    assert pred.is_land is True
    assert pred.probability_land is not None
    assert pred.probability_land > 0.9
    await provider.close()


@pytest.mark.asyncio
async def test_openrouter_d1_provider():
    from latent_earth.providers.openrouter import OpenRouterD1Provider

    coord = Coordinate(latitude=35.0, longitude=135.0)

    async def mock_post(*args, **kwargs):
        payload = kwargs.get("json", {})
        assert "questions" in payload
        assert payload["questions"]["is_land"]["type"] == "noul"
        assert "criteria" in payload["questions"]["is_land"]
        return httpx.Response(
            200,
            json={
                "model": "liquid/d1-20260930",
                "answers": {
                    "is_land": {
                        "type": "noul",
                        "noul": 0.884,
                    }
                },
                "usage": {"input_tokens": 60, "output_tokens": 1},
            },
        )

    provider = OpenRouterD1Provider(api_key="sk-or-test-key")
    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        pred = await provider.predict_coordinate(coord)

    assert pred.is_land is True
    assert pred.probability_land == 0.884
    assert pred.provider == "openrouter"
    assert pred.model == "liquid/d1-20260930"
    await provider.close()


@pytest.mark.asyncio
async def test_fastino_provider():
    from latent_earth.providers.fastino import FastinoProvider

    coord = Coordinate(latitude=45.0, longitude=-73.0)

    async def mock_post(*args, **kwargs):
        payload = kwargs.get("json", {})
        assert "schema" in payload
        assert "classifications" in payload["schema"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "model": "fastino/GLiNER-2.5-Decide",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"surface_type": {"label": "land", "confidence": 0.99968}}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = FastinoProvider(api_key="fast_sk_test")
    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        pred = await provider.predict_coordinate(coord)

    assert pred.is_land is True
    assert pred.probability_land == 0.99968
    assert pred.provider == "fastino"
    assert pred.model == "fastino/GLiNER-2.5-Decide"
    await provider.close()


@pytest.mark.asyncio
async def test_fastino_glide_provider():
    from latent_earth.providers.fastino_glide import FastinoGlideProvider

    coord = Coordinate(latitude=50.0, longitude=10.0)

    async def mock_post(*args, **kwargs):
        payload = kwargs.get("json", {})
        assert payload["model"] == "fastino/glide"
        assert "questions" in payload
        assert payload["questions"]["is_land"]["type"] == "choice"
        return httpx.Response(
            200,
            json={
                "model": "glide",
                "answers": {
                    "is_land": {
                        "type": "choice",
                        "choice": "land",
                        "confidence": 0.5758,
                        "probabilities": {
                            "land": 0.7879,
                            "water": 0.2121,
                        },
                    }
                },
                "usage": {"input_tokens": 137, "output_tokens": 1},
            },
        )

    provider = FastinoGlideProvider(api_key="fast_sk_test")
    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        pred = await provider.predict_coordinate(coord)

    assert pred.is_land is True
    assert pred.probability_land == 0.7879
    assert pred.provider == "fastino_glide"
    assert pred.model == "glide"
    await provider.close()



