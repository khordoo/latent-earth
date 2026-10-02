"""Tests for async inference runner."""

from __future__ import annotations

from pathlib import Path
import pytest

from latent_earth.cache import PredictionCache
from latent_earth.config import AppConfig
from latent_earth.grid import generate_grid
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.mock import MockProvider
from latent_earth.runner import Runner, chunk_list


def test_chunk_list():
    coords = [Coordinate(float(i), 0.0) for i in range(100)]
    chunks = chunk_list(coords, 16)
    assert len(chunks) == 7  # 6 * 16 = 96, + 1 * 4 = 100
    assert sum(len(c) for c in chunks) == 100


@pytest.mark.asyncio
async def test_runner_processes_exact_coordinates(tmp_path: Path):
    cache_file = tmp_path / "test_run.jsonl"
    cache = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="mock-v1")

    config = AppConfig()
    config.grid.step_degrees = 18.0  # 18° grid -> 10 x 20 = 200 coordinates
    config.inference.batch_size = 16
    config.inference.concurrency = 8

    provider = MockProvider(model="mock-v1", prompt_version="mock-v1")
    runner = Runner(config=config, provider=provider, cache=cache)

    # Run with limit=100 coordinates
    preds = await runner.run(step_degrees=18.0, limit=100)

    assert len(preds) == 100
    assert len(cache) == 100

    # Ensure every coordinate ID is distinct and processed exactly once
    ids = [p.coordinate.id for p in preds]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_runner_resumes_cleanly(tmp_path: Path):
    cache_file = tmp_path / "test_resume.jsonl"
    cache = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="mock-v1")

    config = AppConfig()
    config.grid.step_degrees = 18.0
    config.inference.batch_size = 16

    provider1 = MockProvider(model="mock-v1", prompt_version="mock-v1")
    runner1 = Runner(config=config, provider=provider1, cache=cache)

    # First run processes 40 coordinates
    preds1 = await runner1.run(step_degrees=18.0, limit=40)
    assert len(preds1) == 40
    assert len(cache) == 40

    # Second run with limit=80
    provider2 = MockProvider(model="mock-v1", prompt_version="mock-v1")
    runner2 = Runner(config=config, provider=provider2, cache=cache)

    preds2 = await runner2.run(step_degrees=18.0, limit=80)
    assert len(preds2) == 120
    assert len(cache) == 120


@pytest.mark.asyncio
async def test_runner_dry_run_makes_no_calls(tmp_path: Path):
    cache_file = tmp_path / "test_dry_run.jsonl"
    cache = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="mock-v1")

    config = AppConfig()
    provider = MockProvider(model="mock-v1", prompt_version="mock-v1")
    runner = Runner(config=config, provider=provider, cache=cache)

    preds = await runner.run(step_degrees=30.0, dry_run=True)
    assert len(preds) == 0
    assert not cache_file.exists()
