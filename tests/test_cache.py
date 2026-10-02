"""Tests for caching and resumability."""

from __future__ import annotations

from pathlib import Path
import pytest
from latent_earth.cache import PredictionCache
from latent_earth.models import Coordinate, Prediction


def test_write_and_reload_predictions(tmp_path: Path):
    cache_file = tmp_path / "test_run.jsonl"
    cache = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="v1")

    preds = [
        Prediction(
            coordinate=Coordinate(latitude=10.0, longitude=20.0),
            probability_land=0.9,
            is_land=True,
            provider="mock",
            model="mock-v1",
            prompt_version="v1",
        ),
        Prediction(
            coordinate=Coordinate(latitude=-30.0, longitude=-50.0),
            probability_land=0.1,
            is_land=False,
            provider="mock",
            model="mock-v1",
            prompt_version="v1",
        ),
    ]

    cache.append_predictions(preds)
    assert cache_file.exists()

    # Create new cache instance pointing to same file
    cache2 = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="v1")
    loaded = cache2.load()

    assert len(loaded) == 2
    c1_id = Coordinate(10.0, 20.0).id
    assert c1_id in loaded
    assert loaded[c1_id].probability_land == 0.9
    assert loaded[c1_id].is_land is True


def test_skip_existing_and_resume(tmp_path: Path):
    cache_file = tmp_path / "resume_run.jsonl"
    cache = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="v1")

    coord1 = Coordinate(latitude=0.0, longitude=0.0)
    coord2 = Coordinate(latitude=10.0, longitude=10.0)

    # First run saves 1 prediction
    cache.append_predictions([
        Prediction(
            coordinate=coord1,
            probability_land=0.8,
            is_land=True,
            provider="mock",
            model="mock-v1",
            prompt_version="v1",
        )
    ])

    completed = cache.get_completed_ids()
    assert coord1.id in completed
    assert coord2.id not in completed

    # Resume run adds second prediction
    cache.append_predictions([
        Prediction(
            coordinate=coord2,
            probability_land=0.2,
            is_land=False,
            provider="mock",
            model="mock-v1",
            prompt_version="v1",
        )
    ])

    cache_reloaded = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="v1")
    assert len(cache_reloaded.load()) == 2
    assert cache_reloaded.get_completed_ids() == {coord1.id, coord2.id}


def test_incompatible_prompt_version_ignored(tmp_path: Path):
    cache_file = tmp_path / "incompatible_prompt.jsonl"
    cache = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="v1")

    # Save with prompt_version "v1"
    cache.append_predictions([
        Prediction(
            coordinate=Coordinate(latitude=0.0, longitude=0.0),
            probability_land=0.8,
            is_land=True,
            provider="mock",
            model="mock-v1",
            prompt_version="v1",
        )
    ])

    # Now load with expected_prompt_version "v2"
    cache_v2 = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="v2", strict_validation=True)
    loaded = cache_v2.load()
    assert len(loaded) == 0, "Old prompt version cache should not be reused"


def test_incompatible_provider_ignored(tmp_path: Path):
    cache_file = tmp_path / "incompatible_provider.jsonl"
    cache = PredictionCache(cache_file, expected_provider="mock", expected_prompt_version="v1")

    cache.append_predictions([
        Prediction(
            coordinate=Coordinate(latitude=0.0, longitude=0.0),
            probability_land=0.8,
            is_land=True,
            provider="mock",
            model="mock-v1",
            prompt_version="v1",
        )
    ])

    # Load with expected_provider "jev"
    cache_jev = PredictionCache(cache_file, expected_provider="jev", expected_prompt_version="v1", strict_validation=True)
    loaded = cache_jev.load()
    assert len(loaded) == 0, "Incompatible provider cache should not be reused"
