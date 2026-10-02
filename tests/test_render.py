"""Tests for map rendering."""

from __future__ import annotations

from pathlib import Path
import pytest

from latent_earth.grid import generate_grid
from latent_earth.models import Prediction
from latent_earth.providers.mock import MockProvider
from latent_earth.render import render_all


@pytest.mark.asyncio
async def test_render_all_generates_image_files(tmp_path: Path):
    grid = generate_grid(step_degrees=20.0)  # 9 x 18 = 162 coords
    provider = MockProvider(mode="stylized")
    predictions = await provider.predict_coordinates(grid)

    output_dir = tmp_path / "renders"
    paths = render_all(
        predictions=predictions,
        step_degrees=20.0,
        output_dir=output_dir,
        base_name="test_render",
        provider="mock",
        model="mock-v1",
        dpi=100,  # Lower dpi for fast test execution
    )

    assert "binary" in paths
    assert "probability" in paths
    assert "uncertainty" in paths
    assert "composite" in paths
    assert "composite_clean" in paths

    for key, path in paths.items():
        assert path.exists(), f"Image {key} at {path} was not created"
        assert path.stat().st_size > 1000, f"Image {key} file size suspiciously small"
