"""Tests for FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from latent_earth.api import app


def test_api_health():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_list_runs():
    client = TestClient(app)
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        run = data[0]
        assert "run_id" in run
        assert "point_count" in run


def test_api_get_ground_truth():
    client = TestClient(app)
    response = client.get("/api/ground-truth")
    assert response.status_code == 200
    geojson = response.json()
    assert geojson.get("type") in ("FeatureCollection", "GeometryCollection")


def test_api_serves_frontend_index():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Latent Earth" in response.text


def test_api_get_run_with_ground_truth():
    client = TestClient(app)
    runs_res = client.get("/api/runs")
    assert runs_res.status_code == 200
    runs = runs_res.json()
    if runs:
        run_id = runs[0]["run_id"]
        res = client.get(f"/api/runs/{run_id}")
        assert res.status_code == 200
        data = res.json()
        assert "points" in data
        assert len(data["points"]) > 0
        first_pt = data["points"][0]
        assert "actual_land" in first_pt
        assert "error_category" in first_pt
        assert first_pt["actual_land"] is not None
        assert first_pt["error_category"] in ("correct_land", "correct_water", "false_land", "false_water")

