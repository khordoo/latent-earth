"""Tests for ground truth evaluation."""

from __future__ import annotations

from pathlib import Path
import pytest

from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions, render_error_map
from latent_earth.models import Coordinate, Prediction


def test_ground_truth_evaluator_landmarks():
    evaluator = GroundTruthEvaluator()

    # Central Sahara (Egypt / Libya area) -> Land
    assert evaluator.is_land(lat=25.0, lon=20.0) is True

    # Central Australia -> Land
    assert evaluator.is_land(lat=-25.0, lon=135.0) is True

    # Central Pacific -> Water
    assert evaluator.is_land(lat=0.0, lon=-150.0) is False

    # Central South Atlantic -> Water
    assert evaluator.is_land(lat=-25.0, lon=-20.0) is False


def test_evaluate_predictions_metrics(tmp_path: Path):
    evaluator = GroundTruthEvaluator()

    # Synthetic test predictions
    # Point 1: Sahara (actual Land) -> Model predicts Land (TP)
    # Point 2: Pacific (actual Water) -> Model predicts Water (TN)
    # Point 3: Atlantic (actual Water) -> Model predicts Land (FP)
    # Point 4: Australia (actual Land) -> Model predicts Water (FN)
    preds = [
        Prediction(
            coordinate=Coordinate(25.0, 20.0),
            probability_land=0.95,
            is_land=True,
            provider="test",
        ),
        Prediction(
            coordinate=Coordinate(0.0, -150.0),
            probability_land=0.05,
            is_land=False,
            provider="test",
        ),
        Prediction(
            coordinate=Coordinate(-25.0, -20.0),
            probability_land=0.80,
            is_land=True,
            provider="test",
        ),
        Prediction(
            coordinate=Coordinate(-25.0, 135.0),
            probability_land=0.10,
            is_land=False,
            provider="test",
        ),
    ]

    results = evaluate_predictions(preds, evaluator=evaluator)
    cm = results["confusion_matrix"]
    assert cm["true_positive_land"] == 1
    assert cm["true_negative_water"] == 1
    assert cm["false_positive_land"] == 1
    assert cm["false_negative_water"] == 1

    metrics = results["metrics"]
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["brier_score"] is not None

    # Error map generation test
    err_map = tmp_path / "test_error_map.png"
    render_error_map(preds, step_degrees=30.0, output_path=err_map, evaluator=evaluator, dpi=100)
    assert err_map.exists()
    assert err_map.stat().st_size > 1000


def test_coastal_distance_and_metrics():
    evaluator = GroundTruthEvaluator()

    # Tokyo bay coastal point
    dist_tokyo = evaluator.distance_to_coastline(lat=35.5, lon=139.8)
    assert dist_tokyo < 0.5  # Very close to coastline
    assert evaluator.is_coastal(lat=35.5, lon=139.8, threshold_degrees=1.0) is True

    # Deep Pacific Ocean point
    dist_pacific = evaluator.distance_to_coastline(lat=0.0, lon=-150.0)
    assert dist_pacific > 10.0
    assert evaluator.is_coastal(lat=0.0, lon=-150.0, threshold_degrees=1.0) is False

    # Test predictions with coastal classification
    preds = [
        Prediction(
            coordinate=Coordinate(35.5, 139.8),
            probability_land=0.9,
            is_land=True,
            provider="test",
        ),
        Prediction(
            coordinate=Coordinate(0.0, -150.0),
            probability_land=0.02,
            is_land=False,
            provider="test",
        ),
    ]

    results = evaluate_predictions(preds, evaluator=evaluator, coastal_distance_degrees=1.0)
    assert results["coastal_cell_count"] == 1
    assert results["coastal_cell_fraction"] == 0.5
    assert results["coastal_metrics"]["accuracy"] == 1.0
    assert "area_weighted_metrics" in results
    assert results["area_weighted_metrics"]["accuracy"] == 1.0
