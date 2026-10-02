"""Tests for CLI parsing and command execution."""

from __future__ import annotations

from pathlib import Path
import pytest
from latent_earth.cli import build_parser, infer_step_from_predictions
from latent_earth.models import Coordinate, Prediction


def test_cli_parser_run_defaults():
    parser = build_parser()
    args = parser.parse_args(["run", "--step", "10", "--limit", "20", "--dry-run"])
    assert args.command == "run"
    assert args.step == 10.0
    assert args.limit == 20
    assert args.dry_run is True


def test_cli_parser_render():
    parser = build_parser()
    args = parser.parse_args(["render", "--input", "outputs/test.jsonl", "--dpi", "150"])
    assert args.command == "render"
    assert args.input == "outputs/test.jsonl"
    assert args.dpi == 150


def test_cli_parser_evaluate():
    parser = build_parser()
    args = parser.parse_args(["evaluate", "--input", "outputs/test.jsonl", "--no-error-map"])
    assert args.command == "evaluate"
    assert args.input == "outputs/test.jsonl"
    assert args.no_error_map is True


def test_infer_step_from_predictions():
    coords = [
        Coordinate(latitude=85.0, longitude=-175.0),
        Coordinate(latitude=75.0, longitude=-175.0),
        Coordinate(latitude=65.0, longitude=-175.0),
    ]
    preds = [Prediction(coordinate=c, probability_land=0.5, is_land=True) for c in coords]
    step = infer_step_from_predictions(preds)
    assert step == 10.0
