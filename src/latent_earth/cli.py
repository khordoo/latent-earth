"""Command line interface for Latent Earth."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Sequence

from latent_earth.cache import PredictionCache, get_default_output_path
from latent_earth.config import AppConfig
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions, render_error_map
from latent_earth.models import Prediction
from latent_earth.providers import get_provider
from latent_earth.render import render_all
from latent_earth.runner import Runner

logger = logging.getLogger("latent_earth")


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def infer_step_from_predictions(predictions: Sequence[Prediction]) -> float:
    """Attempt to infer grid step size from prediction coordinates."""
    if len(predictions) < 2:
        return 4.0

    lats = sorted({round(p.coordinate.latitude, 6) for p in predictions})
    if len(lats) > 1:
        diffs = [round(abs(lats[i] - lats[i - 1]), 4) for i in range(1, len(lats))]
        min_diff = min(diffs)
        if min_diff > 0:
            return min_diff
    return 4.0


def estimate_jev_cost(remaining_coords: int) -> str:
    """Calculate approximate Jev API cost estimate based on current public pricing ($0.042 / Mtok)."""
    # Each coordinate prompt + systemone wrapper is approx 60 tokens
    estimated_tokens = remaining_coords * 60
    cost = (estimated_tokens / 1_000_000.0) * 0.042
    return f"~{estimated_tokens:,} tokens (~${cost:.4f} USD at current $0.042/Mtok rate)"


async def async_run(args: argparse.Namespace) -> None:
    """Execute model inference run."""
    config = AppConfig.load(args.config)

    # CLI Overrides
    if args.step:
        config.grid.step_degrees = float(args.step)
    if args.provider:
        config.provider.name = args.provider
        if not args.model and args.provider == "mock":
            config.provider.model = "mock-earth-v1"
            config.provider.prompt_version = "mock-v1"
    if args.model:
        config.provider.model = args.model
    if args.concurrency:
        config.inference.concurrency = int(args.concurrency)
    if args.batch_size:
        config.inference.batch_size = int(args.batch_size)

    # Determine output cache file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = get_default_output_path(
            output_dir=config.output.dir,
            provider=config.provider.name,
            model=config.provider.model,
            step_degrees=config.grid.step_degrees,
            prompt_version=config.provider.prompt_version,
        )

    cache = PredictionCache(
        filepath=output_file,
        expected_provider=config.provider.name,
        expected_prompt_version=config.provider.prompt_version,
    )

    provider = get_provider(config.provider, config.inference)
    runner = Runner(config=config, provider=provider, cache=cache)

    # Cost estimate for Jev
    estimated_cost = None
    if config.provider.name == "jev":
        estimated_cost = estimate_jev_cost(
            int(round((180.0 / config.grid.step_degrees) * (360.0 / config.grid.step_degrees)))
        )

    preds = await runner.run(
        step_degrees=config.grid.step_degrees,
        limit=args.limit,
        dry_run=args.dry_run,
        estimated_cost=estimated_cost,
    )

    if args.dry_run:
        return

    # Render visualizations if enabled
    if not args.no_render and len(preds) > 0:
        base_name = output_file.stem
        out_dir = output_file.parent
        print(f"\nGenerating map visual reconstructions in {out_dir}...")
        render_all(
            predictions=preds,
            step_degrees=config.grid.step_degrees,
            output_dir=out_dir,
            base_name=base_name,
            provider=config.provider.name,
            model=config.provider.model,
            convention=config.grid.convention,
        )
        print("Rendering complete.")

    # Ground-truth evaluation if requested
    if args.evaluate and len(preds) > 0:
        print("\nEvaluating against Natural Earth ground truth...")
        evaluator = GroundTruthEvaluator()
        results = evaluate_predictions(preds, evaluator=evaluator)
        eval_json_path = output_file.parent / f"{output_file.stem}_evaluation.json"
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics to {eval_json_path}")

        err_map_path = output_file.parent / f"{output_file.stem}_error_map.png"
        render_error_map(preds, config.grid.step_degrees, err_map_path, evaluator=evaluator)
        print(f"Saved error map to {err_map_path}")


def cmd_render(args: argparse.Namespace) -> None:
    """Render existing JSONL prediction file."""
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    print(f"Loading predictions from {input_path}...")
    cache = PredictionCache(input_path, strict_validation=False)
    preds_dict = cache.load()
    preds = list(preds_dict.values())
    if not preds:
        sys.exit("No valid predictions found in input file.")

    step = args.step if args.step else infer_step_from_predictions(preds)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    base_name = input_path.stem

    provider = preds[0].provider or "unknown"
    model = preds[0].model or "unknown"

    print(f"Rendering maps (Step: {step:g}°, Points: {len(preds):,})...")
    render_all(
        predictions=preds,
        step_degrees=step,
        output_dir=out_dir,
        base_name=base_name,
        provider=provider,
        model=model,
        dpi=args.dpi,
    )
    print("Rendering complete.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate existing JSONL predictions against ground truth."""
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    print(f"Loading predictions from {input_path}...")
    cache = PredictionCache(input_path, strict_validation=False)
    preds = list(cache.load().values())
    if not preds:
        sys.exit("No valid predictions found in input file.")

    step = args.step if args.step else infer_step_from_predictions(preds)
    evaluator = GroundTruthEvaluator()
    results = evaluate_predictions(preds, evaluator=evaluator)

    out_json = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== GROUND TRUTH EVALUATION METRICS ===")
    metrics = results.get("metrics", {})
    for k, v in metrics.items():
        print(f"{k.capitalize():<14}: {v}")
    print(f"\nSaved full results to {out_json}")

    if not args.no_error_map:
        err_map_path = input_path.parent / f"{input_path.stem}_error_map.png"
        render_error_map(preds, step, err_map_path, evaluator=evaluator)
        print(f"Saved error map to {err_map_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latent-earth",
        description="Latent Earth: Testing geographic structure inside decision models.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run model inference over coordinate grid.")
    run_parser.add_argument("--config", default="config.yaml", help="Path to config YAML file.")
    run_parser.add_argument("--step", type=float, help="Grid step in degrees (e.g. 10, 4, 2).")
    run_parser.add_argument("--limit", type=int, help="Limit number of queries (for smoke test / dev).")
    run_parser.add_argument("--provider", choices=["jev", "mock", "openai_compatible"], help="Provider name.")
    run_parser.add_argument("--model", help="Model name.")
    run_parser.add_argument("--concurrency", type=int, help="Concurrent request limit.")
    run_parser.add_argument("--batch-size", type=int, help="Batch size for runner.")
    run_parser.add_argument("--output", help="Custom output JSONL filepath.")
    run_parser.add_argument("--dry-run", action="store_true", help="Print run plan without calling API.")
    run_parser.add_argument("--no-render", action="store_true", help="Skip automatic map rendering.")
    run_parser.add_argument("--evaluate", action="store_true", help="Run ground-truth evaluation after inference.")

    # 'render' subcommand
    render_parser = subparsers.add_parser("render", help="Render maps from JSONL predictions.")
    render_parser.add_argument("--input", required=True, help="Input predictions JSONL file.")
    render_parser.add_argument("--step", type=float, help="Grid step in degrees (optional, auto-inferred).")
    render_parser.add_argument("--output-dir", help="Directory to save rendered images.")
    render_parser.add_argument("--dpi", type=int, default=300, help="Image DPI (default 300).")

    # 'evaluate' subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against ground truth.")
    eval_parser.add_argument("--input", required=True, help="Input predictions JSONL file.")
    eval_parser.add_argument("--step", type=float, help="Grid step in degrees (optional, auto-inferred).")
    eval_parser.add_argument("--output", help="Output JSON evaluation file path.")
    eval_parser.add_argument("--no-error-map", action="store_true", help="Do not render error map.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "run":
        asyncio.run(async_run(args))
    elif args.command == "render":
        cmd_render(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
