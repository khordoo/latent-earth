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


REGION_PRESETS: dict[str, dict[str, float]] = {
    "japan_korea": {
        "lat_min": 24.0,
        "lat_max": 46.0,
        "lon_min": 122.0,
        "lon_max": 146.0,
    }
}


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
    step_val = args.grid_resolution if getattr(args, "grid_resolution", None) is not None else args.step
    if step_val is not None:
        config.grid.step_degrees = float(step_val)

    if getattr(args, "region", None):
        config.region.name = args.region
        preset = REGION_PRESETS.get(args.region.lower())
        if preset:
            config.region.lat_min = preset["lat_min"]
            config.region.lat_max = preset["lat_max"]
            config.region.lon_min = preset["lon_min"]
            config.region.lon_max = preset["lon_max"]

    if getattr(args, "bbox", None):
        config.region.lat_min = float(args.bbox[0])
        config.region.lat_max = float(args.bbox[1])
        config.region.lon_min = float(args.bbox[2])
        config.region.lon_max = float(args.bbox[3])

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
            region_name=config.region.name,
        )

    cache = PredictionCache(
        filepath=output_file,
        expected_provider=config.provider.name,
        expected_prompt_version=config.provider.prompt_version,
    )

    provider = get_provider(config.provider, config.inference)
    runner = Runner(config=config, provider=provider, cache=cache)

    # Calculate exact total coordinates for cost estimate
    from latent_earth.grid import generate_grid
    expected_coords = generate_grid(
        step_degrees=config.grid.step_degrees,
        convention=config.grid.convention,
        lat_min=config.region.lat_min,
        lat_max=config.region.lat_max,
        lon_min=config.region.lon_min,
        lon_max=config.region.lon_max,
    )
    estimated_cost = None
    if config.provider.name == "jev":
        estimated_cost = estimate_jev_cost(len(expected_coords))

    preds = await runner.run(
        step_degrees=config.grid.step_degrees,
        limit=args.limit,
        dry_run=args.dry_run,
        estimated_cost=estimated_cost,
    )

    if args.dry_run:
        return

    # Persist experiment metadata independently
    metadata = {
        "provider": config.provider.name,
        "model": config.provider.model,
        "model_version": config.provider.model,
        "region_name": config.region.name,
        "bounding_box": {
            "lat_min": config.region.lat_min,
            "lat_max": config.region.lat_max,
            "lon_min": config.region.lon_min,
            "lon_max": config.region.lon_max,
        },
        "grid_resolution": config.grid.step_degrees,
        "step_degrees": config.grid.step_degrees,
        "query_count": len(preds),
        "timestamp": preds[0].timestamp if preds else "",
        "prompt_version": config.provider.prompt_version,
        "convention": config.grid.convention,
    }
    meta_path = output_file.parent / f"{output_file.stem}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)
    print(f"Saved run metadata to {meta_path}")

    # Render visualizations if enabled (for global runs or regional if requested)
    if not args.no_render and len(preds) > 0 and config.region.is_global:
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
        coastal_threshold = getattr(args, "coastal_threshold", 1.0) or 1.0
        print(f"\nEvaluating against Natural Earth ground truth (coastal band: {coastal_threshold:g}°)...")
        evaluator = GroundTruthEvaluator()
        results = evaluate_predictions(preds, evaluator=evaluator, coastal_distance_degrees=coastal_threshold)
        results["metadata"] = metadata
        eval_json_path = output_file.parent / f"{output_file.stem}_evaluation.json"
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics to {eval_json_path}")

        # Print summary table
        print("\n=== EVALUATION METRICS ===")
        print(f"Total Evaluated:        {results['total_evaluated']:,}")
        print(f"Grid Accuracy:          {results['grid_metrics']['accuracy']:.1%}")
        print(f"Area-Weighted Accuracy: {results['area_weighted_metrics']['accuracy']:.1%}")
        if results['grid_metrics']['brier_score'] is not None:
            print(f"Grid Brier Score:       {results['grid_metrics']['brier_score']:.4f}")
            print(f"Area-Weighted Brier:    {results['area_weighted_metrics']['brier_score']:.4f}")
        if results['coastal_metrics']['accuracy'] is not None:
            c_cnt = results['coastal_metrics']['coastal_cell_count']
            c_frac = results['coastal_metrics']['coastal_cell_fraction']
            print(f"Coastal Cells:          {c_cnt:,} ({c_frac:.1%} of region)")
            print(f"Coastal Accuracy:       {results['coastal_metrics']['accuracy']:.1%}")
            if results['coastal_metrics']['brier_score'] is not None:
                print(f"Coastal Brier Score:    {results['coastal_metrics']['brier_score']:.4f}")
        print("==========================\n")


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
    coastal_threshold = getattr(args, "coastal_threshold", 1.0) or 1.0
    evaluator = GroundTruthEvaluator()
    results = evaluate_predictions(preds, evaluator=evaluator, coastal_distance_degrees=coastal_threshold)

    out_json = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== GROUND TRUTH EVALUATION METRICS ===")
    print(f"Total Evaluated:        {results['total_evaluated']:,}")
    print(f"Grid Accuracy:          {results['grid_metrics']['accuracy']:.1%}")
    print(f"Area-Weighted Accuracy: {results['area_weighted_metrics']['accuracy']:.1%}")
    if results['grid_metrics']['brier_score'] is not None:
        print(f"Grid Brier Score:       {results['grid_metrics']['brier_score']:.4f}")
        print(f"Area-Weighted Brier:    {results['area_weighted_metrics']['brier_score']:.4f}")
    if results['coastal_metrics']['accuracy'] is not None:
        c_cnt = results['coastal_metrics']['coastal_cell_count']
        c_frac = results['coastal_metrics']['coastal_cell_fraction']
        print(f"Coastal Cells:          {c_cnt:,} ({c_frac:.1%} of region)")
        print(f"Coastal Accuracy:       {results['coastal_metrics']['accuracy']:.1%}")
        if results['coastal_metrics']['brier_score'] is not None:
            print(f"Coastal Brier Score:    {results['coastal_metrics']['brier_score']:.4f}")
    print("=======================================\n")
    print(f"Saved full results to {out_json}")

    if not args.no_error_map:
        err_map_path = input_path.parent / f"{input_path.stem}_error_map.png"
        render_error_map(preds, step, err_map_path, evaluator=evaluator)
        print(f"Saved error map to {err_map_path}")



def cmd_serve(args: argparse.Namespace) -> None:
    """Launch FastAPI server to serve API and interactive frontend."""
    import uvicorn
    print(f"\nStarting Latent Earth Interactive Visualization Server at http://{args.host}:{args.port}")
    print(f"Open http://{args.host}:{args.port} in your browser to view the interactive globe.\n")
    uvicorn.run("latent_earth.api:app", host=args.host, port=args.port, reload=args.reload)


def cmd_export(args: argparse.Namespace) -> None:
    """Export predictions and evaluation into a self-contained frontend JSON bundle."""
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    cache = PredictionCache(input_path, strict_validation=False)
    preds = list(cache.load().values())
    if not preds:
        sys.exit("No valid predictions found in input file.")

    step = args.step if args.step else infer_step_from_predictions(preds)
    out_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_bundle.json"

    eval_file = input_path.parent / f"{input_path.stem}_evaluation.json"
    meta_file = input_path.parent / f"{input_path.stem}_metadata.json"
    eval_data: dict[str, Any] | None = None
    meta_data: dict[str, Any] | None = None

    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                meta_data = json.load(mf)
        except Exception:
            pass

    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as ef:
                eval_data = json.load(ef)
        except Exception:
            pass

    try:
        from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions
        evaluator = GroundTruthEvaluator()
        eval_metrics = evaluate_predictions(preds, evaluator=evaluator)
        if not eval_data:
            eval_data = eval_metrics
    except Exception:
        pass

    first_pred = preds[0]
    points_data = [p.to_dict() for p in preds]

    manifest = {
        "run_id": input_path.stem,
        "provider": first_pred.provider,
        "model": first_pred.model,
        "prompt_version": first_pred.prompt_version,
        "step_degrees": step,
        "convention": "cell_center",
        "total_coordinates": len(preds),
        "sampled_count": len(preds),
        "created_at": first_pred.timestamp,
        "region": meta_data.get("bounding_box") if meta_data else None,
        "metrics": eval_data.get("metrics") if eval_data else None,
        "points": points_data,
    }
    if eval_data:
        manifest["full_evaluation"] = eval_data

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported frontend-ready bundle with {len(points_data):,} points to {out_path}")


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
    run_parser.add_argument("--step", type=float, help="Grid step in degrees (e.g. 10, 4, 2, 1, 0.5).")
    run_parser.add_argument("--grid-resolution", type=float, help="Grid resolution step in degrees (alias for --step).")
    run_parser.add_argument("--region", help="Region preset name (e.g. 'japan_korea') or custom name.")
    run_parser.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"), help="Bounding box: lat_min lat_max lon_min lon_max.")
    run_parser.add_argument("--coastal-threshold", type=float, default=1.0, help="Distance threshold in degrees for coastal band evaluation (default: 1.0).")
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
    eval_parser.add_argument("--coastal-threshold", type=float, default=1.0, help="Distance threshold in degrees for coastal band evaluation (default: 1.0).")
    eval_parser.add_argument("--no-error-map", action="store_true", help="Do not render error map.")

    # 'serve' subcommand
    serve_parser = subparsers.add_parser("serve", help="Launch web visualization server.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default 127.0.0.1).")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    serve_parser.add_argument("--reload", action="store_true", help="Enable hot reloading.")

    # 'export' subcommand
    export_parser = subparsers.add_parser("export", help="Export JSONL run to frontend JSON bundle.")
    export_parser.add_argument("--input", required=True, help="Input predictions JSONL file.")
    export_parser.add_argument("--step", type=float, help="Grid step in degrees (optional, auto-inferred).")
    export_parser.add_argument("--output", help="Output JSON bundle file path.")

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
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
