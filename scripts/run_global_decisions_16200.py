#!/usr/bin/env python3
"""
Run Global 16,200-Point Planetary Benchmark (2.0° resolution)
Sequentially evaluates:
1. Microsoft Decision-1 (microsoft/microsoft-decision-1)
2. Perplexity Pplx-Decider (perplexity/pplx-decider-v1.1-27b)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions
from latent_earth.models import Coordinate, Prediction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("global_decisions_16200")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY not found in environment.")
    sys.exit(1)

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def build_payload(model: str, coordinate: Coordinate) -> dict[str, Any]:
    cardinal = coordinate.format_cardinal()
    state = (
        f"Geographic coordinate: {cardinal} "
        f"(Latitude: {coordinate.latitude:+.4f}, Longitude: {coordinate.longitude:+.4f})"
    )
    return {
        "model": model,
        "state": state,
        "questions": {
            "is_land": {
                "type": "noul",
                "instructions": "Is this geographic coordinate located on land rather than water?",
                "criteria": {
                    "true": "The coordinate is located on land (continents, islands, landmasses).",
                    "false": "The coordinate is located on water (oceans, seas, lakes).",
                },
            }
        },
    }


async def send_single_query(
    client: httpx.AsyncClient,
    model: str,
    coordinate: Coordinate,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> Prediction:
    payload = build_payload(model, coordinate)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/latent-earth",
        "X-Title": "Latent Earth Global Benchmark",
    }

    t0 = time.monotonic()
    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
                resp = await client.post(DECISIONS_URL, json=payload, headers=headers)
            latency = time.monotonic() - t0

            if resp.status_code == 200:
                data = resp.json()
                data["latency_seconds"] = round(latency, 4)
                data["attempt_count"] = attempt

                answers = data.get("answers", {})
                is_land_data = answers.get("is_land", {})
                prob = is_land_data.get("noul")
                if prob is None:
                    prob = is_land_data.get("probability", 0.5)

                prob_float = float(prob)
                is_land = prob_float >= 0.5
                actual_model = data.get("model", model)

                return Prediction(
                    coordinate=coordinate,
                    probability_land=prob_float,
                    is_land=is_land,
                    provider="openrouter",
                    model=actual_model,
                    prompt_version="land-water-v1",
                    raw_response=data,
                )

            if resp.status_code in {429, 500, 502, 503, 504}:
                delay = 1.0 * (2 ** (attempt - 1)) + random.uniform(0.1, 0.4)
                logger.warning("HTTP %d for %s (attempt %d/%d). Sleeping %.2fs...", resp.status_code, coordinate.id, attempt, max_retries, delay)
                await asyncio.sleep(delay)
                continue

            logger.error("HTTP %d error for %s: %s", resp.status_code, coordinate.id, resp.text[:200])
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        except httpx.RequestError as exc:
            delay = 1.0 * (2 ** (attempt - 1)) + random.uniform(0.1, 0.4)
            logger.warning("Network error %s for %s. Sleeping %.2fs...", type(exc).__name__, coordinate.id, delay)
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed coordinate {coordinate.id} after {max_retries} attempts.")


async def benchmark_model_global(
    model_name: str,
    run_id: str,
    target_coords: list[Coordinate],
    evaluator: GroundTruthEvaluator,
    concurrency: int = 24,
    chunk_size: int = 48,
):
    jsonl_path = OUTPUTS_DIR / f"{run_id}.jsonl"
    eval_path = OUTPUTS_DIR / f"{run_id}_evaluation.json"
    meta_path = OUTPUTS_DIR / f"{run_id}_metadata.json"

    cached_preds: dict[str, Prediction] = {}
    if jsonl_path.exists():
        try:
            cached_preds = PredictionCache(jsonl_path, strict_validation=False).load()
            logger.info("[%s] Found %d existing cached predictions.", model_name, len(cached_preds))
        except Exception as e:
            logger.warning("[%s] Could not load partial cache (%s), continuing", model_name, e)

    remaining = [c for c in target_coords if c.id not in cached_preds]
    logger.info("[%s] Global run: %d total, %d cached, %d remaining.", model_name, len(target_coords), len(cached_preds), len(remaining))

    all_preds: dict[str, Prediction] = dict(cached_preds)
    latencies: list[float] = []
    wall_clock_start = time.monotonic()

    if remaining:
        limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)
        timeout = httpx.Timeout(45.0, connect=10.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            sem = asyncio.Semaphore(concurrency)
            chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]

            for ch_idx, chunk in enumerate(chunks, start=1):
                ch_start = time.monotonic()
                tasks = [send_single_query(client, model_name, coord, sem) for coord in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                valid_preds: list[Prediction] = []
                for res in results:
                    if isinstance(res, Prediction):
                        valid_preds.append(res)
                        all_preds[res.coordinate.id] = res
                        raw = res.raw_response or {}
                        if "latency_seconds" in raw:
                            latencies.append(raw["latency_seconds"])
                    else:
                        logger.error("Query failed: %s", res)

                # Append to JSONL immediately
                with open(jsonl_path, "a", encoding="utf-8") as f_out:
                    for pred in valid_preds:
                        f_out.write(json.dumps(pred.to_dict()) + "\n")

                ch_elapsed = time.monotonic() - ch_start
                rate = len(valid_preds) / ch_elapsed if ch_elapsed > 0 else 0
                pct = len(all_preds) / len(target_coords) * 100
                total_elapsed = time.monotonic() - wall_clock_start
                est_rem = (total_elapsed / len(all_preds) * (len(target_coords) - len(all_preds))) if len(all_preds) > 0 else 0

                if ch_idx % 10 == 0 or ch_idx == len(chunks):
                    logger.info(
                        "[%s] Chunk %d/%d (%.1f%%) | Rate: %.1f req/s | Total: %d/%d | Elapsed: %.1fm | ETA: %.1fm",
                        model_name, ch_idx, len(chunks), pct, rate, len(all_preds), len(target_coords),
                        total_elapsed / 60.0, est_rem / 60.0
                    )

    wall_total = time.monotonic() - wall_clock_start
    logger.info("[%s] Completed in %.2fs. Total evaluated: %d.", model_name, wall_total, len(all_preds))

    # Save metadata for UI
    metadata = {
        "run_id": run_id,
        "provider": "openrouter",
        "model": model_name,
        "prompt_version": "land-water-v1",
        "region_name": "global",
        "grid_resolution": 2.0,
        "step_degrees": 2.0,
        "point_count": len(all_preds),
        "wall_clock_seconds": round(wall_total, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    # Ground Truth Evaluation
    ordered_preds = [all_preds[c.id] for c in target_coords if c.id in all_preds]
    eval_results = evaluate_predictions(ordered_preds, evaluator, coastal_distance_km=100.0)

    cm = eval_results["confusion_matrix"]
    tp = cm["true_positive_land"]
    tn = cm["true_negative_water"]
    fp = cm["false_positive_land"]
    fn = cm["false_negative_water"]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0

    grid_m = eval_results["metrics"]
    area_m = eval_results.get("area_weighted_metrics", {})
    coastal_m = eval_results.get("coastal_metrics", {})

    full_eval = {
        "benchmark": "Global 16,200-Point Planetary Benchmark (2.0°)",
        "provider": "openrouter",
        "model": model_name,
        "total_evaluated": len(ordered_preds),
        "balanced_accuracy": round(balanced_acc, 4),
        "metrics": {
            "accuracy": grid_m["accuracy"],
            "area_weighted_accuracy": area_m.get("accuracy", grid_m["accuracy"]),
            "brier_score": grid_m.get("brier_score"),
            "area_weighted_brier_score": area_m.get("brier_score"),
            "coastal_accuracy": coastal_m.get("accuracy"),
            "coastal_brier_score": coastal_m.get("brier_score"),
            "precision": grid_m["precision"],
            "recall": grid_m["recall"],
            "f1": grid_m["f1"],
        },
        "grid_metrics": grid_m,
        "area_weighted_metrics": area_m,
        "coastal_metrics": coastal_m,
        "confusion_matrix": cm,
        "performance": {
            "wall_clock_time_seconds": round(wall_total, 2),
            "queries_executed": len(remaining),
            "queries_cached": len(cached_preds),
            "total_benchmark_points": len(target_coords),
            "effective_throughput_req_per_sec": round(len(remaining) / wall_total, 2) if wall_total > 0 and remaining else 0.0,
            "latency_stats": {
                "median_seconds": round(float(np.median(latencies)), 4) if latencies else None,
                "p90_seconds": round(float(np.percentile(latencies, 90)), 4) if latencies else None,
                "p95_seconds": round(float(np.percentile(latencies, 95)), 4) if latencies else None,
                "min_seconds": round(float(np.min(latencies)), 4) if latencies else None,
                "max_seconds": round(float(np.max(latencies)), 4) if latencies else None,
                "sample_count": len(latencies),
            },
        },
    }

    with open(eval_path, "w", encoding="utf-8") as f_eval:
        json.dump(full_eval, f_eval, indent=2)

    logger.info("[%s] Saved evaluation to %s", model_name, eval_path)
    print(f"\n[{model_name}] Global 16,200 Benchmark Results:")
    print(f"Grid Accuracy: {grid_m['accuracy']*100:.2f}% | Area-Weighted: {area_m.get('accuracy',0)*100:.2f}% | Bal Acc: {balanced_acc*100:.2f}% | Brier: {grid_m.get('brier_score',0):.4f}")


async def main():
    # 1. Load frozen global 16,200 coordinates
    jev_path = OUTPUTS_DIR / "jev_jev-latest_2deg_land-water-v1.jsonl"
    if not jev_path.exists():
        logger.error("Frozen JEV file %s not found.", jev_path)
        sys.exit(1)

    jev_cache = PredictionCache(jev_path, strict_validation=False).load()
    target_coords = [p.coordinate for p in jev_cache.values()]
    logger.info("Loaded %d target global coordinates from JEV.", len(target_coords))

    evaluator = GroundTruthEvaluator()

    # Step 1: Run Microsoft Decision-1
    logger.info(">>> Starting Microsoft Decision-1 Global 16,200 Benchmark...")
    await benchmark_model_global(
        model_name="microsoft/microsoft-decision-1",
        run_id="openrouter_microsoft_microsoft-decision-1_global_16200",
        target_coords=target_coords,
        evaluator=evaluator,
        concurrency=24,
        chunk_size=48,
    )

    # Step 2: Run Perplexity Pplx-Decider
    logger.info(">>> Starting Perplexity Pplx-Decider Global 16,200 Benchmark...")
    await benchmark_model_global(
        model_name="perplexity/pplx-decider-v1.1-27b",
        run_id="openrouter_perplexity_pplx-decider-v1.1-27b_global_16200",
        target_coords=target_coords,
        evaluator=evaluator,
        concurrency=24,
        chunk_size=48,
    )

    logger.info(">>> ALL GLOBAL BENCHMARKS COMPLETED!")


if __name__ == "__main__":
    asyncio.run(main())
