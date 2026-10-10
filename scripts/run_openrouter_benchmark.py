#!/usr/bin/env python3
"""
Benchmark OpenRouter models (e.g. openai/gpt-6-luna, google/gemma-4-31b-it)
on the full Japan/Korea 2,112-point regional benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
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
logger = logging.getLogger("openrouter_benchmark")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY not found in environment.")
    sys.exit(1)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_chat_payload(model: str, coordinate: Coordinate) -> dict[str, Any]:
    cardinal = coordinate.format_cardinal()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Earth scientist and geospatial classification system. "
                    "Given a geographic coordinate on Earth, determine whether it is on land or water. "
                    "Consider physical geography, islands, and open water bodies. "
                    "You MUST reply ONLY with a valid JSON object: {\"choice\": \"land\" | \"water\", \"confidence\": float between 0.0 and 1.0}."
                ),
            },
            {
                "role": "user",
                "content": f"Geographic coordinate: {cardinal} (Latitude: {coordinate.latitude:+.4f}, Longitude: {coordinate.longitude:+.4f})",
            },
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 40,
    }
    if "qwen" in model.lower() or "deepseek" in model.lower():
        payload["reasoning"] = {"effort": "none"}
    return payload


async def send_single_query(
    client: httpx.AsyncClient,
    model: str,
    coordinate: Coordinate,
    prompt_version: str = "land-water-v1",
    max_retries: int = 5,
) -> Prediction:
    payload = build_chat_payload(model, coordinate)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mahmoodkhordoo/latent-earth",
        "X-Title": "Latent Earth Benchmark",
    }

    t0 = time.monotonic()
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(OPENROUTER_CHAT_URL, json=payload, headers=headers)
            latency = time.monotonic() - t0

            if resp.status_code == 200:
                data = resp.json()
                data["latency_seconds"] = round(latency, 4)
                data["attempt_count"] = attempt

                choice_str = "water"
                confidence = 0.5
                try:
                    content_str = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content_str)
                    choice_str = str(parsed.get("choice", "water")).strip().lower()
                    confidence = float(parsed.get("confidence", 0.5))
                except Exception:
                    # fallback to substring check if JSON parsing fails
                    raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
                    if "land" in raw_text and "water" not in raw_text:
                        choice_str = "land"
                        confidence = 0.7
                    elif "water" in raw_text and "land" not in raw_text:
                        choice_str = "water"
                        confidence = 0.7

                is_land = (choice_str == "land")
                # Map confidence to probability of land
                if is_land:
                    prob_land = max(0.50, min(1.0, confidence))
                else:
                    prob_land = max(0.0, min(0.49, 1.0 - confidence))

                return Prediction(
                    coordinate=coordinate,
                    probability_land=prob_land,
                    is_land=is_land,
                    provider="openrouter",
                    model=model,
                    prompt_version=prompt_version,
                    raw_response=data,
                )

            if resp.status_code in {429, 500, 502, 503, 504}:
                delay = 1.0 * (2 ** (attempt - 1))
                logger.warning("HTTP %d for %s (attempt %d/%d). Sleeping %.1fs...", resp.status_code, coordinate.id, attempt, max_retries, delay)
                await asyncio.sleep(delay)
                continue

            logger.error("HTTP %d error for %s: %s", resp.status_code, coordinate.id, resp.text[:200])
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        except httpx.RequestError as exc:
            delay = 1.0 * (2 ** (attempt - 1))
            logger.warning("Network error %s for %s. Sleeping %.1fs...", exc, coordinate.id, delay)
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed coordinate {coordinate.id} after {max_retries} attempts.")


async def execute_openrouter_queries(
    target_coords: list[Coordinate],
    cached_preds: dict[str, Prediction],
    model: str,
    output_path: Path,
    concurrency: int = 24,
    chunk_size: int = 48,
    timeout_seconds: float = 45.0,
) -> tuple[dict[str, Prediction], dict[str, Any]]:
    remaining = [c for c in target_coords if c.id not in cached_preds]
    logger.info(
        "OpenRouter [%s] Execution: %d target coordinates, %d cached, %d to query.",
        model,
        len(target_coords),
        len(cached_preds),
        len(remaining),
    )

    all_preds = dict(cached_preds)
    latencies: list[float] = []
    tokens_list: list[int] = []
    retries = 0
    failures = 0

    wall_clock_start = time.monotonic()

    if remaining:
        limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)
        timeout = httpx.Timeout(timeout_seconds, connect=10.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]
            sem = asyncio.Semaphore(concurrency)

            async def bounded_query(c):
                async with sem:
                    return await send_single_query(client, model, c)

            for ch_idx, chunk in enumerate(chunks, start=1):
                ch_start = time.monotonic()
                logger.info("Starting chunk %d/%d (%d coordinates)...", ch_idx, len(chunks), len(chunk))
                tasks = [bounded_query(c) for c in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                chunk_preds = []
                for res in results:
                    if isinstance(res, Prediction):
                        chunk_preds.append(res)
                        all_preds[res.coordinate.id] = res
                        raw = res.raw_response or {}
                        if "latency_seconds" in raw:
                            latencies.append(raw["latency_seconds"])
                        usage = raw.get("usage", {})
                        if "total_tokens" in usage:
                            tokens_list.append(usage["total_tokens"])
                    else:
                        logger.error("Query failed: %s", res)
                        failures += 1

                with open(output_path, "a", encoding="utf-8") as f_out:
                    for p in chunk_preds:
                        f_out.write(json.dumps(p.to_dict()) + "\n")

                ch_elapsed = time.monotonic() - ch_start
                rate = len(chunk_preds) / ch_elapsed if ch_elapsed > 0 else 0
                logger.info(
                    "Completed chunk %d/%d in %.1fs (%.2f req/s) | %d/%d complete",
                    ch_idx,
                    len(chunks),
                    ch_elapsed,
                    rate,
                    len(all_preds),
                    len(target_coords),
                )

    wall_clock_total = time.monotonic() - wall_clock_start

    perf_stats = {
        "wall_clock_time_seconds": round(wall_clock_total, 2),
        "queries_executed": len(remaining),
        "queries_cached": len(cached_preds),
        "total_benchmark_points": len(target_coords),
        "effective_throughput_req_per_sec": round(len(remaining) / wall_clock_total, 2) if wall_clock_total > 0 else 0,
        "latency_stats": {
            "median_seconds": round(float(np.median(latencies)), 4) if latencies else None,
            "p90_seconds": round(float(np.percentile(latencies, 90)), 4) if latencies else None,
            "p95_seconds": round(float(np.percentile(latencies, 95)), 4) if latencies else None,
            "p99_seconds": round(float(np.percentile(latencies, 99)), 4) if latencies else None,
            "min_seconds": round(float(np.min(latencies)), 4) if latencies else None,
            "max_seconds": round(float(np.max(latencies)), 4) if latencies else None,
            "sample_count": len(latencies),
        },
        "token_stats": {
            "mean_total_tokens": round(float(np.mean(tokens_list)), 1) if tokens_list else None,
            "median_total_tokens": round(float(np.median(tokens_list)), 1) if tokens_list else None,
            "sample_count": len(tokens_list),
        },
    }

    return all_preds, perf_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-6-luna", help="OpenRouter model ID")
    parser.add_argument("--concurrency", type=int, default=24, help="Concurrent HTTP requests")
    parser.add_argument("--chunk-size", type=int, default=48, help="Batch chunk size")
    args = parser.parse_args()

    model = args.model
    model_slug = model.replace("/", "_").replace(":", "_").replace(".", "_")

    outputs_dir = Path("outputs")
    jev_2112_path = outputs_dir / "jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl"
    out_jsonl = outputs_dir / f"openrouter_{model_slug}_japan_korea_2112.jsonl"

    # Load 2,112 target coordinates
    jev_preds = PredictionCache(jev_2112_path, strict_validation=False).load()
    target_coords = [p.coordinate for p in jev_preds.values()]
    target_ids = set(c.id for c in target_coords)

    cached_preds: dict[str, Prediction] = {}
    if out_jsonl.exists():
        cached_preds = PredictionCache(out_jsonl, strict_validation=False).load()
        logger.info("Found %d existing cached predictions in %s", len(cached_preds), out_jsonl)

    # Run queries
    all_preds, perf_stats = asyncio.run(
        execute_openrouter_queries(
            target_coords=target_coords,
            cached_preds=cached_preds,
            model=model,
            output_path=out_jsonl,
            concurrency=args.concurrency,
            chunk_size=args.chunk_size,
        )
    )

    # Evaluate against Ground Truth
    evaluator = GroundTruthEvaluator()
    preds_list = [all_preds[cid] for cid in target_ids]
    ev = evaluate_predictions(preds_list, evaluator=evaluator, coastal_distance_km=100.0)
    cm = ev["confusion_matrix"]

    # Sea of Japan breakdown (lat 35 to 43, lon 130 to 138, GT water)
    soj_coords = [
        c for c in target_coords
        if 35.0 <= c.latitude <= 43.0 and 130.0 <= c.longitude <= 138.0 and not evaluator.is_land(c.latitude, c.longitude)
    ]
    correct_soj = sum(1 for c in soj_coords if not all_preds[c.id].is_land)
    soj_pct = round(correct_soj / len(soj_coords) * 100.0, 2) if soj_coords else 0

    evaluation_report = {
        "benchmark": "Japan/Korea 2,112-Point Regional Benchmark",
        "provider": "openrouter",
        "model": model,
        "total_evaluated": len(preds_list),
        "grid_metrics": ev["grid_metrics"],
        "area_weighted_metrics": ev["area_weighted_metrics"],
        "coastal_metrics": ev["coastal_metrics"],
        "confusion_matrix": cm,
        "sea_of_japan_inspection": {
            "total_probes": len(soj_coords),
            "correct_water_count": correct_soj,
            "correct_water_pct": soj_pct,
            "false_land_count": len(soj_coords) - correct_soj,
        },
        "performance": perf_stats,
    }

    out_eval_json = outputs_dir / f"openrouter_{model_slug}_japan_korea_2112_evaluation.json"
    with open(out_eval_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("Saved evaluation report to %s", out_eval_json)
    print("\n=======================================================")
    print(f"=== {model} BENCHMARK COMPLETE ===")
    print("=======================================================")
    print(f"Grid Accuracy:         {ev['grid_metrics']['accuracy']:.2%}")
    print(f"Area-Weighted Acc:     {ev['area_weighted_metrics']['accuracy']:.2%}")
    print(f"Brier Score:           {ev['grid_metrics']['brier_score']:.4f}")
    print(f"Coastal Accuracy:      {ev['coastal_metrics']['accuracy']:.2%}")
    print(f"Precision:             {ev['grid_metrics']['precision']:.2%}")
    print(f"Recall:                {ev['grid_metrics']['recall']:.2%}")
    print(f"F1:                    {ev['grid_metrics']['f1']:.2%}")
    print(f"Sea of Japan Water:    {soj_pct:.1f}% ({correct_soj}/{len(soj_coords)})")
    print(f"Wall-Clock Time:       {perf_stats['wall_clock_time_seconds']}s")
    print(f"Median Latency:        {perf_stats['latency_stats']['median_seconds']}s")
    print(f"Throughput:            {perf_stats['effective_throughput_req_per_sec']} req/s")


if __name__ == "__main__":
    main()
