#!/usr/bin/env python3
"""
Benchmark local llama.cpp System One decision models (e.g. Julia-1, Laya, Kev-4B)
on the Japan/Korea 2,112-point regional benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any
import httpx
import numpy as np

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.llama_systemone import LlamaSystemOneProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("local_llama_benchmark")


async def execute_local_benchmark(
    target_coords: list[Coordinate],
    cached_preds: dict[str, Prediction],
    model_name: str,
    output_path: Path,
    base_url: str = "http://127.0.0.1:8080/v1/systemone",
    concurrency: int = 16,
    chunk_size: int = 64,
) -> tuple[dict[str, Prediction], dict[str, Any]]:
    provider = LlamaSystemOneProvider(
        model=model_name,
        base_url=base_url,
        concurrency=concurrency,
        timeout_seconds=30.0,
        max_retries=3,
    )

    remaining = [c for c in target_coords if c.id not in cached_preds]
    logger.info(
        "Local llama.cpp [%s] Execution: %d target coordinates, %d cached, %d to query.",
        model_name,
        len(target_coords),
        len(cached_preds),
        len(remaining),
    )

    all_preds = dict(cached_preds)
    latencies: list[float] = []
    failures = 0

    wall_clock_start = time.monotonic()

    if remaining:
        chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]
        for ch_idx, chunk in enumerate(chunks, start=1):
            ch_start = time.monotonic()
            chunk_results = await provider.predict_coordinates(chunk)

            for pred in chunk_results:
                all_preds[pred.coordinate.id] = pred
                raw = pred.raw_response or {}
                if "latency_seconds" in raw:
                    latencies.append(raw["latency_seconds"])

            with open(output_path, "a", encoding="utf-8") as f_out:
                for pred in chunk_results:
                    f_out.write(json.dumps(pred.to_dict()) + "\n")

            ch_elapsed = time.monotonic() - ch_start
            rate = len(chunk) / ch_elapsed if ch_elapsed > 0 else 0
            logger.info(
                "Completed chunk %d/%d (%d coords) in %.2fs (%.1f req/s) | %d/%d total",
                ch_idx,
                len(chunks),
                len(chunk),
                ch_elapsed,
                rate,
                len(all_preds),
                len(target_coords),
            )

    wall_clock_total = time.monotonic() - wall_clock_start
    await provider.close()

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
    }

    return all_preds, perf_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Julia-1-144M", help="Model name label")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080/v1/systemone", help="llama.cpp systemone URL")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent HTTP requests")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size")
    args = parser.parse_args()

    model_slug = args.model.lower().replace("/", "_").replace("-", "_").replace(".", "_")

    outputs_dir = Path("outputs")
    jev_2112_path = outputs_dir / "jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl"
    out_jsonl = outputs_dir / f"local_llama_{model_slug}_japan_korea_2112.jsonl"

    jev_preds = PredictionCache(jev_2112_path, strict_validation=False).load()
    target_coords = [p.coordinate for p in jev_preds.values()]
    target_ids = set(c.id for c in target_coords)

    cached_preds: dict[str, Prediction] = {}
    if out_jsonl.exists():
        cached_preds = PredictionCache(out_jsonl, strict_validation=False).load()
        logger.info("Found %d existing cached predictions in %s", len(cached_preds), out_jsonl)

    all_preds, perf_stats = asyncio.run(
        execute_local_benchmark(
            target_coords=target_coords,
            cached_preds=cached_preds,
            model_name=args.model,
            output_path=out_jsonl,
            base_url=args.url,
            concurrency=args.concurrency,
            chunk_size=args.chunk_size,
        )
    )

    evaluator = GroundTruthEvaluator()
    preds_list = [all_preds[cid] for cid in target_ids]
    ev = evaluate_predictions(preds_list, evaluator=evaluator, coastal_distance_km=100.0)
    cm = ev["confusion_matrix"]

    soj_coords = [
        c for c in target_coords
        if 35.0 <= c.latitude <= 43.0 and 130.0 <= c.longitude <= 138.0 and not evaluator.is_land(c.latitude, c.longitude)
    ]
    correct_soj = sum(1 for c in soj_coords if not all_preds[c.id].is_land)
    soj_pct = round(correct_soj / len(soj_coords) * 100.0, 2) if soj_coords else 0

    evaluation_report = {
        "benchmark": "Japan/Korea 2,112-Point Regional Benchmark",
        "provider": "local_llama_cpp",
        "model": args.model,
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

    out_eval_json = outputs_dir / f"local_llama_{model_slug}_japan_korea_2112_evaluation.json"
    with open(out_eval_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("Saved evaluation report to %s", out_eval_json)
    print("\n=======================================================")
    print(f"=== LOCAL LLAMA.CPP [{args.model}] BENCHMARK COMPLETE ===")
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
