#!/usr/bin/env python3
"""
Benchmark OpenAI Decisions API (gpt-6-luna) on the full Global 16,200-point planetary benchmark (2.0°).
Queries https://api.openai.com/v1/decisions with high concurrency and incremental disk persistence.
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dotenv import load_dotenv

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions
from latent_earth.models import Coordinate, Prediction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("openai_global_decision_benchmark")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment.")
    sys.exit(1)

MODEL_NAME = "gpt-6-luna"
DECISIONS_URL = "https://api.openai.com/v1/decisions"
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_RUNS_DIR = Path("frontend/public/data/runs")
FRONTEND_RUNS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "openai_gpt-6-luna-decision_global_16200"
JSONL_PATH = OUTPUTS_DIR / f"{RUN_ID}.jsonl"
EVAL_PATH = OUTPUTS_DIR / f"{RUN_ID}_evaluation.json"
META_PATH = OUTPUTS_DIR / f"{RUN_ID}_metadata.json"
FIGURE_PATH = OUTPUTS_DIR / f"{RUN_ID}_comparison.png"
FRONTEND_JSON_PATH = FRONTEND_RUNS_DIR / f"{RUN_ID}.json"


def build_payload(coordinate: Coordinate) -> dict[str, Any]:
    cardinal = coordinate.format_cardinal()
    return {
        "model": MODEL_NAME,
        "input": f"Geographic coordinate: {cardinal} (Latitude: {coordinate.latitude:+.4f}, Longitude: {coordinate.longitude:+.4f})",
        "questions": [
            {
                "type": "predicate",
                "name": "is_land",
                "instructions": "Is this geographic coordinate located on land rather than water?",
            }
        ],
    }


async def send_single_query(
    client: httpx.AsyncClient,
    coordinate: Coordinate,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> Prediction:
    payload = build_payload(coordinate)
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
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

                answers = data.get("answers", [])
                prob = 0.5
                if answers and len(answers) > 0:
                    prob = answers[0].get("probability", 0.5)

                prob_float = float(prob)
                is_land = prob_float >= 0.5

                return Prediction(
                    coordinate=coordinate,
                    probability_land=prob_float,
                    is_land=is_land,
                    provider="openai",
                    model="gpt-6-luna-decision",
                    prompt_version="land-water-v1",
                    raw_response=data,
                )

            if resp.status_code in {429, 500, 502, 503, 504}:
                delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0.05, 0.2)
                logger.warning("HTTP %d for %s (attempt %d/%d). Sleeping %.2fs...", resp.status_code, coordinate.id, attempt, max_retries, delay)
                await asyncio.sleep(delay)
                continue

            logger.error("HTTP %d error for %s: %s", resp.status_code, coordinate.id, resp.text[:200])
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        except httpx.RequestError as exc:
            delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0.05, 0.2)
            logger.warning("Network error %s for %s. Sleeping %.2fs...", type(exc).__name__, coordinate.id, delay)
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed coordinate {coordinate.id} after {max_retries} attempts.")


async def run_benchmark():
    # 1. Load frozen global 16,200 coordinates
    jev_path = OUTPUTS_DIR / "jev_jev-latest_2deg_land-water-v1.jsonl"
    if not jev_path.exists():
        logger.error("Reference JEV file not found at %s", jev_path)
        sys.exit(1)

    jev_cache = PredictionCache(jev_path, strict_validation=False).load()
    target_coords = [p.coordinate for p in jev_cache.values()]
    logger.info("Loaded %d target global coordinates from frozen JEV benchmark.", len(target_coords))

    # 2. Check existing cached predictions
    cached_preds: dict[str, Prediction] = {}
    if JSONL_PATH.exists():
        try:
            cached_preds = PredictionCache(JSONL_PATH, strict_validation=False).load()
            logger.info("Found %d existing cached predictions in %s", len(cached_preds), JSONL_PATH)
        except Exception as e:
            logger.warning("Could not load partial cache (%s), starting fresh", e)

    remaining = [c for c in target_coords if c.id not in cached_preds]
    logger.info(
        "Benchmarking Global 16,200 OpenAI Decisions API (%s): %d total, %d cached, %d remaining.",
        MODEL_NAME, len(target_coords), len(cached_preds), len(remaining)
    )

    all_preds: dict[str, Prediction] = dict(cached_preds)
    latencies: list[float] = []
    wall_clock_start = time.monotonic()

    concurrency = 36
    chunk_size = 72

    if remaining:
        limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)
        timeout = httpx.Timeout(30.0, connect=10.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            sem = asyncio.Semaphore(concurrency)
            chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]

            for ch_idx, chunk in enumerate(chunks, start=1):
                ch_start = time.monotonic()
                tasks = [send_single_query(client, coord, sem) for coord in chunk]
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
                with open(JSONL_PATH, "a", encoding="utf-8") as f_out:
                    for pred in valid_preds:
                        f_out.write(json.dumps(pred.to_dict()) + "\n")

                ch_elapsed = time.monotonic() - ch_start
                rate = len(valid_preds) / ch_elapsed if ch_elapsed > 0 else 0
                pct = len(all_preds) / len(target_coords) * 100
                total_elapsed = time.monotonic() - wall_clock_start
                est_rem = (total_elapsed / len(all_preds) * (len(target_coords) - len(all_preds))) if len(all_preds) > 0 else 0

                if ch_idx % 10 == 0 or ch_idx == len(chunks):
                    logger.info(
                        "Chunk %d/%d (%.1f%%) | Rate: %.1f req/s | Total: %d/%d | Elapsed: %.1fm | ETA: %.1fm",
                        ch_idx, len(chunks), pct, rate, len(all_preds), len(target_coords),
                        total_elapsed / 60.0, est_rem / 60.0
                    )

    wall_total = time.monotonic() - wall_clock_start
    logger.info("Completed all queries in %.2fs. Total evaluated: %d.", wall_total, len(all_preds))

    # Save metadata JSON
    metadata = {
        "run_id": RUN_ID,
        "provider": "openai",
        "model": "gpt-6-luna-decision",
        "prompt_version": "land-water-v1",
        "region_name": "global",
        "grid_resolution": 2.0,
        "step_degrees": 2.0,
        "point_count": len(all_preds),
        "wall_clock_seconds": round(wall_total, 2),
    }
    with open(META_PATH, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    # 3. Ground Truth Evaluation
    evaluator = GroundTruthEvaluator()
    ordered_preds = [all_preds[c.id] for c in target_coords if c.id in all_preds]

    eval_results = evaluate_predictions(ordered_preds, evaluator, coastal_distance_km=100.0)

    # Calculate Balanced Accuracy
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
        "provider": "openai",
        "model": "gpt-6-luna-decision",
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
        "coastal_metrics": {
            "coastal_distance_threshold_km": 100.0,
            "coastal_cell_count": eval_results.get("coastal_cell_count", 0),
            "coastal_cell_fraction": eval_results.get("coastal_cell_fraction", 0.0),
            "accuracy": coastal_m.get("accuracy"),
            "brier_score": coastal_m.get("brier_score"),
            "area_weighted_accuracy": coastal_m.get("area_weighted_accuracy"),
            "area_weighted_brier_score": coastal_m.get("area_weighted_brier_score"),
        },
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

    with open(EVAL_PATH, "w", encoding="utf-8") as f_eval:
        json.dump(full_eval, f_eval, indent=2)

    logger.info("Saved evaluation JSON to %s", EVAL_PATH)

    # 4. Generate Frontend Static JSON
    frontend_points = []
    for pred in ordered_preds:
        frontend_points.append(pred.to_dict())

    frontend_bundle = {
        "run_id": RUN_ID,
        "provider": "openai",
        "model": "gpt-6-luna-decision",
        "prompt_version": "land-water-v1",
        "step_degrees": 2.0,
        "convention": "cardinal",
        "total_coordinates": len(target_coords),
        "sampled_count": len(ordered_preds),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "region": {
            "lat_min": -90.0,
            "lat_max": 90.0,
            "lon_min": -180.0,
            "lon_max": 180.0,
            "name": "global",
        },
        "metrics": grid_m,
        "grid_metrics": grid_m,
        "area_weighted_metrics": area_m,
        "coastal_metrics": {
            "coastal_distance_threshold_km": 100.0,
            "coastal_distance_threshold_deg": 0.901,
            "coastal_cell_count": eval_results.get("coastal_cell_count", 0),
            "coastal_cell_fraction": eval_results.get("coastal_cell_fraction", 0.0),
            "accuracy": coastal_m.get("accuracy"),
            "brier_score": coastal_m.get("brier_score"),
            "area_weighted_accuracy": coastal_m.get("area_weighted_accuracy"),
            "area_weighted_brier_score": coastal_m.get("area_weighted_brier_score"),
        },
        "points": frontend_points,
    }

    with open(FRONTEND_JSON_PATH, "w", encoding="utf-8") as f_front:
        json.dump(frontend_bundle, f_front)
    logger.info("Exported frontend bundle to %s (size: %.1f MB)", FRONTEND_JSON_PATH, FRONTEND_JSON_PATH.stat().st_size / 1e6)

    # 5. Generate Comparison Plot
    generate_global_plot(ordered_preds, evaluator, full_eval, FIGURE_PATH)

    # 6. Print Summary
    print("\n" + "=" * 70)
    print("OPENAI DECISIONS API (GPT-6 LUNA) — GLOBAL 16,200 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Overall Grid Accuracy:    {grid_m['accuracy']:.2%}")
    print(f"Balanced Accuracy:        {balanced_acc:.2%}")
    print(f"Area-Weighted Accuracy:   {area_m.get('accuracy', 0):.2%}")
    print(f"Coastal Accuracy (≤100km): {coastal_m.get('accuracy', 0):.2%}")
    print(f"Brier Score:              {grid_m.get('brier_score', 0):.4f}")
    print(f"Precision:                {grid_m.get('precision', 0):.2%}")
    print(f"Recall:                   {grid_m.get('recall', 0):.2%}")
    print(f"F1 Score:                 {grid_m.get('f1', 0):.4f}")
    print(f"Wall Clock Time:          {wall_total:.1f}s ({wall_total/60.0:.2f} min)")
    if latencies:
        print(f"Median Latency:           {np.median(latencies):.3f}s")
        print(f"P90 Latency:              {np.percentile(latencies, 90):.3f}s")
    print("=" * 70 + "\n")


def generate_global_plot(
    preds: list[Prediction],
    evaluator: GroundTruthEvaluator,
    eval_dict: dict[str, Any],
    out_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.0), facecolor="#0a0e17")

    sorted_coords = [p.coordinate for p in preds]
    lats = [c.latitude for c in sorted_coords]
    lons = [c.longitude for c in sorted_coords]
    gt_land = [evaluator.is_land(c.latitude, c.longitude) for c in sorted_coords]
    pred_probs = [p.probability_land if p.probability_land is not None else 0.5 for p in preds]
    pred_is_land = [p.is_land for p in preds]

    color_correct = "#10b981"
    color_wrong = "#ef4444"

    # Panel 1: Probability Map
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    cmap = plt.cm.plasma
    sc1 = ax1.scatter(lons, lats, c=pred_probs, cmap=cmap, vmin=0.0, vmax=1.0, s=4, alpha=0.9, edgecolors="none")
    cbar = plt.colorbar(sc1, ax=ax1, orientation="horizontal", pad=0.08, shrink=0.7)
    cbar.set_label("Predicted P(Land)", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

    grid_acc = eval_dict["grid_metrics"]["accuracy"]
    bal_acc = eval_dict.get("balanced_accuracy", 0)
    area_acc = eval_dict["area_weighted_metrics"].get("accuracy", 0)
    brier = eval_dict["grid_metrics"].get("brier_score", 0)

    ax1.set_title(
        f"OpenAI Decisions API (gpt-6-luna)\nGlobal Planetary Probability Map (16,200 points)",
        fontsize=13, fontweight="bold", pad=10, color="white",
    )
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")

    # Panel 2: Error Classification Map
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    correct_mask = [pred_is_land[i] == gt_land[i] for i in range(len(sorted_coords))]
    c_err = [color_correct if corr else color_wrong for corr in correct_mask]
    ax2.scatter(lons, lats, c=c_err, s=4, alpha=0.9, edgecolors="none")

    ax2.set_title(
        f"Classification Performance\nAcc: {grid_acc:.1%} | Area-Weighted: {area_acc:.1%} | Bal Acc: {bal_acc:.1%} | Brier: {brier:.4f}",
        fontsize=13, fontweight="bold", pad=10, color="white",
    )
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    ax2.legend(
        handles=[mpatches.Patch(color=color_correct, label="Correct"), mpatches.Patch(color=color_wrong, label="Error")],
        loc="lower left", facecolor="#1f2937", edgecolor="#374151", fontsize=9,
    )

    plt.suptitle("OpenAI Decisions API (gpt-6-luna) on Global 16,200 Grid (2.0° Planetary Resolution)", fontsize=16, fontweight="bold", y=0.98, color="white")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
