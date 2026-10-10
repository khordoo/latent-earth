#!/usr/bin/env python3
"""
Benchmark Perplexity pplx-decider-v1.1-27b on the full Japan/Korea 2,112-point regional benchmark (0.5°).
Queries OpenRouter Decisions API with high concurrency and incremental disk persistence.
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
logger = logging.getLogger("pplx_decider_benchmark")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY not found in environment.")
    sys.exit(1)

MODEL_NAME = "perplexity/pplx-decider-v1.1-27b"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "openrouter_perplexity_pplx-decider-v1.1-27b_japan_korea_2112"
JSONL_PATH = OUTPUTS_DIR / f"{RUN_ID}.jsonl"
EVAL_PATH = OUTPUTS_DIR / f"{RUN_ID}_evaluation.json"
META_PATH = OUTPUTS_DIR / f"{RUN_ID}_metadata.json"
FIGURE_PATH = OUTPUTS_DIR / f"{RUN_ID}_comparison.png"


def build_payload(coordinate: Coordinate) -> dict[str, Any]:
    cardinal = coordinate.format_cardinal()
    state = (
        f"Geographic coordinate: {cardinal} "
        f"(Latitude: {coordinate.latitude:+.4f}, Longitude: {coordinate.longitude:+.4f})"
    )
    return {
        "model": MODEL_NAME,
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
    coordinate: Coordinate,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> Prediction:
    payload = build_payload(coordinate)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/latent-earth",
        "X-Title": "Latent Earth Benchmark",
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
                actual_model = data.get("model", MODEL_NAME)

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


async def run_benchmark():
    # 1. Load target 2,112 coordinates from frozen JEV benchmark
    jev_path = OUTPUTS_DIR / "jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl"
    if not jev_path.exists():
        logger.error("Reference JEV file not found at %s", jev_path)
        sys.exit(1)

    jev_cache = PredictionCache(jev_path, strict_validation=False).load()
    target_coords = [p.coordinate for p in jev_cache.values()]
    logger.info("Loaded %d target coordinates from frozen JEV benchmark.", len(target_coords))

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
        "Benchmarking %s: %d total, %d cached, %d remaining.",
        MODEL_NAME, len(target_coords), len(cached_preds), len(remaining)
    )

    all_preds: dict[str, Prediction] = dict(cached_preds)
    latencies: list[float] = []
    wall_clock_start = time.monotonic()

    concurrency = 24
    chunk_size = 48

    if remaining:
        limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)
        timeout = httpx.Timeout(45.0, connect=10.0)

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
                logger.info(
                    "Chunk %d/%d (%d coords) in %.1fs (%.1f req/s) | Total done: %d/%d",
                    ch_idx, len(chunks), len(valid_preds), ch_elapsed, rate, len(all_preds), len(target_coords)
                )

    wall_total = time.monotonic() - wall_clock_start
    logger.info("Completed all queries in %.2fs. Total evaluated: %d.", wall_total, len(all_preds))

    # Save metadata JSON for the frontend
    metadata = {
        "run_id": RUN_ID,
        "provider": "openrouter",
        "model": MODEL_NAME,
        "prompt_version": "land-water-v1",
        "region_name": "japan_korea",
        "grid_resolution": 0.5,
        "step_degrees": 0.5,
        "point_count": len(all_preds),
        "wall_clock_seconds": round(wall_total, 2),
    }
    with open(META_PATH, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    # 3. Ground Truth Evaluation
    evaluator = GroundTruthEvaluator()
    ordered_preds = [all_preds[c.id] for c in target_coords if c.id in all_preds]

    eval_results = evaluate_predictions(ordered_preds, evaluator, coastal_distance_km=100.0)

    # Calculate Sea of Japan deep water probes (226 points)
    # Sea of Japan bounds: lat 37.0 to 42.0, lon 130.0 to 138.0
    soj_coords = [
        c for c in target_coords
        if 37.0 <= c.latitude <= 42.0 and 130.0 <= c.longitude <= 138.0 and not evaluator.is_land(c.latitude, c.longitude)
    ]
    soj_correct_water = sum(1 for c in soj_coords if not all_preds[c.id].is_land)
    soj_false_land = len(soj_coords) - soj_correct_water

    # Calculate Balanced Accuracy
    cm = eval_results["confusion_matrix"]
    tp = cm["true_positive_land"]
    tn = cm["true_negative_water"]
    fp = cm["false_positive_land"]
    fn = cm["false_negative_water"]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0

    # Japan Land Recovery (lat 30-46, lon 128-146)
    japan_land_coords = [
        c for c in target_coords
        if 30.0 <= c.latitude <= 46.0 and 128.0 <= c.longitude <= 146.0 and evaluator.is_land(c.latitude, c.longitude)
    ]
    japan_land_hits = sum(1 for c in japan_land_coords if all_preds[c.id].is_land)
    japan_land_recall = japan_land_hits / len(japan_land_coords) if japan_land_coords else 0.0

    grid_m = eval_results["metrics"]
    area_m = eval_results.get("area_weighted_metrics", {})
    coastal_m = eval_results.get("coastal_metrics", {})

    full_eval = {
        "benchmark": "Japan/Korea 2,112-Point Regional Benchmark",
        "provider": "openrouter",
        "model": MODEL_NAME,
        "total_evaluated": len(ordered_preds),
        "balanced_accuracy": round(balanced_acc, 4),
        "japan_land_recovery": {
            "total_japan_land_points": len(japan_land_coords),
            "correct_predicted_land": japan_land_hits,
            "recovery_rate": round(japan_land_recall, 4),
        },
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
        "sea_of_japan_inspection": {
            "total_probes": len(soj_coords),
            "correct_water_count": soj_correct_water,
            "correct_water_pct": round(soj_correct_water / len(soj_coords) * 100, 2) if soj_coords else 0.0,
            "false_land_count": soj_false_land,
        },
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

    # 4. Render 4-Panel Visualization Plot
    render_comparison_figure(target_coords, all_preds, evaluator, full_eval, FIGURE_PATH)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETED: {MODEL_NAME}")
    print("=" * 60)
    print(f"Grid Accuracy:         {grid_m['accuracy'] * 100:.2f}%")
    print(f"Area-Weighted Acc:     {area_m.get('accuracy', 0) * 100:.2f}%")
    print(f"Balanced Accuracy:     {balanced_acc * 100:.2f}%")
    print(f"Brier Score:           {grid_m.get('brier_score', 0):.4f}")
    print(f"Area-Weighted Brier:   {area_m.get('brier_score', 0):.4f}")
    print(f"Coastal Acc (≤100km):  {coastal_m.get('accuracy', 0) * 100:.2f}%")
    print(f"Precision / Recall:    {grid_m['precision'] * 100:.2f}% / {grid_m['recall'] * 100:.2f}% (F1: {grid_m['f1']:.3f})")
    print(f"True Positive Land:    {tp} / {tp + fn} ({tpr * 100:.1f}%)")
    print(f"True Negative Water:   {tn} / {tn + fp} ({tnr * 100:.1f}%)")
    print(f"Sea of Japan False Land: {soj_false_land} / {len(soj_coords)} ({soj_correct_water / len(soj_coords) * 100:.1f}% correct water)")
    print(f"Japan Land Recovery:   {japan_land_hits} / {len(japan_land_coords)} ({japan_land_recall * 100:.1f}%)")
    print("=" * 60 + "\n")


def render_comparison_figure(
    target_coords: list[Coordinate],
    preds: dict[str, Prediction],
    evaluator: GroundTruthEvaluator,
    eval_dict: dict[str, Any],
    out_path: Path,
):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    sorted_coords = sorted(target_coords, key=lambda c: (c.latitude, c.longitude))
    lats = [c.latitude for c in sorted_coords]
    lons = [c.longitude for c in sorted_coords]
    gt_land = [evaluator.is_land(c.latitude, c.longitude) for c in sorted_coords]
    pred_probs = [preds[c.id].probability_land for c in sorted_coords]
    pred_is_land = [preds[c.id].is_land for c in sorted_coords]

    color_water = "#1e3a8a"
    color_land = "#15803d"
    color_wrong = "#ef4444"
    color_correct = "#10b981"

    # Panel 1: Ground Truth
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    c_gt = [color_land if is_l else color_water for is_l in gt_land]
    ax1.scatter(lons, lats, c=c_gt, s=24, alpha=0.9, edgecolors="none")
    ax1.set_title("Ground Truth (10m Land Polygons)\n2,112 Points (0.5° Resolution)", fontsize=13, fontweight="bold", pad=10, color="white")
    ax1.set_xlim(121.5, 146.5)
    ax1.set_ylim(23.5, 46.5)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    ax1.legend(
        handles=[mpatches.Patch(color=color_land, label="Land (489)"), mpatches.Patch(color=color_water, label="Water (1,623)")],
        loc="lower right", facecolor="#1f2937", edgecolor="#374151", fontsize=9,
    )

    # Panel 2: Continuous Probability Prediction
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    cmap = plt.cm.plasma
    sc2 = ax2.scatter(lons, lats, c=pred_probs, cmap=cmap, vmin=0.0, vmax=1.0, s=24, alpha=0.9, edgecolors="none")
    cbar = plt.colorbar(sc2, ax=ax2, orientation="horizontal", pad=0.08, shrink=0.7)
    cbar.set_label("Predicted P(Land)", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

    ax2.set_title(
        f"Perplexity Pplx-Decider (Qwen3.8-27B)\nContinuous Probability Map (P ≥ 0.5 = Land)",
        fontsize=13, fontweight="bold", pad=10, color="white",
    )
    ax2.set_xlim(121.5, 146.5)
    ax2.set_ylim(23.5, 46.5)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")

    # Panel 3: Error Classification Map
    ax3 = axes[2]
    ax3.set_facecolor("#111827")
    correct_mask = [pred_is_land[i] == gt_land[i] for i in range(len(sorted_coords))]
    c_err = [color_correct if corr else color_wrong for corr in correct_mask]
    ax3.scatter(lons, lats, c=c_err, s=24, alpha=0.9, edgecolors="none")
    grid_acc = eval_dict["grid_metrics"]["accuracy"]
    brier = eval_dict["grid_metrics"].get("brier_score", 0)
    bal_acc = eval_dict.get("balanced_accuracy", 0)
    ax3.set_title(
        f"Classification Performance\nAcc: {grid_acc:.1%} | Bal Acc: {bal_acc:.1%} | Brier: {brier:.4f}",
        fontsize=13, fontweight="bold", pad=10, color="white",
    )
    ax3.set_xlim(121.5, 146.5)
    ax3.set_ylim(23.5, 46.5)
    ax3.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    ax3.legend(
        handles=[mpatches.Patch(color=color_correct, label="Correct"), mpatches.Patch(color=color_wrong, label="Error")],
        loc="lower right", facecolor="#1f2937", edgecolor="#374151", fontsize=9,
    )

    plt.suptitle("Perplexity pplx-decider-v1.1-27b on Japan & Korea Regional Grid (2,112 Coords, 0.5°)", fontsize=16, fontweight="bold", y=0.98, color="white")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
