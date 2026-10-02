#!/usr/bin/env python3
"""
Stratified 512-Coordinate Benchmark Execution & Comparative Analysis:
JEV vs. Liquid D1 vs. Fastino GLiDE on identical Japan/Korea coordinates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.fastino_glide import FastinoGlideProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stratified_512_benchmark")


async def execute_glide_queries(
    target_coords: list[Coordinate],
    cached_preds: dict[str, Prediction],
    stratified_output_path: Path,
    full_cache_output_path: Path,
    concurrency: int = 12,
    timeout_seconds: float = 90.0,
) -> tuple[dict[str, Prediction], dict[str, Any]]:
    """Run GLiDE on missing coordinates with concurrency, retry tracking, and incremental persistence."""
    provider = FastinoGlideProvider(
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        max_retries=5,
    )

    remaining = [c for c in target_coords if c.id not in cached_preds]
    logger.info(
        "GLiDE Execution: %d total target coordinates, %d already cached, %d to query.",
        len(target_coords),
        len(cached_preds),
        len(remaining),
    )

    all_preds = dict(cached_preds)
    latencies: list[float] = []
    output_tokens_list: list[int] = []
    retries_count = 0
    failures_count = 0

    wall_clock_start = time.monotonic()

    if remaining:
        # Query in chunks of 16 for incremental file flushes
        chunk_size = 16
        chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]

        for ch_idx, chunk in enumerate(chunks, start=1):
            ch_start = time.monotonic()
            logger.info("Starting chunk %d/%d (%d coordinates)...", ch_idx, len(chunks), len(chunk))
            try:
                # Process chunk concurrently
                chunk_results = await provider.predict_coordinates(chunk)
                for pred in chunk_results:
                    all_preds[pred.coordinate.id] = pred
                    raw = pred.raw_response or {}
                    if "latency_seconds" in raw:
                        latencies.append(float(raw["latency_seconds"]))
                    if "attempt_count" in raw and raw["attempt_count"] > 1:
                        retries_count += (raw["attempt_count"] - 1)
                    usage = raw.get("usage", {})
                    out_tok = usage.get("output_tokens")
                    if out_tok is not None:
                        output_tokens_list.append(int(out_tok))

                # Append to both stratified output and full cache output
                with open(stratified_output_path, "a", encoding="utf-8") as f_strat:
                    for pred in chunk_results:
                        f_strat.write(json.dumps(pred.to_dict()) + "\n")

                with open(full_cache_output_path, "a", encoding="utf-8") as f_full:
                    for pred in chunk_results:
                        f_full.write(json.dumps(pred.to_dict()) + "\n")

                ch_elapsed = time.monotonic() - ch_start
                logger.info(
                    "Completed chunk %d/%d in %.1fs (%d/%d total complete)",
                    ch_idx,
                    len(chunks),
                    ch_elapsed,
                    len(all_preds),
                    len(target_coords),
                )
            except Exception as exc:
                failures_count += len(chunk)
                logger.error("Chunk %d/%d failed with error: %s", ch_idx, len(chunks), exc, exc_info=True)
                raise exc

    wall_clock_total = time.monotonic() - wall_clock_start
    await provider.close()

    # Collect existing latencies / tokens from cached predictions if present
    for cid, p in cached_preds.items():
        raw = p.raw_response or {}
        if "latency_seconds" in raw and raw["latency_seconds"]:
            latencies.append(float(raw["latency_seconds"]))
        usage = raw.get("usage", {})
        out_tok = usage.get("output_tokens")
        if out_tok is not None:
            output_tokens_list.append(int(out_tok))

    perf_stats = {
        "wall_clock_time_seconds": round(wall_clock_total, 2),
        "queries_executed": len(remaining),
        "queries_cached": len(cached_preds),
        "total_benchmark_points": len(target_coords),
        "retries_count": retries_count,
        "failures_count": failures_count,
        "latency_stats": {
            "median_seconds": round(float(np.median(latencies)), 4) if latencies else None,
            "p90_seconds": round(float(np.percentile(latencies, 90)), 4) if latencies else None,
            "p95_seconds": round(float(np.percentile(latencies, 95)), 4) if latencies else None,
            "p99_seconds": round(float(np.percentile(latencies, 99)), 4) if latencies else None,
            "min_seconds": round(float(np.min(latencies)), 4) if latencies else None,
            "max_seconds": round(float(np.max(latencies)), 4) if latencies else None,
            "sample_count": len(latencies),
        },
        "thinking_token_stats": {
            "mean_tokens": round(float(np.mean(output_tokens_list)), 1) if output_tokens_list else None,
            "median_tokens": round(float(np.median(output_tokens_list)), 1) if output_tokens_list else None,
            "p90_tokens": round(float(np.percentile(output_tokens_list, 90)), 1) if output_tokens_list else None,
            "p95_tokens": round(float(np.percentile(output_tokens_list, 95)), 1) if output_tokens_list else None,
            "sample_count": len(output_tokens_list),
        },
    }

    return all_preds, perf_stats


def render_stratified_figure(
    coords_dict: dict[str, dict[str, Any]],
    jev_preds: dict[str, Prediction],
    d1_preds: dict[str, Prediction],
    glide_preds: dict[str, Prediction],
    evaluator: GroundTruthEvaluator,
    metrics: dict[str, Any],
    out_path: Path,
):
    """Render a 4-panel visual comparison on the 512 stratified coordinates."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    sorted_ids = sorted(coords_dict.keys(), key=lambda cid: (coords_dict[cid]["latitude"], coords_dict[cid]["longitude"]))
    lats = [coords_dict[cid]["latitude"] for cid in sorted_ids]
    lons = [coords_dict[cid]["longitude"] for cid in sorted_ids]
    gt_land = [coords_dict[cid]["is_land"] for cid in sorted_ids]

    color_water = "#1e3a8a"  # blue
    color_land = "#15803d"   # green
    color_wrong = "#ef4444"  # red
    color_correct = "#10b981" # green

    # Panel 1: Ground Truth Strata
    ax1 = axes[0, 0]
    ax1.set_facecolor("#111827")
    strata_colors = {
        "coastal_land": "#22c55e",
        "interior_land": "#15803d",
        "coastal_water": "#38bdf8",
        "open_water": "#1d4ed8",
    }
    c_list = [strata_colors[coords_dict[cid]["stratum"]] for cid in sorted_ids]
    ax1.scatter(lons, lats, c=c_list, s=36, alpha=0.9, edgecolors="none")
    ax1.set_title("Ground Truth (Natural Earth 10m) - 512 Stratified Probes", fontsize=13, fontweight="bold", pad=10, color="white")
    ax1.set_xlim(121.5, 146.5)
    ax1.set_ylim(23.5, 46.5)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    patches1 = [
        mpatches.Patch(color=strata_colors["coastal_land"], label="Coastal Land (71)"),
        mpatches.Patch(color=strata_colors["interior_land"], label="Interior Land (48)"),
        mpatches.Patch(color=strata_colors["coastal_water"], label="Coastal Water (169)"),
        mpatches.Patch(color=strata_colors["open_water"], label="Open Water (224)"),
    ]
    ax1.legend(handles=patches1, loc="lower right", facecolor="#1f2937", edgecolor="#374151", fontsize=9)

    # Panel 2: JEV Error Map
    ax2 = axes[0, 1]
    ax2.set_facecolor("#111827")
    jev_correct = [jev_preds[cid].is_land == gt_land[idx] for idx, cid in enumerate(sorted_ids)]
    c_jev = [color_correct if corr else color_wrong for corr in jev_correct]
    ax2.scatter(lons, lats, c=c_jev, s=36, alpha=0.9, edgecolors="none")
    m_jev = metrics["jev"]
    ax2.set_title(
        f"JEV (jev-latest)\nAcc: {m_jev['grid_accuracy']:.1%} | Brier: {m_jev['grid_brier']:.3f} | Coastal Acc: {m_jev['coastal_accuracy']:.1%}",
        fontsize=12, fontweight="bold", pad=10, color="white",
    )
    ax2.set_xlim(121.5, 146.5)
    ax2.set_ylim(23.5, 46.5)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    ax2.legend(
        handles=[mpatches.Patch(color=color_correct, label="Correct"), mpatches.Patch(color=color_wrong, label="Error (Land Inflation)")],
        loc="lower right", facecolor="#1f2937", edgecolor="#374151", fontsize=9,
    )

    # Panel 3: Liquid D1 Error Map
    ax3 = axes[1, 0]
    ax3.set_facecolor("#111827")
    d1_correct = [d1_preds[cid].is_land == gt_land[idx] for idx, cid in enumerate(sorted_ids)]
    c_d1 = [color_correct if corr else color_wrong for corr in d1_correct]
    ax3.scatter(lons, lats, c=c_d1, s=36, alpha=0.9, edgecolors="none")
    m_d1 = metrics["liquid_d1"]
    ax3.set_title(
        f"Liquid D1 (liquid/d1)\nAcc: {m_d1['grid_accuracy']:.1%} | Brier: {m_d1['grid_brier']:.3f} | Coastal Acc: {m_d1['coastal_accuracy']:.1%}",
        fontsize=12, fontweight="bold", pad=10, color="white",
    )
    ax3.set_xlim(121.5, 146.5)
    ax3.set_ylim(23.5, 46.5)
    ax3.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    ax3.legend(
        handles=[mpatches.Patch(color=color_correct, label="Correct"), mpatches.Patch(color=color_wrong, label="Error")],
        loc="lower right", facecolor="#1f2937", edgecolor="#374151", fontsize=9,
    )

    # Panel 4: Fastino GLiDE Error Map
    ax4 = axes[1, 1]
    ax4.set_facecolor("#111827")
    glide_correct = [glide_preds[cid].is_land == gt_land[idx] for idx, cid in enumerate(sorted_ids)]
    c_glide = [color_correct if corr else color_wrong for corr in glide_correct]
    ax4.scatter(lons, lats, c=c_glide, s=36, alpha=0.9, edgecolors="none")
    m_glide = metrics["fastino_glide"]
    ax4.set_title(
        f"Fastino GLiDE (fastino/glide)\nAcc: {m_glide['grid_accuracy']:.1%} | Brier: {m_glide['grid_brier']:.3f} | Coastal Acc: {m_glide['coastal_accuracy']:.1%}",
        fontsize=12, fontweight="bold", pad=10, color="white",
    )
    ax4.set_xlim(121.5, 146.5)
    ax4.set_ylim(23.5, 46.5)
    ax4.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    ax4.legend(
        handles=[mpatches.Patch(color=color_correct, label="Correct"), mpatches.Patch(color=color_wrong, label="Error")],
        loc="lower right", facecolor="#1f2937", edgecolor="#374151", fontsize=9,
    )

    plt.suptitle("512-Point Stratified Benchmark: Japan & Korea (0.5° Resolution)", fontsize=16, fontweight="bold", y=0.98, color="white")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved stratified benchmark figure to %s", out_path)


def run_full_stratified_evaluation():
    outputs_dir = Path("outputs")
    coords_json_path = outputs_dir / "stratified_512_coordinates.json"
    with open(coords_json_path) as f:
        meta_data = json.load(f)
    coords_dict = meta_data["coordinates"]

    target_coords = [
        Coordinate(latitude=c["latitude"], longitude=c["longitude"])
        for cid, c in coords_dict.items()
    ]
    target_ids = set(coords_dict.keys())

    # File paths
    jev_512_path = outputs_dir / "jev_japan_korea_stratified_512.jsonl"
    d1_512_path = outputs_dir / "openrouter_liquid_d1_japan_korea_stratified_512.jsonl"
    glide_512_path = outputs_dir / "fastino_glide_japan_korea_stratified_512.jsonl"
    glide_full_path = outputs_dir / "fastino_glide_fastino_glide_japan_korea_0_5deg_land-water-v1.jsonl"

    # Load JEV and D1
    jev_preds = PredictionCache(jev_512_path, strict_validation=False).load()
    d1_preds = PredictionCache(d1_512_path, strict_validation=False).load()

    # Load any already-completed GLiDE predictions
    cached_glide = {}
    if glide_512_path.exists():
        cached_glide.update(PredictionCache(glide_512_path, strict_validation=False).load())
    if glide_full_path.exists():
        full_cached = PredictionCache(glide_full_path, strict_validation=False).load()
        for cid, pred in full_cached.items():
            if cid in target_ids and cid not in cached_glide:
                cached_glide[cid] = pred
                # Write to stratified file
                with open(glide_512_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(pred.to_dict()) + "\n")

    logger.info("GLiDE initially cached for 512 subset: %d / 512", len(cached_glide))

    # Run remaining GLiDE queries
    glide_preds, perf_stats = asyncio.run(
        execute_glide_queries(
            target_coords=target_coords,
            cached_preds=cached_glide,
            stratified_output_path=glide_512_path,
            full_cache_output_path=glide_full_path,
            concurrency=12,
            timeout_seconds=90.0,
        )
    )

    # Evaluate all 3 models on the identical 512 coordinates
    evaluator = GroundTruthEvaluator()
    models_preds = {
        "jev": ("jev-latest", [jev_preds[cid] for cid in target_ids]),
        "liquid_d1": ("liquid/d1", [d1_preds[cid] for cid in target_ids]),
        "fastino_glide": ("fastino/glide", [glide_preds[cid] for cid in target_ids]),
    }

    metrics = {}
    for key, (mname, preds) in models_preds.items():
        ev = evaluate_predictions(preds, evaluator=evaluator, coastal_distance_km=100.0)
        cm = ev["confusion_matrix"]
        metrics[key] = {
            "model": mname,
            "total_points": len(preds),
            "grid_accuracy": ev["grid_metrics"]["accuracy"],
            "area_weighted_accuracy": ev["area_weighted_metrics"]["accuracy"],
            "grid_brier": ev["grid_metrics"]["brier_score"],
            "area_weighted_brier": ev["area_weighted_metrics"]["brier_score"],
            "coastal_accuracy": ev["coastal_metrics"]["accuracy"],
            "coastal_brier": ev["coastal_metrics"]["brier_score"],
            "land_precision": ev["grid_metrics"]["precision"],
            "land_recall": ev["grid_metrics"]["recall"],
            "f1": ev["grid_metrics"]["f1"],
            "tp": cm["true_positive_land"],
            "tn": cm["true_negative_water"],
            "fp": cm["false_positive_land"],
            "fn": cm["false_negative_water"],
            "confusion_matrix": cm,
        }

    # Pairwise disagreement matrices
    def compute_pairwise(model_a, model_b):
        a_dict = {p.coordinate.id: p for p in models_preds[model_a][1]}
        b_dict = {p.coordinate.id: p for p in models_preds[model_b][1]}
        counts = {"both_correct": 0, "both_wrong": 0, f"{model_a}_correct_{model_b}_wrong": 0, f"{model_b}_correct_{model_a}_wrong": 0}
        for cid in target_ids:
            gt = coords_dict[cid]["is_land"]
            a_correct = (a_dict[cid].is_land == gt)
            b_correct = (b_dict[cid].is_land == gt)
            if a_correct and b_correct:
                counts["both_correct"] += 1
            elif not a_correct and not b_correct:
                counts["both_wrong"] += 1
            elif a_correct and not b_correct:
                counts[f"{model_a}_correct_{model_b}_wrong"] += 1
            else:
                counts[f"{model_b}_correct_{model_a}_wrong"] += 1
        pcts = {k: round(v / len(target_ids) * 100.0, 2) for k, v in counts.items()}
        return {"counts": counts, "percentages": pcts}

    pairwise_jev_glide = compute_pairwise("jev", "fastino_glide")
    pairwise_d1_glide = compute_pairwise("liquid_d1", "fastino_glide")
    pairwise_jev_d1 = compute_pairwise("jev", "liquid_d1")

    # Sea of Japan breakdown within the 512 subset (lat 35 to 43, lon 130 to 138, GT water)
    soj_ids = [
        cid for cid, c in coords_dict.items()
        if 35.0 <= c["latitude"] <= 43.0 and 130.0 <= c["longitude"] <= 138.0 and not c["is_land"]
    ]
    soj_breakdown = {}
    for key, (mname, preds) in models_preds.items():
        p_dict = {p.coordinate.id: p for p in preds}
        correct_water = sum(1 for cid in soj_ids if not p_dict[cid].is_land)
        false_land = len(soj_ids) - correct_water
        soj_breakdown[key] = {
            "model": mname,
            "total_soj_water_probes": len(soj_ids),
            "correct_water_count": correct_water,
            "correct_water_pct": round(correct_water / len(soj_ids) * 100.0, 2) if soj_ids else 0,
            "false_land_count": false_land,
            "false_land_pct": round(false_land / len(soj_ids) * 100.0, 2) if soj_ids else 0,
        }

    results = {
        "benchmark": "512-point stratified benchmark",
        "parent_benchmark": "Japan/Korea 0.5-degree frozen benchmark (2,112 coordinates)",
        "metadata": meta_data["metadata"],
        "metrics": metrics,
        "pairwise_disagreements": {
            "jev_vs_fastino_glide": pairwise_jev_glide,
            "liquid_d1_vs_fastino_glide": pairwise_d1_glide,
            "jev_vs_liquid_d1": pairwise_jev_d1,
        },
        "sea_of_japan_inspection": {
            "total_probes": len(soj_ids),
            "breakdown": soj_breakdown,
        },
        "provider_performance": {
            "fastino_glide": perf_stats,
        },
    }

    out_json = outputs_dir / "three_models_stratified_512_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved 512 comparison results to %s", out_json)

    # Render figure
    fig_path = outputs_dir / "three_models_stratified_512_comparison.png"
    render_stratified_figure(
        coords_dict=coords_dict,
        jev_preds={p.coordinate.id: p for p in models_preds["jev"][1]},
        d1_preds={p.coordinate.id: p for p in models_preds["liquid_d1"][1]},
        glide_preds={p.coordinate.id: p for p in models_preds["fastino_glide"][1]},
        evaluator=evaluator,
        metrics=metrics,
        out_path=fig_path,
    )

    return results


if __name__ == "__main__":
    run_full_stratified_evaluation()
