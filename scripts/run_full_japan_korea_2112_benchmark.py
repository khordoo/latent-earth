#!/usr/bin/env python3
"""
Full Japan/Korea 2,112-Point Regional Benchmark:
JEV vs. Liquid D1 vs. Fastino GLiDE on identical coordinates (0.5° resolution).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.fastino_glide import FastinoGlideProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("japan_korea_2112_benchmark")


async def execute_glide_2112(
    target_coords: list[Coordinate],
    cached_preds: dict[str, Prediction],
    output_path: Path,
    concurrency: int = 48,
    chunk_size: int = 48,
    timeout_seconds: float = 90.0,
) -> tuple[dict[str, Prediction], dict[str, Any]]:
    """Run GLiDE on missing coordinates with high concurrency and incremental disk flushes."""
    provider = FastinoGlideProvider(
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        max_retries=5,
        rate_limit_per_second=0.0,
    )

    remaining = [c for c in target_coords if c.id not in cached_preds]
    logger.info(
        "GLiDE 2,112 Execution: %d total coordinates, %d already cached, %d remaining to query.",
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
        chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]
        for ch_idx, chunk in enumerate(chunks, start=1):
            ch_start = time.monotonic()
            logger.info("Starting chunk %d/%d (%d coordinates)...", ch_idx, len(chunks), len(chunk))
            try:
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

                # Append to output file
                with open(output_path, "a", encoding="utf-8") as f_out:
                    for pred in chunk_results:
                        f_out.write(json.dumps(pred.to_dict()) + "\n")

                ch_elapsed = time.monotonic() - ch_start
                rate = len(chunk) / ch_elapsed if ch_elapsed > 0 else 0
                logger.info(
                    "Completed chunk %d/%d in %.1fs (%.2f req/s) | %d/%d total complete",
                    ch_idx,
                    len(chunks),
                    ch_elapsed,
                    rate,
                    len(all_preds),
                    len(target_coords),
                )
            except Exception as exc:
                logger.error("Error processing chunk %d/%d: %s", ch_idx, len(chunks), exc)
                failures_count += len(chunk)

    wall_clock_total = time.monotonic() - wall_clock_start
    await provider.close()

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


def render_2112_figure(
    target_coords: list[Coordinate],
    jev_preds: dict[str, Prediction],
    d1_preds: dict[str, Prediction],
    glide_preds: dict[str, Prediction],
    evaluator: GroundTruthEvaluator,
    metrics: dict[str, Any],
    out_path: Path,
):
    """Render a 4-panel visual comparison on the full 2,112 coordinates."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    sorted_coords = sorted(target_coords, key=lambda c: (c.latitude, c.longitude))
    lats = [c.latitude for c in sorted_coords]
    lons = [c.longitude for c in sorted_coords]
    gt_land = [evaluator.is_land(c.latitude, c.longitude) for c in sorted_coords]

    color_water = "#1e3a8a"  # blue
    color_land = "#15803d"   # green
    color_wrong = "#ef4444"  # red
    color_correct = "#10b981" # green

    # Panel 1: Ground Truth
    ax1 = axes[0, 0]
    ax1.set_facecolor("#111827")
    c_gt = [color_land if is_l else color_water for is_l in gt_land]
    ax1.scatter(lons, lats, c=c_gt, s=24, alpha=0.9, edgecolors="none")
    ax1.set_title(
        f"Ground Truth (Natural Earth 1:10m) - 2,112 Coordinates\nLand: {sum(gt_land)} | Water: {len(gt_land) - sum(gt_land)}",
        fontsize=12, fontweight="bold", pad=10, color="white",
    )
    ax1.set_xlim(121.5, 146.5)
    ax1.set_ylim(23.5, 46.5)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#9ca3af")
    ax1.legend(
        handles=[mpatches.Patch(color=color_land, label="Land"), mpatches.Patch(color=color_water, label="Water")],
        loc="lower right", facecolor="#1f2937", edgecolor="#374151", fontsize=9,
    )

    # Panel 2: JEV Error Map
    ax2 = axes[0, 1]
    ax2.set_facecolor("#111827")
    jev_correct = [jev_preds[c.id].is_land == gt_land[idx] for idx, c in enumerate(sorted_coords)]
    c_jev = [color_correct if corr else color_wrong for corr in jev_correct]
    ax2.scatter(lons, lats, c=c_jev, s=24, alpha=0.9, edgecolors="none")
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
    d1_correct = [d1_preds[c.id].is_land == gt_land[idx] for idx, c in enumerate(sorted_coords)]
    c_d1 = [color_correct if corr else color_wrong for corr in d1_correct]
    ax3.scatter(lons, lats, c=c_d1, s=24, alpha=0.9, edgecolors="none")
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
    glide_correct = [glide_preds[c.id].is_land == gt_land[idx] for idx, c in enumerate(sorted_coords)]
    c_glide = [color_correct if corr else color_wrong for corr in glide_correct]
    ax4.scatter(lons, lats, c=c_glide, s=24, alpha=0.9, edgecolors="none")
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

    plt.suptitle("Full Japan & Korea Benchmark (2,112 Coordinates at 0.5° Resolution)", fontsize=16, fontweight="bold", y=0.98, color="white")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 2,112 benchmark figure to %s", out_path)


def main():
    outputs_dir = Path("outputs")
    jev_2112_path = outputs_dir / "jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl"
    d1_2112_path = outputs_dir / "openrouter_liquid_d1_japan_korea_0_5deg_land-water-v1.jsonl"
    glide_2112_path = outputs_dir / "fastino_glide_japan_korea_2112.jsonl"

    # Load all 2,112 target coordinates from JEV dataset
    jev_preds = PredictionCache(jev_2112_path, strict_validation=False).load()
    d1_preds = PredictionCache(d1_2112_path, strict_validation=False).load()

    target_coords = [p.coordinate for p in jev_preds.values()]
    target_ids = set(c.id for c in target_coords)

    logger.info("Loaded %d target coordinates from frozen JEV benchmark.", len(target_coords))

    # Initialize / load existing GLiDE predictions
    cached_glide: dict[str, Prediction] = {}

    if glide_2112_path.exists():
        cached_glide.update(PredictionCache(glide_2112_path, strict_validation=False).load())

    # Import from stratified 512 if not already in 2112 cache
    strat_512_path = outputs_dir / "fastino_glide_japan_korea_stratified_512.jsonl"
    if strat_512_path.exists():
        s512_preds = PredictionCache(strat_512_path, strict_validation=False).load()
        new_from_s512 = []
        for cid, pred in s512_preds.items():
            if cid in target_ids and cid not in cached_glide:
                cached_glide[cid] = pred
                new_from_s512.append(pred)
        if new_from_s512:
            with open(glide_2112_path, "a", encoding="utf-8") as f:
                for pred in new_from_s512:
                    f.write(json.dumps(pred.to_dict()) + "\n")
            logger.info("Imported %d predictions from stratified 512 into 2,112 dataset.", len(new_from_s512))

    logger.info("Currently cached GLiDE predictions: %d / %d", len(cached_glide), len(target_coords))

    # Run remaining GLiDE queries with concurrency 48
    glide_preds, perf_stats = asyncio.run(
        execute_glide_2112(
            target_coords=target_coords,
            cached_preds=cached_glide,
            output_path=glide_2112_path,
            concurrency=48,
            chunk_size=48,
            timeout_seconds=90.0,
        )
    )

    logger.info("GLiDE execution finished. Total predictions available: %d", len(glide_preds))

    # Comprehensive evaluation against Ground Truth
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

    # Pairwise disagreement analysis
    def compute_pairwise(model_a, model_b):
        a_dict = {p.coordinate.id: p for p in models_preds[model_a][1]}
        b_dict = {p.coordinate.id: p for p in models_preds[model_b][1]}
        counts = {"both_correct": 0, "both_wrong": 0, f"{model_a}_correct_{model_b}_wrong": 0, f"{model_b}_correct_{model_a}_wrong": 0}
        for c in target_coords:
            cid = c.id
            gt = evaluator.is_land(c.latitude, c.longitude)
            a_corr = (a_dict[cid].is_land == gt)
            b_corr = (b_dict[cid].is_land == gt)
            if a_corr and b_corr:
                counts["both_correct"] += 1
            elif not a_corr and not b_corr:
                counts["both_wrong"] += 1
            elif a_corr and not b_corr:
                counts[f"{model_a}_correct_{model_b}_wrong"] += 1
            else:
                counts[f"{model_b}_correct_{model_a}_wrong"] += 1
        pcts = {k: round(v / len(target_coords) * 100.0, 2) for k, v in counts.items()}
        return {"counts": counts, "percentages": pcts}

    pairwise_jev_glide = compute_pairwise("jev", "fastino_glide")
    pairwise_d1_glide = compute_pairwise("liquid_d1", "fastino_glide")
    pairwise_jev_d1 = compute_pairwise("jev", "liquid_d1")

    # Sea of Japan breakdown (lat 35 to 43, lon 130 to 138, ground truth water)
    soj_coords = [
        c for c in target_coords
        if 35.0 <= c.latitude <= 43.0 and 130.0 <= c.longitude <= 138.0 and not evaluator.is_land(c.latitude, c.longitude)
    ]
    soj_breakdown = {}
    for key, (mname, preds) in models_preds.items():
        p_dict = {p.coordinate.id: p for p in preds}
        correct_water = sum(1 for c in soj_coords if not p_dict[c.id].is_land)
        false_land = len(soj_coords) - correct_water
        soj_breakdown[key] = {
            "model": mname,
            "total_soj_water_probes": len(soj_coords),
            "correct_water_count": correct_water,
            "correct_water_pct": round(correct_water / len(soj_coords) * 100.0, 2) if soj_coords else 0,
            "false_land_count": false_land,
            "false_land_pct": round(false_land / len(soj_coords) * 100.0, 2) if soj_coords else 0,
        }

    results = {
        "benchmark": "Full Japan/Korea 2,112-Point Regional Benchmark",
        "resolution_deg": 0.5,
        "total_coordinates": len(target_coords),
        "bbox": {"lat_min": 24.0, "lat_max": 46.0, "lon_min": 122.0, "lon_max": 146.0},
        "metrics": metrics,
        "pairwise_disagreements": {
            "jev_vs_fastino_glide": pairwise_jev_glide,
            "liquid_d1_vs_fastino_glide": pairwise_d1_glide,
            "jev_vs_liquid_d1": pairwise_jev_d1,
        },
        "sea_of_japan_inspection": {
            "total_probes": len(soj_coords),
            "breakdown": soj_breakdown,
        },
        "provider_performance": {
            "fastino_glide": perf_stats,
        },
    }

    out_json = outputs_dir / "three_models_japan_korea_2112_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved 2,112 comparison metrics to %s", out_json)

    fig_path = outputs_dir / "three_models_japan_korea_2112_comparison.png"
    render_2112_figure(
        target_coords=target_coords,
        jev_preds={p.coordinate.id: p for p in models_preds["jev"][1]},
        d1_preds={p.coordinate.id: p for p in models_preds["liquid_d1"][1]},
        glide_preds={p.coordinate.id: p for p in models_preds["fastino_glide"][1]},
        evaluator=evaluator,
        metrics=metrics,
        out_path=fig_path,
    )

    # Copy figure to artifact directory
    artifact_dir = Path("/Users/mahmoodkhordoo/.gemini/antigravity-cli/brain/c769cda4-e9a1-4263-bea1-72bbcc85b8f1")
    if artifact_dir.exists():
        shutil.copy(fig_path, artifact_dir / "three_models_japan_korea_2112_comparison.png")
        logger.info("Copied 2,112 comparison figure to artifacts directory.")

    print("\n=======================================================")
    print("=== FULL 2,112 JAPAN/KOREA BENCHMARK COMPLETE ===")
    print("=======================================================")
    print(f"JEV Accuracy:          {metrics['jev']['grid_accuracy']:.2%}")
    print(f"Liquid D1 Accuracy:    {metrics['liquid_d1']['grid_accuracy']:.2%}")
    print(f"Fastino GLiDE Accuracy:{metrics['fastino_glide']['grid_accuracy']:.2%}")
    print(f"Figure: {fig_path}")
    print(f"JSON:   {out_json}")


if __name__ == "__main__":
    main()
