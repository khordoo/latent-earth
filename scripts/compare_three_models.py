#!/usr/bin/env python3
"""
Three-model comparative analysis on Japan/Korea 0.5-degree grid (2,112 coordinates):
1. JEV (jev-latest)
2. Liquid D1 (liquid/d1)
3. Fastino GLiNER (fastino/GLiNER-2.5-Decide)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions


def run_three_model_comparison():
    outputs_dir = Path("outputs")
    jev_path = outputs_dir / "jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl"
    d1_path = outputs_dir / "openrouter_liquid_d1_japan_korea_0_5deg_land-water-v1.jsonl"
    fastino_path = outputs_dir / "fastino_fastino_GLiNER-2.5-Decide_japan_korea_0_5deg_land-water-v1.jsonl"

    for p in (jev_path, d1_path, fastino_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    evaluator = GroundTruthEvaluator()

    # Load caches
    models = {
        "jev": ("jev-latest", list(PredictionCache(jev_path, strict_validation=False).load().values())),
        "liquid_d1": ("liquid/d1", list(PredictionCache(d1_path, strict_validation=False).load().values())),
        "fastino": ("GLiNER-2.5-Decide", list(PredictionCache(fastino_path, strict_validation=False).load().values())),
    }

    metrics_by_model = {}
    dict_by_model = {}

    for key, (model_name, preds) in models.items():
        print(f"Evaluating {model_name} ({len(preds):,} points)...")
        ev = evaluate_predictions(preds, evaluator=evaluator, coastal_distance_km=100.0)
        metrics_by_model[key] = {
            "model": model_name,
            "total_points": len(preds),
            "grid_accuracy": ev["grid_metrics"]["accuracy"],
            "area_weighted_accuracy": ev["area_weighted_metrics"]["accuracy"],
            "grid_brier": ev["grid_metrics"]["brier_score"],
            "area_weighted_brier": ev["area_weighted_metrics"]["brier_score"],
            "coastal_accuracy": ev["coastal_metrics"]["accuracy"],
            "coastal_brier": ev["coastal_metrics"]["brier_score"],
            "land_precision": ev["grid_metrics"]["precision"],
            "land_recall": ev["grid_metrics"]["recall"],
            "confusion_matrix": ev["confusion_matrix"],
        }
        dict_by_model[key] = {(round(p.coordinate.latitude, 6), round(p.coordinate.longitude, 6)): p for p in preds}

    # Inspect Sea of Japan (35N to 43N, 130E to 138E)
    common_coords = sorted(set(dict_by_model["jev"].keys()) & set(dict_by_model["d1"].keys() if "d1" in dict_by_model else dict_by_model["liquid_d1"].keys()) & set(dict_by_model["fastino"].keys()))
    
    soj_coords = []
    for coord in common_coords:
        lat, lon = coord
        gt_land = evaluator.is_land(lat, lon)
        if 35.0 <= lat <= 43.0 and 130.0 <= lon <= 138.0 and not gt_land:
            soj_coords.append(coord)

    soj_breakdown = {}
    for key in ("jev", "liquid_d1", "fastino"):
        c_dict = dict_by_model[key]
        correct_water = sum(1 for c in soj_coords if not c_dict[c].is_land)
        false_land = len(soj_coords) - correct_water
        soj_breakdown[key] = {
            "model": models[key][0],
            "correct_water_count": correct_water,
            "correct_water_pct": round(correct_water / len(soj_coords) * 100.0, 2) if soj_coords else 0,
            "false_land_count": false_land,
        }

    comparison_results = {
        "metadata": {
            "region": "japan_korea",
            "resolution_degrees": 0.5,
            "total_points": len(common_coords),
            "coastal_threshold_km": 100.0,
            "ground_truth": "Natural Earth 1:10m Land",
        },
        "metrics": metrics_by_model,
        "sea_of_japan_inspection": {
            "total_water_probes": len(soj_coords),
            "breakdown": soj_breakdown,
        },
    }

    out_json = outputs_dir / "three_models_japan_korea_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved three-model comparison JSON to {out_json}")

    # Render Visual Comparison Map
    render_three_model_figure(
        common_coords=common_coords,
        dict_by_model=dict_by_model,
        evaluator=evaluator,
        metrics_by_model=metrics_by_model,
        out_path=outputs_dir / "three_models_japan_korea_comparison.png",
    )

    return comparison_results


def render_three_model_figure(common_coords, dict_by_model, evaluator, metrics_by_model, out_path):
    lats = sorted({c[0] for c in common_coords}, reverse=True)
    lons = sorted({c[1] for c in common_coords})
    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}
    n_lat, n_lon = len(lats), len(lons)

    gt_grid = np.zeros((n_lat, n_lon))
    jev_grid = np.zeros((n_lat, n_lon))
    d1_grid = np.zeros((n_lat, n_lon))
    fastino_grid = np.zeros((n_lat, n_lon))

    for c in common_coords:
        r = lat_idx[c[0]]
        col = lon_idx[c[1]]
        gt_grid[r, col] = 1 if evaluator.is_land(c[0], c[1]) else 0
        jev_grid[r, col] = 1 if dict_by_model["jev"][c].is_land else 0
        d1_grid[r, col] = 1 if dict_by_model["liquid_d1"][c].is_land else 0
        fastino_grid[r, col] = 1 if dict_by_model["fastino"][c].is_land else 0

    extent = [min(lons) - 0.25, max(lons) + 0.25, min(lats) - 0.25, max(lats) + 0.25]
    cmap = mcolors.ListedColormap(["#0F2942", "#10B981"])  # Ocean Blue, Emerald Land
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(16, 15), facecolor="#0B0F19")
    fig.suptitle(
        "Latent Earth: Three-Model Comparison\n"
        "JEV vs. Liquid D1 vs. Fastino GLiNER-2.5-Decide (Japan/Korea 0.5° Grid)",
        color="#F3F4F6",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    panels = [
        (axes[0, 0], gt_grid, "Ground Truth: Natural Earth 1:10m", None),
        (axes[0, 1], jev_grid, f"JEV (jev-latest) — Acc: {metrics_by_model['jev']['grid_accuracy']:.1%}", metrics_by_model['jev']),
        (axes[1, 0], d1_grid, f"Liquid D1 (liquid/d1) — Acc: {metrics_by_model['liquid_d1']['grid_accuracy']:.1%}", metrics_by_model['liquid_d1']),
        (axes[1, 1], fastino_grid, f"Fastino (GLiNER-2.5-Decide) — Acc: {metrics_by_model['fastino']['grid_accuracy']:.1%}", metrics_by_model['fastino']),
    ]

    for ax, grid, title, m in panels:
        ax.set_facecolor("#0B0F19")
        ax.imshow(grid, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, color="#E5E7EB", fontsize=12, fontweight="semibold", pad=8)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude (°E)", color="#9CA3AF", fontsize=9)
        ax.set_ylabel("Latitude (°N)", color="#9CA3AF", fontsize=9)
        ax.tick_params(colors="#9CA3AF", labelsize=8)
        ax.grid(color="#1F2937", linestyle=":", linewidth=0.5, alpha=0.6)

        # Highlight Sea of Japan
        rect = plt.Rectangle((130.0, 35.0), 8.0, 8.0, linewidth=1.2, edgecolor="#E2E8F0", facecolor="none", linestyle="--", alpha=0.5)
        ax.add_patch(rect)
        ax.text(130.5, 42.2, "Sea of Japan", color="#E2E8F0", fontsize=8, alpha=0.7, style="italic")

        patches = [
            mpatches.Patch(color="#0F2942", label="Water"),
            mpatches.Patch(color="#10B981", label="Land"),
        ]
        ax.legend(handles=patches, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved three-model comparison figure to {out_path}")


if __name__ == "__main__":
    run_three_model_comparison()
