#!/usr/bin/env python3
"""
Comprehensive comparative analysis between JEV (jev-latest) and Liquid D1 (liquid/d1)
on the identical Japan/Korea 0.5-degree grid (2,112 coordinates).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions


def run_comparison():
    outputs_dir = Path("outputs")
    jev_path = outputs_dir / "jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl"
    d1_path = outputs_dir / "openrouter_liquid_d1_japan_korea_0_5deg_land-water-v1.jsonl"

    if not jev_path.exists():
        raise FileNotFoundError(f"Missing JEV file: {jev_path}")
    if not d1_path.exists():
        raise FileNotFoundError(f"Missing Liquid D1 file: {d1_path}")

    # Load predictions
    jev_cache = PredictionCache(jev_path, strict_validation=False)
    jev_preds = list(jev_cache.load().values())
    d1_cache = PredictionCache(d1_path, strict_validation=False)
    d1_preds = list(d1_cache.load().values())

    print(f"Loaded {len(jev_preds):,} JEV predictions.")
    print(f"Loaded {len(d1_preds):,} Liquid D1 predictions.")

    # Build coordinate lookup
    jev_dict = {(round(p.coordinate.latitude, 6), round(p.coordinate.longitude, 6)): p for p in jev_preds}
    d1_dict = {(round(p.coordinate.latitude, 6), round(p.coordinate.longitude, 6)): p for p in d1_preds}

    common_coords = sorted(set(jev_dict.keys()) & set(d1_dict.keys()))
    print(f"Common coordinates: {len(common_coords):,}")

    evaluator = GroundTruthEvaluator()
    jev_eval = evaluate_predictions(jev_preds, evaluator=evaluator, coastal_distance_km=100.0)
    d1_eval = evaluate_predictions(d1_preds, evaluator=evaluator, coastal_distance_km=100.0)

    # Disagreement categorization
    disagreement_counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "jev_correct_d1_wrong": 0,
        "d1_correct_jev_wrong": 0,
    }

    # Sea of Japan bounds: ~35N to 43N, 130E to 138E
    soj_coords = []
    category_by_coord = {}

    for coord in common_coords:
        lat, lon = coord
        jp = jev_dict[coord]
        dp = d1_dict[coord]

        gt_land = evaluator.is_land(lat, lon)
        jev_land = jp.is_land
        d1_land = dp.is_land

        jev_correct = (jev_land == gt_land)
        d1_correct = (d1_land == gt_land)

        if jev_correct and d1_correct:
            cat = "both_correct"
            cat_code = 0
        elif not jev_correct and not d1_correct:
            cat = "both_wrong"
            cat_code = 1
        elif jev_correct and not d1_correct:
            cat = "jev_correct_d1_wrong"
            cat_code = 2
        else:
            cat = "d1_correct_jev_wrong"
            cat_code = 3

        disagreement_counts[cat] += 1
        category_by_coord[coord] = {
            "code": cat_code,
            "category": cat,
            "gt_land": gt_land,
            "jev_land": jev_land,
            "jev_prob": jp.probability_land,
            "d1_land": d1_land,
            "d1_prob": dp.probability_land,
        }

        # Check Sea of Japan
        if 35.0 <= lat <= 43.0 and 130.0 <= lon <= 138.0 and not gt_land:
            soj_coords.append({
                "lat": lat,
                "lon": lon,
                "jev_pred": "LAND" if jev_land else "WATER",
                "jev_prob": jp.probability_land,
                "d1_pred": "LAND" if d1_land else "WATER",
                "d1_prob": dp.probability_land,
            })

    total_pts = len(common_coords)
    disagreement_percentages = {k: round(v / total_pts * 100.0, 2) for k, v in disagreement_counts.items()}

    # Sea of Japan breakdown
    soj_total_water = len(soj_coords)
    soj_jev_water_count = sum(1 for p in soj_coords if p["jev_pred"] == "WATER")
    soj_d1_water_count = sum(1 for p in soj_coords if p["d1_pred"] == "WATER")

    comparison_results = {
        "metadata": {
            "region": "japan_korea",
            "resolution_degrees": 0.5,
            "total_points": total_pts,
            "coastal_threshold_km": 100.0,
            "ground_truth": "Natural Earth 1:10m Land",
        },
        "metrics_comparison": {
            "jev": {
                "model": "jev-latest",
                "grid_accuracy": jev_eval["grid_metrics"]["accuracy"],
                "area_weighted_accuracy": jev_eval["area_weighted_metrics"]["accuracy"],
                "grid_brier": jev_eval["grid_metrics"]["brier_score"],
                "area_weighted_brier": jev_eval["area_weighted_metrics"]["brier_score"],
                "coastal_accuracy": jev_eval["coastal_metrics"]["accuracy"],
                "coastal_brier": jev_eval["coastal_metrics"]["brier_score"],
                "land_precision": jev_eval["grid_metrics"]["precision"],
                "land_recall": jev_eval["grid_metrics"]["recall"],
                "confusion_matrix": jev_eval["confusion_matrix"],
            },
            "liquid_d1": {
                "model": "liquid/d1",
                "grid_accuracy": d1_eval["grid_metrics"]["accuracy"],
                "area_weighted_accuracy": d1_eval["area_weighted_metrics"]["accuracy"],
                "grid_brier": d1_eval["grid_metrics"]["brier_score"],
                "area_weighted_brier": d1_eval["area_weighted_metrics"]["brier_score"],
                "coastal_accuracy": d1_eval["coastal_metrics"]["accuracy"],
                "coastal_brier": d1_eval["coastal_metrics"]["brier_score"],
                "land_precision": d1_eval["grid_metrics"]["precision"],
                "land_recall": d1_eval["grid_metrics"]["recall"],
                "confusion_matrix": d1_eval["confusion_matrix"],
            },
        },
        "disagreement_matrix": {
            "counts": disagreement_counts,
            "percentages": disagreement_percentages,
        },
        "sea_of_japan_inspection": {
            "bounding_box": {"lat_min": 35.0, "lat_max": 43.0, "lon_min": 130.0, "lon_max": 138.0},
            "total_ground_truth_water_points": soj_total_water,
            "jev_correct_water_count": soj_jev_water_count,
            "jev_correct_water_percentage": round(soj_jev_water_count / soj_total_water * 100.0, 2) if soj_total_water else 0,
            "jev_false_land_count": soj_total_water - soj_jev_water_count,
            "liquid_d1_correct_water_count": soj_d1_water_count,
            "liquid_d1_correct_water_percentage": round(soj_d1_water_count / soj_total_water * 100.0, 2) if soj_total_water else 0,
            "liquid_d1_false_land_count": soj_total_water - soj_d1_water_count,
        },
    }

    out_json = outputs_dir / "jev_vs_liquid_d1_japan_korea_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved comparison JSON to {out_json}")

    # Render Visual Comparison Figure
    render_comparison_figure(
        common_coords=common_coords,
        category_by_coord=category_by_coord,
        evaluator=evaluator,
        jev_dict=jev_dict,
        d1_dict=d1_dict,
        comparison_results=comparison_results,
        out_path=outputs_dir / "jev_vs_liquid_d1_japan_korea.png",
    )

    return comparison_results


def render_comparison_figure(
    common_coords,
    category_by_coord,
    evaluator,
    jev_dict,
    d1_dict,
    comparison_results,
    out_path,
):
    lats = sorted({c[0] for c in common_coords}, reverse=True)
    lons = sorted({c[1] for c in common_coords})

    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}

    n_lat = len(lats)
    n_lon = len(lons)

    # 4 Grids: GT, JEV, Liquid D1, Disagreement Map
    gt_grid = np.zeros((n_lat, n_lon))
    jev_grid = np.zeros((n_lat, n_lon))
    d1_grid = np.zeros((n_lat, n_lon))
    disagree_grid = np.zeros((n_lat, n_lon))

    for coord, data in category_by_coord.items():
        r = lat_idx[coord[0]]
        c = lon_idx[coord[1]]
        gt_grid[r, c] = 1 if data["gt_land"] else 0
        jev_grid[r, c] = 1 if data["jev_land"] else 0
        d1_grid[r, c] = 1 if data["d1_land"] else 0
        disagree_grid[r, c] = data["code"]

    extent = [min(lons) - 0.25, max(lons) + 0.25, min(lats) - 0.25, max(lats) + 0.25]

    fig, axes = plt.subplots(2, 2, figsize=(16, 15), facecolor="#0B0F19")
    fig.suptitle(
        "Latent Earth: Model Comparison — JEV (jev-latest) vs. Liquid D1 (liquid/d1)\n"
        "Japan / Korea 0.5° Resolution (2,112 Probes)",
        color="#F3F4F6",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # Colormaps
    land_water_cmap = mcolors.ListedColormap(["#0F2942", "#10B981"])  # Ocean Blue, Emerald Land
    norm_binary = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], land_water_cmap.N)

    # Disagreement Colormap:
    # 0: Both Correct (Subtle neutral slate gray #334155)
    # 1: Both Wrong (Muted crimson #991B1B)
    # 2: JEV Correct / D1 Wrong (Amber/Orange #F59E0B)
    # 3: D1 Correct / JEV Wrong (Vibrant Emerald/Cyan #06B6D4)
    disagree_cmap = mcolors.ListedColormap(["#334155", "#991B1B", "#F59E0B", "#06B6D4"])
    norm_disagree = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], disagree_cmap.N)

    panels = [
        (axes[0, 0], gt_grid, land_water_cmap, norm_binary, "Ground Truth: Natural Earth 1:10m", "gt"),
        (axes[0, 1], jev_grid, land_water_cmap, norm_binary, f"JEV (jev-latest) — Acc: {comparison_results['metrics_comparison']['jev']['grid_accuracy']:.1%}", "jev"),
        (axes[1, 0], d1_grid, land_water_cmap, norm_binary, f"Liquid D1 (liquid/d1) — Acc: {comparison_results['metrics_comparison']['liquid_d1']['grid_accuracy']:.1%}", "d1"),
        (axes[1, 1], disagree_grid, disagree_cmap, norm_disagree, "Model Agreement / Disagreement Map", "disagree"),
    ]

    for ax, data, cmap, norm, title, ptype in panels:
        ax.set_facecolor("#0B0F19")
        ax.imshow(data, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, color="#E5E7EB", fontsize=12, fontweight="semibold", pad=8)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude (°E)", color="#9CA3AF", fontsize=9)
        ax.set_ylabel("Latitude (°N)", color="#9CA3AF", fontsize=9)
        ax.tick_params(colors="#9CA3AF", labelsize=8)
        ax.grid(color="#1F2937", linestyle=":", linewidth=0.5, alpha=0.6)

        # Highlight Sea of Japan box
        rect = plt.Rectangle((130.0, 35.0), 8.0, 8.0, linewidth=1.2, edgecolor="#E2E8F0", facecolor="none", linestyle="--", alpha=0.5)
        ax.add_patch(rect)
        ax.text(130.5, 42.2, "Sea of Japan", color="#E2E8F0", fontsize=8, alpha=0.7, style="italic")

    # Legends
    # Binary legend for first 3 panels
    patches_bin = [
        mpatches.Patch(color="#0F2942", label="Water"),
        mpatches.Patch(color="#10B981", label="Land"),
    ]
    axes[0, 0].legend(handles=patches_bin, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)
    axes[0, 1].legend(handles=patches_bin, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)
    axes[1, 0].legend(handles=patches_bin, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)

    # Disagreement legend
    dc = comparison_results["disagreement_matrix"]["counts"]
    dp = comparison_results["disagreement_matrix"]["percentages"]
    patches_disagree = [
        mpatches.Patch(color="#334155", label=f"Both Correct ({dc['both_correct']} | {dp['both_correct']}%)"),
        mpatches.Patch(color="#991B1B", label=f"Both Wrong ({dc['both_wrong']} | {dp['both_wrong']}%)"),
        mpatches.Patch(color="#F59E0B", label=f"JEV Correct / D1 Wrong ({dc['jev_correct_d1_wrong']} | {dp['jev_correct_d1_wrong']}%)"),
        mpatches.Patch(color="#06B6D4", label=f"D1 Correct / JEV Wrong ({dc['d1_correct_jev_wrong']} | {dp['d1_correct_jev_wrong']}%)"),
    ]
    axes[1, 1].legend(handles=patches_disagree, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved visual comparison figure to {out_path}")


if __name__ == "__main__":
    run_comparison()
