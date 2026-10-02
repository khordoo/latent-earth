#!/usr/bin/env python3
"""
Comprehensive comparative benchmark:
JEV (jev-latest) vs. Liquid D1 (liquid/d1) vs. Fastino GLiDE (fastino/glide)
on the identical Japan/Korea 0.5° grid (2,112 coordinates).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from latent_earth.cache import PredictionCache
from latent_earth.evaluation import GroundTruthEvaluator, evaluate_predictions


def run_benchmark():
    outputs_dir = Path("outputs")
    jev_path = outputs_dir / "jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl"
    d1_path = outputs_dir / "openrouter_liquid_d1_japan_korea_0_5deg_land-water-v1.jsonl"
    glide_path = outputs_dir / "fastino_glide_fastino_glide_japan_korea_0_5deg_land-water-v1.jsonl"

    for p in (jev_path, d1_path, glide_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    evaluator = GroundTruthEvaluator()

    # Load predictions
    models_data = {
        "jev": ("jev-latest", list(PredictionCache(jev_path, strict_validation=False).load().values())),
        "liquid_d1": ("liquid/d1", list(PredictionCache(d1_path, strict_validation=False).load().values())),
        "fastino_glide": ("fastino/glide", list(PredictionCache(glide_path, strict_validation=False).load().values())),
    }

    metrics = {}
    dict_by_model = {}

    for key, (model_name, preds) in models_data.items():
        print(f"Evaluating {model_name} ({len(preds):,} points)...")
        ev = evaluate_predictions(preds, evaluator=evaluator, coastal_distance_km=100.0)
        metrics[key] = {
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
            "f1": ev["grid_metrics"]["f1"],
            "confusion_matrix": ev["confusion_matrix"],
        }
        dict_by_model[key] = {(round(p.coordinate.latitude, 6), round(p.coordinate.longitude, 6)): p for p in preds}

    common_coords = sorted(set(dict_by_model["jev"].keys()) & set(dict_by_model["liquid_d1"].keys()) & set(dict_by_model["fastino_glide"].keys()))
    print(f"Common coordinates across all 3 models: {len(common_coords):,}")

    # Pairwise disagreement matrices
    def compute_pairwise(model_a, model_b):
        counts = {"both_correct": 0, "both_wrong": 0, f"{model_a}_correct_{model_b}_wrong": 0, f"{model_b}_correct_{model_a}_wrong": 0}
        for c in common_coords:
            gt = evaluator.is_land(c[0], c[1])
            a_correct = (dict_by_model[model_a][c].is_land == gt)
            b_correct = (dict_by_model[model_b][c].is_land == gt)
            if a_correct and b_correct:
                counts["both_correct"] += 1
            elif not a_correct and not b_correct:
                counts["both_wrong"] += 1
            elif a_correct and not b_correct:
                counts[f"{model_a}_correct_{model_b}_wrong"] += 1
            else:
                counts[f"{model_b}_correct_{model_a}_wrong"] += 1
        pcts = {k: round(v / len(common_coords) * 100.0, 2) for k, v in counts.items()}
        return {"counts": counts, "percentages": pcts}

    pairwise_jev_glide = compute_pairwise("jev", "fastino_glide")
    pairwise_d1_glide = compute_pairwise("liquid_d1", "fastino_glide")

    # Sea of Japan breakdown (35N to 43N, 130E to 138E)
    soj_coords = [c for c in common_coords if 35.0 <= c[0] <= 43.0 and 130.0 <= c[1] <= 138.0 and not evaluator.is_land(c[0], c[1])]
    soj_breakdown = {}
    for key, (mname, _) in models_data.items():
        c_dict = dict_by_model[key]
        correct_water = sum(1 for c in soj_coords if not c_dict[c].is_land)
        false_land = len(soj_coords) - correct_water
        soj_breakdown[key] = {
            "model": mname,
            "correct_water_count": correct_water,
            "correct_water_pct": round(correct_water / len(soj_coords) * 100.0, 2) if soj_coords else 0,
            "false_land_count": false_land,
            "false_land_pct": round(false_land / len(soj_coords) * 100.0, 2) if soj_coords else 0,
        }

    results = {
        "metadata": {
            "region": "japan_korea",
            "resolution_degrees": 0.5,
            "total_points": len(common_coords),
            "coastal_threshold_km": 100.0,
            "ground_truth": "Natural Earth 1:10m Land",
        },
        "metrics": metrics,
        "pairwise_disagreements": {
            "jev_vs_fastino_glide": pairwise_jev_glide,
            "liquid_d1_vs_fastino_glide": pairwise_d1_glide,
        },
        "sea_of_japan_inspection": {
            "bounding_box": {"lat_min": 35.0, "lat_max": 43.0, "lon_min": 130.0, "lon_max": 138.0},
            "total_water_probes": len(soj_coords),
            "breakdown": soj_breakdown,
        },
    }

    out_json = outputs_dir / "glide_three_model_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark JSON to {out_json}")

    # Render Visual Comparison Figure
    render_figure(common_coords, dict_by_model, evaluator, metrics, pairwise_jev_glide, pairwise_d1_glide, outputs_dir / "glide_vs_jev_vs_liquid_d1_japan_korea.png")
    return results


def render_figure(common_coords, dict_by_model, evaluator, metrics, pw_jev, pw_d1, out_path):
    lats = sorted({c[0] for c in common_coords}, reverse=True)
    lons = sorted({c[1] for c in common_coords})
    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}
    n_lat, n_lon = len(lats), len(lons)

    gt_grid = np.zeros((n_lat, n_lon))
    jev_grid = np.zeros((n_lat, n_lon))
    d1_grid = np.zeros((n_lat, n_lon))
    glide_grid = np.zeros((n_lat, n_lon))

    # Disagreement grids: 0=both correct, 1=both wrong, 2=model_a correct, 3=glide correct
    dis_jev_grid = np.zeros((n_lat, n_lon))
    dis_d1_grid = np.zeros((n_lat, n_lon))

    for c in common_coords:
        r = lat_idx[c[0]]
        col = lon_idx[c[1]]
        gt = evaluator.is_land(c[0], c[1])
        gt_grid[r, col] = 1 if gt else 0
        jev_land = dict_by_model["jev"][c].is_land
        d1_land = dict_by_model["liquid_d1"][c].is_land
        glide_land = dict_by_model["fastino_glide"][c].is_land

        jev_grid[r, col] = 1 if jev_land else 0
        d1_grid[r, col] = 1 if d1_land else 0
        glide_grid[r, col] = 1 if glide_land else 0

        # JEV vs GLiDE
        j_corr = (jev_land == gt)
        g_corr = (glide_land == gt)
        if j_corr and g_corr:
            dis_jev_grid[r, col] = 0
        elif not j_corr and not g_corr:
            dis_jev_grid[r, col] = 1
        elif j_corr and not g_corr:
            dis_jev_grid[r, col] = 2
        else:
            dis_jev_grid[r, col] = 3

        # D1 vs GLiDE
        d_corr = (d1_land == gt)
        if d_corr and g_corr:
            dis_d1_grid[r, col] = 0
        elif not d_corr and not g_corr:
            dis_d1_grid[r, col] = 1
        elif d_corr and not g_corr:
            dis_d1_grid[r, col] = 2
        else:
            dis_d1_grid[r, col] = 3

    extent = [min(lons) - 0.25, max(lons) + 0.25, min(lats) - 0.25, max(lats) + 0.25]
    cmap_bin = mcolors.ListedColormap(["#0F2942", "#10B981"])  # Ocean Blue, Emerald Land
    norm_bin = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap_bin.N)

    # Disagreement cmap: 0=slate gray, 1=crimson, 2=amber (incumbent correct), 3=cyan (glide correct)
    cmap_dis = mcolors.ListedColormap(["#334155", "#991B1B", "#F59E0B", "#06B6D4"])
    norm_dis = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_dis.N)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor="#0B0F19")
    fig.suptitle(
        "Latent Earth: Ground-Truth Benchmark — JEV vs. Liquid D1 vs. Fastino GLiDE\n"
        "Japan / Korea 0.5° Grid (2,112 Probes, Natural Earth 1:10m)",
        color="#F3F4F6",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    panels = [
        (axes[0, 0], gt_grid, cmap_bin, norm_bin, "Ground Truth: Natural Earth 1:10m"),
        (axes[0, 1], jev_grid, cmap_bin, norm_bin, f"JEV (jev-latest) — Acc: {metrics['jev']['grid_accuracy']:.1%}"),
        (axes[0, 2], d1_grid, cmap_bin, norm_bin, f"Liquid D1 (liquid/d1) — Acc: {metrics['liquid_d1']['grid_accuracy']:.1%}"),
        (axes[1, 0], glide_grid, cmap_bin, norm_bin, f"Fastino GLiDE (fastino/glide) — Acc: {metrics['fastino_glide']['grid_accuracy']:.1%}"),
        (axes[1, 1], dis_jev_grid, cmap_dis, norm_dis, "Pairwise: JEV vs. Fastino GLiDE"),
        (axes[1, 2], dis_d1_grid, cmap_dis, norm_dis, "Pairwise: Liquid D1 vs. Fastino GLiDE"),
    ]

    for ax, data, cm, nm, title in panels:
        ax.set_facecolor("#0B0F19")
        ax.imshow(data, extent=extent, origin="upper", cmap=cm, norm=nm, interpolation="nearest")
        ax.set_title(title, color="#E5E7EB", fontsize=11, fontweight="semibold", pad=8)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude (°E)", color="#9CA3AF", fontsize=8)
        ax.set_ylabel("Latitude (°N)", color="#9CA3AF", fontsize=8)
        ax.tick_params(colors="#9CA3AF", labelsize=8)
        ax.grid(color="#1F2937", linestyle=":", linewidth=0.5, alpha=0.6)

        rect = plt.Rectangle((130.0, 35.0), 8.0, 8.0, linewidth=1.1, edgecolor="#E2E8F0", facecolor="none", linestyle="--", alpha=0.5)
        ax.add_patch(rect)
        ax.text(130.5, 42.2, "Sea of Japan", color="#E2E8F0", fontsize=7.5, alpha=0.7, style="italic")

    # Binary Legend for first 4 panels
    patches_bin = [mpatches.Patch(color="#0F2942", label="Water"), mpatches.Patch(color="#10B981", label="Land")]
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.legend(handles=patches_bin, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=7.5)

    # JEV vs GLiDE Legend
    c_j = pw_jev["counts"]
    p_j = pw_jev["percentages"]
    patches_jev = [
        mpatches.Patch(color="#334155", label=f"Both Correct ({c_j['both_correct']} | {p_j['both_correct']}%)"),
        mpatches.Patch(color="#991B1B", label=f"Both Wrong ({c_j['both_wrong']} | {p_j['both_wrong']}%)"),
        mpatches.Patch(color="#F59E0B", label=f"JEV Correct / GLiDE Wrong ({c_j['jev_correct_fastino_glide_wrong']} | {p_j['jev_correct_fastino_glide_wrong']}%)"),
        mpatches.Patch(color="#06B6D4", label=f"GLiDE Correct / JEV Wrong ({c_j['fastino_glide_correct_jev_wrong']} | {p_j['fastino_glide_correct_jev_wrong']}%)"),
    ]
    axes[1, 1].legend(handles=patches_jev, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=7.5)

    # D1 vs GLiDE Legend
    c_d = pw_d1["counts"]
    p_d = pw_d1["percentages"]
    patches_d1 = [
        mpatches.Patch(color="#334155", label=f"Both Correct ({c_d['both_correct']} | {p_d['both_correct']}%)"),
        mpatches.Patch(color="#991B1B", label=f"Both Wrong ({c_d['both_wrong']} | {p_d['both_wrong']}%)"),
        mpatches.Patch(color="#F59E0B", label=f"D1 Correct / GLiDE Wrong ({c_d['liquid_d1_correct_fastino_glide_wrong']} | {p_d['liquid_d1_correct_fastino_glide_wrong']}%)"),
        mpatches.Patch(color="#06B6D4", label=f"GLiDE Correct / D1 Wrong ({c_d['fastino_glide_correct_liquid_d1_wrong']} | {p_d['fastino_glide_correct_liquid_d1_wrong']}%)"),
    ]
    axes[1, 2].legend(handles=patches_d1, loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=7.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved visual comparison figure to {out_path}")


if __name__ == "__main__":
    run_benchmark()
