"""Enrich existing evaluation runs with distance stratification and selective classification curves."""

import json
from pathlib import Path
from typing import Any
import numpy as np

from latent_earth.evaluation import GroundTruthEvaluator

RUNS_DIR = Path("frontend/public/data/runs")
OUTPUTS_DIR = Path("outputs")


def main():
    evaluator = GroundTruthEvaluator()
    print("Loaded GroundTruthEvaluator successfully.")

    # Cache grid ground truths: (lat, lon) -> (is_land, dist_km)
    coord_cache: dict[tuple[float, float], tuple[bool, float]] = {}

    run_files = sorted(RUNS_DIR.glob("*.json"))
    print(f"Found {len(run_files)} runs in {RUNS_DIR}")

    summary_results: dict[str, Any] = {}

    for fpath in run_files:
        run_id = fpath.stem
        print(f"\nProcessing {run_id}...")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = data.get("points", [])
        if not points:
            continue

        # Ensure ground truth and distances are set
        total = len(points)
        for p in points:
            lat = round(float(p["latitude"]), 4)
            lon = round(float(p["longitude"]), 4)
            key = (lat, lon)
            if key not in coord_cache:
                act = evaluator.is_land(lat, lon)
                d_km = evaluator.distance_to_coastline_km(lat, lon)
                coord_cache[key] = (act, d_km)

            act, d_km = coord_cache[key]
            p["actual_land"] = act
            p["distance_to_coastline_km"] = round(d_km, 2)
            p["distance_to_coastline"] = round(d_km, 2)
            p["is_coastal"] = (d_km <= 100.0)

            # Error category
            is_land = bool(p["is_land"])
            if is_land and act:
                p["error_category"] = "correct_land"
            elif not is_land and not act:
                p["error_category"] = "correct_water"
            elif is_land and not act:
                p["error_category"] = "false_land"
            else:
                p["error_category"] = "false_water"

        # 1. Distance Stratification Bins
        dist_bins_spec = [
            ("coastal_0_50km", 0.0, 50.0),
            ("near_coastal_50_200km", 50.0, 200.0),
            ("intermediate_200_500km", 200.0, 500.0),
            ("deep_interior_gt_500km", 500.0, 1e9),
        ]
        dist_strat: dict[str, Any] = {}
        for b_name, min_d, max_d in dist_bins_spec:
            b_pts = [p for p in points if min_d <= p["distance_to_coastline_km"] < max_d]
            if b_pts:
                tp = sum(1 for p in b_pts if p["is_land"] and p["actual_land"])
                tn = sum(1 for p in b_pts if not p["is_land"] and not p["actual_land"])
                fp = sum(1 for p in b_pts if p["is_land"] and not p["actual_land"])
                fn = sum(1 for p in b_pts if not p["is_land"] and p["actual_land"])
                acc = (tp + tn) / len(b_pts)
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bal_acc = (tpr + tnr) / 2.0
                dist_strat[b_name] = {
                    "cell_count": len(b_pts),
                    "cell_fraction": round(len(b_pts) / total, 4),
                    "accuracy": round(acc, 4),
                    "balanced_accuracy": round(bal_acc, 4),
                    "true_positive": tp,
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                }

        # 2. Selective Classification (Abstention)
        scored_pts = []
        for p in points:
            prob = p.get("probability_land")
            if prob is not None:
                conf = abs(float(prob) - 0.5) * 2.0
                corr = (bool(p["is_land"]) == bool(p["actual_land"]))
                scored_pts.append((conf, bool(p["is_land"]), bool(p["actual_land"]), corr))

        selective_curve: list[dict[str, Any]] = []
        if scored_pts:
            scored_pts.sort(key=lambda x: x[0], reverse=True)
            for cov_pct in [100, 95, 90, 85, 80, 75, 70, 60, 50]:
                take_n = max(1, int(len(scored_pts) * (cov_pct / 100.0)))
                sub = scored_pts[:take_n]
                sub_acc = sum(x[3] for x in sub) / len(sub)
                s_tp = sum(1 for x in sub if x[1] and x[2])
                s_tn = sum(1 for x in sub if not x[1] and not x[2])
                s_fp = sum(1 for x in sub if x[1] and not x[2])
                s_fn = sum(1 for x in sub if not x[1] and x[2])
                s_tpr = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
                s_tnr = s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0
                s_bal = (s_tpr + s_tnr) / 2.0
                min_conf = sub[-1][0]
                selective_curve.append({
                    "coverage_percent": cov_pct,
                    "cell_count": take_n,
                    "accuracy": round(sub_acc, 4),
                    "balanced_accuracy": round(s_bal, 4),
                    "min_confidence_threshold": round(min_conf, 4),
                })

        # Save back into data structure
        data["distance_stratification"] = dist_strat
        data["selective_classification"] = selective_curve

        with open(fpath, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out)

        # Also update evaluation.json in outputs/ if exists
        eval_fpath = OUTPUTS_DIR / f"{run_id}_evaluation.json"
        if eval_fpath.exists():
            with open(eval_fpath, "r", encoding="utf-8") as ef:
                ef_data = json.load(ef)
            ef_data["distance_stratification"] = dist_strat
            ef_data["selective_classification"] = selective_curve
            with open(eval_fpath, "w", encoding="utf-8") as ef:
                json.dump(ef_data, ef, indent=2)

        summary_results[run_id] = {
            "model": data.get("model", run_id),
            "total_points": total,
            "distance_stratification": dist_strat,
            "selective_classification": selective_curve,
        }

    # Write summary
    summary_out = OUTPUTS_DIR / "distance_and_selective_metrics_summary.json"
    with open(summary_out, "w", encoding="utf-8") as sf:
        json.dump(summary_results, sf, indent=2)

    print(f"\nAll runs enriched! Summary written to {summary_out}")


if __name__ == "__main__":
    main()
