"""Ground-truth geographic evaluation and error analysis."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    from shapely import prepare
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

from latent_earth.grid import get_latitudes, get_longitudes
from latent_earth.models import Coordinate, Prediction

logger = logging.getLogger(__name__)

DEFAULT_GEOJSON_PATH = Path(__file__).parent / "data" / "ne_110m_land.geojson"


class GroundTruthEvaluator:
    """Evaluates model predictions against Natural Earth land polygons."""

    def __init__(self, geojson_path: str | Path | None = None) -> None:
        if not HAS_SHAPELY:
            raise ImportError(
                "Shapely is required for ground-truth evaluation. Install it with: uv add shapely"
            )

        self.path = Path(geojson_path) if geojson_path else DEFAULT_GEOJSON_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"Natural Earth GeoJSON not found at: {self.path}")

        logger.info("Loading ground truth land polygons from %s...", self.path)
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        polygons = [shape(feature["geometry"]) for feature in features if feature.get("geometry")]
        self.land_union = unary_union(polygons)
        # Prepare geometry for fast spatial point-in-polygon queries
        prepare(self.land_union)
        self.coastline = self.land_union.boundary
        prepare(self.coastline)
        logger.info("Ground truth geometry and coastline prepared.")

    def is_land(self, lat: float, lon: float) -> bool:
        """Check whether a (latitude, longitude) coordinate intersects land ground truth."""
        # Longitude is X, Latitude is Y
        pt = Point(lon, lat)
        return bool(self.land_union.contains(pt) or self.land_union.touches(pt))

    def distance_to_coastline(self, lat: float, lon: float) -> float:
        """Calculate minimum Euclidean distance in degrees to the ground truth coastline."""
        pt = Point(lon, lat)
        return float(pt.distance(self.coastline))

    def is_coastal(self, lat: float, lon: float, threshold_degrees: float = 1.0) -> bool:
        """Check whether a coordinate is within threshold degrees of the coastline."""
        return self.distance_to_coastline(lat, lon) <= threshold_degrees


def evaluate_predictions(
    predictions: Sequence[Prediction],
    evaluator: GroundTruthEvaluator | None = None,
    geojson_path: str | Path | None = None,
    coastal_distance_degrees: float = 1.0,
) -> dict[str, Any]:
    """Calculate full evaluation metrics and error analysis, including area-weighting and coastal metrics."""
    if evaluator is None:
        evaluator = GroundTruthEvaluator(geojson_path)

    total = len(predictions)
    if total == 0:
        return {"error": "No predictions provided to evaluate"}

    tp = 0  # Predicted land, actual land
    tn = 0  # Predicted water, actual water
    fp = 0  # Predicted land, actual water
    fn = 0  # Predicted water, actual land

    # Area-weighted sums (cos(lat) weighting)
    weighted_tp = 0.0
    weighted_tn = 0.0
    weighted_fp = 0.0
    weighted_fn = 0.0
    total_weight = 0.0

    brier_sum = 0.0
    weighted_brier_sum = 0.0
    log_loss_sum = 0.0
    weighted_log_loss_sum = 0.0
    has_probs = False

    # Coastal metrics tracking
    coastal_count = 0
    coastal_correct = 0
    coastal_brier_sum = 0.0
    coastal_weighted_correct = 0.0
    coastal_weighted_brier_sum = 0.0
    coastal_total_weight = 0.0

    confidently_wrong: list[dict[str, Any]] = []
    uncertain_coords: list[dict[str, Any]] = []

    # 10 calibration bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bins = [[] for _ in range(10)]

    for pred in predictions:
        lat = pred.coordinate.latitude
        lon = pred.coordinate.longitude
        actual_land = evaluator.is_land(lat, lon)
        actual_y = 1.0 if actual_land else 0.0
        predicted_land = pred.is_land
        prob = pred.probability_land if pred.probability_land is not None else actual_y

        # Area weight proportional to cos(latitude)
        lat_rad = math.radians(lat)
        weight = max(0.0, math.cos(lat_rad))
        total_weight += weight

        # Coastal proximity evaluation
        dist_to_coast = evaluator.distance_to_coastline(lat, lon)
        is_coastal_pt = dist_to_coast <= coastal_distance_degrees
        pred.is_coastal = is_coastal_pt
        pred.distance_to_coastline = round(dist_to_coast, 4)

        is_correct = (predicted_land == actual_land)

        if is_coastal_pt:
            coastal_count += 1
            coastal_total_weight += weight
            if is_correct:
                coastal_correct += 1
                coastal_weighted_correct += weight

        if pred.probability_land is not None:
            has_probs = True
            brier_pt = (prob - actual_y) ** 2
            brier_sum += brier_pt
            weighted_brier_sum += weight * brier_pt

            if is_coastal_pt:
                coastal_brier_sum += brier_pt
                coastal_weighted_brier_sum += weight * brier_pt

            p_clipped = max(1e-7, min(1.0 - 1e-7, prob))
            ll_pt = -(actual_y * math.log(p_clipped) + (1.0 - actual_y) * math.log(1.0 - p_clipped))
            log_loss_sum += ll_pt
            weighted_log_loss_sum += weight * ll_pt

            # Calibration bin
            b_idx = min(9, int(prob * 10))
            bins[b_idx].append((prob, actual_y))

            # Uncertainty
            uncertain_coords.append({
                "latitude": lat,
                "longitude": lon,
                "probability_land": prob,
                "uncertainty": pred.uncertainty,
                "actual_land": actual_land,
                "is_coastal": is_coastal_pt,
                "distance_to_coastline": round(dist_to_coast, 4),
            })

            # Check if wrongly classified
            if not is_correct:
                confidence = prob if predicted_land else (1.0 - prob)
                confidently_wrong.append({
                    "latitude": lat,
                    "longitude": lon,
                    "probability_land": prob,
                    "predicted_land": predicted_land,
                    "actual_land": actual_land,
                    "is_coastal": is_coastal_pt,
                    "confidence_in_error": round(confidence, 4),
                })

        pred.actual_land = actual_land
        if predicted_land and actual_land:
            pred.error_category = "correct_land"
            tp += 1
            weighted_tp += weight
        elif not predicted_land and not actual_land:
            pred.error_category = "correct_water"
            tn += 1
            weighted_tn += weight
        elif predicted_land and not actual_land:
            pred.error_category = "false_land"
            fp += 1
            weighted_fp += weight
        else:
            pred.error_category = "false_water"
            fn += 1
            weighted_fn += weight

    # Unweighted Grid Metrics
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    brier_score = brier_sum / total if has_probs else None
    log_loss = log_loss_sum / total if has_probs else None

    # Area-Weighted Metrics
    weighted_acc = (weighted_tp + weighted_tn) / total_weight if total_weight > 0 else 0.0
    weighted_prec = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    weighted_rec = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    weighted_f1 = (
        2 * weighted_prec * weighted_rec / (weighted_prec + weighted_rec)
        if (weighted_prec + weighted_rec) > 0
        else 0.0
    )
    weighted_brier = weighted_brier_sum / total_weight if (has_probs and total_weight > 0) else None
    weighted_log_loss = weighted_log_loss_sum / total_weight if (has_probs and total_weight > 0) else None

    # Coastal Metrics
    coastal_acc = coastal_correct / coastal_count if coastal_count > 0 else None
    coastal_brier = coastal_brier_sum / coastal_count if (has_probs and coastal_count > 0) else None
    coastal_weighted_acc = coastal_weighted_correct / coastal_total_weight if coastal_total_weight > 0 else None
    coastal_weighted_brier = (
        coastal_weighted_brier_sum / coastal_total_weight
        if (has_probs and coastal_total_weight > 0)
        else None
    )

    # Sort analysis lists
    confidently_wrong.sort(key=lambda x: x["confidence_in_error"], reverse=True)
    uncertain_coords.sort(key=lambda x: abs(x["probability_land"] - 0.5))

    # Calibration table
    calibration_curve: list[dict[str, Any]] = []
    for i, b in enumerate(bins):
        if b:
            mean_prob = sum(p for p, _ in b) / len(b)
            observed_frac = sum(y for _, y in b) / len(b)
            calibration_curve.append({
                "bin": f"{i/10:.1f}-{(i+1)/10:.1f}",
                "mean_predicted_prob": round(mean_prob, 4),
                "observed_land_fraction": round(observed_frac, 4),
                "count": len(b),
            })

    results = {
        "total_evaluated": total,
        "coastal_threshold_degrees": coastal_distance_degrees,
        "coastal_cell_count": coastal_count,
        "coastal_cell_fraction": round(coastal_count / total, 4),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "brier_score": round(brier_score, 4) if brier_score is not None else None,
            "log_loss": round(log_loss, 4) if log_loss is not None else None,
            "area_weighted_accuracy": round(weighted_acc, 4),
            "area_weighted_brier_score": round(weighted_brier, 4) if weighted_brier is not None else None,
            "coastal_accuracy": round(coastal_acc, 4) if coastal_acc is not None else None,
            "coastal_brier_score": round(coastal_brier, 4) if coastal_brier is not None else None,
            "coastal_area_weighted_accuracy": round(coastal_weighted_acc, 4) if coastal_weighted_acc is not None else None,
        },
        "grid_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "brier_score": round(brier_score, 4) if brier_score is not None else None,
            "log_loss": round(log_loss, 4) if log_loss is not None else None,
        },
        "area_weighted_metrics": {
            "accuracy": round(weighted_acc, 4),
            "precision": round(weighted_prec, 4),
            "recall": round(weighted_rec, 4),
            "f1": round(weighted_f1, 4),
            "brier_score": round(weighted_brier, 4) if weighted_brier is not None else None,
            "log_loss": round(weighted_log_loss, 4) if weighted_log_loss is not None else None,
        },
        "coastal_metrics": {
            "coastal_distance_threshold_deg": coastal_distance_degrees,
            "coastal_cell_count": coastal_count,
            "coastal_cell_fraction": round(coastal_count / total, 4),
            "accuracy": round(coastal_acc, 4) if coastal_acc is not None else None,
            "brier_score": round(coastal_brier, 4) if coastal_brier is not None else None,
            "area_weighted_accuracy": round(coastal_weighted_acc, 4) if coastal_weighted_acc is not None else None,
            "area_weighted_brier_score": round(coastal_weighted_brier, 4) if coastal_weighted_brier is not None else None,
        },
        "confusion_matrix": {
            "true_positive_land": tp,
            "true_negative_water": tn,
            "false_positive_land": fp,
            "false_negative_water": fn,
        },
        "calibration": calibration_curve,
        "most_confidently_wrong": confidently_wrong[:20],
        "most_uncertain": uncertain_coords[:20],
    }
    return results


def render_error_map(
    predictions: Sequence[Prediction],
    step_degrees: float,
    output_path: str | Path,
    evaluator: GroundTruthEvaluator,
    convention: str = "cell_center",
    dpi: int = 300,
) -> Path:
    """Render 4-class error map: Correct Land, Correct Water, False Land, False Water."""
    lats = get_latitudes(step_degrees, convention=convention, north_to_south=True)
    lons = get_longitudes(step_degrees, convention=convention)

    n_lat = len(lats)
    n_lon = len(lons)
    extent = (-180.0, 180.0, -90.0, 90.0)

    # 0 = TN (Correct Water), 1 = TP (Correct Land), 2 = FP (False Land), 3 = FN (False Water)
    error_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    lat_map = {lat: i for i, lat in enumerate(lats)}
    lon_map = {lon: j for j, lon in enumerate(lons)}

    for pred in predictions:
        lat = round(pred.coordinate.latitude, 6)
        lon = round(pred.coordinate.longitude, 6)
        if lat in lat_map and lon in lon_map:
            r = lat_map[lat]
            c = lon_map[lon]
            actual_land = evaluator.is_land(lat, lon)
            pred_land = pred.is_land

            if not pred_land and not actual_land:
                cat = 0  # TN Correct Water
            elif pred_land and actual_land:
                cat = 1  # TP Correct Land
            elif pred_land and not actual_land:
                cat = 2  # FP False Land
            else:
                cat = 3  # FN False Water
            error_grid[r, c] = cat

    # Visual palette
    # Dark blue (Water), Emerald green (Land), Vibrant red (False Land), Amber orange (False Water)
    cmap = mcolors.ListedColormap(["#0F2942", "#10B981", "#EF4444", "#F59E0B"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#090D16")
    ax.set_facecolor("#090D16")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_xticklabels(["180°W", "120°W", "60°W", "0°", "60°E", "120°E", "180°E"], fontsize=9, color="#9CA3AF")
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"], fontsize=9, color="#9CA3AF")
    ax.grid(color="#1F2937", linestyle=":", linewidth=0.6, alpha=0.7)

    im = ax.imshow(
        error_grid,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title("Latent Earth: Ground-Truth Error Map (Natural Earth 110m)", fontsize=14, color="#F3F4F6", pad=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.65, ticks=[0, 1, 2, 3])
    cbar.ax.set_xticklabels([
        "Correct Water (TN)",
        "Correct Land (TP)",
        "False Land (FP)",
        "False Water (FN)",
    ], color="#F3F4F6", fontsize=9)
    cbar.ax.tick_params(size=0)
    cbar.outline.set_edgecolor("#374151")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi, facecolor="#090D16", edgecolor="none")
    plt.close(fig)
    logger.info("Saved ground truth error map to %s", p)
    return p
