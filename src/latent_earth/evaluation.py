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
    from shapely.geometry import Point, box, shape
    from shapely.ops import nearest_points, unary_union
    from shapely import prepare
    import shapely.affinity
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from latent_earth.grid import get_latitudes, get_longitudes
from latent_earth.models import Coordinate, Prediction

logger = logging.getLogger(__name__)

DEFAULT_HIGH_RES_GEOJSON_PATH = Path(__file__).parent / "data" / "ne_10m_land.geojson"
DEFAULT_GEOJSON_PATH = Path(__file__).parent / "data" / "ne_110m_land.geojson"


class GroundTruthEvaluator:
    """Evaluates model predictions against Natural Earth land polygons."""

    def __init__(self, geojson_path: str | Path | None = None) -> None:
        if not HAS_SHAPELY:
            raise ImportError(
                "Shapely is required for ground-truth evaluation. Install it with: uv add shapely"
            )

        if geojson_path:
            self.path = Path(geojson_path)
        elif DEFAULT_HIGH_RES_GEOJSON_PATH.exists():
            self.path = DEFAULT_HIGH_RES_GEOJSON_PATH
        else:
            self.path = DEFAULT_GEOJSON_PATH

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

        if HAS_PYPROJ:
            self.geod = pyproj.Geod(ellps="WGS84")
        else:
            self.geod = None
        logger.info("Ground truth geometry and WGS84 geodesic coastline prepared (%s).", self.path.name)

    def is_land(self, lat: float, lon: float) -> bool:
        """Check whether a (latitude, longitude) coordinate intersects land ground truth."""
        # Longitude is X, Latitude is Y
        pt = Point(lon, lat)
        return bool(self.land_union.contains(pt) or self.land_union.touches(pt))

    def distance_to_coastline(self, lat: float, lon: float) -> float:
        """Calculate minimum Euclidean distance in degrees to the ground truth coastline."""
        pt = Point(lon, lat)
        return float(pt.distance(self.coastline))

    def distance_to_coastline_km(self, lat: float, lon: float) -> float:
        """Calculate exact WGS84 geodesic distance in kilometers to the coastline.

        Combines initial spherical GEOS nearest-point query with local aspect-ratio
        cosine refinement within 250km to eliminate Euclidean degree distortions at higher latitudes.
        """
        pt = Point(lon, lat)
        np_pt = nearest_points(pt, self.coastline)[1]

        if self.geod is None:
            # Fallback to mean Earth radius spherical approximation
            deg_dist = float(pt.distance(self.coastline))
            return deg_dist * 111.139

        _, _, d_meters = self.geod.inv(lon, lat, np_pt.x, np_pt.y)
        d_km = d_meters / 1000.0

        cos_lat = math.cos(math.radians(lat))
        if d_km < 250.0 and abs(cos_lat - 1.0) > 0.02:
            d_deg_lat = (d_km * 1.30) / 110.57
            d_deg_lon = (d_km * 1.30) / (111.32 * max(0.05, cos_lat))
            s_box = box(lon - d_deg_lon, lat - d_deg_lat, lon + d_deg_lon, lat + d_deg_lat)
            local_c = self.coastline.intersection(s_box)
            if not local_c.is_empty:
                scaled_c = shapely.affinity.scale(local_c, xfact=cos_lat, yfact=1.0, origin=(lon, lat))
                np_scaled = nearest_points(pt, scaled_c)[1]
                unscaled_x = lon + (np_scaled.x - lon) / cos_lat
                unscaled_y = np_scaled.y
                _, _, refined_d_meters = self.geod.inv(lon, lat, unscaled_x, unscaled_y)
                return refined_d_meters / 1000.0

        return d_km

    def is_coastal(
        self,
        lat: float,
        lon: float,
        threshold_km: float = 100.0,
        threshold_degrees: float | None = None,
    ) -> bool:
        """Check whether a coordinate is within the coastal band threshold."""
        if threshold_degrees is not None:
            return self.distance_to_coastline(lat, lon) <= threshold_degrees
        return self.distance_to_coastline_km(lat, lon) <= threshold_km


def evaluate_predictions(
    predictions: Sequence[Prediction],
    evaluator: GroundTruthEvaluator | None = None,
    geojson_path: str | Path | None = None,
    coastal_distance_km: float = 100.0,
    coastal_distance_degrees: float | None = None,
) -> dict[str, Any]:
    """Calculate full evaluation metrics and error analysis, including area-weighting and geodesic coastal metrics."""
    if evaluator is None:
        evaluator = GroundTruthEvaluator(geojson_path)

    # Backward compatibility with degree thresholds
    if coastal_distance_degrees is not None and coastal_distance_km == 100.0:
        coastal_distance_km = coastal_distance_degrees * 111.0

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

    # Coastal metrics tracking (geodesic distance <= threshold_km)
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

        # WGS84 geodesic coastal proximity evaluation in kilometers
        dist_to_coast_km = evaluator.distance_to_coastline_km(lat, lon)
        is_coastal_pt = dist_to_coast_km <= coastal_distance_km
        pred.is_coastal = is_coastal_pt
        pred.distance_to_coastline_km = round(dist_to_coast_km, 2)
        pred.distance_to_coastline = round(dist_to_coast_km, 2)

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
                "distance_to_coastline_km": round(dist_to_coast_km, 2),
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
                    "distance_to_coastline_km": round(dist_to_coast_km, 2),
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

    coastal_deg_equiv = round(coastal_distance_km / 111.0, 3)

    results = {
        "total_evaluated": total,
        "coastal_threshold_km": coastal_distance_km,
        "coastal_threshold_degrees": coastal_deg_equiv,
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
            "coastal_distance_threshold_km": coastal_distance_km,
            "coastal_distance_threshold_deg": coastal_deg_equiv,
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


def compute_cross_resolution_agreement(
    coarse_predictions: Sequence[Prediction | dict[str, Any]],
    fine_predictions: Sequence[Prediction | dict[str, Any]],
    coarse_step: float,
    fine_step: float,
    evaluator: GroundTruthEvaluator | None = None,
) -> dict[str, Any]:
    """Compute spatial and probabilistic agreement between coarse and fine nested grids.

    In the cell_center quadtree hierarchy:
    Each coarse parent cell of step S contains (S / s)^2 nested fine child cells of step s.
    For example:
    - 2.0° -> 1.0°: 4 child cells per parent (528 evaluations across 132 parents)
    - 1.0° -> 0.5°: 4 child cells per parent (2,112 evaluations across 528 parents)
    - 2.0° -> 0.5°: 16 grandchild cells per parent (2,112 evaluations across 132 parents)
    """
    if not coarse_predictions or not fine_predictions:
        return {"error": "Empty predictions provided"}

    if evaluator is None:
        try:
            evaluator = GroundTruthEvaluator()
        except Exception:
            evaluator = None

    def to_item(p: Prediction | dict[str, Any]) -> dict[str, Any]:
        if isinstance(p, dict):
            return p
        return p.to_dict()

    coarse_items = [to_item(p) for p in coarse_predictions]
    fine_items = [to_item(p) for p in fine_predictions]

    # Index fine predictions by (lat, lon) rounded to 4 decimals
    fine_by_coord: dict[tuple[float, float], dict[str, Any]] = {
        (round(float(p["latitude"]), 4), round(float(p["longitude"]), 4)): p
        for p in fine_items
    }

    subdiv = int(round(coarse_step / fine_step))
    if subdiv <= 1:
        raise ValueError(f"coarse_step ({coarse_step}) must be greater than fine_step ({fine_step})")

    total_parents = len(coarse_items)
    children_per_parent = subdiv * subdiv

    total_child_evals = 0
    agree_child_evals = 0
    majority_agree_count = 0

    both_land = 0
    both_water = 0
    parent_land_child_water = 0
    parent_water_child_land = 0

    # Ground-truth transition tracking
    w_to_l_beneficial = 0  # Parent=water, child=land, GT is land
    w_to_l_erroneous = 0   # Parent=water, child=land, GT is water (erroneous land expansion)
    l_to_w_beneficial = 0  # Parent=land, child=water, GT is water (beneficial boundary refinement)
    l_to_w_erroneous = 0   # Parent=land, child=water, GT is land (erroneous water hole)

    coastal_child_evals = 0
    coastal_agree_evals = 0
    interior_child_evals = 0
    interior_agree_evals = 0

    prob_deltas: list[float] = []
    parent_probs: list[float] = []
    mean_child_probs: list[float] = []

    half_coarse = coarse_step / 2.0
    half_fine = fine_step / 2.0

    for parent in coarse_items:
        p_lat = float(parent["latitude"])
        p_lon = float(parent["longitude"])
        p_land = bool(parent["is_land"])
        p_prob = parent.get("probability_land")
        if p_prob is None:
            p_prob = 1.0 if p_land else 0.0
        p_coastal = parent.get("is_coastal")

        # Find child cells strictly inside parent's spatial cell extent
        children: list[dict[str, Any]] = []
        for i in range(subdiv):
            c_lat = (p_lat - half_coarse) + half_fine + i * fine_step
            for j in range(subdiv):
                c_lon = (p_lon - half_coarse) + half_fine + j * fine_step
                child = fine_by_coord.get((round(c_lat, 4), round(c_lon, 4)))
                if child:
                    children.append(child)

        c_lands = [bool(c["is_land"]) for c in children]
        c_probs = [
            float(c.get("probability_land") if c.get("probability_land") is not None else (1.0 if c["is_land"] else 0.0))
            for c in children
        ]

        for idx_c, c_land in enumerate(c_lands):
            child_dict = children[idx_c]
            total_child_evals += 1
            if c_land == p_land:
                agree_child_evals += 1
                if p_land:
                    both_land += 1
                else:
                    both_water += 1
            else:
                c_lat_val = float(child_dict["latitude"])
                c_lon_val = float(child_dict["longitude"])
                c_gt = evaluator.is_land(c_lat_val, c_lon_val) if evaluator is not None else None

                if p_land:
                    parent_land_child_water += 1
                    if c_gt is not None:
                        if not c_gt:
                            l_to_w_beneficial += 1
                        else:
                            l_to_w_erroneous += 1
                else:
                    parent_water_child_land += 1
                    if c_gt is not None:
                        if c_gt:
                            w_to_l_beneficial += 1
                        else:
                            w_to_l_erroneous += 1

            if p_coastal is not None:
                if p_coastal:
                    coastal_child_evals += 1
                    if c_land == p_land:
                        coastal_agree_evals += 1
                else:
                    interior_child_evals += 1
                    if c_land == p_land:
                        interior_agree_evals += 1

        if c_lands:
            land_votes = sum(1 for c in c_lands if c)
            water_votes = len(c_lands) - land_votes
            if land_votes > water_votes:
                maj_decision = True
            elif water_votes > land_votes:
                maj_decision = False
            else:
                maj_decision = p_land

            if maj_decision == p_land:
                majority_agree_count += 1

            mean_c_prob = sum(c_probs) / len(c_probs)
            prob_deltas.append(abs(mean_c_prob - p_prob))
            parent_probs.append(p_prob)
            mean_child_probs.append(mean_c_prob)

    child_agree_rate = agree_child_evals / total_child_evals if total_child_evals > 0 else 0.0
    maj_agree_rate = majority_agree_count / total_parents if total_parents > 0 else 0.0
    mean_delta = sum(prob_deltas) / len(prob_deltas) if prob_deltas else 0.0

    n = len(parent_probs)
    r = 0.0
    if n > 1:
        mean_p = sum(parent_probs) / n
        mean_c = sum(mean_child_probs) / n
        num = sum((p - mean_p) * (c - mean_c) for p, c in zip(parent_probs, mean_child_probs))
        den = math.sqrt(sum((p - mean_p) ** 2 for p in parent_probs) * sum((c - mean_c) ** 2 for c in mean_child_probs))
        r = num / den if den > 0 else 0.0

    res = {
        "coarse_resolution_deg": coarse_step,
        "fine_resolution_deg": fine_step,
        "subdivision_factor": subdiv,
        "children_per_parent": children_per_parent,
        "total_parents": total_parents,
        "total_child_evaluations": total_child_evals,
        "child_agreement_rate": round(child_agree_rate, 4),
        "majority_agreement_rate": round(maj_agree_rate, 4),
        "mean_probability_delta": round(mean_delta, 4),
        "probability_correlation": round(r, 4),
        "confusion_breakdown": {
            "both_land": both_land,
            "both_water": both_water,
            "parent_land_child_water": parent_land_child_water,
            "parent_water_child_land": parent_water_child_land,
        },
    }

    if evaluator is not None:
        res["transitions"] = {
            "water_to_land": {
                "total": parent_water_child_land,
                "beneficial_refinement": w_to_l_beneficial,
                "erroneous_expansion": w_to_l_erroneous,
                "beneficial_pct": round(w_to_l_beneficial / parent_water_child_land * 100, 1) if parent_water_child_land > 0 else 0.0,
                "erroneous_pct": round(w_to_l_erroneous / parent_water_child_land * 100, 1) if parent_water_child_land > 0 else 0.0,
            },
            "land_to_water": {
                "total": parent_land_child_water,
                "beneficial_refinement": l_to_w_beneficial,
                "erroneous_reduction": l_to_w_erroneous,
                "beneficial_pct": round(l_to_w_beneficial / parent_land_child_water * 100, 1) if parent_land_child_water > 0 else 0.0,
                "erroneous_pct": round(l_to_w_erroneous / parent_land_child_water * 100, 1) if parent_land_child_water > 0 else 0.0,
            }
        }

    if coastal_child_evals > 0:
        res["coastal_child_agreement_rate"] = round(coastal_agree_evals / coastal_child_evals, 4)
    if interior_child_evals > 0:
        res["interior_child_agreement_rate"] = round(interior_agree_evals / interior_child_evals, 4)

    return res


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
