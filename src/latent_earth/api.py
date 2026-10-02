"""FastAPI application for serving Latent Earth experiment results to web frontends."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from latent_earth.cache import PredictionCache
from latent_earth.models import RunManifest

logger = logging.getLogger(__name__)


def find_repo_root() -> Path:
    """Find repository root containing outputs/ and frontend/dist/."""
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (parent / "frontend" / "dist").exists() or (parent / "outputs").exists():
            return parent
    cwd = Path.cwd()
    if (cwd / "frontend" / "dist").exists() or (cwd / "outputs").exists():
        return cwd
    if (cwd / "latent-earth" / "frontend" / "dist").exists():
        return cwd / "latent-earth"
    return here.parent.parent.parent


REPO_ROOT = find_repo_root()
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"
LAND_GEOJSON_PATH = Path(__file__).resolve().parent / "data" / "ne_110m_land.geojson"

app = FastAPI(
    title="Latent Earth API",
    description="Backend API serving decision model geographic reconstructions and evaluation metrics.",
    version="0.1.0",
)

_evaluator: Any = None


def get_evaluator() -> Any:
    global _evaluator
    if _evaluator is None:
        try:
            from latent_earth.evaluation import GroundTruthEvaluator
            _evaluator = GroundTruthEvaluator(LAND_GEOJSON_PATH)
        except Exception as exc:
            logger.warning("Failed to initialize GroundTruthEvaluator: %s", exc)
    return _evaluator


# Enable CORS for local development (e.g. Vite dev server on localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "app": "latent-earth", "version": "0.1.0"}


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    """List all available experiment runs in the outputs directory."""
    if not DEFAULT_OUTPUTS_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []
    jsonl_files = sorted(DEFAULT_OUTPUTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    for f in jsonl_files:
        run_id = f.stem
        eval_file = DEFAULT_OUTPUTS_DIR / f"{run_id}_evaluation.json"
        meta_file = DEFAULT_OUTPUTS_DIR / f"{run_id}_metadata.json"

        provider = "unknown"
        model = "unknown"
        prompt_version = "unknown"
        point_count = 0
        meta_data: dict[str, Any] = {}

        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta_data = json.load(mf)
                    provider = meta_data.get("provider", provider)
                    model = meta_data.get("model", model)
                    prompt_version = meta_data.get("prompt_version", prompt_version)
            except Exception:
                pass

        with open(f, "r", encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                line_str = line.strip()
                if not line_str:
                    continue
                point_count += 1
                if i == 0 and not meta_data:
                    try:
                        first_data = json.loads(line_str)
                        provider = first_data.get("provider", provider)
                        model = first_data.get("model", model)
                        prompt_version = first_data.get("prompt_version", prompt_version)
                    except Exception:
                        pass

        stat = f.stat()
        has_eval = eval_file.exists()

        eval_summary = None
        if has_eval:
            try:
                with open(eval_file, "r", encoding="utf-8") as ef:
                    eval_data = json.load(ef)
                    eval_summary = eval_data.get("metrics")
            except Exception:
                pass

        step = meta_data.get("grid_resolution") or meta_data.get("step_degrees")
        if not step:
            # Infer from filename e.g. _2deg_ or _0_5deg_
            m = re.search(r"_(\d+(?:_\d+)?)deg_", run_id)
            if m:
                step = float(m.group(1).replace("_", "."))
            else:
                step = 4.0

        region_name = meta_data.get("region_name", "global")
        if region_name == "global" and "japan_korea" in run_id:
            region_name = "japan_korea"

        runs.append({
            "run_id": run_id,
            "filename": f.name,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "region_name": region_name,
            "step_degrees": step,
            "point_count": point_count,
            "has_evaluation": has_eval,
            "metrics": eval_summary,
            "modified_time": stat.st_mtime,
            "size_bytes": stat.st_size,
        })

    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Retrieve full run details, points, and evaluation metrics."""
    jsonl_file = DEFAULT_OUTPUTS_DIR / f"{run_id}.jsonl"
    if not jsonl_file.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    cache = PredictionCache(jsonl_file, strict_validation=False)
    preds = list(cache.load().values())

    meta_file = DEFAULT_OUTPUTS_DIR / f"{run_id}_metadata.json"
    meta_data: dict[str, Any] = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                meta_data = json.load(mf)
        except Exception as exc:
            logger.warning("Could not read metadata file %s: %s", meta_file, exc)

    eval_file = DEFAULT_OUTPUTS_DIR / f"{run_id}_evaluation.json"
    eval_data: dict[str, Any] | None = None
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as ef:
                eval_data = json.load(ef)
        except Exception as exc:
            logger.warning("Could not read evaluation file %s: %s", eval_file, exc)

    # Attach ground truth and error categories to predictions so UI Ground Truth and Error Overlay work
    evaluator = get_evaluator()
    if evaluator is not None and preds:
        try:
            from latent_earth.evaluation import evaluate_predictions
            eval_metrics = evaluate_predictions(preds, evaluator=evaluator)
            if not eval_data:
                eval_data = eval_metrics
        except Exception as exc:
            logger.warning("Could not evaluate predictions against ground truth: %s", exc)

    first_pred = preds[0] if preds else None
    provider = meta_data.get("provider") or (first_pred.provider if first_pred else "unknown")
    model = meta_data.get("model") or (first_pred.model if first_pred else "unknown")
    prompt_version = meta_data.get("prompt_version") or (first_pred.prompt_version if first_pred else "unknown")

    step = meta_data.get("grid_resolution") or meta_data.get("step_degrees")
    if not step:
        if len(preds) > 1:
            lats = sorted({round(p.coordinate.latitude, 6) for p in preds})
            if len(lats) > 1:
                diffs = [round(abs(lats[i] - lats[i - 1]), 4) for i in range(1, len(lats))]
                step = min(diffs)
            else:
                step = 4.0
        else:
            step = 4.0

    points_data = [p.to_dict() for p in preds]

    region_obj = meta_data.get("bounding_box") or (
        {"name": meta_data.get("region_name")} if meta_data.get("region_name") else None
    )

    manifest = RunManifest(
        run_id=run_id,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        step_degrees=float(step),
        convention="cell_center",
        total_coordinates=len(preds),
        sampled_count=len(preds),
        created_at=preds[0].timestamp if preds else "",
        region=region_obj,
        metrics=eval_data.get("metrics") if eval_data else None,
        points=points_data,
    )

    response_payload = manifest.to_dict()
    if eval_data:
        response_payload["full_evaluation"] = eval_data
        response_payload["grid_metrics"] = eval_data.get("grid_metrics")
        response_payload["area_weighted_metrics"] = eval_data.get("area_weighted_metrics")
        response_payload["coastal_metrics"] = eval_data.get("coastal_metrics")

    return response_payload


@app.get("/api/ground-truth")
async def get_ground_truth_geojson():
    """Serve bundled Natural Earth 110m land polygons GeoJSON."""
    if not LAND_GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="Ground truth GeoJSON not found.")
    return FileResponse(LAND_GEOJSON_PATH, media_type="application/json")


# Direct route for root index.html
@app.get("/")
async def serve_root():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "message": "Latent Earth API is running. Build frontend with 'cd frontend && npm run build'.",
        "status": "ok",
    })


# Favicon handler
@app.get("/favicon.ico")
async def favicon():
    fav_file = FRONTEND_DIST_DIR / "favicon.ico"
    if fav_file.exists():
        return FileResponse(fav_file)
    return Response(status_code=204)


# Mount static assets if dist exists
if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")

# Fallback mount for other static files
if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
