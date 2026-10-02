"""Visualization and map reconstruction pipeline using Matplotlib."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from latent_earth.grid import get_latitudes, get_longitudes
from latent_earth.models import Prediction

logger = logging.getLogger(__name__)

# Dark theme palette
BG_DARK = "#090D16"
SURFACE_DARK = "#111827"
TEXT_LIGHT = "#F3F4F6"
TEXT_MUTED = "#9CA3AF"
ACCENT_BLUE = "#38BDF8"

# Earth colormap: Deep ocean navy -> Cyan shore -> Amber beach -> Lush land green
EARTH_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "earth_latent",
    [
        (0.00, "#040814"),  # Deep abyss
        (0.20, "#0C2340"),  # Open ocean
        (0.40, "#1D4ED8"),  # Coastal waters
        (0.50, "#06B6D4"),  # Shelf / Shoreline threshold
        (0.60, "#D97706"),  # Coastal sands
        (0.75, "#22C55E"),  # Vegetated plains
        (0.90, "#16A34A"),  # Forest highlands
        (1.00, "#14532D"),  # Dense continental heartland
    ],
    N=256,
)


UNCERTAINTY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "uncertainty_glow",
    [
        (0.00, "#4A355A"),  # Certain: Muted plum / lavender-gray
        (0.28, "#7E22CE"),  # Purple
        (0.55, "#C026D3"),  # Magenta
        (0.80, "#EA580C"),  # Orange
        (1.00, "#FDE047"),  # Max uncertainty: Bright yellow
    ],
    N=256,
)
UNCERTAINTY_CMAP = UNCERTAINTY_CMAP.with_extremes(bad=BG_DARK)


def predictions_to_grid(
    predictions: Sequence[Prediction],
    step_degrees: float,
    convention: str = "cell_center",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Convert an unordered or ordered sequence of predictions into 2D grids.

    Returns:
        (prob_grid, binary_grid, uncertainty_grid, extent)
    """
    lats = get_latitudes(step_degrees, convention=convention, north_to_south=True)
    lons = get_longitudes(step_degrees, convention=convention)

    n_lat = len(lats)
    n_lon = len(lons)
    half_step = step_degrees / 2.0

    extent = (-180.0, 180.0, -90.0, 90.0)

    prob_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    binary_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    uncertainty_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    # Index lookups
    lat_indices = {lat: idx for idx, lat in enumerate(lats)}
    lon_indices = {lon: idx for idx, lon in enumerate(lons)}

    for pred in predictions:
        lat = round(pred.coordinate.latitude, 6)
        lon = round(pred.coordinate.longitude, 6)
        if lat in lat_indices and lon in lon_indices:
            r = lat_indices[lat]
            c = lon_indices[lon]
            p = pred.probability_land if pred.probability_land is not None else (1.0 if pred.is_land else 0.0)
            prob_grid[r, c] = p
            binary_grid[r, c] = 1.0 if pred.is_land else 0.0
            uncertainty_grid[r, c] = pred.uncertainty

    return prob_grid, binary_grid, uncertainty_grid, extent


def _apply_map_styling(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    """Apply consistent dark-mode styling and axes labels."""
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    # Set ticks and labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_xticklabels(["180°W", "120°W", "60°W", "0°", "60°E", "120°E", "180°E"], fontsize=9, color=TEXT_MUTED)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"], fontsize=9, color=TEXT_MUTED)

    ax.grid(color="#1F2937", linestyle=":", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(0.8)


def render_binary_map(
    binary_grid: np.ndarray,
    extent: tuple[float, float, float, float],
    output_path: str | Path,
    title: str = "Latent Earth: Binary Reconstruction",
    dpi: int = 300,
) -> Path:
    """Render binary classification map (Land = light, Water = dark)."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_DARK)
    _apply_map_styling(ax, extent)

    # Custom binary colormap: Dark navy for water (0), Ivory for land (1)
    binary_cmap = mcolors.ListedColormap(["#0B132B", "#F1F5F9"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, binary_cmap.N)

    im = ax.imshow(
        binary_grid,
        extent=extent,
        origin="upper",
        cmap=binary_cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=14, color=TEXT_LIGHT, pad=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.35, ticks=[0, 1])
    cbar.ax.set_xticklabels(["Water (P < 0.5)", "Land (P ≥ 0.5)"], color=TEXT_LIGHT, fontsize=10)
    cbar.ax.tick_params(size=0)
    cbar.outline.set_edgecolor("#374151")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi, facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    logger.info("Saved binary map to %s", p)
    return p


def render_probability_map(
    prob_grid: np.ndarray,
    extent: tuple[float, float, float, float],
    output_path: str | Path,
    title: str = "Latent Earth: Decision Probability P(Land)",
    dpi: int = 300,
) -> Path:
    """Render continuous land probability map."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_DARK)
    _apply_map_styling(ax, extent)

    im = ax.imshow(
        prob_grid,
        extent=extent,
        origin="upper",
        cmap=EARTH_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear" if prob_grid.shape[0] > 30 else "nearest",
    )

    ax.set_title(title, fontsize=14, color=TEXT_LIGHT, pad=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.45)
    cbar.set_label("P(Coordinate is Land)", color=TEXT_LIGHT, fontsize=10, labelpad=8)
    cbar.ax.tick_params(colors=TEXT_LIGHT, labelsize=9)
    cbar.outline.set_edgecolor("#374151")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi, facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    logger.info("Saved probability map to %s", p)
    return p


def render_uncertainty_map(
    uncertainty_grid: np.ndarray,
    extent: tuple[float, float, float, float],
    output_path: str | Path,
    title: str = "Latent Earth: Decision Uncertainty",
    dpi: int = 300,
) -> Path:
    """Render decision uncertainty map (highlights coastlines, islands, borders)."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_DARK)
    _apply_map_styling(ax, extent)

    im = ax.imshow(
        uncertainty_grid,
        extent=extent,
        origin="upper",
        cmap=UNCERTAINTY_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=14, color=TEXT_LIGHT, pad=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.45)
    cbar.set_label("Uncertainty: 1 - 2|P(Land) - 0.5|", color=TEXT_LIGHT, fontsize=10, labelpad=8)
    cbar.ax.tick_params(colors=TEXT_LIGHT, labelsize=9)
    cbar.outline.set_edgecolor("#374151")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi, facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    logger.info("Saved uncertainty map to %s", p)
    return p


def render_composite(
    prob_grid: np.ndarray,
    uncertainty_grid: np.ndarray,
    extent: tuple[float, float, float, float],
    output_path: str | Path,
    provider: str,
    model: str,
    total_queries: int,
    clean: bool = False,
    dpi: int = 300,
) -> Path:
    """Create a high-impact side-by-side composite suitable for GitHub and social media."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG_DARK)

    # Left: Probability map
    _apply_map_styling(ax1, extent)
    im1 = ax1.imshow(
        prob_grid,
        extent=extent,
        origin="upper",
        cmap=EARTH_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear" if prob_grid.shape[0] > 30 else "nearest",
    )
    ax1.set_title("Reconstructed P(Land)", fontsize=13, color=TEXT_LIGHT, pad=10, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.09, shrink=0.6)
    cbar1.set_label("P(Land)", color=TEXT_MUTED, fontsize=9)
    cbar1.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar1.outline.set_edgecolor("#374151")

    # Right: Uncertainty map
    _apply_map_styling(ax2, extent)
    im2 = ax2.imshow(
        uncertainty_grid,
        extent=extent,
        origin="upper",
        cmap=UNCERTAINTY_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax2.set_title("Decision Uncertainty (Coastlines & Ambiguity)", fontsize=13, color=TEXT_LIGHT, pad=10, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.09, shrink=0.6)
    cbar2.set_label("Uncertainty [0=Certain, 1=Ambiguous]", color=TEXT_MUTED, fontsize=9)
    cbar2.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar2.outline.set_edgecolor("#374151")

    if not clean:
        provider_disp = provider.upper()
        fig.suptitle(
            f"CAN A TEXT-ONLY DECISION MODEL RECONSTRUCT EARTH?\n"
            f"Latent Earth Experiment • Model: {model} ({provider_disp})",
            fontsize=16,
            color=TEXT_LIGHT,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.5,
            0.02,
            f"{total_queries:,} independent coordinate decisions • Pure coordinates only (no map, polygons, or coastline context)",
            ha="center",
            fontsize=11,
            color=ACCENT_BLUE,
            fontstyle="italic",
        )
        plt.subplots_adjust(top=0.88, bottom=0.10, left=0.04, right=0.96, wspace=0.12)
    else:
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.04, right=0.96, wspace=0.12)

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=dpi, facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    logger.info("Saved composite image to %s", p)
    return p


def render_all(
    predictions: Sequence[Prediction],
    step_degrees: float,
    output_dir: str | Path,
    base_name: str,
    provider: str,
    model: str,
    convention: str = "cell_center",
    dpi: int = 300,
) -> dict[str, Path]:
    """Generate the complete suite of visualizations."""
    prob_grid, bin_grid, unc_grid, extent = predictions_to_grid(
        predictions, step_degrees=step_degrees, convention=convention
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # 1. Binary Map
    bin_path = out / f"{base_name}_binary.png"
    paths["binary"] = render_binary_map(
        bin_grid,
        extent,
        bin_path,
        title=f"Latent Earth: Binary Map ({model}, {len(predictions):,} pts)",
        dpi=dpi,
    )

    # 2. Probability Map
    prob_path = out / f"{base_name}_probability.png"
    paths["probability"] = render_probability_map(
        prob_grid,
        extent,
        prob_path,
        title=f"Latent Earth: Reconstructed P(Land) ({model})",
        dpi=dpi,
    )

    # 3. Uncertainty Map
    unc_path = out / f"{base_name}_uncertainty.png"
    paths["uncertainty"] = render_uncertainty_map(
        unc_grid,
        extent,
        unc_path,
        title=f"Latent Earth: Model Uncertainty ({model})",
        dpi=dpi,
    )

    # 4. Social Media Composite (with title & metadata)
    comp_path = out / f"{base_name}_composite.png"
    paths["composite"] = render_composite(
        prob_grid,
        unc_grid,
        extent,
        comp_path,
        provider=provider,
        model=model,
        total_queries=len(predictions),
        clean=False,
        dpi=dpi,
    )

    # 5. Clean Composite (headless, for custom typography/design)
    comp_clean_path = out / f"{base_name}_composite_clean.png"
    paths["composite_clean"] = render_composite(
        prob_grid,
        unc_grid,
        extent,
        comp_clean_path,
        provider=provider,
        model=model,
        total_queries=len(predictions),
        clean=True,
        dpi=dpi,
    )

    return paths
