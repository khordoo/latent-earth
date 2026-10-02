"""Latitude/Longitude grid generation for Latent Earth."""

from __future__ import annotations

import numpy as np
from latent_earth.models import Coordinate


def get_grid_dimensions(step_degrees: float, convention: str = "cell_center") -> tuple[int, int]:
    """Calculate the expected (n_lat, n_lon) dimensions for a given step size."""
    if step_degrees <= 0:
        raise ValueError(f"step_degrees must be positive, got {step_degrees}")

    if convention == "cell_center":
        n_lat = int(round(180.0 / step_degrees))
        n_lon = int(round(360.0 / step_degrees))
        return n_lat, n_lon
    elif convention == "boundary":
        n_lat = int(round(180.0 / step_degrees)) + 1
        n_lon = int(round(360.0 / step_degrees))  # excludes 180 to prevent duplicate meridian
        return n_lat, n_lon
    else:
        raise ValueError(f"Unknown convention: {convention}")


def get_latitudes(step_degrees: float, convention: str = "cell_center", north_to_south: bool = True) -> list[float]:
    """Generate ordered latitude positions."""
    if step_degrees <= 0:
        raise ValueError(f"step_degrees must be positive, got {step_degrees}")

    if convention == "cell_center":
        n_lat = int(round(180.0 / step_degrees))
        half_step = step_degrees / 2.0
        # From +90 - half_step down to -90 + half_step
        if north_to_south:
            lats = [round(90.0 - half_step - i * step_degrees, 6) for i in range(n_lat)]
        else:
            lats = [round(-90.0 + half_step + i * step_degrees, 6) for i in range(n_lat)]
        return lats
    elif convention == "boundary":
        n_lat = int(round(180.0 / step_degrees)) + 1
        if north_to_south:
            lats = [round(90.0 - i * step_degrees, 6) for i in range(n_lat)]
        else:
            lats = [round(-90.0 + i * step_degrees, 6) for i in range(n_lat)]
        return lats
    else:
        raise ValueError(f"Unknown convention: {convention}")


def get_longitudes(step_degrees: float, convention: str = "cell_center") -> list[float]:
    """Generate ordered longitude positions from West (-180) to East (+180), avoiding duplicate boundary."""
    if step_degrees <= 0:
        raise ValueError(f"step_degrees must be positive, got {step_degrees}")

    if convention == "cell_center":
        n_lon = int(round(360.0 / step_degrees))
        half_step = step_degrees / 2.0
        # From -180 + half_step up to 180 - half_step
        lons = [round(-180.0 + half_step + i * step_degrees, 6) for i in range(n_lon)]
        return lons
    elif convention == "boundary":
        n_lon = int(round(360.0 / step_degrees))
        # From -180 up to strictly < 180
        lons = [round(-180.0 + i * step_degrees, 6) for i in range(n_lon)]
        return lons
    else:
        raise ValueError(f"Unknown convention: {convention}")


def generate_grid(
    step_degrees: float = 4.0,
    convention: str = "cell_center",
    north_to_south: bool = True,
) -> list[Coordinate]:
    """Generate a deterministic grid of coordinates covering the Earth.

    By default, uses the 'cell_center' convention:
    - 4-degree: 45 latitudes x 90 longitudes = 4,050 coordinates
    - 2-degree: 90 latitudes x 180 longitudes = 16,200 coordinates
    - 10-degree: 18 latitudes x 36 longitudes = 648 coordinates

    Ordering:
    - Outer loop: Latitude (North to South by default)
    - Inner loop: Longitude (West to East)
    """
    lats = get_latitudes(step_degrees, convention=convention, north_to_south=north_to_south)
    lons = get_longitudes(step_degrees, convention=convention)

    coordinates: list[Coordinate] = []
    for lat in lats:
        for lon in lons:
            coordinates.append(Coordinate(latitude=lat, longitude=lon))

    return coordinates
