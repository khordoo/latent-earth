"""Tests for grid generation."""

from __future__ import annotations

import pytest
from latent_earth.grid import generate_grid, get_grid_dimensions, get_latitudes, get_longitudes


def test_grid_dimensions_step_10():
    n_lat, n_lon = get_grid_dimensions(10.0, convention="cell_center")
    assert n_lat == 18
    assert n_lon == 36
    assert n_lat * n_lon == 648

    grid = generate_grid(step_degrees=10.0, convention="cell_center")
    assert len(grid) == 648


def test_grid_dimensions_step_4():
    n_lat, n_lon = get_grid_dimensions(4.0, convention="cell_center")
    assert n_lat == 45
    assert n_lon == 90
    assert n_lat * n_lon == 4050

    grid = generate_grid(step_degrees=4.0, convention="cell_center")
    assert len(grid) == 4050


def test_grid_dimensions_step_2():
    n_lat, n_lon = get_grid_dimensions(2.0, convention="cell_center")
    assert n_lat == 90
    assert n_lon == 180
    assert n_lat * n_lon == 16200

    grid = generate_grid(step_degrees=2.0, convention="cell_center")
    assert len(grid) == 16200


def test_no_duplicate_longitude_endpoint():
    lons = get_longitudes(4.0, convention="cell_center")
    assert -180.0 not in lons
    assert 180.0 not in lons
    assert len(lons) == len(set(lons))

    # Also test boundary convention
    lons_b = get_longitudes(4.0, convention="boundary")
    assert -180.0 in lons_b
    assert 180.0 not in lons_b
    assert len(lons_b) == len(set(lons_b))


def test_coordinate_valid_ranges():
    for step in [10.0, 4.0, 2.0]:
        grid = generate_grid(step_degrees=step, convention="cell_center")
        for coord in grid:
            assert -90.0 <= coord.latitude <= 90.0, f"Lat {coord.latitude} out of bounds"
            assert -180.0 <= coord.longitude < 180.0, f"Lon {coord.longitude} out of bounds"


def test_deterministic_ordering():
    grid1 = generate_grid(step_degrees=4.0, convention="cell_center")
    grid2 = generate_grid(step_degrees=4.0, convention="cell_center")
    assert len(grid1) == len(grid2)
    for c1, c2 in zip(grid1, grid2):
        assert c1.id == c2.id
        assert c1.latitude == c2.latitude
        assert c1.longitude == c2.longitude


def test_coordinate_formatting():
    from latent_earth.models import Coordinate

    c = Coordinate(latitude=45.0, longitude=-73.0)
    assert "45.000° N" in c.format_cardinal()
    assert "73.000° W" in c.format_cardinal()
    assert c.id == "lat_+45.000_lon_-73.000"
