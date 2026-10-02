"""Tiny smoke test for TypeSafe Jev provider with landmark coordinates."""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from latent_earth.models import Coordinate
from latent_earth.providers.jev import JevProvider

# Landmark coordinates: 3 known land points, 3 known ocean points
TEST_COORDINATES = [
    # Land landmarks
    ("Central Sahara (Algeria/Libya)", Coordinate(latitude=25.0, longitude=15.0), True),
    ("Central Australia (Outback)", Coordinate(latitude=-25.0, longitude=133.0), True),
    ("Central North America (Kansas)", Coordinate(latitude=39.0, longitude=-98.0), True),
    ("Central Europe (Germany)", Coordinate(latitude=51.0, longitude=10.0), True),
    # Ocean landmarks
    ("Central Pacific Ocean", Coordinate(latitude=0.0, longitude=-140.0), False),
    ("Central North Atlantic Ocean", Coordinate(latitude=35.0, longitude=-40.0), False),
    ("Southern Ocean", Coordinate(latitude=-55.0, longitude=90.0), False),
    ("Indian Ocean (Equatorial)", Coordinate(latitude=0.0, longitude=75.0), False),
]


async def main():
    print("=== JEV LIVE SMOKE TEST ===")
    provider = JevProvider(model="jev-latest", concurrency=4)
    print(f"Provider: {provider.provider_name}, Model: {provider.model}")
    print(f"Querying {len(TEST_COORDINATES)} test coordinates...\n")

    results = []
    correct_count = 0

    for name, coord, expected_land in TEST_COORDINATES:
        try:
            pred = await provider.predict_coordinate(coord)
            p_land = pred.probability_land
            is_land = pred.is_land
            is_correct = (is_land == expected_land)
            if is_correct:
                correct_count += 1

            status = "✓ PASS" if is_correct else "✗ FAIL"
            exp_str = "LAND" if expected_land else "WATER"
            pred_str = "LAND" if is_land else "WATER"
            prob_pct = f"{p_land * 100:.1f}%" if p_land is not None else "N/A"

            print(f"[{status}] {name:<32}")
            print(f"        Coord: {coord.format_cardinal()} (Lat: {coord.latitude}, Lon: {coord.longitude})")
            print(f"        Expected: {exp_str} | Predicted: {pred_str} | P(Land): {prob_pct} | Uncertainty: {pred.uncertainty:.3f}")
            results.append((name, coord, expected_land, pred))
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            import traceback
            traceback.print_exc()

    await provider.close()

    total = len(TEST_COORDINATES)
    print("\n" + "=" * 50)
    print(f"Smoke Test Summary: {correct_count} / {total} correct ({correct_count / total * 100:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
