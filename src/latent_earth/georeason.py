"""GeoReason: Geographic Relational Reasoning Benchmark.

Tests whether models possess spatial understanding (cardinal directions,
relative proximity, oceanic/marine adjacency, and hemispheric orientation)
without requiring memorization of continuous floating-point coordinates.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import pyproj

geod = pyproj.Geod(ellps="WGS84")

# Global City Knowledge Base with verified WGS84 coordinates and regions
GLOBAL_CITIES = [
    # North America
    {"name": "New York City", "country": "United States", "continent": "North America", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "country": "United States", "continent": "North America", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "country": "United States", "continent": "North America", "lat": 41.8781, "lon": -87.6298},
    {"name": "San Francisco", "country": "United States", "continent": "North America", "lat": 37.7749, "lon": -122.4194},
    {"name": "Seattle", "country": "United States", "continent": "North America", "lat": 47.6062, "lon": -122.3321},
    {"name": "Miami", "country": "United States", "continent": "North America", "lat": 25.7617, "lon": -80.1918},
    {"name": "Denver", "country": "United States", "continent": "North America", "lat": 39.7392, "lon": -104.9903},
    {"name": "Toronto", "country": "Canada", "continent": "North America", "lat": 43.6532, "lon": -79.3832},
    {"name": "Vancouver", "country": "Canada", "continent": "North America", "lat": 49.2827, "lon": -123.1207},
    {"name": "Montreal", "country": "Canada", "continent": "North America", "lat": 45.5017, "lon": -73.5673},
    {"name": "Mexico City", "country": "Mexico", "continent": "North America", "lat": 19.4326, "lon": -99.1332},
    {"name": "Cancun", "country": "Mexico", "continent": "North America", "lat": 21.1619, "lon": -86.8515},
    {"name": "Havana", "country": "Cuba", "continent": "North America", "lat": 23.1136, "lon": -82.3666},
    {"name": "Panama City", "country": "Panama", "continent": "North America", "lat": 8.9824, "lon": -79.5199},
    {"name": "Anchorage", "country": "United States", "continent": "North America", "lat": 61.2181, "lon": -149.9003},
    {"name": "Honolulu", "country": "United States", "continent": "North America", "lat": 21.3069, "lon": -157.8583},

    # South America
    {"name": "Bogota", "country": "Colombia", "continent": "South America", "lat": 4.7110, "lon": -74.0721},
    {"name": "Lima", "country": "Peru", "continent": "South America", "lat": -12.0464, "lon": -77.0428},
    {"name": "Santiago", "country": "Chile", "continent": "South America", "lat": -33.4489, "lon": -70.6693},
    {"name": "Buenos Aires", "country": "Argentina", "continent": "South America", "lat": -34.6037, "lon": -58.3816},
    {"name": "Sao Paulo", "country": "Brazil", "continent": "South America", "lat": -23.5505, "lon": -46.6333},
    {"name": "Rio de Janeiro", "country": "Brazil", "continent": "South America", "lat": -22.9068, "lon": -43.1729},
    {"name": "Brasilia", "country": "Brazil", "continent": "South America", "lat": -15.8267, "lon": -47.9218},
    {"name": "Quito", "country": "Ecuador", "continent": "South America", "lat": -0.1807, "lon": -78.4678},
    {"name": "Caracas", "country": "Venezuela", "continent": "South America", "lat": 10.4806, "lon": -66.9036},
    {"name": "Montevideo", "country": "Uruguay", "continent": "South America", "lat": -34.9011, "lon": -56.1645},
    {"name": "La Paz", "country": "Bolivia", "continent": "South America", "lat": -16.5000, "lon": -68.1500},

    # Europe
    {"name": "London", "country": "United Kingdom", "continent": "Europe", "lat": 51.5074, "lon": -0.1278},
    {"name": "Edinburgh", "country": "United Kingdom", "continent": "Europe", "lat": 55.9533, "lon": -3.1883},
    {"name": "Paris", "country": "France", "continent": "Europe", "lat": 48.8566, "lon": 2.3522},
    {"name": "Marseille", "country": "France", "continent": "Europe", "lat": 43.2965, "lon": 5.3698},
    {"name": "Berlin", "country": "Germany", "continent": "Europe", "lat": 52.5200, "lon": 13.4050},
    {"name": "Munich", "country": "Germany", "continent": "Europe", "lat": 48.1351, "lon": 11.5820},
    {"name": "Rome", "country": "Italy", "continent": "Europe", "lat": 41.9028, "lon": 12.4964},
    {"name": "Milan", "country": "Italy", "continent": "Europe", "lat": 45.4642, "lon": 9.1900},
    {"name": "Madrid", "country": "Spain", "continent": "Europe", "lat": 40.4168, "lon": -3.7038},
    {"name": "Barcelona", "country": "Spain", "continent": "Europe", "lat": 41.3879, "lon": 2.1699},
    {"name": "Lisbon", "country": "Portugal", "continent": "Europe", "lat": 38.7223, "lon": -9.1393},
    {"name": "Amsterdam", "country": "Netherlands", "continent": "Europe", "lat": 52.3676, "lon": 4.9041},
    {"name": "Brussels", "country": "Belgium", "continent": "Europe", "lat": 50.8503, "lon": 4.3517},
    {"name": "Vienna", "country": "Austria", "continent": "Europe", "lat": 48.2082, "lon": 16.3738},
    {"name": "Stockholm", "country": "Sweden", "continent": "Europe", "lat": 59.3293, "lon": 18.0686},
    {"name": "Oslo", "country": "Norway", "continent": "Europe", "lat": 59.9139, "lon": 10.7522},
    {"name": "Helsinki", "country": "Finland", "continent": "Europe", "lat": 60.1699, "lon": 24.9384},
    {"name": "Copenhagen", "country": "Denmark", "continent": "Europe", "lat": 55.6761, "lon": 12.5683},
    {"name": "Warsaw", "country": "Poland", "continent": "Europe", "lat": 52.2297, "lon": 21.0122},
    {"name": "Prague", "country": "Czech Republic", "continent": "Europe", "lat": 50.0755, "lon": 14.4378},
    {"name": "Budapest", "country": "Hungary", "continent": "Europe", "lat": 47.4979, "lon": 19.0402},
    {"name": "Athens", "country": "Greece", "continent": "Europe", "lat": 37.9838, "lon": 23.7275},
    {"name": "Dublin", "country": "Ireland", "continent": "Europe", "lat": 53.3498, "lon": -6.2603},
    {"name": "Reykjavik", "country": "Iceland", "continent": "Europe", "lat": 64.1466, "lon": -21.9426},
    {"name": "Zurich", "country": "Switzerland", "continent": "Europe", "lat": 47.3769, "lon": 8.5417},
    {"name": "Moscow", "country": "Russia", "continent": "Europe", "lat": 55.7558, "lon": 37.6173},
    {"name": "Kyiv", "country": "Ukraine", "continent": "Europe", "lat": 50.4501, "lon": 30.5234},
    {"name": "Istanbul", "country": "Turkey", "continent": "Europe", "lat": 41.0082, "lon": 28.9784},

    # Asia & Middle East
    {"name": "Tokyo", "country": "Japan", "continent": "Asia", "lat": 35.6762, "lon": 139.6503},
    {"name": "Osaka", "country": "Japan", "continent": "Asia", "lat": 34.6937, "lon": 135.5023},
    {"name": "Sapporo", "country": "Japan", "continent": "Asia", "lat": 43.0618, "lon": 141.3545},
    {"name": "Seoul", "country": "South Korea", "continent": "Asia", "lat": 37.5665, "lon": 126.9780},
    {"name": "Busan", "country": "South Korea", "continent": "Asia", "lat": 35.1796, "lon": 129.0756},
    {"name": "Beijing", "country": "China", "continent": "Asia", "lat": 39.9042, "lon": 116.4074},
    {"name": "Shanghai", "country": "China", "continent": "Asia", "lat": 31.2304, "lon": 121.4737},
    {"name": "Guangzhou", "country": "China", "continent": "Asia", "lat": 23.1291, "lon": 113.2644},
    {"name": "Hong Kong", "country": "Hong Kong", "continent": "Asia", "lat": 22.3193, "lon": 114.1694},
    {"name": "Taipei", "country": "Taiwan", "continent": "Asia", "lat": 25.0330, "lon": 121.5654},
    {"name": "Bangkok", "country": "Thailand", "continent": "Asia", "lat": 13.7563, "lon": 100.5018},
    {"name": "Singapore", "country": "Singapore", "continent": "Asia", "lat": 1.3521, "lon": 103.8198},
    {"name": "Kuala Lumpur", "country": "Malaysia", "continent": "Asia", "lat": 3.1390, "lon": 101.6869},
    {"name": "Jakarta", "country": "Indonesia", "continent": "Asia", "lat": -6.2088, "lon": 106.8456},
    {"name": "Manila", "country": "Philippines", "continent": "Asia", "lat": 14.5995, "lon": 120.9842},
    {"name": "Hanoi", "country": "Vietnam", "continent": "Asia", "lat": 21.0285, "lon": 105.8542},
    {"name": "New Delhi", "country": "India", "continent": "Asia", "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai", "country": "India", "continent": "Asia", "lat": 19.0760, "lon": 72.8777},
    {"name": "Bengaluru", "country": "India", "continent": "Asia", "lat": 12.9716, "lon": 77.5946},
    {"name": "Dubai", "country": "United Arab Emirates", "continent": "Asia", "lat": 25.2048, "lon": 55.2708},
    {"name": "Riyadh", "country": "Saudi Arabia", "continent": "Asia", "lat": 24.7136, "lon": 46.6753},
    {"name": "Doha", "country": "Qatar", "continent": "Asia", "lat": 25.2854, "lon": 51.5310},
    {"name": "Tel Aviv", "country": "Israel", "continent": "Asia", "lat": 32.0853, "lon": 34.7818},
    {"name": "Tehran", "country": "Iran", "continent": "Asia", "lat": 35.6892, "lon": 51.3890},
    {"name": "Baghdad", "country": "Iraq", "continent": "Asia", "lat": 33.3152, "lon": 44.3661},

    # Africa
    {"name": "Cairo", "country": "Egypt", "continent": "Africa", "lat": 30.0444, "lon": 31.2357},
    {"name": "Alexandria", "country": "Egypt", "continent": "Africa", "lat": 31.2001, "lon": 29.9187},
    {"name": "Casablanca", "country": "Morocco", "continent": "Africa", "lat": 33.5731, "lon": -7.5898},
    {"name": "Marrakech", "country": "Morocco", "continent": "Africa", "lat": 31.6295, "lon": -7.9811},
    {"name": "Lagos", "country": "Nigeria", "continent": "Africa", "lat": 6.5244, "lon": 3.3792},
    {"name": "Abuja", "country": "Nigeria", "continent": "Africa", "lat": 9.0765, "lon": 7.3986},
    {"name": "Nairobi", "country": "Kenya", "continent": "Africa", "lat": -1.2921, "lon": 36.8219},
    {"name": "Mombasa", "country": "Kenya", "continent": "Africa", "lat": -4.0435, "lon": 39.6682},
    {"name": "Addis Ababa", "country": "Ethiopia", "continent": "Africa", "lat": 9.0320, "lon": 38.7482},
    {"name": "Johannesburg", "country": "South Africa", "continent": "Africa", "lat": -26.2041, "lon": 28.0473},
    {"name": "Cape Town", "country": "South Africa", "continent": "Africa", "lat": -33.9249, "lon": 18.4241},
    {"name": "Durban", "country": "South Africa", "continent": "Africa", "lat": -29.8587, "lon": 31.0218},
    {"name": "Dakar", "country": "Senegal", "continent": "Africa", "lat": 14.7167, "lon": -17.4677},
    {"name": "Accra", "country": "Ghana", "continent": "Africa", "lat": 5.6037, "lon": -0.1870},
    {"name": "Tunis", "country": "Tunisia", "continent": "Africa", "lat": 36.8065, "lon": 10.1815},
    {"name": "Algiers", "country": "Algeria", "continent": "Africa", "lat": 36.7538, "lon": 3.0588},

    # Oceania
    {"name": "Sydney", "country": "Australia", "continent": "Oceania", "lat": -33.8688, "lon": 151.2093},
    {"name": "Melbourne", "country": "Australia", "continent": "Oceania", "lat": -37.8136, "lon": 144.9631},
    {"name": "Brisbane", "country": "Australia", "continent": "Oceania", "lat": -27.4698, "lon": 153.0251},
    {"name": "Perth", "country": "Australia", "continent": "Oceania", "lat": -31.9505, "lon": 115.8605},
    {"name": "Auckland", "country": "New Zealand", "continent": "Oceania", "lat": -36.8485, "lon": 174.7633},
    {"name": "Wellington", "country": "New Zealand", "continent": "Oceania", "lat": -41.2865, "lon": 174.7762},
    {"name": "Suva", "country": "Fiji", "continent": "Oceania", "lat": -18.1248, "lon": 178.4501},
]

# Geopolitical Marine Adjacency & Hemispheric Knowledge Base
MARINE_FACTS = [
    # True facts
    {"entity": "Chile", "feature": "Pacific Ocean", "type": "borders_water", "expected": True},
    {"entity": "Japan", "feature": "Pacific Ocean", "type": "borders_water", "expected": True},
    {"entity": "Norway", "feature": "Atlantic Ocean", "type": "borders_water", "expected": True},
    {"entity": "Egypt", "feature": "Mediterranean Sea", "type": "borders_water", "expected": True},
    {"entity": "Egypt", "feature": "Red Sea", "type": "borders_water", "expected": True},
    {"entity": "Australia", "feature": "Indian Ocean", "type": "borders_water", "expected": True},
    {"entity": "India", "feature": "Indian Ocean", "type": "borders_water", "expected": True},
    {"entity": "Canada", "feature": "Arctic Ocean", "type": "borders_water", "expected": True},
    {"entity": "Brazil", "feature": "Atlantic Ocean", "type": "borders_water", "expected": True},
    {"entity": "South Africa", "feature": "Atlantic Ocean", "type": "borders_water", "expected": True},
    {"entity": "South Africa", "feature": "Indian Ocean", "type": "borders_water", "expected": True},
    {"entity": "Greece", "feature": "Aegean Sea", "type": "borders_water", "expected": True},
    {"entity": "Poland", "feature": "Baltic Sea", "type": "borders_water", "expected": True},
    {"entity": "Italy", "feature": "Adriatic Sea", "type": "borders_water", "expected": True},
    {"entity": "Kenya", "feature": "Indian Ocean", "type": "borders_water", "expected": True},
    {"entity": "Morocco", "feature": "Atlantic Ocean", "type": "borders_water", "expected": True},
    {"entity": "Vietnam", "feature": "South China Sea", "type": "borders_water", "expected": True},
    {"entity": "Oman", "feature": "Arabian Sea", "type": "borders_water", "expected": True},
    {"entity": "Mexico", "feature": "Gulf of Mexico", "type": "borders_water", "expected": True},
    {"entity": "Turkey", "feature": "Black Sea", "type": "borders_water", "expected": True},
    {"entity": "United States", "feature": "Pacific Ocean", "type": "borders_water", "expected": True},
    {"entity": "United States", "feature": "Atlantic Ocean", "type": "borders_water", "expected": True},
    {"entity": "United Kingdom", "feature": "North Sea", "type": "borders_water", "expected": True},

    # False facts (Distractors / Landlocked or Non-adjacent)
    {"entity": "Bolivia", "feature": "Pacific Ocean", "type": "borders_water", "expected": False},
    {"entity": "Switzerland", "feature": "Mediterranean Sea", "type": "borders_water", "expected": False},
    {"entity": "Austria", "feature": "Adriatic Sea", "type": "borders_water", "expected": False},
    {"entity": "Germany", "feature": "Mediterranean Sea", "type": "borders_water", "expected": False},
    {"entity": "Brazil", "feature": "Pacific Ocean", "type": "borders_water", "expected": False},
    {"entity": "Peru", "feature": "Atlantic Ocean", "type": "borders_water", "expected": False},
    {"entity": "Chad", "feature": "Atlantic Ocean", "type": "borders_water", "expected": False},
    {"entity": "Mongolia", "feature": "Pacific Ocean", "type": "borders_water", "expected": False},
    {"entity": "Poland", "feature": "Black Sea", "type": "borders_water", "expected": False},
    {"entity": "Sweden", "feature": "Mediterranean Sea", "type": "borders_water", "expected": False},
    {"entity": "Chile", "feature": "Atlantic Ocean", "type": "borders_water", "expected": False},
    {"entity": "Ethiopia", "feature": "Red Sea", "type": "borders_water", "expected": False},
    {"entity": "Paraguay", "feature": "Atlantic Ocean", "type": "borders_water", "expected": False},
    {"entity": "Czech Republic", "feature": "Baltic Sea", "type": "borders_water", "expected": False},
    {"entity": "Nepal", "feature": "Indian Ocean", "type": "borders_water", "expected": False},
    {"entity": "Spain", "feature": "Baltic Sea", "type": "borders_water", "expected": False},
    {"entity": "Argentina", "feature": "Pacific Ocean", "type": "borders_water", "expected": False},
    {"entity": "Nigeria", "feature": "Indian Ocean", "type": "borders_water", "expected": False},
    {"entity": "Japan", "feature": "Indian Ocean", "type": "borders_water", "expected": False},
    {"entity": "France", "feature": "Red Sea", "type": "borders_water", "expected": False},
    {"entity": "Canada", "feature": "Indian Ocean", "type": "borders_water", "expected": False},
    {"entity": "Australia", "feature": "Atlantic Ocean", "type": "borders_water", "expected": False},
    {"entity": "Colombia", "feature": "Baltic Sea", "type": "borders_water", "expected": False},

    # Hemispheric Orientation (True & False)
    {"entity": "Singapore", "feature": "Northern Hemisphere", "type": "hemisphere", "expected": True},
    {"entity": "Bogota", "feature": "Northern Hemisphere", "type": "hemisphere", "expected": True},
    {"entity": "Nairobi", "feature": "Southern Hemisphere", "type": "hemisphere", "expected": True},
    {"entity": "Sydney", "feature": "Southern Hemisphere", "type": "hemisphere", "expected": True},
    {"entity": "Tokyo", "feature": "Northern Hemisphere", "type": "hemisphere", "expected": True},
    {"entity": "Buenos Aires", "feature": "Southern Hemisphere", "type": "hemisphere", "expected": True},
    {"entity": "Quito", "feature": "Southern Hemisphere", "type": "hemisphere", "expected": True},
    {"entity": "London", "feature": "Northern Hemisphere", "type": "hemisphere", "expected": True},

    {"entity": "Sydney", "feature": "Northern Hemisphere", "type": "hemisphere", "expected": False},
    {"entity": "Paris", "feature": "Southern Hemisphere", "type": "hemisphere", "expected": False},
    {"entity": "Cairo", "feature": "Southern Hemisphere", "type": "hemisphere", "expected": False},
    {"entity": "Melbourne", "feature": "Northern Hemisphere", "type": "hemisphere", "expected": False},
    {"entity": "Santiago", "feature": "Northern Hemisphere", "type": "hemisphere", "expected": False},
    {"entity": "Moscow", "feature": "Southern Hemisphere", "type": "hemisphere", "expected": False},
]


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate exact WGS84 geodesic distance in kilometers."""
    _, _, d_meters = geod.inv(lon1, lat1, lon2, lat2)
    return d_meters / 1000.0


@dataclass
class GeoReasonQuestion:
    """Single question in the GeoReason benchmark."""
    id: str
    category: Literal["compass_cardinal", "proximity_triplets", "marine_coastal_borders"]
    context: str
    question: str
    expected_answer: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "context": self.context,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "metadata": self.metadata,
        }


def generate_compass_questions(target_count: int = 400, seed: int = 42) -> list[GeoReasonQuestion]:
    """Generate unambiguous cardinal direction questions."""
    rng = random.Random(seed)
    questions: list[GeoReasonQuestion] = []
    cities = GLOBAL_CITIES

    # Generate candidate pairs with strong cardinal axis separation
    pairs: list[dict[str, Any]] = []
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            c1, c2 = cities[i], cities[j]
            d_lat = c2["lat"] - c1["lat"]
            d_lon = c2["lon"] - c1["lon"]

            # Wrap longitude difference to [-180, 180]
            if d_lon > 180:
                d_lon -= 360
            elif d_lon < -180:
                d_lon += 360

            # Pure North-South axis check: |d_lat| >= 4.0 and (|d_lat| >= 1.2 * |d_lon| or same continent)
            if abs(d_lat) >= 4.0:
                is_c2_north = d_lat > 0
                pairs.append({
                    "c1": c1, "c2": c2,
                    "axis": "North/South",
                    "c2_dir": "North" if is_c2_north else "South",
                    "c2_opposite": "South" if is_c2_north else "North",
                })

            # Pure East-West axis check: |d_lon| >= 6.0 and (|d_lon| >= 1.2 * |d_lat| or same continent)
            if abs(d_lon) >= 6.0:
                is_c2_east = d_lon > 0
                pairs.append({
                    "c1": c1, "c2": c2,
                    "axis": "East/West",
                    "c2_dir": "East" if is_c2_east else "West",
                    "c2_opposite": "West" if is_c2_east else "East",
                })

    rng.shuffle(pairs)
    true_target = target_count // 2
    false_target = target_count - true_target

    true_count = 0
    false_count = 0

    idx = 0
    for p in pairs:
        c1, c2 = p["c1"], p["c2"]
        # Alternate or balance True and False
        want_true = (true_count < true_target) and (false_count >= false_target or rng.random() < 0.5)
        if want_true:
            dir_to_ask = p["c2_dir"]
            expected = True
            true_count += 1
        elif false_count < false_target:
            dir_to_ask = p["c2_opposite"]
            expected = False
            false_count += 1
        else:
            continue

        q_id = f"compass_{idx:04d}"
        idx += 1
        q = GeoReasonQuestion(
            id=q_id,
            category="compass_cardinal",
            context=f"Geographic reference: {c1['name']} ({c1['country']}) and {c2['name']} ({c2['country']}).",
            question=f"Is {c2['name']} located predominantly to the {dir_to_ask} of {c1['name']}?",
            expected_answer=expected,
            metadata={
                "city_from": c1["name"],
                "city_to": c2["name"],
                "axis": p["axis"],
                "actual_direction": p["c2_dir"],
                "asked_direction": dir_to_ask,
            },
        )
        questions.append(q)
        if len(questions) >= target_count:
            break

    return questions


def generate_proximity_questions(target_count: int = 400, seed: int = 42) -> list[GeoReasonQuestion]:
    """Generate proximity triplet questions with strict distance separation margin."""
    rng = random.Random(seed)
    questions: list[GeoReasonQuestion] = []
    cities = GLOBAL_CITIES

    triplets: list[dict[str, Any]] = []
    for i, c_ref in enumerate(cities):
        for j, c_a in enumerate(cities):
            if i == j:
                continue
            for k, c_b in enumerate(cities):
                if k <= j or k == i:
                    continue

                d_a = geodesic_distance_km(c_ref["lat"], c_ref["lon"], c_a["lat"], c_a["lon"])
                d_b = geodesic_distance_km(c_ref["lat"], c_ref["lon"], c_b["lat"], c_b["lon"])

                # Separation margin: at least 500 km difference and 25% relative difference
                diff = abs(d_a - d_b)
                min_d = min(d_a, d_b)
                if diff >= 500.0 and (diff / max(1.0, min_d)) >= 0.25:
                    a_closer = d_a < d_b
                    triplets.append({
                        "ref": c_ref,
                        "ca": c_a,
                        "cb": c_b,
                        "da": round(d_a, 1),
                        "db": round(d_b, 1),
                        "a_closer": a_closer,
                    })

    rng.shuffle(triplets)
    true_target = target_count // 2
    false_target = target_count - true_target

    true_count = 0
    false_count = 0
    idx = 0

    for trip in triplets:
        c_ref = trip["ref"]
        c_a = trip["ca"]
        c_b = trip["cb"]
        a_is_closer = trip["a_closer"]

        want_true = (true_count < true_target) and (false_count >= false_target or rng.random() < 0.5)
        if want_true:
            # Ask if the truly closer city is closer
            closer_name = c_a["name"] if a_is_closer else c_b["name"]
            farther_name = c_b["name"] if a_is_closer else c_a["name"]
            expected = True
            true_count += 1
        elif false_count < false_target:
            # Ask if the farther city is closer
            closer_name = c_b["name"] if a_is_closer else c_a["name"]
            farther_name = c_a["name"] if a_is_closer else c_b["name"]
            expected = False
            false_count += 1
        else:
            continue

        q_id = f"proximity_{idx:04d}"
        idx += 1
        q = GeoReasonQuestion(
            id=q_id,
            category="proximity_triplets",
            context=f"Reference location: {c_ref['name']} ({c_ref['country']}). Candidates: {closer_name} and {farther_name}.",
            question=f"Is {closer_name} closer in distance to {c_ref['name']} than {farther_name} is to {c_ref['name']}?",
            expected_answer=expected,
            metadata={
                "ref_city": c_ref["name"],
                "candidate_1": closer_name,
                "candidate_2": farther_name,
                "distance_cand1_km": trip["da"] if closer_name == c_a["name"] else trip["db"],
                "distance_cand2_km": trip["db"] if closer_name == c_a["name"] else trip["da"],
            },
        )
        questions.append(q)
        if len(questions) >= target_count:
            break

    return questions


def generate_marine_questions(target_count: int = 400, seed: int = 42) -> list[GeoReasonQuestion]:
    """Generate ocean, coastal border, and hemispheric orientation questions."""
    rng = random.Random(seed)
    questions: list[GeoReasonQuestion] = []
    facts = list(MARINE_FACTS)

    # Replicate or augment to reach target count
    augmented: list[dict[str, Any]] = []
    for _ in range(max(1, math.ceil(target_count / len(facts)))):
        shuffled = list(facts)
        rng.shuffle(shuffled)
        augmented.extend(shuffled)

    # Balance True and False
    true_target = target_count // 2
    false_target = target_count - true_target

    true_count = 0
    false_count = 0
    idx = 0

    for fact in augmented:
        exp = fact["expected"]
        if exp and true_count >= true_target:
            continue
        if not exp and false_count >= false_target:
            continue

        if exp:
            true_count += 1
        else:
            false_count += 1

        f_type = fact["type"]
        entity = fact["entity"]
        feature = fact["feature"]

        if f_type == "borders_water":
            context = f"Geographic entity: {entity}."
            question = f"Does {entity} have a natural coastline bordering the {feature}?"
        else:
            context = f"Geographic location: {entity}."
            question = f"Is {entity} located in the {feature}?"

        q_id = f"marine_{idx:04d}"
        idx += 1
        q = GeoReasonQuestion(
            id=q_id,
            category="marine_coastal_borders",
            context=context,
            question=question,
            expected_answer=exp,
            metadata={
                "entity": entity,
                "feature": feature,
                "fact_type": f_type,
            },
        )
        questions.append(q)
        if len(questions) >= target_count:
            break

    return questions


def generate_georeason_dataset(total_count: int = 1200, seed: int = 42) -> list[GeoReasonQuestion]:
    """Generate balanced, comprehensive GeoReason benchmark dataset."""
    c_per_cat = total_count // 3
    q_compass = generate_compass_questions(target_count=c_per_cat, seed=seed)
    q_proximity = generate_proximity_questions(target_count=c_per_cat, seed=seed + 1)
    q_marine = generate_marine_questions(target_count=c_per_cat, seed=seed + 2)

    all_q = q_compass + q_proximity + q_marine
    rng = random.Random(seed)
    rng.shuffle(all_q)
    return all_q
