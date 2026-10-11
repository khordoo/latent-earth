"""GeoTopology: Geopolitical Adjacency, Maritime Connectivity, and 3D Hypsometry Benchmark.

Evaluates:
1. border_adjacency: Sovereign land border adjacency and graph connectivity (True & False pairs).
2. landlocked_maritime: Landlocked sovereignty vs direct oceanic/sea access.
3. elevation_hypsometry: 3D vertical terrain reasoning (city elevations, mountain summits, sub-sea-level depressions).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Literal

# 1. Sovereign Border Adjacency Knowledge Base
# Pairs of countries with verified sovereign land borders (True) and plausible non-borders (False).
LAND_BORDER_PAIRS = [
    # True Land Borders - Europe
    {"country_a": "France", "country_b": "Spain", "has_border": True},
    {"country_a": "France", "country_b": "Germany", "has_border": True},
    {"country_a": "France", "country_b": "Italy", "has_border": True},
    {"country_a": "France", "country_b": "Belgium", "has_border": True},
    {"country_a": "France", "country_b": "Switzerland", "has_border": True},
    {"country_a": "Germany", "country_b": "Poland", "has_border": True},
    {"country_a": "Germany", "country_b": "Austria", "has_border": True},
    {"country_a": "Germany", "country_b": "Denmark", "has_border": True},
    {"country_a": "Germany", "country_b": "Netherlands", "has_border": True},
    {"country_a": "Germany", "country_b": "Czech Republic", "has_border": True},
    {"country_a": "Spain", "country_b": "Portugal", "has_border": True},
    {"country_a": "Italy", "country_b": "Switzerland", "has_border": True},
    {"country_a": "Italy", "country_b": "Austria", "has_border": True},
    {"country_a": "Italy", "country_b": "Slovenia", "has_border": True},
    {"country_a": "Poland", "country_b": "Czech Republic", "has_border": True},
    {"country_a": "Poland", "country_b": "Slovakia", "has_border": True},
    {"country_a": "Poland", "country_b": "Ukraine", "has_border": True},
    {"country_a": "Norway", "country_b": "Sweden", "has_border": True},
    {"country_a": "Norway", "country_b": "Finland", "has_border": True},
    {"country_a": "Finland", "country_b": "Sweden", "has_border": True},
    {"country_a": "Greece", "country_b": "Turkey", "has_border": True},
    {"country_a": "Greece", "country_b": "Bulgaria", "has_border": True},
    {"country_a": "Greece", "country_b": "Albania", "has_border": True},
    {"country_a": "United Kingdom", "country_b": "Ireland", "has_border": True},

    # True Land Borders - Americas
    {"country_a": "United States", "country_b": "Canada", "has_border": True},
    {"country_a": "United States", "country_b": "Mexico", "has_border": True},
    {"country_a": "Mexico", "country_b": "Guatemala", "has_border": True},
    {"country_a": "Mexico", "country_b": "Belize", "has_border": True},
    {"country_a": "Brazil", "country_b": "Argentina", "has_border": True},
    {"country_a": "Brazil", "country_b": "Uruguay", "has_border": True},
    {"country_a": "Brazil", "country_b": "Paraguay", "has_border": True},
    {"country_a": "Brazil", "country_b": "Bolivia", "has_border": True},
    {"country_a": "Brazil", "country_b": "Peru", "has_border": True},
    {"country_a": "Brazil", "country_b": "Colombia", "has_border": True},
    {"country_a": "Brazil", "country_b": "Venezuela", "has_border": True},
    {"country_a": "Argentina", "country_b": "Chile", "has_border": True},
    {"country_a": "Argentina", "country_b": "Bolivia", "has_border": True},
    {"country_a": "Argentina", "country_b": "Paraguay", "has_border": True},
    {"country_a": "Argentina", "country_b": "Uruguay", "has_border": True},
    {"country_a": "Chile", "country_b": "Peru", "has_border": True},
    {"country_a": "Chile", "country_b": "Bolivia", "has_border": True},
    {"country_a": "Colombia", "country_b": "Ecuador", "has_border": True},
    {"country_a": "Colombia", "country_b": "Panama", "has_border": True},
    {"country_a": "Peru", "country_b": "Ecuador", "has_border": True},

    # True Land Borders - Asia & Middle East
    {"country_a": "China", "country_b": "Russia", "has_border": True},
    {"country_a": "China", "country_b": "Mongolia", "has_border": True},
    {"country_a": "China", "country_b": "India", "has_border": True},
    {"country_a": "China", "country_b": "Pakistan", "has_border": True},
    {"country_a": "China", "country_b": "Vietnam", "has_border": True},
    {"country_a": "China", "country_b": "Myanmar", "has_border": True},
    {"country_a": "China", "country_b": "North Korea", "has_border": True},
    {"country_a": "India", "country_b": "Pakistan", "has_border": True},
    {"country_a": "India", "country_b": "Bangladesh", "has_border": True},
    {"country_a": "India", "country_b": "Nepal", "has_border": True},
    {"country_a": "India", "country_b": "Bhutan", "has_border": True},
    {"country_a": "India", "country_b": "Myanmar", "has_border": True},
    {"country_a": "Thailand", "country_b": "Malaysia", "has_border": True},
    {"country_a": "Thailand", "country_b": "Cambodia", "has_border": True},
    {"country_a": "Thailand", "country_b": "Laos", "has_border": True},
    {"country_a": "Thailand", "country_b": "Myanmar", "has_border": True},
    {"country_a": "Saudi Arabia", "country_b": "United Arab Emirates", "has_border": True},
    {"country_a": "Saudi Arabia", "country_b": "Oman", "has_border": True},
    {"country_a": "Saudi Arabia", "country_b": "Yemen", "has_border": True},
    {"country_a": "Saudi Arabia", "country_b": "Jordan", "has_border": True},
    {"country_a": "Saudi Arabia", "country_b": "Iraq", "has_border": True},
    {"country_a": "Saudi Arabia", "country_b": "Kuwait", "has_border": True},
    {"country_a": "Turkey", "country_b": "Iran", "has_border": True},
    {"country_a": "Turkey", "country_b": "Iraq", "has_border": True},
    {"country_a": "Turkey", "country_b": "Syria", "has_border": True},
    {"country_a": "Turkey", "country_b": "Georgia", "has_border": True},

    # True Land Borders - Africa
    {"country_a": "Egypt", "country_b": "Sudan", "has_border": True},
    {"country_a": "Egypt", "country_b": "Libya", "has_border": True},
    {"country_a": "Egypt", "country_b": "Israel", "has_border": True},
    {"country_a": "South Africa", "country_b": "Namibia", "has_border": True},
    {"country_a": "South Africa", "country_b": "Botswana", "has_border": True},
    {"country_a": "South Africa", "country_b": "Zimbabwe", "has_border": True},
    {"country_a": "South Africa", "country_b": "Mozambique", "has_border": True},
    {"country_a": "South Africa", "country_b": "Lesotho", "has_border": True},
    {"country_a": "South Africa", "country_b": "Eswatini", "has_border": True},
    {"country_a": "Kenya", "country_b": "Tanzania", "has_border": True},
    {"country_a": "Kenya", "country_b": "Uganda", "has_border": True},
    {"country_a": "Kenya", "country_b": "Ethiopia", "has_border": True},
    {"country_a": "Kenya", "country_b": "Somalia", "has_border": True},
    {"country_a": "Nigeria", "country_b": "Niger", "has_border": True},
    {"country_a": "Nigeria", "country_b": "Benin", "has_border": True},
    {"country_a": "Nigeria", "country_b": "Cameroon", "has_border": True},
    {"country_a": "Nigeria", "country_b": "Chad", "has_border": True},
    {"country_a": "Morocco", "country_b": "Algeria", "has_border": True},

    # False Land Borders (Plausible Non-Adjacencies / Distractors)
    {"country_a": "France", "country_b": "Portugal", "has_border": False},
    {"country_a": "Germany", "country_b": "Italy", "has_border": False},
    {"country_a": "Spain", "country_b": "Italy", "has_border": False},
    {"country_a": "Poland", "country_b": "Austria", "has_border": False},
    {"country_a": "Poland", "country_b": "Hungary", "has_border": False},
    {"country_a": "United Kingdom", "country_b": "France", "has_border": False},
    {"country_a": "Sweden", "country_b": "Denmark", "has_border": False},
    {"country_a": "Belgium", "country_b": "Switzerland", "has_border": False},
    {"country_a": "Austria", "country_b": "Croatia", "has_border": False},
    {"country_a": "Netherlands", "country_b": "France", "has_border": False},
    {"country_a": "Norway", "country_b": "Denmark", "has_border": False},
    {"country_a": "Greece", "country_b": "Romania", "has_border": False},

    {"country_a": "Brazil", "country_b": "Chile", "has_border": False},
    {"country_a": "Brazil", "country_b": "Ecuador", "has_border": False},
    {"country_a": "Argentina", "country_b": "Peru", "has_border": False},
    {"country_a": "Chile", "country_b": "Paraguay", "has_border": False},
    {"country_a": "Colombia", "country_b": "Bolivia", "has_border": False},
    {"country_a": "Uruguay", "country_b": "Paraguay", "has_border": False},
    {"country_a": "Venezuela", "country_b": "Ecuador", "has_border": False},
    {"country_a": "Canada", "country_b": "Mexico", "has_border": False},
    {"country_a": "United States", "country_b": "Guatemala", "has_border": False},

    {"country_a": "China", "country_b": "Japan", "has_border": False},
    {"country_a": "China", "country_b": "South Korea", "has_border": False},
    {"country_a": "Japan", "country_b": "South Korea", "has_border": False},
    {"country_a": "India", "country_b": "Thailand", "has_border": False},
    {"country_a": "India", "country_b": "Afghanistan", "has_border": False},
    {"country_a": "India", "country_b": "Sri Lanka", "has_border": False},
    {"country_a": "Thailand", "country_b": "Vietnam", "has_border": False},
    {"country_a": "Vietnam", "country_b": "Myanmar", "has_border": False},
    {"country_a": "Singapore", "country_b": "Indonesia", "has_border": False},
    {"country_a": "Turkey", "country_b": "Saudi Arabia", "has_border": False},
    {"country_a": "Saudi Arabia", "country_b": "Iran", "has_border": False},
    {"country_a": "Iraq", "country_b": "Lebanon", "has_border": False},

    {"country_a": "Egypt", "country_b": "Ethiopia", "has_border": False},
    {"country_a": "Egypt", "country_b": "Chad", "has_border": False},
    {"country_a": "South Africa", "country_b": "Zambia", "has_border": False},
    {"country_a": "South Africa", "country_b": "Angola", "has_border": False},
    {"country_a": "Kenya", "country_b": "Rwanda", "has_border": False},
    {"country_a": "Kenya", "country_b": "Sudan", "has_border": False},
    {"country_a": "Nigeria", "country_b": "Ghana", "has_border": False},
    {"country_a": "Morocco", "country_b": "Tunisia", "has_border": False},
    {"country_a": "Algeria", "country_b": "Egypt", "has_border": False},
    {"country_a": "Madagascar", "country_b": "Mozambique", "has_border": False},
    {"country_a": "Australia", "country_b": "New Zealand", "has_border": False},
]

# 2. Landlocked Sovereignty & Maritime Connectivity Knowledge Base
LANDLOCKED_MARITIME_FACTS = [
    # Landlocked Nations (is_landlocked = True)
    {"country": "Switzerland", "is_landlocked": True, "continent": "Europe"},
    {"country": "Austria", "is_landlocked": True, "continent": "Europe"},
    {"country": "Czech Republic", "is_landlocked": True, "continent": "Europe"},
    {"country": "Slovakia", "is_landlocked": True, "continent": "Europe"},
    {"country": "Hungary", "is_landlocked": True, "continent": "Europe"},
    {"country": "Luxembourg", "is_landlocked": True, "continent": "Europe"},
    {"country": "Liechtenstein", "is_landlocked": True, "continent": "Europe"},
    {"country": "San Marino", "is_landlocked": True, "continent": "Europe"},
    {"country": "Vatican City", "is_landlocked": True, "continent": "Europe"},
    {"country": "Serbia", "is_landlocked": True, "continent": "Europe"},
    {"country": "North Macedonia", "is_landlocked": True, "continent": "Europe"},
    {"country": "Belarus", "is_landlocked": True, "continent": "Europe"},
    {"country": "Moldova", "is_landlocked": True, "continent": "Europe"},

    {"country": "Bolivia", "is_landlocked": True, "continent": "South America"},
    {"country": "Paraguay", "is_landlocked": True, "continent": "South America"},

    {"country": "Mongolia", "is_landlocked": True, "continent": "Asia"},
    {"country": "Nepal", "is_landlocked": True, "continent": "Asia"},
    {"country": "Bhutan", "is_landlocked": True, "continent": "Asia"},
    {"country": "Laos", "is_landlocked": True, "continent": "Asia"},
    {"country": "Afghanistan", "is_landlocked": True, "continent": "Asia"},
    {"country": "Kazakhstan", "is_landlocked": True, "continent": "Asia"},
    {"country": "Uzbekistan", "is_landlocked": True, "continent": "Asia"},
    {"country": "Kyrgyzstan", "is_landlocked": True, "continent": "Asia"},
    {"country": "Tajikistan", "is_landlocked": True, "continent": "Asia"},
    {"country": "Armenia", "is_landlocked": True, "continent": "Asia"},

    {"country": "Ethiopia", "is_landlocked": True, "continent": "Africa"},
    {"country": "Chad", "is_landlocked": True, "continent": "Africa"},
    {"country": "Niger", "is_landlocked": True, "continent": "Africa"},
    {"country": "Mali", "is_landlocked": True, "continent": "Africa"},
    {"country": "Burkina Faso", "is_landlocked": True, "continent": "Africa"},
    {"country": "Central African Republic", "is_landlocked": True, "continent": "Africa"},
    {"country": "South Sudan", "is_landlocked": True, "continent": "Africa"},
    {"country": "Uganda", "is_landlocked": True, "continent": "Africa"},
    {"country": "Rwanda", "is_landlocked": True, "continent": "Africa"},
    {"country": "Burundi", "is_landlocked": True, "continent": "Africa"},
    {"country": "Zambia", "is_landlocked": True, "continent": "Africa"},
    {"country": "Zimbabwe", "is_landlocked": True, "continent": "Africa"},
    {"country": "Malawi", "is_landlocked": True, "continent": "Africa"},
    {"country": "Botswana", "is_landlocked": True, "continent": "Africa"},
    {"country": "Lesotho", "is_landlocked": True, "continent": "Africa"},
    {"country": "Eswatini", "is_landlocked": True, "continent": "Africa"},

    # Coastal / Maritime / Island Nations (is_landlocked = False)
    {"country": "Poland", "is_landlocked": False, "continent": "Europe"},
    {"country": "Germany", "is_landlocked": False, "continent": "Europe"},
    {"country": "France", "is_landlocked": False, "continent": "Europe"},
    {"country": "Spain", "is_landlocked": False, "continent": "Europe"},
    {"country": "Portugal", "is_landlocked": False, "continent": "Europe"},
    {"country": "Italy", "is_landlocked": False, "continent": "Europe"},
    {"country": "Greece", "is_landlocked": False, "continent": "Europe"},
    {"country": "Norway", "is_landlocked": False, "continent": "Europe"},
    {"country": "Sweden", "is_landlocked": False, "continent": "Europe"},
    {"country": "Denmark", "is_landlocked": False, "continent": "Europe"},
    {"country": "Netherlands", "is_landlocked": False, "continent": "Europe"},
    {"country": "Belgium", "is_landlocked": False, "continent": "Europe"},
    {"country": "United Kingdom", "is_landlocked": False, "continent": "Europe"},
    {"country": "Ireland", "is_landlocked": False, "continent": "Europe"},
    {"country": "Croatia", "is_landlocked": False, "continent": "Europe"},
    {"country": "Romania", "is_landlocked": False, "continent": "Europe"},
    {"country": "Bulgaria", "is_landlocked": False, "continent": "Europe"},

    {"country": "Chile", "is_landlocked": False, "continent": "South America"},
    {"country": "Peru", "is_landlocked": False, "continent": "South America"},
    {"country": "Argentina", "is_landlocked": False, "continent": "South America"},
    {"country": "Brazil", "is_landlocked": False, "continent": "South America"},
    {"country": "Uruguay", "is_landlocked": False, "continent": "South America"},
    {"country": "Colombia", "is_landlocked": False, "continent": "South America"},
    {"country": "Ecuador", "is_landlocked": False, "continent": "South America"},
    {"country": "Venezuela", "is_landlocked": False, "continent": "South America"},

    {"country": "Japan", "is_landlocked": False, "continent": "Asia"},
    {"country": "South Korea", "is_landlocked": False, "continent": "Asia"},
    {"country": "Vietnam", "is_landlocked": False, "continent": "Asia"},
    {"country": "Thailand", "is_landlocked": False, "continent": "Asia"},
    {"country": "Cambodia", "is_landlocked": False, "continent": "Asia"},
    {"country": "India", "is_landlocked": False, "continent": "Asia"},
    {"country": "Pakistan", "is_landlocked": False, "continent": "Asia"},
    {"country": "Indonesia", "is_landlocked": False, "continent": "Asia"},
    {"country": "Philippines", "is_landlocked": False, "continent": "Asia"},
    {"country": "Singapore", "is_landlocked": False, "continent": "Asia"},
    {"country": "Malaysia", "is_landlocked": False, "continent": "Asia"},
    {"country": "Turkey", "is_landlocked": False, "continent": "Asia"},
    {"country": "Saudi Arabia", "is_landlocked": False, "continent": "Asia"},
    {"country": "Oman", "is_landlocked": False, "continent": "Asia"},

    {"country": "Egypt", "is_landlocked": False, "continent": "Africa"},
    {"country": "Morocco", "is_landlocked": False, "continent": "Africa"},
    {"country": "Algeria", "is_landlocked": False, "continent": "Africa"},
    {"country": "Tunisia", "is_landlocked": False, "continent": "Africa"},
    {"country": "Kenya", "is_landlocked": False, "continent": "Africa"},
    {"country": "Tanzania", "is_landlocked": False, "continent": "Africa"},
    {"country": "South Africa", "is_landlocked": False, "continent": "Africa"},
    {"country": "Namibia", "is_landlocked": False, "continent": "Africa"},
    {"country": "Angola", "is_landlocked": False, "continent": "Africa"},
    {"country": "Mozambique", "is_landlocked": False, "continent": "Africa"},
    {"country": "Nigeria", "is_landlocked": False, "continent": "Africa"},
    {"country": "Ghana", "is_landlocked": False, "continent": "Africa"},
    {"country": "Senegal", "is_landlocked": False, "continent": "Africa"},
    {"country": "Madagascar", "is_landlocked": False, "continent": "Africa"},

    {"country": "Australia", "is_landlocked": False, "continent": "Oceania"},
    {"country": "New Zealand", "is_landlocked": False, "continent": "Oceania"},
    {"country": "Fiji", "is_landlocked": False, "continent": "Oceania"},
    {"country": "Papua New Guinea", "is_landlocked": False, "continent": "Oceania"},
]

# 3. 3D Elevation & Hypsometric Knowledge Base (meters above sea level)
ELEVATION_ENTITIES = [
    # Below sea level depressions
    {"name": "Dead Sea depression", "type": "depression", "elevation_m": -430, "country": "Jordan/Israel"},
    {"name": "Lake Assal depression", "type": "depression", "elevation_m": -155, "country": "Djibouti"},
    {"name": "Turpan Depression", "type": "depression", "elevation_m": -154, "country": "China"},
    {"name": "Qattara Depression", "type": "depression", "elevation_m": -133, "country": "Egypt"},
    {"name": "Death Valley (Badwater Basin)", "type": "depression", "elevation_m": -86, "country": "United States"},
    {"name": "Salton Sink", "type": "depression", "elevation_m": -69, "country": "United States"},
    {"name": "Caspian Sea shoreline (Baku coast)", "type": "coastal", "elevation_m": -28, "country": "Azerbaijan"},
    {"name": "Amsterdam (Zuidplaspolder area)", "type": "city", "elevation_m": -2, "country": "Netherlands"},

    # Coastal & low-lying cities (< 100m)
    {"name": "Miami", "type": "city", "elevation_m": 2, "country": "United States"},
    {"name": "Alexandria", "type": "city", "elevation_m": 5, "country": "Egypt"},
    {"name": "New York City", "type": "city", "elevation_m": 10, "country": "United States"},
    {"name": "Mumbai", "type": "city", "elevation_m": 14, "country": "India"},
    {"name": "Singapore", "type": "city", "elevation_m": 15, "country": "Singapore"},
    {"name": "London", "type": "city", "elevation_m": 15, "country": "United Kingdom"},
    {"name": "Tokyo", "type": "city", "elevation_m": 17, "country": "Japan"},
    {"name": "Sydney", "type": "city", "elevation_m": 19, "country": "Australia"},
    {"name": "Rome", "type": "city", "elevation_m": 21, "country": "Italy"},
    {"name": "Cairo", "type": "city", "elevation_m": 23, "country": "Egypt"},
    {"name": "Buenos Aires", "type": "city", "elevation_m": 25, "country": "Argentina"},
    {"name": "Berlin", "type": "city", "elevation_m": 34, "country": "Germany"},
    {"name": "Paris", "type": "city", "elevation_m": 35, "country": "France"},
    {"name": "Seoul", "type": "city", "elevation_m": 38, "country": "South Korea"},
    {"name": "Beijing", "type": "city", "elevation_m": 43, "country": "China"},
    {"name": "Lisbon", "type": "city", "elevation_m": 50, "country": "Portugal"},

    # Moderate elevation cities (150m - 900m)
    {"name": "Vienna", "type": "city", "elevation_m": 171, "country": "Austria"},
    {"name": "Geneva", "type": "city", "elevation_m": 375, "country": "Switzerland"},
    {"name": "Zurich", "type": "city", "elevation_m": 408, "country": "Switzerland"},
    {"name": "Munich", "type": "city", "elevation_m": 519, "country": "Germany"},
    {"name": "Bern", "type": "city", "elevation_m": 542, "country": "Switzerland"},
    {"name": "Madrid", "type": "city", "elevation_m": 667, "country": "Spain"},
    {"name": "Jerusalem", "type": "city", "elevation_m": 754, "country": "Israel"},
    {"name": "Amman", "type": "city", "elevation_m": 773, "country": "Jordan"},

    # High elevation cities & plateaus (> 1,000m)
    {"name": "Calgary", "type": "city", "elevation_m": 1045, "country": "Canada"},
    {"name": "Tehran", "type": "city", "elevation_m": 1200, "country": "Iran"},
    {"name": "Kathmandu", "type": "city", "elevation_m": 1400, "country": "Nepal"},
    {"name": "Denver", "type": "city", "elevation_m": 1609, "country": "United States"},
    {"name": "Johannesburg", "type": "city", "elevation_m": 1753, "country": "South Africa"},
    {"name": "Kabul", "type": "city", "elevation_m": 1790, "country": "Afghanistan"},
    {"name": "Nairobi", "type": "city", "elevation_m": 1795, "country": "Kenya"},
    {"name": "Mexico City", "type": "city", "elevation_m": 2240, "country": "Mexico"},
    {"name": "Addis Ababa", "type": "city", "elevation_m": 2355, "country": "Ethiopia"},
    {"name": "Bogota", "type": "city", "elevation_m": 2640, "country": "Colombia"},
    {"name": "Quito", "type": "city", "elevation_m": 2850, "country": "Ecuador"},
    {"name": "La Paz", "type": "city", "elevation_m": 3640, "country": "Bolivia"},
    {"name": "Lhasa", "type": "city", "elevation_m": 3656, "country": "China (Tibet)"},

    # Prominent Mountain Summits
    {"name": "Ben Nevis", "type": "peak", "elevation_m": 1345, "country": "United Kingdom (Scotland)"},
    {"name": "Mount Kosciuszko", "type": "peak", "elevation_m": 2228, "country": "Australia"},
    {"name": "Mount Olympus", "type": "peak", "elevation_m": 2917, "country": "Greece"},
    {"name": "Mount Cook (Aoraki)", "type": "peak", "elevation_m": 3724, "country": "New Zealand"},
    {"name": "Mount Fuji", "type": "peak", "elevation_m": 3776, "country": "Japan"},
    {"name": "Mount Whitney", "type": "peak", "elevation_m": 4421, "country": "United States"},
    {"name": "Matterhorn", "type": "peak", "elevation_m": 4478, "country": "Switzerland/Italy"},
    {"name": "Mont Blanc", "type": "peak", "elevation_m": 4808, "country": "France/Italy"},
    {"name": "Mount Elbrus", "type": "peak", "elevation_m": 5642, "country": "Russia"},
    {"name": "Mount Kilimanjaro", "type": "peak", "elevation_m": 5895, "country": "Tanzania"},
    {"name": "Denali (Mount McKinley)", "type": "peak", "elevation_m": 6190, "country": "United States (Alaska)"},
    {"name": "Aconcagua", "type": "peak", "elevation_m": 6961, "country": "Argentina"},
    {"name": "K2", "type": "peak", "elevation_m": 8611, "country": "Pakistan/China"},
    {"name": "Mount Everest", "type": "peak", "elevation_m": 8849, "country": "Nepal/China"},
]


@dataclass
class GeoTopologyQuestion:
    """Single question in the GeoTopology benchmark."""
    id: str
    category: Literal["border_adjacency", "landlocked_maritime", "elevation_hypsometry"]
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


def generate_border_questions(target_count: int = 400, seed: int = 42) -> list[GeoTopologyQuestion]:
    """Generate sovereign land border adjacency questions."""
    rng = random.Random(seed)
    questions: list[GeoTopologyQuestion] = []

    true_pairs = [p for p in LAND_BORDER_PAIRS if p["has_border"]]
    false_pairs = [p for p in LAND_BORDER_PAIRS if not p["has_border"]]

    target_each = target_count // 2

    # Templates
    # Positive query: "Does CountryA share a direct land border with CountryB?"
    # Negative query: "Does CountryA share a direct land border with CountryB?"
    def create_q(pair: dict[str, Any], query_has_border: bool, q_id: str) -> GeoTopologyQuestion:
        c_a = pair["country_a"]
        c_b = pair["country_b"]
        # Randomly flip order for symmetric phrasing
        if rng.random() > 0.5:
            c_a, c_b = c_b, c_a

        if query_has_border:
            text = f"Does {c_a} share a direct sovereign land border with {c_b}?"
            expected = pair["has_border"]
        else:
            text = f"Is {c_a} geographically separated from {c_b} without sharing a direct sovereign land border?"
            expected = not pair["has_border"]

        return GeoTopologyQuestion(
            id=q_id,
            category="border_adjacency",
            context=f"Geopolitical boundary assessment between {c_a} and {c_b}.",
            question=text,
            expected_answer=expected,
            metadata={"country_a": c_a, "country_b": c_b, "actual_has_border": pair["has_border"]},
        )

    # Sample balanced set
    idx = 0
    # Also create border triplets: (Country, Neighbor1, Neighbor2)
    # e.g., "Does Germany share a land border with both France and Austria?" (True)
    # vs "Does Germany share a land border with both France and Italy?" (False)
    country_neighbors: dict[str, set[str]] = {}
    country_non_neighbors: dict[str, set[str]] = {}
    all_countries = set()
    for p in LAND_BORDER_PAIRS:
        ca, cb = p["country_a"], p["country_b"]
        all_countries.add(ca)
        all_countries.add(cb)
        if p["has_border"]:
            country_neighbors.setdefault(ca, set()).add(cb)
            country_neighbors.setdefault(cb, set()).add(ca)
        else:
            country_non_neighbors.setdefault(ca, set()).add(cb)
            country_non_neighbors.setdefault(cb, set()).add(ca)

    seen_questions = set()
    while len(questions) < target_count:
        want_expected_true = (sum(1 for q in questions if q.expected_answer) < target_count // 2)

        # 60% direct pair query, 40% joint/triplet query
        if rng.random() < 0.6:
            if want_expected_true:
                if rng.random() > 0.5 and true_pairs:
                    pair = rng.choice(true_pairs)
                    q = create_q(pair, query_has_border=True, q_id=f"topoborder_{idx:04d}")
                else:
                    pair = rng.choice(false_pairs)
                    q = create_q(pair, query_has_border=False, q_id=f"topoborder_{idx:04d}")
            else:
                if rng.random() > 0.5 and false_pairs:
                    pair = rng.choice(false_pairs)
                    q = create_q(pair, query_has_border=True, q_id=f"topoborder_{idx:04d}")
                else:
                    pair = rng.choice(true_pairs)
                    q = create_q(pair, query_has_border=False, q_id=f"topoborder_{idx:04d}")
        else:
            # Triplet query: pick a hub country with at least 2 neighbors
            hubs = [c for c, nbs in country_neighbors.items() if len(nbs) >= 2]
            hub = rng.choice(hubs)
            nbs = list(country_neighbors[hub])
            non_nbs = list(country_non_neighbors.get(hub, []))

            if want_expected_true:
                # pick 2 actual neighbors
                n1, n2 = rng.sample(nbs, 2)
                text = f"Does {hub} share a direct sovereign land border with both {n1} and {n2}?"
                q = GeoTopologyQuestion(
                    id=f"topoborder_{idx:04d}",
                    category="border_adjacency",
                    context=f"Multi-border connectivity assessment for {hub}.",
                    question=text,
                    expected_answer=True,
                    metadata={"hub": hub, "neighbor_1": n1, "neighbor_2": n2, "actual": True},
                )
            else:
                # pick 1 actual neighbor and 1 non-neighbor if available, or 2 non-neighbors
                if non_nbs:
                    n1 = rng.choice(nbs)
                    nn = rng.choice(non_nbs)
                    text = f"Does {hub} share a direct sovereign land border with both {n1} and {nn}?"
                    q = GeoTopologyQuestion(
                        id=f"topoborder_{idx:04d}",
                        category="border_adjacency",
                        context=f"Multi-border connectivity assessment for {hub}.",
                        question=text,
                        expected_answer=False,
                        metadata={"hub": hub, "neighbor_1": n1, "non_neighbor": nn, "actual": False},
                    )
                else:
                    continue

        if q.question not in seen_questions:
            seen_questions.add(q.question)
            questions.append(q)
            idx += 1

    return questions


def generate_landlocked_questions(target_count: int = 400, seed: int = 42) -> list[GeoTopologyQuestion]:
    """Generate landlocked vs maritime ocean access questions."""
    rng = random.Random(seed)
    questions: list[GeoTopologyQuestion] = []

    facts = LANDLOCKED_MARITIME_FACTS
    landlocked_list = [f["country"] for f in facts if f["is_landlocked"]]
    coastal_list = [f["country"] for f in facts if not f["is_landlocked"]]

    seen_questions = set()
    idx = 0

    while len(questions) < target_count:
        want_expected_true = (sum(1 for q in questions if q.expected_answer) < target_count // 2)

        # 50% single country classification, 50% comparative pair classification
        q_type = rng.choice(["single", "comparative_cross", "joint_same"])

        if q_type == "single":
            f = rng.choice(facts)
            c = f["country"]
            is_ll = f["is_landlocked"]
            mode = rng.choice([1, 2, 3, 4])
            if mode == 1:
                text = f"Is {c} a landlocked nation without direct sovereign coastline access to the ocean or open sea?"
                expected = is_ll
            elif mode == 2:
                text = f"Does {c} possess a sovereign coastline with direct access to an ocean or open sea?"
                expected = not is_ll
            elif mode == 3:
                text = f"Is {c} completely landlocked with no sovereign coastline borders on the world's oceans?"
                expected = is_ll
            else:
                text = f"Is {c} a coastal or island nation possessing sovereign maritime coastlines?"
                expected = not is_ll

            if expected != want_expected_true:
                continue

            q = GeoTopologyQuestion(
                id=f"topolandlocked_{idx:04d}",
                category="landlocked_maritime",
                context=f"Maritime sovereignty and oceanic coastline classification for {c}.",
                question=text,
                expected_answer=expected,
                metadata={"country": c, "is_landlocked": is_ll, "continent": f["continent"]},
            )

        elif q_type == "comparative_cross":
            ll_c = rng.choice(landlocked_list)
            co_c = rng.choice(coastal_list)

            if want_expected_true:
                # True: "Is Bolivia landlocked while Chile possesses an open ocean coastline?"
                text = f"Is {ll_c} a landlocked nation, whereas {co_c} possesses sovereign open ocean coastline access?"
                expected = True
                meta = {"landlocked_country": ll_c, "coastal_country": co_c}
            else:
                # False: inverted
                text = f"Is {co_c} a landlocked nation, whereas {ll_c} possesses sovereign open ocean coastline access?"
                expected = False
                meta = {"landlocked_country": ll_c, "coastal_country": co_c}

            q = GeoTopologyQuestion(
                id=f"topolandlocked_{idx:04d}",
                category="landlocked_maritime",
                context=f"Comparative oceanic access between {ll_c} and {co_c}.",
                question=text,
                expected_answer=expected,
                metadata=meta,
            )

        else:  # joint_same
            if want_expected_true:
                if rng.random() > 0.5:
                    c1, c2 = rng.sample(landlocked_list, 2)
                    text = f"Are both {c1} and {c2} landlocked nations lacking direct ocean coastlines?"
                    expected = True
                else:
                    c1, c2 = rng.sample(coastal_list, 2)
                    text = f"Do both {c1} and {c2} possess sovereign ocean or sea coastlines?"
                    expected = True
            else:
                # Mix one landlocked and one coastal
                ll_c = rng.choice(landlocked_list)
                co_c = rng.choice(coastal_list)
                if rng.random() > 0.5:
                    text = f"Are both {ll_c} and {co_c} landlocked nations lacking direct ocean coastlines?"
                    expected = False
                else:
                    text = f"Do both {ll_c} and {co_c} possess sovereign ocean or sea coastlines?"
                    expected = False

            q = GeoTopologyQuestion(
                id=f"topolandlocked_{idx:04d}",
                category="landlocked_maritime",
                context="Joint maritime coastline classification.",
                question=text,
                expected_answer=expected,
                metadata={"type": "joint_same"},
            )

        if q.question not in seen_questions:
            seen_questions.add(q.question)
            questions.append(q)
            idx += 1

    return questions


def generate_elevation_questions(target_count: int = 400, seed: int = 42) -> list[GeoTopologyQuestion]:
    """Generate 3D elevation comparison and depression questions."""
    rng = random.Random(seed)
    questions: list[GeoTopologyQuestion] = []
    entities = ELEVATION_ENTITIES

    # Pre-generate valid comparison pairs with at least 300m elevation separation
    comp_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue
            e1, e2 = entities[i], entities[j]
            diff = e1["elevation_m"] - e2["elevation_m"]
            if abs(diff) >= 300:
                comp_pairs.append((e1, e2))

    depressions = [e for e in entities if e["elevation_m"] < 0]
    non_depressions = [e for e in entities if e["elevation_m"] > 100]

    idx = 0
    while len(questions) < target_count:
        current_trues = sum(1 for q in questions if q.expected_answer)
        current_falses = len(questions) - current_trues

        # 80% relative elevation comparisons, 20% below sea level checks
        use_depression = (rng.random() < 0.2)

        if use_depression:
            if rng.random() > 0.5:
                # True depression check
                dep = rng.choice(depressions)
                text = f"Is {dep['name']} geographically situated below sea level (negative elevation)?"
                expected = True
                meta = {"entity": dep["name"], "elevation_m": dep["elevation_m"]}
            else:
                # False depression check
                non_dep = rng.choice(non_depressions)
                text = f"Is {non_dep['name']} geographically situated below sea level (negative elevation)?"
                expected = False
                meta = {"entity": non_dep["name"], "elevation_m": non_dep["elevation_m"]}
        else:
            e1, e2 = rng.choice(comp_pairs)
            is_higher = (e1["elevation_m"] > e2["elevation_m"])
            # Phrasing: "higher than" or "lower than"
            if rng.random() > 0.5:
                text = f"Is the elevation of {e1['name']} significantly higher than the elevation of {e2['name']}?"
                expected = is_higher
            else:
                text = f"Is the elevation of {e1['name']} significantly lower than the elevation of {e2['name']}?"
                expected = not is_higher
            meta = {
                "entity_1": e1["name"],
                "elev_1_m": e1["elevation_m"],
                "entity_2": e2["name"],
                "elev_2_m": e2["elevation_m"],
                "elev_diff_m": e1["elevation_m"] - e2["elevation_m"],
            }

        # Check balance
        if expected and current_trues >= target_count // 2:
            continue
        if not expected and current_falses >= target_count // 2:
            continue

        if not any(eq.question == text for eq in questions):
            questions.append(
                GeoTopologyQuestion(
                    id=f"topoelev_{idx:04d}",
                    category="elevation_hypsometry",
                    context="Hypsometric physical geography and terrain elevation analysis.",
                    question=text,
                    expected_answer=expected,
                    metadata=meta,
                )
            )
            idx += 1

    return questions


def generate_geotopology_dataset(total_count: int = 1200, seed: int = 42) -> list[GeoTopologyQuestion]:
    """Generate balanced GeoTopology dataset across all 3 categories."""
    per_cat = total_count // 3
    q_border = generate_border_questions(target_count=per_cat, seed=seed)
    q_landlocked = generate_landlocked_questions(target_count=per_cat, seed=seed + 1)
    q_elev = generate_elevation_questions(target_count=per_cat, seed=seed + 2)

    combined = q_border + q_landlocked + q_elev
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined
