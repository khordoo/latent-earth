"""Data models for Latent Earth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any


@dataclass(frozen=True)
class Coordinate:
    """Represents a geographic point on Earth."""

    latitude: float
    longitude: float

    @property
    def id(self) -> str:
        """Stable deterministic string identifier."""
        return f"lat_{self.latitude:+.3f}_lon_{self.longitude:+.3f}"

    def format_cardinal(self) -> str:
        """Format coordinate with cardinal directions (e.g., 45.000° N, 73.000° W)."""
        lat_dir = "N" if self.latitude >= 0 else "S"
        lon_dir = "E" if self.longitude >= 0 else "W"
        return f"{abs(self.latitude):.3f}° {lat_dir}, {abs(self.longitude):.3f}° {lon_dir}"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "id": self.id,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Coordinate:
        return cls(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
        )


@dataclass
class Prediction:
    """Represents a model's prediction for a single coordinate."""

    coordinate: Coordinate
    probability_land: float | None
    is_land: bool
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_response: dict[str, Any] | None = None

    @property
    def uncertainty(self) -> float:
        """Normalized uncertainty score from P(land) in [0, 1].

        P = 0.50 -> maximum uncertainty (1.0)
        P = 0.00 -> minimum uncertainty (0.0)
        P = 1.00 -> minimum uncertainty (0.0)
        """
        if self.probability_land is None:
            return 1.0
        return max(0.0, min(1.0, 1.0 - 2.0 * abs(self.probability_land - 0.5)))

    @property
    def entropy(self) -> float:
        """Binary entropy in bits: - (p * log2(p) + (1-p) * log2(1-p))."""
        if self.probability_land is None:
            return 1.0
        p = max(1e-9, min(1.0 - 1e-9, self.probability_land))
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to JSON-serializable dictionary."""
        d = {
            "coordinate_id": self.coordinate.id,
            "latitude": self.coordinate.latitude,
            "longitude": self.coordinate.longitude,
            "probability_land": self.probability_land,
            "is_land": self.is_land,
            "uncertainty": self.uncertainty,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp,
        }
        if self.raw_response is not None:
            d["raw_response"] = self.raw_response
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        """Reconstruct prediction from dictionary."""
        coord = Coordinate(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
        )
        prob = data.get("probability_land")
        if prob is not None:
            prob = float(prob)
        return cls(
            coordinate=coord,
            probability_land=prob,
            is_land=bool(data["is_land"]),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            prompt_version=str(data.get("prompt_version", "")),
            timestamp=str(data.get("timestamp", "")),
            raw_response=data.get("raw_response"),
        )
