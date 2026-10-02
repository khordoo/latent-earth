"""Configuration management for Latent Earth."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import yaml

# Load .env file automatically
load_dotenv()


@dataclass
class GridConfig:
    step_degrees: float = 4.0
    convention: str = "cell_center"  # 'cell_center' or 'boundary'


@dataclass
class ProviderConfig:
    name: str = "jev"
    model: str = "jev-latest"
    base_url: str = "https://api.typesafe.ai/v1/systemone"
    prompt_version: str = "land-water-v1"
    api_key: str | None = None

    def resolve_api_key(self) -> str | None:
        """Resolve API key based on provider name and environment variables."""
        if self.api_key:
            return self.api_key

        if self.name == "jev":
            return os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")
        elif self.name in ("fastino", "gliner", "fastino_gliner", "glide", "fastino_glide"):
            return os.getenv("FASTINO_API_KEY")
        elif self.name in ("openrouter", "openrouter_d1", "openrouterd1", "liquid"):
            return os.getenv("OPENROUTER_API_KEY")
        elif self.name == "openai_compatible":
            return os.getenv("OPENAI_API_KEY")
        return None


@dataclass
class InferenceConfig:
    batch_size: int = 64
    concurrency: int = 16
    max_retries: int = 4
    rate_limit_per_second: float = 35.0
    timeout_seconds: float = 30.0


@dataclass
class RegionConfig:
    name: str = "global"
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0

    @property
    def is_global(self) -> bool:
        return (
            self.lat_min <= -90.0
            and self.lat_max >= 90.0
            and self.lon_min <= -180.0
            and self.lon_max >= 180.0
            and self.name.lower() in ("global", "world", "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


@dataclass
class OutputConfig:
    dir: str = "outputs"
    save_interval: int = 50


@dataclass
class AppConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> AppConfig:
        """Load configuration from a YAML file with defaults."""
        data: dict[str, Any] = {}
        if config_path:
            p = Path(config_path)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        data = loaded

        grid_data = data.get("grid", {})
        region_data = data.get("region", {})
        provider_data = data.get("provider", {})
        inference_data = data.get("inference", {})
        output_data = data.get("output", {})

        # Support grid_resolution as top-level or grid-level alias
        step = data.get("grid_resolution", grid_data.get("step_degrees", 4.0))

        return cls(
            grid=GridConfig(
                step_degrees=float(step),
                convention=str(grid_data.get("convention", "cell_center")),
            ),
            region=RegionConfig(
                name=str(region_data.get("name", "global")),
                lat_min=float(region_data.get("lat_min", -90.0)),
                lat_max=float(region_data.get("lat_max", 90.0)),
                lon_min=float(region_data.get("lon_min", -180.0)),
                lon_max=float(region_data.get("lon_max", 180.0)),
            ),
            provider=ProviderConfig(
                name=str(provider_data.get("name", "jev")),
                model=str(provider_data.get("model", "jev-latest")),
                base_url=str(provider_data.get("base_url", "https://api.typesafe.ai/v1/systemone")),
                prompt_version=str(provider_data.get("prompt_version", "land-water-v1")),
                api_key=provider_data.get("api_key"),
            ),
            inference=InferenceConfig(
                batch_size=int(inference_data.get("batch_size", 64)),
                concurrency=int(inference_data.get("concurrency", 16)),
                max_retries=int(inference_data.get("max_retries", 4)),
                rate_limit_per_second=float(inference_data.get("rate_limit_per_second", 35.0)),
                timeout_seconds=float(inference_data.get("timeout_seconds", 30.0)),
            ),
            output=OutputConfig(
                dir=str(output_data.get("dir", "outputs")),
                save_interval=int(output_data.get("save_interval", 50)),
            ),
        )
