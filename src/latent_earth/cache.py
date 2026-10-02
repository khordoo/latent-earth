"""Caching and resumability for model predictions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any
from latent_earth.models import Prediction

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitize string for filesystem paths."""
    return re.sub(r"[^\w\-.]", "_", name)


def get_default_output_path(
    output_dir: str | Path,
    provider: str,
    model: str,
    step_degrees: float,
    prompt_version: str,
) -> Path:
    """Generate a structured, descriptive filename for run outputs."""
    p_dir = Path(output_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    step_str = f"{step_degrees:g}".replace(".", "_")
    safe_model = sanitize_filename(model)
    safe_prompt = sanitize_filename(prompt_version)
    filename = f"{provider}_{safe_model}_{step_str}deg_{safe_prompt}.jsonl"
    return p_dir / filename


class PredictionCache:
    """Manages incremental persistence and loading of predictions in JSONL format."""

    def __init__(
        self,
        filepath: str | Path,
        expected_provider: str | None = None,
        expected_model: str | None = None,
        expected_prompt_version: str | None = None,
        strict_validation: bool = True,
    ) -> None:
        self.filepath = Path(filepath)
        self.expected_provider = expected_provider
        self.expected_model = expected_model
        self.expected_prompt_version = expected_prompt_version
        self.strict_validation = strict_validation
        self._cached_predictions: dict[str, Prediction] = {}
        self._loaded = False

    def load(self) -> dict[str, Prediction]:
        """Load predictions from disk, verifying compatibility."""
        if not self.filepath.exists():
            self._cached_predictions = {}
            self._loaded = True
            return self._cached_predictions

        predictions: dict[str, Prediction] = {}
        incompatible_count = 0
        corrupt_lines = 0

        with open(self.filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    pred = Prediction.from_dict(data)

                    # Validate compatibility if specified
                    if self.strict_validation:
                        if self.expected_provider and pred.provider and pred.provider != self.expected_provider:
                            incompatible_count += 1
                            continue
                        if self.expected_prompt_version and pred.prompt_version and pred.prompt_version != self.expected_prompt_version:
                            incompatible_count += 1
                            continue

                    predictions[pred.coordinate.id] = pred
                except Exception as exc:
                    corrupt_lines += 1
                    logger.warning("Corrupt line %d in %s: %s", line_no, self.filepath, exc)

        if corrupt_lines > 0:
            logger.warning("Skipped %d corrupt lines in %s", corrupt_lines, self.filepath)
        if incompatible_count > 0:
            logger.warning(
                "Ignored %d predictions in %s due to incompatible provider/model/prompt_version",
                incompatible_count,
                self.filepath,
            )

        self._cached_predictions = predictions
        self._loaded = True
        return self._cached_predictions

    def get_completed_ids(self) -> set[str]:
        """Get set of coordinate IDs already cached."""
        if not self._loaded:
            self.load()
        return set(self._cached_predictions.keys())

    def append_predictions(self, predictions: list[Prediction]) -> None:
        """Append predictions incrementally to the JSONL file safely."""
        if not predictions:
            return

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "a", encoding="utf-8") as f:
            for pred in predictions:
                line = json.dumps(pred.to_dict()) + "\n"
                f.write(line)
                self._cached_predictions[pred.coordinate.id] = pred
            f.flush()
            os.fsync(f.fileno())

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._cached_predictions)
