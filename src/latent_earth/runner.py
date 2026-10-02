"""Inference execution runner with concurrency, caching, and progress tracking."""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from typing import Sequence
from latent_earth.cache import PredictionCache
from latent_earth.config import AppConfig
from latent_earth.grid import generate_grid
from latent_earth.models import Coordinate, Prediction
from latent_earth.providers.base import DecisionProvider

logger = logging.getLogger(__name__)


def chunk_list(items: Sequence[Coordinate], chunk_size: int) -> list[list[Coordinate]]:
    """Split a list into chunks of given size."""
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def print_run_summary(
    provider_name: str,
    model: str,
    step_degrees: float,
    total_coords: int,
    cached_coords: int,
    remaining_coords: int,
    batch_size: int,
    concurrency: int,
    dry_run: bool = False,
    estimated_cost: str | None = None,
) -> None:
    """Print standard formatted run summary before starting execution."""
    expected_batches = math.ceil(remaining_coords / batch_size) if batch_size > 0 and remaining_coords > 0 else 0
    header = "=== LATENT EARTH RUN PLAN ===" if not dry_run else "=== LATENT EARTH [DRY RUN] ==="
    print("\n" + header)
    print(f"Provider:           {provider_name}")
    print(f"Model:              {model}")
    print(f"Grid spacing:       {step_degrees:g}°")
    print(f"Total coordinates:  {total_coords:,}")
    print(f"Already cached:     {cached_coords:,}")
    print(f"Remaining to query: {remaining_coords:,}")
    print(f"Batch size:         {batch_size}")
    print(f"Concurrency:        {concurrency}")
    print(f"Expected batches:   {expected_batches:,}")
    if estimated_cost:
        print(f"Estimated Cost:     {estimated_cost}")
    print("=" * len(header) + "\n")


class Runner:
    """Orchestrates grid generation, caching, provider inference, and result storage."""

    def __init__(
        self,
        config: AppConfig,
        provider: DecisionProvider,
        cache: PredictionCache,
    ) -> None:
        self.config = config
        self.provider = provider
        self.cache = cache

    async def run(
        self,
        step_degrees: float | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        estimated_cost: str | None = None,
    ) -> list[Prediction]:
        """Execute the grid queries, respecting caching and limits."""
        step = step_degrees if step_degrees is not None else self.config.grid.step_degrees
        all_coords = generate_grid(
            step_degrees=step,
            convention=self.config.grid.convention,
        )
        total_coords = len(all_coords)

        # Identify already cached coordinates
        cached_predictions = self.cache.load()
        cached_ids = set(cached_predictions.keys())

        # Determine remaining coordinates
        remaining_coords = [c for c in all_coords if c.id not in cached_ids]

        if limit is not None and limit > 0:
            remaining_coords = remaining_coords[:limit]

        cached_count = len(cached_ids)
        remaining_count = len(remaining_coords)
        batch_size = self.config.inference.batch_size
        concurrency = self.config.inference.concurrency

        print_run_summary(
            provider_name=self.provider.provider_name,
            model=self.provider.model,
            step_degrees=step,
            total_coords=total_coords,
            cached_coords=cached_count,
            remaining_coords=remaining_count,
            batch_size=batch_size,
            concurrency=concurrency,
            dry_run=dry_run,
            estimated_cost=estimated_cost,
        )

        if dry_run:
            logger.info("Dry run requested. Exiting without sending API requests.")
            # Return current cached predictions matching the grid
            return [cached_predictions[c.id] for c in all_coords if c.id in cached_predictions]

        if not remaining_coords:
            logger.info("All %d coordinates already cached! Nothing to query.", total_coords)
            return [cached_predictions[c.id] for c in all_coords if c.id in cached_predictions]

        # Chunk coordinates into batches
        batches = chunk_list(remaining_coords, batch_size)
        total_batches = len(batches)

        completed_count = cached_count
        failed_count = 0
        requests_sent = 0
        start_time = time.monotonic()

        logger.info(
            "Starting execution: %d coordinates to query in %d batches...",
            remaining_count,
            total_batches,
        )

        new_predictions: list[Prediction] = []

        try:
            for b_idx, batch in enumerate(batches, start=1):
                batch_start = time.monotonic()
                try:
                    # Run batch concurrently using provider's concurrency semaphore
                    batch_preds = await self.provider.predict_coordinates(batch)
                    for i, pred in enumerate(batch_preds):
                        pred.sequence_index = completed_count + i

                    requests_sent += len(batch)
                    completed_count += len(batch_preds)
                    new_predictions.extend(batch_preds)

                    # Incremental cache write
                    self.cache.append_predictions(batch_preds)

                    elapsed = time.monotonic() - start_time
                    rate = (completed_count - cached_count) / elapsed if elapsed > 0 else 0.0

                    logger.info(
                        "Progress: Completed %d / %d | Cached %d | Failed %d | Requests sent %d (Batch %d/%d, %.1f coords/sec)",
                        completed_count,
                        total_coords,
                        cached_count,
                        failed_count,
                        requests_sent,
                        b_idx,
                        total_batches,
                        rate,
                    )
                except Exception as exc:
                    failed_count += len(batch)
                    logger.error(
                        "Batch %d/%d failed with error: %s",
                        b_idx,
                        total_batches,
                        exc,
                        exc_info=True,
                    )
                    # Re-raise to prevent infinite failing or silent data corruption
                    raise exc

        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.warning("Execution interrupted! Saved progress in %s.", self.cache.filepath)
            raise
        finally:
            await self.provider.close()

        logger.info(
            "Execution completed: Total %d predictions available (New: %d, Previously Cached: %d).",
            len(self.cache),
            len(new_predictions),
            cached_count,
        )

        # Assemble full list of predictions in grid order
        final_cache = self.cache.load()
        return [final_cache[c.id] for c in all_coords if c.id in final_cache]
