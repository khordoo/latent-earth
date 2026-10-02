"""Test 64-request concurrent batch against Fastino GLiDE API.

Tests whether the upgraded GPU and serving infrastructure handles 64
concurrent in-flight queries without socket drops, rate limits (429), or gateway errors.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
import httpx
from dotenv import load_dotenv

load_dotenv()

FASTINO_API_KEY = os.getenv("FASTINO_API_KEY")
if not FASTINO_API_KEY:
    print("ERROR: FASTINO_API_KEY not found in environment or .env")
    sys.exit(1)

FASTINO_URL = "https://api.fastino.ai/v1/systemone"
BATCH_SIZE = 64


def get_64_untested_coordinates() -> list[dict[str, Any]]:
    """Select 64 deterministic, untested coordinates from Japan/Korea grid."""
    with open("outputs/stratified_512_coordinates.json") as f:
        tested_512 = set(json.load(f)["coordinates"].keys())

    candidates = []
    with open("outputs/jev_jev-latest_japan_korea_0_5deg_land-water-v1.jsonl") as f:
        for line in f:
            item = json.loads(line)
            cid = item["coordinate_id"]
            if cid not in tested_512:
                candidates.append(item)

    # Pick 64 evenly distributed across candidates
    step = len(candidates) // BATCH_SIZE
    selected = [candidates[i * step] for i in range(BATCH_SIZE)]
    return selected


def build_payload(item: dict[str, Any]) -> dict[str, Any]:
    lat = item["latitude"]
    lon = item["longitude"]
    lat_card = f"{abs(lat):.3f}° {'N' if lat >= 0 else 'S'}"
    lon_card = f"{abs(lon):.3f}° {'E' if lon >= 0 else 'W'}"
    cardinal = f"{lat_card}, {lon_card}"
    state = f"Geographic coordinate: {cardinal} (Latitude: {lat:+.4f}, Longitude: {lon:+.4f})"

    return {
        "model": "fastino/glide",
        "state": {
            "scenario": (
                "You are an expert Earth scientist and geospatial intelligence system. "
                "Determine whether the given geographic coordinate on Earth is located on land or water. "
                "Consider physical Earth geography, continental landmasses, islands, and open water bodies."
            ),
            "facts": [state],
        },
        "questions": {
            "is_land": {
                "type": "choice",
                "instructions": (
                    f"Is the geographic coordinate {cardinal} on land or water? "
                    "Choose 'land' if the coordinate falls on continental land, an island, or inland water enclosed by land. "
                    "Choose 'water' if it falls in the ocean, seas, or major coastal water bodies."
                ),
                "criteria": {
                    "land": "The coordinate is located on land.",
                    "water": "The coordinate is located on open water or ocean.",
                },
            }
        },
    }


async def send_single_query(
    client: httpx.AsyncClient,
    coord_info: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    cid = coord_info["coordinate_id"]
    payload = build_payload(coord_info)
    headers = {
        "Authorization": f"Bearer {FASTINO_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    result = {
        "index": index,
        "coordinate_id": cid,
        "latitude": coord_info["latitude"],
        "longitude": coord_info["longitude"],
        "status_code": None,
        "latency_seconds": None,
        "success": False,
        "error": None,
        "output_tokens": None,
        "input_tokens": None,
        "choice": None,
        "confidence": None,
        "rate_limit_headers": {},
    }

    try:
        response = await client.post(FASTINO_URL, json=payload, headers=headers)
        elapsed = time.monotonic() - t0
        result["latency_seconds"] = round(elapsed, 4)
        result["status_code"] = response.status_code

        # Capture any rate-limit headers if present
        for h, val in response.headers.items():
            if "ratelimit" in h.lower() or "retry-after" in h.lower():
                result["rate_limit_headers"][h] = val

        if response.status_code == 200:
            data = response.json()
            result["success"] = True
            usage = data.get("usage", {})
            result["input_tokens"] = usage.get("input_tokens")
            result["output_tokens"] = usage.get("output_tokens")
            answers = data.get("answers", {}).get("is_land", {})
            result["choice"] = answers.get("choice")
            result["confidence"] = answers.get("confidence")
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"

    except Exception as exc:
        elapsed = time.monotonic() - t0
        result["latency_seconds"] = round(elapsed, 4)
        result["error"] = f"{type(exc).__name__}: {str(exc)}"

    return result


async def main():
    coords = get_64_untested_coordinates()
    print(f"=== TESTING 64 CONCURRENT BATCH AGAINST FASTINO GLiDE ===")
    print(f"Selected {len(coords)} untested Japan/Korea coordinates.")
    print(f"Endpoint: {FASTINO_URL}")
    print(f"Launching all {BATCH_SIZE} requests simultaneously...\n")

    limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
    timeout = httpx.Timeout(120.0, connect=15.0)

    t_start = time.monotonic()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Launch all 64 requests concurrently
        tasks = [send_single_query(client, c, i) for i, c in enumerate(coords)]
        results = await asyncio.gather(*tasks)

    total_wall_clock = time.monotonic() - t_start

    # Analyze results
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency_seconds"] for r in results if r["latency_seconds"] is not None]

    print(f"\n=======================================================")
    print(f"=== BATCH 64 EXECUTION SUMMARY ===")
    print(f"=======================================================")
    print(f"Total Requests Sent: {len(results)}")
    print(f"Successful (200 OK): {len(successes)} / {len(results)} ({len(successes)/len(results)*100:.1f}%)")
    print(f"Failures / Errors:   {len(failures)} / {len(results)} ({len(failures)/len(results)*100:.1f}%)")
    print(f"Total Wall-Clock:    {total_wall_clock:.2f} seconds ({total_wall_clock/60:.2f} min)")
    print(f"Effective Throughput:{len(results)/total_wall_clock:.2f} req / second")

    if failures:
        print(f"\n--- Failure Details ({len(failures)} errors) ---")
        for f in failures:
            print(f"  [{f['index']}] {f['coordinate_id']}: {f['error']} (after {f['latency_seconds']}s)")

    if latencies:
        sorted_lats = sorted(latencies)
        p50 = sorted_lats[len(sorted_lats) // 2]
        p90 = sorted_lats[int(len(sorted_lats) * 0.90)]
        p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
        print(f"\n--- Latency Distribution ---")
        print(f"  Min:    {min(latencies):.2f}s")
        print(f"  Median: {p50:.2f}s")
        print(f"  Mean:   {sum(latencies)/len(latencies):.2f}s")
        print(f"  P90:    {p90:.2f}s")
        print(f"  P95:    {p95:.2f}s")
        print(f"  Max:    {max(latencies):.2f}s")

    if successes:
        fast_path = [s for s in successes if (s["output_tokens"] or 0) <= 1]
        thinking_path = [s for s in successes if (s["output_tokens"] or 0) > 1]
        print(f"\n--- Thinking vs. Fast Path ---")
        print(f"  Fast Path (<=1 token):   {len(fast_path)} ({len(fast_path)/len(successes)*100:.1f}%)")
        print(f"  Thinking Path (>1 token):{len(thinking_path)} ({len(thinking_path)/len(successes)*100:.1f}%)")
        if thinking_path:
            avg_tok = sum(s["output_tokens"] for s in thinking_path) / len(thinking_path)
            print(f"  Average Thinking Tokens (when engaged): {avg_tok:.1f}")

    # Check for rate limit headers
    rate_headers = {}
    for r in results:
        if r.get("rate_limit_headers"):
            rate_headers.update(r["rate_limit_headers"])
    if rate_headers:
        print(f"\n--- Observed Rate Limit Headers ---")
        for k, v in rate_headers.items():
            print(f"  {k}: {v}")
    else:
        print(f"\nRate Limit Headers: None emitted by server.")

    # Save summary to outputs
    output_path = "outputs/batch_64_concurrency_test_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "batch_size": BATCH_SIZE,
                "total_wall_clock_seconds": total_wall_clock,
                "success_count": len(successes),
                "failure_count": len(failures),
                "latencies": {
                    "min": min(latencies) if latencies else None,
                    "median": p50 if latencies else None,
                    "mean": sum(latencies)/len(latencies) if latencies else None,
                    "p90": p90 if latencies else None,
                    "p95": p95 if latencies else None,
                    "max": max(latencies) if latencies else None,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
