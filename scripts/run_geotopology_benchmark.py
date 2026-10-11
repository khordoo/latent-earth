"""Runner script for GeoTopology: 3D Relief & Border Connectivity Benchmark.

Supports:
- Cloudflare Clef (27B) via OpenRouter Decisions API
- Perplexity Pplx-Decider (27B) via OpenRouter Decisions API
- Microsoft Decision-1 via OpenRouter Decisions API
- OpenAI GPT-6-Luna Decision via OpenAI Decisions API
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv

from latent_earth.geotopology import GeoTopologyQuestion, generate_geotopology_dataset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("geotopology")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENAI_DECISIONS_URL = "https://api.openai.com/v1/decisions"
TYPESAFE_JEV_URL = "https://api.typesafe.ai/v1/systemone"

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DATA_DIR = Path("frontend/public/data")
FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"id": "clef", "name": "Cloudflare Clef (27B)", "provider": "openrouter", "model_id": "cloudflare/clef"},
    {"id": "pplx", "name": "Perplexity Pplx-Decider", "provider": "openrouter", "model_id": "perplexity/pplx-decider-v1.1-27b"},
    {"id": "msft", "name": "Microsoft Decision-1", "provider": "openrouter", "model_id": "microsoft/microsoft-decision-1"},
    {"id": "jev", "name": "TypeSafe Jev", "provider": "typesafe", "model_id": "jev-latest"},
    {"id": "openai", "name": "OpenAI Decisions (Luna)", "provider": "openai", "model_id": "gpt-6-luna"},
]


async def query_model_single(
    client: httpx.AsyncClient,
    q: GeoTopologyQuestion,
    model_spec: dict[str, Any],
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> dict[str, Any]:
    provider = model_spec["provider"]
    model_id = model_spec["model_id"]

    if provider == "openrouter":
        url = OPENROUTER_DECISIONS_URL
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/latent-earth",
            "X-Title": "GeoTopology Benchmark",
        }
        payload = {
            "model": model_id,
            "state": q.context,
            "questions": {
                "statement_is_true": {
                    "type": "noul",
                    "instructions": q.question,
                    "criteria": {
                        "true": "The statement is geographically correct, factual, and accurate.",
                        "false": "The statement is geographically false, incorrect, or contradictory.",
                    },
                }
            },
        }
    elif provider == "typesafe":
        url = TYPESAFE_JEV_URL
        headers = {
            "Authorization": f"Bearer {TYPESAFE_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "LatentEarth/0.1.0",
        }
        payload = {
            "model": model_id,
            "state": q.context,
            "questions": {
                "statement_is_true": {
                    "type": "noul",
                    "instructions": q.question,
                    "criteria": {
                        "true": "The statement is geographically correct, factual, and accurate.",
                        "false": "The statement is geographically false, incorrect, or contradictory.",
                    },
                }
            },
        }
    else:  # openai
        url = OPENAI_DECISIONS_URL
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "input": q.context,
            "questions": [
                {
                    "type": "predicate",
                    "name": "statement_is_true",
                    "instructions": q.question,
                }
            ],
        }

    t0 = time.monotonic()
    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
                resp = await client.post(url, json=payload, headers=headers)
            lat = time.monotonic() - t0

            if resp.status_code == 200:
                data = resp.json()
                prob = 0.5
                cost = 0.0

                if provider == "openrouter":
                    ans = data.get("answers", {}).get("statement_is_true", {})
                    p = ans.get("noul", ans.get("probability"))
                    if p is not None:
                        prob = float(p)
                    cost = data.get("usage", {}).get("cost", 0.0)
                elif provider == "typesafe":
                    ans = data.get("answers", {}).get("statement_is_true", {})
                    p = ans.get("noul", ans.get("probability"))
                    if p is not None:
                        prob = float(p)
                    usage = data.get("usage", {})
                    toks = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    cost = toks * 0.042 / 1e6
                else:  # openai
                    ans_list = data.get("answers", [])
                    if ans_list and isinstance(ans_list, list):
                        p = ans_list[0].get("probability")
                        if p is not None:
                            prob = float(p)
                    usage = data.get("usage", {})
                    # Estimated OpenAI cost ~$0.15/1M tokens
                    toks = usage.get("total_tokens", usage.get("prompt_tokens", 45))
                    cost = toks * 0.15 / 1e6

                predicted_true = prob >= 0.5
                is_correct = (predicted_true == q.expected_answer)
                confidence = abs(prob - 0.5) * 2.0

                return {
                    "question_id": q.id,
                    "category": q.category,
                    "expected_answer": q.expected_answer,
                    "predicted_answer": predicted_true,
                    "probability_true": round(prob, 4),
                    "confidence": round(confidence, 4),
                    "is_correct": is_correct,
                    "latency_seconds": round(lat, 4),
                    "cost": cost,
                    "attempt": attempt,
                }

            if resp.status_code in {429, 500, 502, 503, 504}:
                delay = 0.8 * (2 ** (attempt - 1)) + random.uniform(0.1, 0.3)
                logger.warning("HTTP %d for %s (attempt %d). Retrying in %.2fs...", resp.status_code, q.id, attempt, delay)
                await asyncio.sleep(delay)
                continue

            logger.error("HTTP %d error for %s: %s", resp.status_code, q.id, resp.text[:200])
            break

        except httpx.RequestError as exc:
            delay = 0.8 * (2 ** (attempt - 1)) + random.uniform(0.1, 0.3)
            logger.warning("Network error %s for %s. Retrying in %.2fs...", type(exc).__name__, q.id, delay)
            await asyncio.sleep(delay)

    # Fallback default on fatal retry exhaustion
    return {
        "question_id": q.id,
        "category": q.category,
        "expected_answer": q.expected_answer,
        "predicted_answer": False,
        "probability_true": 0.5,
        "confidence": 0.0,
        "is_correct": False,
        "latency_seconds": 9.99,
        "cost": 0.0,
        "attempt": max_retries,
    }


def compute_geotopology_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r["is_correct"])
    acc = correct / total

    tp = sum(1 for r in results if r["predicted_answer"] and r["expected_answer"])
    tn = sum(1 for r in results if not r["predicted_answer"] and not r["expected_answer"])
    fp = sum(1 for r in results if r["predicted_answer"] and not r["expected_answer"])
    fn = sum(1 for r in results if not r["predicted_answer"] and r["expected_answer"])

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (tpr + tnr) / 2.0

    brier = sum((r["probability_true"] - (1.0 if r["expected_answer"] else 0.0)) ** 2 for r in results) / total
    latencies = [r["latency_seconds"] for r in results if r["latency_seconds"] < 9.0]
    cost = sum(r["cost"] for r in results)

    # Per-category metrics
    categories = ["border_adjacency", "landlocked_maritime", "elevation_hypsometry"]
    cat_metrics = {}
    for cat in categories:
        sub = [r for r in results if r["category"] == cat]
        if sub:
            c_corr = sum(1 for r in sub if r["is_correct"])
            c_acc = c_corr / len(sub)
            c_tp = sum(1 for r in sub if r["predicted_answer"] and r["expected_answer"])
            c_tn = sum(1 for r in sub if not r["predicted_answer"] and not r["expected_answer"])
            c_fp = sum(1 for r in sub if r["predicted_answer"] and not r["expected_answer"])
            c_fn = sum(1 for r in sub if not r["predicted_answer"] and r["expected_answer"])
            c_tpr = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            c_tnr = c_tn / (c_tn + c_fp) if (c_tn + c_fp) > 0 else 0.0
            cat_metrics[cat] = {
                "count": len(sub),
                "accuracy": round(c_acc, 4),
                "balanced_accuracy": round((c_tpr + c_tnr) / 2.0, 4),
            }

    return {
        "total_questions": total,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "brier_score": round(brier, 4),
        "categories": cat_metrics,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "median_latency_seconds": round(float(np.median(latencies)), 4) if latencies else None,
        "p90_latency_seconds": round(float(np.percentile(latencies, 90)), 4) if latencies else None,
        "total_cost_usd": round(cost, 6),
    }


async def run_benchmark_for_model(
    model_spec: dict[str, Any],
    questions: list[GeoTopologyQuestion],
    concurrency: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info("Evaluating %s on %d questions (concurrency=%d)...", model_spec["name"], len(questions), concurrency)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()

    async with httpx.AsyncClient(timeout=45.0) as client:
        tasks = [query_model_single(client, q, model_spec, sem) for q in questions]
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    metrics = compute_geotopology_metrics(results)
    metrics["wall_clock_time_seconds"] = round(elapsed, 2)
    metrics["effective_throughput_req_sec"] = round(len(questions) / max(0.1, elapsed), 2)
    logger.info(
        "Finished %s in %.1fs: Balanced Acc: %.2f%% (Borders: %.1f%%, Landlocked: %.1f%%, Elevation: %.1f%%) Cost: $%.5f",
        model_spec["name"], elapsed, metrics["balanced_accuracy"] * 100,
        metrics["categories"].get("border_adjacency", {}).get("balanced_accuracy", 0) * 100,
        metrics["categories"].get("landlocked_maritime", {}).get("balanced_accuracy", 0) * 100,
        metrics["categories"].get("elevation_hypsometry", {}).get("balanced_accuracy", 0) * 100,
        metrics["total_cost_usd"],
    )
    return metrics, results


async def main():
    parser = argparse.ArgumentParser(description="Run GeoTopology benchmark")
    parser.add_argument("--size", type=int, default=300, help="Benchmark dataset size (e.g. 300 or 1200)")
    parser.add_argument("--concurrency", type=int, default=25, help="Concurrent request limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model ids or 'all'")
    args = parser.parse_args()

    questions = generate_geotopology_dataset(total_count=args.size, seed=args.seed)
    logger.info("Generated %d GeoTopology benchmark questions.", len(questions))

    summary_path = OUTPUTS_DIR / f"geotopology_{args.size}_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {
                "benchmark_name": "GeoTopology: 3D Relief & Border Connectivity Benchmark",
                "sample_size": len(questions),
                "seed": args.seed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "models": {},
            }
    else:
        summary = {
            "benchmark_name": "GeoTopology: 3D Relief & Border Connectivity Benchmark",
            "sample_size": len(questions),
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": {},
        }

    all_detailed = {}

    selected_models = MODELS
    if args.models != "all":
        ids = [x.strip() for x in args.models.split(",")]
        selected_models = [m for m in MODELS if m["id"] in ids]

    for spec in selected_models:
        m_id = spec["id"]
        # Preflight test
        test_q = questions[0]
        try:
            async with httpx.AsyncClient(timeout=10.0) as test_client:
                test_sem = asyncio.Semaphore(1)
                res = await query_model_single(test_client, test_q, spec, test_sem, max_retries=1)
                if res.get("attempt") >= 5 or (res.get("latency_seconds") == 9.99 and not res.get("is_correct")):
                    logger.warning("Pre-flight check failed for %s. Skipping this model.", spec["name"])
                    continue
        except Exception as e:
            logger.warning("Pre-flight exception for %s: %s. Skipping this model.", spec["name"], e)
            continue

        metrics, results = await run_benchmark_for_model(spec, questions, concurrency=args.concurrency)
        summary["models"][m_id] = {
            "name": spec["name"],
            "provider": spec["provider"],
            "model_id": spec["model_id"],
            "metrics": metrics,
        }
        all_detailed[m_id] = results

        # Save individual model outputs
        out_f = OUTPUTS_DIR / f"geotopology_{args.size}_{m_id}.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump({"model": spec, "metrics": metrics, "results": results}, f, indent=2)

    # Save summary
    summary_path = OUTPUTS_DIR / f"geotopology_{args.size}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.size >= 1200:
        with open(FRONTEND_DATA_DIR / "geotopology_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    logger.info("Saved summary results to %s", summary_path)


if __name__ == "__main__":
    asyncio.run(main())
