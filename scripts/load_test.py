"""Load test of the running service: latency and requests per second at several concurrency levels.

Each request is one Collection3 test sentence sent to POST /extract.
Start the service first, e.g. `docker run -p 8000:8000 ru-ner`.

Usage: uv run python scripts/load_test.py [--url http://127.0.0.1:8000] [--requests 300]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from ru_ner.benchmark import hardware_info
from ru_ner.data import load_collection3

RESULTS_PATH = Path("results/service_benchmark.json")


async def run_level(client, url, texts, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    latencies, errors = [], 0

    async def one(text):
        nonlocal errors
        async with semaphore:
            start = time.perf_counter()
            response = await client.post(f"{url}/extract", json={"text": text})
            latencies.append(time.perf_counter() - start)
            if response.status_code != 200:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(one(t) for t in texts))
    elapsed = time.perf_counter() - start
    ms = np.array(latencies) * 1000
    return {
        "concurrency": concurrency,
        "requests": len(texts),
        "errors": errors,
        "rps": len(texts) / elapsed,
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
    }


async def main():
    p = argparse.ArgumentParser()
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first, which adds ~45 ms
    # per request with Docker Desktop port forwarding
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--note", default="", help="free text saved with results, e.g. docker limits")
    args = p.parse_args()

    sentences = [" ".join(s) for s in load_collection3()["test"]["tokens"]]
    rng = np.random.default_rng(0)
    texts = [sentences[i] for i in rng.choice(len(sentences), args.requests, replace=False)]

    async with httpx.AsyncClient(timeout=60) as client:
        await run_level(client, args.url, texts[:20], 1)  # warm-up
        levels = [await run_level(client, args.url, texts, c) for c in args.concurrency]

    results = {"client_hardware": hardware_info(), "note": args.note, "levels": levels}
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'conc':>4} {'rps':>7} {'median ms':>10} {'p95 ms':>8} {'errors':>6}")
    for r in levels:
        print(
            f"{r['concurrency']:>4} {r['rps']:>7.1f} {r['median_ms']:>10.1f} "
            f"{r['p95_ms']:>8.1f} {r['errors']:>6}"
        )


if __name__ == "__main__":
    asyncio.run(main())
