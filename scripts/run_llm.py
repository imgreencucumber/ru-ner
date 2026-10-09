"""Run an LLM over the Collection3 test set, then align its answers and save predictions.

Raw responses are appended to results/llm_raw/<name>.jsonl as they arrive, so an interrupted run
resumes where it stopped and already paid requests are never sent again. With --limit only the
first N sentences are processed and nothing is saved to results/predictions (a quick check).

With --pilot N the model runs on N random sentences from the validation split instead of the test
set, which is used to decide whether a model is worth a full test run without looking at test.
Its summary goes to results/llm/pilot/.

Usage: uv run python scripts/run_llm.py --model gigachat3_lightning --mode few [--workers 2]
       uv run python scripts/run_llm.py --model gigachat3_ultra --mode few --pilot 100
"""

import argparse
import json
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ru_ner.data import load_collection3, tag_names
from ru_ner.evaluation import save_predictions
from ru_ner.llm import GigaChatClient, YandexGPTClient
from ru_ner.llm_ner import (
    build_system_prompt,
    build_user_prompt,
    pick_few_shot_examples,
    response_to_tags,
)
from ru_ner.metrics import ENTITY_TYPES, ner_report

MODELS = {
    "gigachat3_lightning": lambda: GigaChatClient("GigaChat-3-Lightning"),
    "gigachat3_pro": lambda: GigaChatClient("GigaChat-3-Pro"),
    "gigachat3_ultra": lambda: GigaChatClient("GigaChat-3-Ultra"),
    "yandexgpt_lite": lambda: YandexGPTClient("yandexgpt-lite"),
    "yandexgpt_pro": lambda: YandexGPTClient("yandexgpt"),
}
RAW_DIR = Path("results/llm_raw")
SUMMARY_DIR = Path("results/llm")
CORPUS = "c3_test"
PILOT_SEED = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--mode", choices=["zero", "few"], required=True)
    p.add_argument("--limit", type=int, help="only the first N sentences, nothing is saved")
    p.add_argument("--pilot", type=int, help="N random validation sentences instead of test")
    p.add_argument("--workers", type=int, default=1, help="parallel requests")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def load_raw(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return {r["i"]: r for r in records}


def summarize(name, args, sentences, y_true, raw, n, summary_dir, save_preds):
    preds, alignment, answered = [], Counter(), []
    for i in range(n):
        tags, stats = response_to_tags(sentences[i], raw[i]["response"])
        preds.append(tags)
        alignment.update(stats)
        if not stats["no_json"] and not stats["parse_error"]:
            answered.append(i)
    report = ner_report(y_true[:n], preds)
    # quality of the model itself, without sentences it refused or answered with broken JSON
    report_answered = ner_report([y_true[i] for i in answered], [preds[i] for i in answered])
    latency_ms = np.array([raw[i]["latency_s"] for i in range(n)]) * 1000
    summary = {
        "model": args.model,
        "mode": args.mode,
        "temperature": args.temperature,
        "n_sentences": n,
        **{k: report[k] for k in ["overall", *ENTITY_TYPES] if k in report},
        "answered_only": {"n_sentences": len(answered), "overall": report_answered["overall"]},
        "alignment": dict(alignment),
        "tokens": {
            "input": sum(raw[i]["input_tokens"] for i in range(n)),
            "output": sum(raw[i]["output_tokens"] for i in range(n)),
            "cached_input": sum(raw[i].get("cached_input_tokens", 0) for i in range(n)),
        },
        "latency": {
            "median_ms": float(np.median(latency_ms)),
            "p95_ms": float(np.percentile(latency_ms, 95)),
        },
    }
    if n == len(sentences):
        if save_preds:
            save_predictions(name, CORPUS, preds)
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / f"{name}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    # GigaChat runs without TLS verification (see GigaChatClient), don't repeat the warning
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    name = f"{args.model}_{args.mode}"

    c3 = load_collection3()
    if args.pilot:
        name += f"_pilot{args.pilot}"
        val = c3["validation"]
        idx = np.random.default_rng(PILOT_SEED).choice(len(val), args.pilot, replace=False)
        val_tokens, val_tags = list(val["tokens"]), tag_names(val)
        sentences = [list(val_tokens[i]) for i in idx]
        y_true = [val_tags[i] for i in idx]
        summary_dir, save_preds = SUMMARY_DIR / "pilot", False
    else:
        sentences, y_true = list(c3["test"]["tokens"]), tag_names(c3["test"])
        summary_dir, save_preds = SUMMARY_DIR, True
    examples = pick_few_shot_examples(c3["train"]) if args.mode == "few" else None
    system = build_system_prompt(examples)

    raw_path = RAW_DIR / f"{name}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_path)
    n = min(args.limit or len(sentences), len(sentences))
    todo = [i for i in range(n) if i not in raw]
    print(f"{name}: {n - len(todo)} cached, {len(todo)} to request")

    client = MODELS[args.model]()

    def call(i):
        r = client.complete(system, build_user_prompt(sentences[i]), temperature=args.temperature)
        return {
            "i": i,
            "response": r.text,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cached_input_tokens": r.cached_input_tokens,
            "latency_s": round(r.latency_s, 3),
        }

    errors = Counter()
    with raw_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(call, i) for i in todo]
        for k, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
            except Exception as e:  # a failed request is retried on the next run
                errors[f"{type(e).__name__}: {str(e)[:120]}"] += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            raw[record["i"]] = record
            if k % 100 == 0:
                print(f"  {k}/{len(todo)}", flush=True)

    for message, count in errors.items():
        print(f"  {count} failed: {message}")
    missing = [i for i in range(n) if i not in raw]
    if missing:
        print(f"{len(missing)} sentences have no response yet, run the same command again")
        return
    summarize(name, args, sentences, y_true, raw, n, summary_dir, save_preds)


if __name__ == "__main__":
    main()
