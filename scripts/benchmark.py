"""Speed of all local models on the same sentences and hardware.

Latency: one sentence per call, median and p95 over 300 random Collection3 test sentences after
a warm-up. Throughput: the whole Collection3 test set in one call (BERT models use batches of 32).
Run export_onnx.py and run_baselines.py first.

Usage: uv run python scripts/benchmark.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ru_ner.benchmark import hardware_info, measure_latency, measure_throughput
from ru_ner.bert import BertNER
from ru_ner.crf import predict_crf
from ru_ner.data import load_collection3
from ru_ner.inference import OnnxBertNER
from ru_ner.natasha_baseline import NatashaNER

MODEL_DIR = Path("models/rubert-ner")
RESULTS_PATH = Path("results/benchmark.json")
NUM_THREADS = 4
N_LATENCY = 300


def models():
    """Yield (name, predict) one at a time, so only one model is loaded at any moment."""
    crf = pickle.loads(Path("models/crf.pkl").read_bytes())
    yield "crf", lambda s: predict_crf(crf, s)
    yield "natasha", NatashaNER().predict
    yield "rubert_pytorch_cpu", BertNER(MODEL_DIR, device="cpu").predict
    if torch.cuda.is_available():
        yield "rubert_pytorch_gpu", BertNER(MODEL_DIR, device="cuda").predict
    for name, filename in [("rubert_onnx", "model.onnx"), ("rubert_onnx_int8", "model_int8.onnx")]:
        onnx_path = MODEL_DIR / "onnx" / filename
        yield name, OnnxBertNER(MODEL_DIR, onnx_path, num_threads=NUM_THREADS).predict


def main():
    torch.set_num_threads(NUM_THREADS)
    sentences = list(load_collection3()["test"]["tokens"])
    rng = np.random.default_rng(0)
    latency_sentences = [sentences[i] for i in rng.choice(len(sentences), N_LATENCY, replace=False)]

    results = {"hardware": hardware_info(), "num_threads": NUM_THREADS, "models": {}}
    for name, predict in models():
        print(f"benchmarking {name}", flush=True)
        results["models"][name] = {
            **measure_latency(predict, latency_sentences),
            "sentences_per_sec": measure_throughput(predict, sentences),
        }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(results["hardware"])
    print(pd.DataFrame(results["models"]).T.round(1).to_string())


if __name__ == "__main__":
    main()
