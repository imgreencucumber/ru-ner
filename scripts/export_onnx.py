"""Export the fine-tuned ruBERT to ONNX, optimize the graph for CPU, quantize it to int8
and evaluate both fp32 and int8 versions on CPU.

Usage: uv run python scripts/export_onnx.py
"""

import json
from pathlib import Path

from ru_ner.bert import OnnxBertNER
from ru_ner.data import load_collection3, load_wikineural_ru
from ru_ner.export import (
    export_onnx,
    max_logit_diff,
    model_size_mb,
    optimize_for_cpu,
    quantize_int8,
)
from ru_ner.metrics import evaluate_on, results_table

MODEL_DIR = Path("models/rubert-ner")
RAW_PATH = MODEL_DIR / "onnx" / "raw.onnx"
FP32_PATH = MODEL_DIR / "onnx" / "model.onnx"
INT8_PATH = MODEL_DIR / "onnx" / "model_int8.onnx"
RESULTS_PATH = Path("results/onnx.json")
NUM_THREADS = 4


def main():
    export_onnx(MODEL_DIR, RAW_PATH)
    fused = optimize_for_cpu(RAW_PATH, FP32_PATH)
    print("fused operators:", {k: v for k, v in fused.items() if v})
    quantize_int8(FP32_PATH, INT8_PATH)

    c3 = load_collection3()
    test_sets = {"c3_test": c3["test"], "wikineural_test": load_wikineural_ru("test")}
    # a few real test sentences of different length, padded into one batch
    check_sentences = list(c3["test"]["tokens"][:16])

    results = {}
    for name, path in [("rubert_onnx", FP32_PATH), ("rubert_onnx_int8", INT8_PATH)]:
        ner = OnnxBertNER(MODEL_DIR, path, num_threads=NUM_THREADS)
        results[name] = evaluate_on(ner.predict, test_sets, model_name=name)
        results[name]["info"] = {
            "size_mb": round(model_size_mb(path), 1),
            "max_logit_diff_vs_pytorch": max_logit_diff(MODEL_DIR, path, check_sentences),
            "num_threads": NUM_THREADS,
        }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(results_table(results).to_string())
    for name, r in results.items():
        print(name, r["info"])


if __name__ == "__main__":
    main()
