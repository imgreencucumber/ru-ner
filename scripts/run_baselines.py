"""Evaluate Natasha and CRF baselines on Collection3 test and WikiNEuRal-ru test.

Usage: uv run python scripts/run_baselines.py
"""

import itertools
import json
import pickle
from pathlib import Path

from ru_ner.crf import predict_crf, train_crf
from ru_ner.data import load_collection3, load_wikineural_ru, tag_names
from ru_ner.metrics import evaluate_on, ner_report, results_table
from ru_ner.natasha_baseline import NatashaNER

RESULTS_PATH = Path("results/baselines.json")
MODEL_PATH = Path("models/crf.pkl")

CRF_GRID = {"c1": [0.05, 0.2], "c2": [0.01, 0.1]}


def tune_crf(train, validation):
    train_x, train_y = list(train["tokens"]), tag_names(train)
    val_x, val_y = list(validation["tokens"]), tag_names(validation)
    best_f1, best = -1.0, None
    for c1, c2 in itertools.product(CRF_GRID["c1"], CRF_GRID["c2"]):
        crf = train_crf(train_x, train_y, c1=c1, c2=c2)
        f1 = ner_report(val_y, predict_crf(crf, val_x))["overall"]["f1"]
        print(f"CRF c1={c1} c2={c2}: validation F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1, best = f1, (crf, {"c1": c1, "c2": c2})
    return best


def main():
    c3 = load_collection3()
    test_sets = {"c3_test": c3["test"], "wikineural_test": load_wikineural_ru("test")}
    results = {}

    natasha = NatashaNER()
    results["natasha"] = evaluate_on(natasha.predict, test_sets)

    crf, params = tune_crf(c3["train"], c3["validation"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(crf))
    results["crf"] = evaluate_on(lambda s: predict_crf(crf, s), test_sets)
    results["crf"]["params"] = params

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(results_table(results).to_string())


if __name__ == "__main__":
    main()
