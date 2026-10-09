"""Extended metrics from saved predictions in results/predictions/:
F1 with a bootstrap 95% CI, F1 without sentences duplicated from train, seen vs unseen entities,
error types and paired bootstrap comparisons against ruBERT.

Usage: uv run python scripts/evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ru_ner.data import load_collection3, load_wikineural_ru, tag_names
from ru_ner.evaluation import (
    ERROR_TYPES,
    PREDICTIONS_DIR,
    bootstrap_ci,
    classify_errors,
    entity_texts,
    f1_from_sums,
    load_predictions,
    paired_bootstrap,
    seen_unseen_report,
    sentence_counts,
)

RESULTS_PATH = Path("results/evaluation.json")
REFERENCE_MODEL = "rubert"


def main():
    c3 = load_collection3()
    corpora = {"c3_test": c3["test"], "wikineural_test": load_wikineural_ru("test")}

    train_sentences = list(c3["train"]["tokens"])
    seen_texts = entity_texts(train_sentences, tag_names(c3["train"]))
    train_keys = {" ".join(s) for s in train_sentences}

    models = sorted(p.name for p in PREDICTIONS_DIR.iterdir() if p.is_dir())
    results, counts = {}, {}
    for corpus, ds in corpora.items():
        sentences = list(ds["tokens"])
        y_true = tag_names(ds)
        not_in_train = np.array([" ".join(s) not in train_keys for s in sentences])
        for model in models:
            if not (PREDICTIONS_DIR / model / f"{corpus}.jsonl").exists():
                continue
            y_pred = load_predictions(model, corpus, [len(s) for s in sentences])
            c = sentence_counts(y_true, y_pred)
            counts[model, corpus] = c
            errors, _ = classify_errors(y_true, y_pred)
            results.setdefault(model, {})[corpus] = {
                "f1": float(f1_from_sums(c.sum(0))),
                "f1_ci95": bootstrap_ci(c),
                "f1_without_train_duplicates": float(f1_from_sums(c[not_in_train].sum(0))),
                "seen_unseen": seen_unseen_report(sentences, y_true, y_pred, seen_texts),
                "errors": errors,
            }

    comparisons = {}
    for (model, corpus), c in counts.items():
        ref = counts.get((REFERENCE_MODEL, corpus))
        if model != REFERENCE_MODEL and ref is not None:
            name = f"{REFERENCE_MODEL}_vs_{model}"
            comparisons.setdefault(name, {})[corpus] = paired_bootstrap(ref, c)

    RESULTS_PATH.write_text(
        json.dumps({"models": results, "comparisons": comparisons}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_tables(results, comparisons)


def print_tables(results, comparisons):
    rows, error_rows = [], []
    for model, by_corpus in results.items():
        for corpus, r in by_corpus.items():
            lo, hi = r["f1_ci95"]
            su = r["seen_unseen"]
            rows.append(
                {
                    "model": model,
                    "corpus": corpus,
                    "f1": r["f1"],
                    "ci95": f"{lo:.3f}-{hi:.3f}",
                    "f1_no_dup": r["f1_without_train_duplicates"],
                    "f1_seen": su["seen"]["f1"],
                    "f1_unseen": su["unseen"]["f1"],
                    "recall_unseen": su["unseen"]["recall"],
                }
            )
            error_rows.append({"model": model, "corpus": corpus, **r["errors"]})

    pd.set_option("display.width", 200)
    metrics = pd.DataFrame(rows).set_index(["model", "corpus"]).round(3)
    errors = pd.DataFrame(error_rows).set_index(["model", "corpus"])[ERROR_TYPES]
    print(metrics.to_string(), end="\n\n")
    print(errors.to_string(), end="\n\n")

    comp_rows = [
        {
            "comparison": name,
            "corpus": corpus,
            "f1_diff": r["diff"],
            "ci95": f"{r['ci'][0]:+.3f}..{r['ci'][1]:+.3f}",
            "p": r["p_a_not_better"],
        }
        for name, by_corpus in comparisons.items()
        for corpus, r in by_corpus.items()
    ]
    print(pd.DataFrame(comp_rows).set_index(["comparison", "corpus"]).round(4).to_string())


if __name__ == "__main__":
    main()
