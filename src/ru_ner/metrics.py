import time

import pandas as pd
from seqeval.metrics import classification_report

from ru_ner.data import tag_names
from ru_ner.evaluation import save_predictions

ENTITY_TYPES = ["PER", "LOC", "ORG"]


def ner_report(y_true, y_pred):
    """Entity-level precision / recall / F1: overall (micro average) and per entity type.

    An entity counts as correct only if both its boundaries and its type match exactly.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    def pick(row):
        return {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        }

    result = {"overall": pick(report["micro avg"])}
    for etype in ENTITY_TYPES:
        if etype in report:
            result[etype] = pick(report[etype])
    return result


def evaluate_on(predict, test_sets, model_name=None):
    """Run `predict(list of token lists) -> list of tag lists` on each test set, with timing.

    If model_name is given, predictions are saved to results/predictions/<model_name>/.
    """
    out = {}
    for name, ds in test_sets.items():
        sentences = list(ds["tokens"])
        start = time.perf_counter()
        pred = predict(sentences)
        elapsed = time.perf_counter() - start
        out[name] = ner_report(tag_names(ds), pred)
        out[name]["sentences_per_sec"] = len(sentences) / elapsed
        if model_name:
            save_predictions(model_name, name, pred)
    return out


def results_table(results):
    rows = []
    for model, by_corpus in results.items():
        for corpus, r in by_corpus.items():
            if "overall" not in r:
                continue
            row = {"model": model, "corpus": corpus}
            row.update({k: r["overall"][k] for k in ("precision", "recall", "f1")})
            row.update({etype: r[etype]["f1"] for etype in ENTITY_TYPES})
            row["sent/s"] = r["sentences_per_sec"]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "corpus"]).round(3)
