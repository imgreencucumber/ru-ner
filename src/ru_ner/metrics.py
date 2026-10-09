from seqeval.metrics import classification_report

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
