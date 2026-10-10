import json
from collections import Counter
from pathlib import Path

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

PREDICTIONS_DIR = Path("results/predictions")

ERROR_TYPES = [
    "correct",
    "wrong_type",
    "wrong_boundary",
    "wrong_boundary_and_type",
    "missed",
    "spurious",
]


def tags_to_spans(tags):
    """BIO tags -> [(start, end, type)], end exclusive. Uses the same chunking rules as seqeval."""
    return [(s, e + 1, t) for t, s, e in get_entities(list(tags))]


def spans_to_tags(spans, n):
    tags = ["O"] * n
    for s, e, t in spans:
        tags[s] = "B-" + t
        for i in range(s + 1, e):
            tags[i] = "I-" + t
    return tags


def save_predictions(model, corpus, predictions):
    """Store predicted entities as one JSON line of spans per sentence.

    Spans are stored instead of tags because it's ~10x smaller. A stray I- tag becomes B- on
    the way back, which doesn't change seqeval scores since seqeval treats it the same way.
    """
    path = PREDICTIONS_DIR / model / f"{corpus}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tags in predictions:
            f.write(json.dumps(tags_to_spans(tags)) + "\n")


def load_predictions(model, corpus, lengths):
    path = PREDICTIONS_DIR / model / f"{corpus}.jsonl"
    with path.open(encoding="utf-8") as f:
        spans = [json.loads(line) for line in f]
    if len(spans) != len(lengths):
        raise ValueError(f"{path}: {len(spans)} sentences, expected {len(lengths)}")
    return [spans_to_tags(s, n) for s, n in zip(spans, lengths, strict=True)]


EDGE_PUNCT = {'"', "«", "»", "(", ")", "'"}


def strip_edge_punct(spans, tokens):
    """Drop quotes and brackets at the edges of entity spans: '" Данас "' -> 'Данас'.

    Collection3 annotators sometimes include surrounding quotes into an entity and sometimes
    don't, this makes a comparison insensitive to that. Spans of only such tokens disappear.
    """
    out = set()
    for s, e, t in spans:
        while s < e and tokens[s] in EDGE_PUNCT:
            s += 1
        while e > s and tokens[e - 1] in EDGE_PUNCT:
            e -= 1
        if s < e:
            out.add((s, e, t))
    return out


def sentence_counts(y_true, y_pred, sentences=None):
    """Per-sentence (true positives, predicted entities, gold entities), shape (n_sentences, 3).

    If sentences are given, quotes and brackets at entity edges are ignored on both sides.
    """
    rows = []
    for k, (t, p) in enumerate(zip(y_true, y_pred, strict=True)):
        gold, pred = set(tags_to_spans(t)), set(tags_to_spans(p))
        if sentences is not None:
            gold, pred = strip_edge_punct(gold, sentences[k]), strip_edge_punct(pred, sentences[k])
        rows.append((len(gold & pred), len(pred), len(gold)))
    return np.array(rows, dtype=np.int64)


def f1_from_sums(sums):
    """F1 = 2TP / (predicted + gold), the same as 2PR / (P + R). Works on (..., 3) arrays."""
    sums = np.asarray(sums)
    tp, n_pred, n_gold = sums[..., 0], sums[..., 1], sums[..., 2]
    denom = n_pred + n_gold
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def _bootstrap_sums(count_arrays, n_resamples, seed, chunk=100):
    """Resample sentences with replacement; the same resamples are applied to every array.

    Instead of materialising resampled indices, each resample is a vector of multinomial weights
    (how many times each sentence was drawn), so a chunk of resamples is a single matrix product.
    """
    rng = np.random.default_rng(seed)
    n = len(count_arrays[0])
    sums = [[] for _ in count_arrays]
    for start in range(0, n_resamples, chunk):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=min(chunk, n_resamples - start))
        for out, counts in zip(sums, count_arrays, strict=True):
            out.append(weights @ counts)
    return [np.vstack(s) for s in sums]


def bootstrap_ci(counts, n_resamples=1000, seed=0, alpha=0.05):
    (sums,) = _bootstrap_sums([counts], n_resamples, seed)
    f1 = f1_from_sums(sums)
    return float(np.quantile(f1, alpha / 2)), float(np.quantile(f1, 1 - alpha / 2))


def paired_bootstrap(counts_a, counts_b, n_resamples=1000, seed=0, alpha=0.05):
    """F1(a) - F1(b) with a confidence interval, both models scored on the same resamples."""
    sums_a, sums_b = _bootstrap_sums([counts_a, counts_b], n_resamples, seed)
    diff = f1_from_sums(sums_a) - f1_from_sums(sums_b)
    point = float(f1_from_sums(counts_a.sum(0)) - f1_from_sums(counts_b.sum(0)))
    return {
        "diff": point,
        "ci": (float(np.quantile(diff, alpha / 2)), float(np.quantile(diff, 1 - alpha / 2))),
        # share of resamples where model a is not better: a rough one-sided p-value
        "p_a_not_better": float((diff <= 0).mean()),
    }


def entity_texts(sentences, tags_list):
    """Surface forms of all entities, e.g. {"Москве", "Георгий Чижов"}."""
    return {
        " ".join(tokens[s:e])
        for tokens, tags in zip(sentences, tags_list, strict=True)
        for s, e, _ in tags_to_spans(tags)
    }


def seen_unseen_report(sentences, y_true, y_pred, seen_texts):
    """Precision / recall / F1 separately for entities whose text was or wasn't seen in train.

    An entity is "seen" if exactly the same string was an entity in train (of any type).
    Predicted entities are split by the same rule, so a correct prediction always lands in the
    same group as its gold entity.
    """
    report = {}
    for group in ("seen", "unseen"):
        tp = n_pred = n_gold = 0
        for tokens, t, p in zip(sentences, y_true, y_pred, strict=True):

            def in_group(span, tokens=tokens, group=group):
                is_seen = " ".join(tokens[span[0] : span[1]]) in seen_texts
                return is_seen == (group == "seen")

            gold = {s for s in tags_to_spans(t) if in_group(s)}
            pred = {s for s in tags_to_spans(p) if in_group(s)}
            tp += len(gold & pred)
            n_pred += len(pred)
            n_gold += len(gold)
        report[group] = {
            "precision": tp / n_pred if n_pred else 0.0,
            "recall": tp / n_gold if n_gold else 0.0,
            "f1": float(f1_from_sums([tp, n_pred, n_gold])),
            "support": n_gold,
        }
    return report


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def classify_errors(y_true, y_pred):
    """Sort every gold entity into correct / wrong_type / wrong_boundary / wrong_boundary_and_type
    / missed, and every predicted entity that overlaps no gold entity into spurious.

    Returns counts and a list of error records for later inspection.
    """
    counts = Counter({e: 0 for e in ERROR_TYPES})
    errors = []
    for i, (t, p) in enumerate(zip(y_true, y_pred, strict=True)):
        gold, pred = tags_to_spans(t), tags_to_spans(p)
        pred_set = set(pred)
        for g in gold:
            overlapping = [q for q in pred if _overlap(g, q)]
            if g in pred_set:
                kind = "correct"
            elif any(q[:2] == g[:2] for q in overlapping):
                kind = "wrong_type"
            elif any(q[2] == g[2] for q in overlapping):
                kind = "wrong_boundary"
            elif overlapping:
                kind = "wrong_boundary_and_type"
            else:
                kind = "missed"
            counts[kind] += 1
            if kind != "correct":
                errors.append({"sent": i, "kind": kind, "gold": g, "pred": overlapping})
        for q in pred:
            if not any(_overlap(q, g) for g in gold):
                counts["spurious"] += 1
                errors.append({"sent": i, "kind": "spurious", "gold": None, "pred": [q]})
    return dict(counts), errors
