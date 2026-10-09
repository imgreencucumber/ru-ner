import random

import numpy as np
import pytest

from ru_ner.evaluation import (
    bootstrap_ci,
    classify_errors,
    f1_from_sums,
    paired_bootstrap,
    seen_unseen_report,
    sentence_counts,
    spans_to_tags,
    tags_to_spans,
)
from ru_ner.metrics import ner_report


def random_tags(rng, n):
    return [rng.choice(["O", "O", "O", "B-PER", "I-PER", "B-LOC", "I-ORG"]) for _ in range(n)]


def test_spans_round_trip():
    tags = ["B-PER", "I-PER", "O", "B-LOC", "B-LOC", "I-LOC"]
    assert spans_to_tags(tags_to_spans(tags), len(tags)) == tags


def test_stray_i_tag_becomes_b_tag():
    assert spans_to_tags(tags_to_spans(["O", "I-ORG", "I-ORG"]), 3) == ["O", "B-ORG", "I-ORG"]


def test_f1_from_counts_matches_seqeval():
    rng = random.Random(0)
    y_true = [random_tags(rng, 15) for _ in range(200)]
    y_pred = [random_tags(rng, 15) for _ in range(200)]
    counts = sentence_counts(y_true, y_pred)

    expected = ner_report(y_true, y_pred)["overall"]["f1"]
    assert f1_from_sums(counts.sum(0)) == pytest.approx(expected)


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    rng = random.Random(1)
    y_true = [random_tags(rng, 10) for _ in range(300)]
    y_pred = [t if rng.random() < 0.7 else random_tags(rng, 10) for t in y_true]
    counts = sentence_counts(y_true, y_pred)
    point = f1_from_sums(counts.sum(0))

    lo, hi = bootstrap_ci(counts, seed=0)
    assert lo < point < hi
    assert bootstrap_ci(counts, seed=0) == (lo, hi)


def test_paired_bootstrap_identical_models():
    counts = np.array([[1, 1, 2], [0, 1, 1], [2, 2, 2]])
    r = paired_bootstrap(counts, counts)

    assert r["diff"] == 0.0 and r["ci"] == (0.0, 0.0) and r["p_a_not_better"] == 1.0


def test_classify_errors():
    y_true = [["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O", "B-PER"]]
    y_pred = [["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "B-ORG", "O", "O"]]
    counts, errors = classify_errors(y_true, y_pred)

    assert counts["correct"] == 1  # PER 0-2
    assert counts["wrong_boundary"] == 1  # ORG 3-5 predicted as ORG 3-4
    assert counts["wrong_type"] == 1  # LOC 6 predicted as ORG 6
    assert counts["missed"] == 1  # PER 8
    assert counts["spurious"] == 1  # LOC 5
    assert len(errors) == 4


def test_seen_unseen_split():
    sentences = [["Иван", "Петров", "в", "Москве"]]
    y_true = [["B-PER", "I-PER", "O", "B-LOC"]]
    y_pred = [["B-PER", "I-PER", "O", "O"]]
    r = seen_unseen_report(sentences, y_true, y_pred, seen_texts={"Иван Петров"})

    assert r["seen"]["recall"] == 1.0 and r["seen"]["support"] == 1
    assert r["unseen"]["recall"] == 0.0 and r["unseen"]["support"] == 1
