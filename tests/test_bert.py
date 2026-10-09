import numpy as np

from ru_ner.bert import (
    ALLOWED,
    ALLOWED_START,
    IGNORE,
    align_labels,
    compute_metrics,
    viterbi_decode,
    word_log_probs,
    word_predictions,
)
from ru_ner.data import LABEL2ID, LABELS

B_PER, I_PER, OUT = LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"]
B_ORG, I_ORG = LABEL2ID["B-ORG"], LABEL2ID["I-ORG"]


def log_probs_from(rows):
    """Turn {label_id: prob} dicts into a (n_words, n_labels) log-prob matrix."""
    probs = np.full((len(rows), len(LABELS)), 1e-6)
    for i, row in enumerate(rows):
        for label, p in row.items():
            probs[i, label] = p
    return np.log(probs / probs.sum(1, keepdims=True))


# [CLS] Сил ##уан ##ов посетил [SEP] for the words ["Силуанов", "посетил"]
WORD_IDS = [None, 0, 0, 0, 1, None]


def test_align_labels_first_subword_only():
    expected = [IGNORE, B_PER, IGNORE, IGNORE, OUT, IGNORE]
    assert align_labels(WORD_IDS, [B_PER, OUT]) == expected


def test_align_labels_repeated_word_after_special_token():
    # the same word id right after None is still a new occurrence of that word
    assert align_labels([None, 0, None], [B_PER]) == [IGNORE, B_PER, IGNORE]


def test_word_predictions_takes_first_subword():
    # subword predictions disagree inside the word, only the first one matters
    pred = [OUT, B_PER, OUT, I_PER, OUT, OUT]

    assert word_predictions(WORD_IDS, pred, n_words=2) == ["B-PER", "O"]


def test_word_predictions_truncated_words_get_o():
    # the third word was cut off by truncation and has no subwords
    pred = [OUT, B_PER, I_PER, OUT]

    assert word_predictions([None, 0, 1, None], pred, n_words=3) == ["B-PER", "I-PER", "O"]


def test_bio_constraints():
    assert not ALLOWED[OUT, I_ORG] and not ALLOWED[B_PER, I_ORG]
    assert ALLOWED[B_ORG, I_ORG] and ALLOWED[I_ORG, I_ORG] and ALLOWED[OUT, B_ORG]
    assert not ALLOWED_START[I_PER] and ALLOWED_START[B_PER]


def test_viterbi_matches_argmax_when_argmax_is_valid():
    lp = log_probs_from([{B_PER: 0.9}, {I_PER: 0.8}, {OUT: 0.9}])
    assert viterbi_decode(lp) == [B_PER, I_PER, OUT]


def test_viterbi_repairs_stray_i_tag():
    # argmax would give O, I-ORG, O; B-ORG is the next best label for the middle word
    lp = log_probs_from([{OUT: 0.9}, {I_ORG: 0.5, B_ORG: 0.3, OUT: 0.2}, {OUT: 0.9}])
    assert [LABELS[i] for i in lp.argmax(1)] == ["O", "I-ORG", "O"]
    assert viterbi_decode(lp) == [OUT, B_ORG, OUT]


def test_viterbi_drops_stray_i_tag_when_o_is_more_likely():
    lp = log_probs_from([{OUT: 0.9}, {I_ORG: 0.45, OUT: 0.4, B_ORG: 0.15}, {OUT: 0.9}])
    assert viterbi_decode(lp) == [OUT, OUT, OUT]


def test_word_log_probs_forces_truncated_words_to_o():
    subword_lp = np.log(np.full((3, len(LABELS)), 1 / len(LABELS)))
    lp = word_log_probs([None, 0, None], subword_lp, n_words=2)

    assert viterbi_decode(lp)[1] == OUT
    assert lp[1, OUT] == 0.0 and np.isneginf(lp[1, B_PER])


def test_compute_metrics_ignores_masked_positions():
    labels = np.array([[IGNORE, B_PER, IGNORE, OUT, IGNORE]])
    logits = np.zeros((1, 5, 7))
    logits[0, :, OUT] = 1.0  # predict O everywhere...
    logits[0, 1, B_PER] = 2.0  # ...except the labelled first subword
    logits[0, 2, B_PER] = 2.0  # this position is masked and must not create an extra entity

    assert compute_metrics((logits, labels))["f1"] == 1.0
