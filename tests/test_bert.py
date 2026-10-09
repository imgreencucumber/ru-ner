import numpy as np

from ru_ner.bert import IGNORE, align_labels, compute_metrics, word_predictions
from ru_ner.data import LABEL2ID

B_PER, I_PER, OUT = LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"]

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


def test_compute_metrics_ignores_masked_positions():
    labels = np.array([[IGNORE, B_PER, IGNORE, OUT, IGNORE]])
    logits = np.zeros((1, 5, 7))
    logits[0, :, OUT] = 1.0  # predict O everywhere...
    logits[0, 1, B_PER] = 2.0  # ...except the labelled first subword
    logits[0, 2, B_PER] = 2.0  # this position is masked and must not create an extra entity

    assert compute_metrics((logits, labels))["f1"] == 1.0
