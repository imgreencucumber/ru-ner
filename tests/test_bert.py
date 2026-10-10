import numpy as np

from ru_ner.bert import IGNORE, align_labels, compute_metrics
from ru_ner.inference import log_softmax
from ru_ner.labels import LABEL2ID

B_PER, OUT = LABEL2ID["B-PER"], LABEL2ID["O"]

# [CLS] Сил ##уан ##ов посетил [SEP] for the words ["Силуанов", "посетил"]
WORD_IDS = [None, 0, 0, 0, 1, None]


def test_align_labels_first_subword_only():
    expected = [IGNORE, B_PER, IGNORE, IGNORE, OUT, IGNORE]
    assert align_labels(WORD_IDS, [B_PER, OUT]) == expected


def test_align_labels_repeated_word_after_special_token():
    # the same word id right after None is still a new occurrence of that word
    assert align_labels([None, 0, None], [B_PER]) == [IGNORE, B_PER, IGNORE]


def test_compute_metrics_ignores_masked_positions():
    labels = np.array([[IGNORE, B_PER, IGNORE, OUT, IGNORE]])
    logits = np.zeros((1, 5, 7))
    logits[0, :, OUT] = 1.0  # predict O everywhere...
    logits[0, 1, B_PER] = 2.0  # ...except the labelled first subword
    logits[0, 2, B_PER] = 2.0  # this position is masked and must not create an extra entity

    assert compute_metrics((logits, labels))["f1"] == 1.0


def test_log_softmax_matches_torch():
    import torch

    x = np.random.default_rng(0).normal(size=(2, 5, 7)) * 10
    expected = torch.log_softmax(torch.from_numpy(x), dim=-1).numpy()
    assert np.allclose(log_softmax(x), expected)
