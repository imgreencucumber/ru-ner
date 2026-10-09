import numpy as np

from ru_ner.bert import (
    ALLOWED,
    ALLOWED_START,
    IGNORE,
    _TokenClassifier,
    align_labels,
    compute_metrics,
    log_softmax,
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


class _FakeClassifier(_TokenClassifier):
    """Predicts B-PER for every word of sentences whose first word is 'имя', so the test can see
    whether outputs come back in the original order after sorting by length."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.decoding = "argmax"
        self.tokenizer = _WhitespaceTokenizer()

    def _logits(self, enc):
        logits = np.zeros((*enc["input_ids"].shape, len(LABELS)))
        logits[..., OUT] = 1.0
        logits[enc["input_ids"] == 1, B_PER] = 2.0
        return logits


class _WhitespaceTokenizer:
    """One token per word, id 1 for the word 'имя' and 2 for anything else."""

    def __call__(self, batch, **kwargs):
        width = max(len(s) for s in batch)
        ids = np.zeros((len(batch), width), dtype=np.int64)
        for i, s in enumerate(batch):
            ids[i, : len(s)] = [1 if w == "имя" else 2 for w in s]
        return _Encoding({"input_ids": ids}, [len(s) for s in batch], width)


class _Encoding(dict):
    def __init__(self, data, lengths, width):
        super().__init__(data)
        self.lengths, self.width = lengths, width

    def word_ids(self, i):
        n = self.lengths[i]
        return list(range(n)) + [None] * (self.width - n)


def test_predict_keeps_original_order_after_sorting_by_length():
    sentences = [["имя", "a", "b", "c"], ["x"], ["имя"], ["y", "z"]]
    out = _FakeClassifier(batch_size=2).predict(sentences)

    assert [len(o) for o in out] == [4, 1, 1, 2]
    assert out[0][0] == "B-PER" and out[2] == ["B-PER"] and out[1] == ["O"]


def test_log_softmax_matches_torch():
    import torch

    x = np.random.default_rng(0).normal(size=(2, 5, 7)) * 10
    expected = torch.log_softmax(torch.from_numpy(x), dim=-1).numpy()
    assert np.allclose(log_softmax(x), expected)


def test_compute_metrics_ignores_masked_positions():
    labels = np.array([[IGNORE, B_PER, IGNORE, OUT, IGNORE]])
    logits = np.zeros((1, 5, 7))
    logits[0, :, OUT] = 1.0  # predict O everywhere...
    logits[0, 1, B_PER] = 2.0  # ...except the labelled first subword
    logits[0, 2, B_PER] = 2.0  # this position is masked and must not create an extra entity

    assert compute_metrics((logits, labels))["f1"] == 1.0
