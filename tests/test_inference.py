import numpy as np

from ru_ner.inference import (
    ALLOWED,
    ALLOWED_START,
    TokenClassifier,
    bio_spans,
    split_sentences,
    tokenize,
    viterbi_decode,
    word_log_probs,
    word_predictions,
)
from ru_ner.labels import LABEL2ID, LABELS

B_PER, I_PER, OUT = LABEL2ID["B-PER"], LABEL2ID["I-PER"], LABEL2ID["O"]
B_ORG, I_ORG, B_LOC = LABEL2ID["B-ORG"], LABEL2ID["I-ORG"], LABEL2ID["B-LOC"]

# [CLS] Сил ##уан ##ов посетил [SEP] for the words ["Силуанов", "посетил"]
WORD_IDS = [None, 0, 0, 0, 1, None]


def log_probs_from(rows):
    """Turn {label_id: prob} dicts into a (n_words, n_labels) log-prob matrix."""
    probs = np.full((len(rows), len(LABELS)), 1e-6)
    for i, row in enumerate(rows):
        for label, p in row.items():
            probs[i, label] = p
    return np.log(probs / probs.sum(1, keepdims=True))


class FakeNER(TokenClassifier):
    """One token per word, no special tokens. Words from WORD_LABELS get that label,
    everything else gets O. Lets us test batching and offsets without a real model."""

    WORD_LABELS = {"Иван": B_PER, "Петров": I_PER, "Москве": B_LOC, "Госдума": B_ORG}

    def __init__(self, batch_size=32, decoding="viterbi"):
        self.batch_size = batch_size
        self.decoding = decoding
        self.vocab = {w: i + 1 for i, w in enumerate(self.WORD_LABELS)}

    def _encode(self, batch):
        width = max(len(s) for s in batch)
        ids = np.zeros((len(batch), width), dtype=np.int64)
        for i, s in enumerate(batch):
            ids[i, : len(s)] = [self.vocab.get(w, len(self.vocab) + 1) for w in s]
        word_ids = [list(range(len(s))) + [None] * (width - len(s)) for s in batch]
        return {"input_ids": ids}, word_ids

    def _logits(self, inputs):
        ids = inputs["input_ids"]
        logits = np.zeros((*ids.shape, len(LABELS)))
        logits[..., OUT] = 1.0
        for word, label in self.WORD_LABELS.items():
            logits[ids == self.vocab[word], label] = 5.0
        return logits


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


def test_bio_spans():
    tags = ["B-PER", "I-PER", "O", "I-ORG", "B-LOC", "B-LOC", "I-PER"]

    expected = [(0, 2, "PER"), (3, 4, "ORG"), (4, 5, "LOC"), (5, 6, "LOC"), (6, 7, "PER")]
    assert bio_spans(tags) == expected


def test_tokenize_splits_punctuation_and_keeps_offsets():
    text = "Пресс-служба «Газпрома», 2024 г."
    tokens = tokenize(text)

    assert [w for w, _, _ in tokens] == [
        "Пресс", "-", "служба", "«", "Газпрома", "»", ",", "2024", "г", ".",
    ]  # fmt: skip
    assert all(text[s:e] == w for w, s, e in tokens)


def test_split_sentences_on_end_punctuation_and_max_length():
    tokens = tokenize("Один два. Три! " + "слово " * 5)
    pieces = split_sentences(tokens, max_words=3)

    assert [[w for w, _, _ in p] for p in pieces] == [
        ["Один", "два", "."], ["Три", "!"], ["слово"] * 3, ["слово"] * 2,
    ]  # fmt: skip


def test_predict_keeps_original_order_after_sorting_by_length():
    sentences = [["Иван", "a", "b", "c"], ["x"], ["Иван"], ["y", "z"]]
    out = FakeNER(batch_size=2).predict(sentences)

    assert out == [["B-PER", "O", "O", "O"], ["O"], ["B-PER"], ["O", "O"]]


def test_extract_returns_character_offsets_into_original_text():
    texts = ["Вчера  Иван Петров был в Москве. Госдума молчит.", "", "Ничего нет"]
    results = FakeNER().extract_batch(texts)

    assert results[0] == [
        {"text": "Иван Петров", "type": "PER", "start": 7, "end": 18},
        {"text": "Москве", "type": "LOC", "start": 25, "end": 31},
        {"text": "Госдума", "type": "ORG", "start": 33, "end": 40},
    ]
    assert results[1] == [] and results[2] == []
