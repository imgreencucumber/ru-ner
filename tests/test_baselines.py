from ru_ner.crf import sent2features, word_shape
from ru_ner.metrics import ner_report
from ru_ner.natasha_baseline import detokenize, spans_to_bio


def test_detokenize_offsets_point_to_tokens():
    tokens = ["Георгий", "Чижов", "из", "Москвы"]
    text, offsets = detokenize(tokens)

    assert [text[s:e] for s, e in offsets] == tokens


def test_spans_to_bio():
    tokens = ["Георгий", "Чижов", "посетил", "Новый", "Уренгой"]
    _, offsets = detokenize(tokens)
    # "Георгий Чижов" = chars 0..13, "Новый Уренгой" = chars 22..35
    spans = [(0, 13, "PER"), (22, 35, "LOC")]

    assert spans_to_bio(offsets, spans) == ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]


def test_spans_to_bio_partial_overlap_marks_whole_token():
    # "Нью" only, while our token is "Нью-Йорк"
    _, offsets = detokenize(["в", "Нью-Йорк"])

    assert spans_to_bio(offsets, [(2, 5, "LOC")]) == ["O", "B-LOC"]


def test_word_shape():
    assert word_shape("Медведев") == "Xx"
    assert word_shape("РФ") == "X"
    assert word_shape("2003г") == "dx"
    assert word_shape("Ёлки") == "Xx"


def test_sent2features_context_and_boundaries():
    features = sent2features(["В", "Москве"])

    assert features[0]["BOS"] and "EOS" not in features[0]
    assert features[1]["EOS"] and features[1]["-1:lower"] == "в"
    assert "+1:lower" not in features[1] and "1:lower" not in features[1]


def test_ner_report_requires_exact_boundaries():
    y_true = [["B-PER", "I-PER", "O", "B-LOC"]]
    # PER is found only partially, LOC is correct
    y_pred = [["B-PER", "O", "O", "B-LOC"]]
    r = ner_report(y_true, y_pred)

    assert r["PER"]["f1"] == 0.0
    assert r["LOC"]["f1"] == 1.0
    assert r["overall"]["precision"] == 0.5 and r["overall"]["recall"] == 0.5
