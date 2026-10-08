from ru_ner.data import LABEL2ID, LABELS, _wikineural_to_our_labels, parse_conll


def test_parse_conll_splits_sentences_and_skips_docstart():
    lines = [
        "<DOCSTART>\n",
        "\n",
        "Д\tB-PER\n",
        ".\tI-PER\n",
        "Медведев\tI-PER\n",
        "\n",
        "в\tO\n",
        "Москве\tB-LOC\n",
    ]
    sentences = parse_conll(lines)

    assert sentences == [
        {"tokens": ["Д", ".", "Медведев"], "ner_tags": ["B-PER", "I-PER", "I-PER"]},
        {"tokens": ["в", "Москве"], "ner_tags": ["O", "B-LOC"]},
    ]


def test_wikineural_labels_are_remapped_and_misc_dropped():
    # WikiNEuRal ids: 3 = B-ORG, 5 = B-LOC, 7 = B-MISC
    example = {"ner_tags": [0, 3, 5, 7, 8]}
    tags = [LABELS[i] for i in _wikineural_to_our_labels(example)["ner_tags"]]

    assert tags == ["O", "B-ORG", "B-LOC", "O", "O"]
    assert LABEL2ID["B-ORG"] != 3
