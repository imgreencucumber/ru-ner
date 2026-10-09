from ru_ner.llm_ner import (
    align_entities,
    build_system_prompt,
    entities_json,
    parse_response,
    response_to_tags,
)


def test_parse_response_plain_and_fenced():
    plain = '{"entities": [{"text": "Москве", "type": "LOC"}]}'
    fenced = 'Вот ответ:\n```json\n{"entities": [{"text": "Москве", "type": "location"}]}\n```'

    assert parse_response(plain) == [("Москве", "LOC")]
    assert parse_response(fenced) == [("Москве", "LOC")]


def test_parse_response_drops_unknown_types_and_rejects_garbage():
    text = '{"entities": [{"text": "ЕГЭ", "type": "MISC"}, {"text": "РАН", "type": "ORG"}]}'

    assert parse_response(text) == [("РАН", "ORG")]
    assert parse_response("Сущностей нет.") is None
    assert parse_response('{"entities": "none"}') is None


def test_align_ignores_spacing_case_yo_and_quote_style():
    tokens = ["Пресс", "-", "служба", "партии", '"', "Единая", "Россия", '"', "едет", "в", "Орёл"]
    entities = [("пресс-служба", "ORG"), ("«Единая Россия»", "ORG"), ("Орел", "LOC")]
    spans, stats = align_entities(tokens, entities)

    assert spans == [(0, 3, "ORG"), (4, 8, "ORG"), (10, 11, "LOC")]
    assert stats["exact"] == 3


def test_align_does_not_match_part_of_a_word():
    # "Росси" is a cut-off "россияне": neither an exact match on token boundaries
    # nor another form of the same word
    spans, stats = align_entities(["россияне", "довольны"], [("Росси", "LOC")])

    assert spans == [] and stats["unmatched"] == 1


def test_align_other_word_form_is_fuzzy_match():
    spans, stats = align_entities(["Встреча", "в", "Москве"], [("Москва", "LOC")])
    assert spans == [(2, 3, "LOC")] and stats["fuzzy"] == 1

    spans, _ = align_entities(["с", "Дмитрием", "Медведевым"], [("Дмитрий Медведев", "PER")])
    assert spans == [(1, 3, "PER")]


def test_align_repeated_mentions_go_to_different_occurrences():
    tokens = ["Иванов", "встретил", "Петрова", ",", "Иванов", "доволен"]
    spans, _ = align_entities(tokens, [("Иванов", "PER"), ("Петрова", "PER"), ("Иванов", "PER")])

    assert spans == [(0, 1, "PER"), (2, 3, "PER"), (4, 5, "PER")]


def test_align_hallucinated_entity_is_dropped():
    spans, stats = align_entities(["Встреча", "в", "Москве"], [("Париж", "LOC")])

    assert spans == [] and stats["unmatched"] == 1


def test_response_to_tags_refusal_and_broken_json_give_all_o():
    tags, stats = response_to_tags(["В", "Москве"], "Я не могу обсуждать эту тему.")
    assert tags == ["O", "O"] and stats["no_json"] == 1

    broken = '{"entities": [{"text": "Москве"\n "type": "LOC"}]}'  # missing comma
    tags, stats = response_to_tags(["В", "Москве"], broken)
    assert tags == ["O", "O"] and stats["parse_error"] == 1


def test_few_shot_prompt_contains_examples_as_json():
    tokens, tags = ["Глава", "Минфина", "Антон", "Силуанов"], ["O", "B-ORG", "B-PER", "I-PER"]
    prompt = build_system_prompt([(tokens, tags)])

    assert "Текст: Глава Минфина Антон Силуанов" in prompt
    assert entities_json(tokens, tags) in prompt
    assert '"Антон Силуанов"' in entities_json(tokens, tags)
