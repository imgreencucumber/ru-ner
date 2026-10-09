"""NER with an LLM: prompt building, response parsing and aligning returned strings to tokens."""

import json
import re
from collections import Counter

import numpy as np

from ru_ner.data import tag_names
from ru_ner.evaluation import spans_to_tags, tags_to_spans

INSTRUCTION = """Ты — система извлечения именованных сущностей из русских новостных текстов.
Найди в тексте все упоминания сущностей трёх типов:
- PER — люди: имя, фамилия, отчество, инициалы. Должности и звания не включай.
- LOC — географические объекты: страны, города, регионы, улицы, реки, горы и т. п.
- ORG — организации: компании, госорганы, министерства, партии, СМИ, учебные заведения, \
спортивные клубы и т. п.

Правила:
- Выписывай каждую сущность точно так, как она написана в тексте, в той же словоформе.
- Если сущность упоминается несколько раз, перечисли каждое упоминание отдельно.
- Перечисляй сущности в порядке появления в тексте.
- Не добавляй сущности, которых нет в тексте.

Ответь только JSON без пояснений в формате:
{"entities": [{"text": "...", "type": "PER"}]}
Если сущностей нет, ответь {"entities": []}."""

TYPE_ALIASES = {
    "PER": "PER",
    "PERSON": "PER",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
}


def sentence_text(tokens):
    return " ".join(tokens)


def entities_json(tokens, tags):
    entities = [{"text": " ".join(tokens[s:e]), "type": t} for s, e, t in tags_to_spans(tags)]
    return json.dumps({"entities": entities}, ensure_ascii=False)


def pick_few_shot_examples(train, n_with_entities=4, seed=0):
    """Fixed examples from train: sentences of 10-30 tokens with entities of at least two types,
    plus one sentence without entities."""
    rng = np.random.default_rng(seed)
    sentences, tags = list(train["tokens"]), tag_names(train)
    rich, empty = [], []
    for i, (s, t) in enumerate(zip(sentences, tags, strict=True)):
        if not 10 <= len(s) <= 30:
            continue
        types = {etype for _, _, etype in tags_to_spans(t)}
        if len(types) >= 2:
            rich.append(i)
        elif not types:
            empty.append(i)
    chosen = list(rng.choice(rich, n_with_entities, replace=False)) + [int(rng.choice(empty))]
    return [(sentences[i], tags[i]) for i in chosen]


def build_system_prompt(examples=None):
    if not examples:
        return INSTRUCTION
    shots = "\n\n".join(
        f"Текст: {sentence_text(tokens)}\nОтвет: {entities_json(tokens, tags)}"
        for tokens, tags in examples
    )
    return f"{INSTRUCTION}\n\nПримеры:\n\n{shots}"


def build_user_prompt(tokens):
    return f"Текст: {sentence_text(tokens)}"


def parse_response(text):
    """LLM answer -> [(entity text, type)]. Returns None if there is no valid JSON object.

    Tolerates markdown code fences and text around the JSON. Unknown types are dropped.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    items = data.get("entities") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    entities = []
    for item in items:
        if not isinstance(item, dict):
            continue
        etext, etype = item.get("text"), TYPE_ALIASES.get(str(item.get("type", "")).upper())
        if isinstance(etext, str) and etext.strip() and etype:
            entities.append((etext, etype))
    return entities


# --- alignment -------------------------------------------------------------------------------

_CHAR_MAP = str.maketrans(
    {"ё": "е", "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "—": "-", "–": "-"}
)


def _norm(text):
    return text.lower().translate(_CHAR_MAP)


def _same_word(a, b):
    """Equal after normalisation, or the same word in another grammatical form: lengths differ
    by at most 2 and only the last 3 letters may differ (Москва / Москве, Медведев / Медведевым).
    A cut-off word like Росси / россияне doesn't match."""
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    if not (a.isalpha() and b.isalpha()) or abs(len(a) - len(b)) > 2:
        return False
    prefix = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix += 1
    return prefix >= max(3, max(len(a), len(b)) - 3)


def _exact_candidates(tokens, etext):
    """Token spans whose text equals etext ignoring whitespace, case, ё/е, quote and dash styles.

    Comparison is done on the concatenation of tokens without spaces, and a match must start
    and end on token boundaries, so 'Росси' is never found inside 'россиян'.
    """
    squeezed_tokens = [_norm(t).replace(" ", "") for t in tokens]
    target = re.sub(r"\s+", "", _norm(etext))
    if not target:
        return []
    starts, pos = {}, 0
    for i, t in enumerate(squeezed_tokens):
        starts[pos] = i
        pos += len(t)
    ends = {}
    pos = 0
    for i, t in enumerate(squeezed_tokens):
        pos += len(t)
        ends[pos] = i + 1
    joined = "".join(squeezed_tokens)
    found, k = [], joined.find(target)
    while k != -1:
        if k in starts and k + len(target) in ends:
            found.append((starts[k], ends[k + len(target)]))
        k = joined.find(target, k + 1)
    return found


def _fuzzy_candidates(tokens, etext):
    """Token spans where every word of etext matches a token by _same_word."""
    words = re.findall(r"\w+|[^\w\s]", etext)
    n = len(words)
    return [
        (i, i + n)
        for i in range(len(tokens) - n + 1)
        if all(_same_word(w, t) for w, t in zip(words, tokens[i : i + n], strict=True))
    ]


def align_entities(tokens, entities):
    """Place (text, type) pairs onto token spans. Returns spans and alignment statistics.

    Each entity takes the leftmost occurrence that doesn't overlap an already placed entity,
    so an entity listed twice lands on two different mentions. Exact matches are tried first,
    then matches allowing a different word form. Entities not found in the text are dropped.
    """
    used = set()
    spans = []
    stats = Counter()
    for etext, etype in entities:
        placed = False
        for kind, finder in (("exact", _exact_candidates), ("fuzzy", _fuzzy_candidates)):
            for s, e in finder(tokens, etext):
                if used.isdisjoint(range(s, e)):
                    spans.append((s, e, etype))
                    used.update(range(s, e))
                    stats[kind] += 1
                    placed = True
                    break
            if placed:
                break
        if not placed:
            stats["unmatched"] += 1
    return sorted(spans), stats


def response_to_tags(tokens, response_text):
    """Tags for one sentence and statistics. An answer without any JSON is counted as no_json
    (in practice a moderation refusal), broken JSON as parse_error; both give all-O tags."""
    entities = parse_response(response_text)
    if entities is None:
        kind = "parse_error" if "{" in response_text else "no_json"
        return ["O"] * len(tokens), Counter({kind: 1})
    spans, stats = align_entities(tokens, entities)
    stats["returned"] += len(entities)
    return spans_to_tags(spans, len(tokens)), stats
