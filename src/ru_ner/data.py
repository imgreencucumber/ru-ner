import gzip

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from seqeval.metrics.sequence_labeling import get_entities

from ru_ner.labels import LABEL2ID, LABELS

FEATURES = Features(
    {
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    }
)

# Revisions are pinned so that results don't change if the upstream repos are updated
COLLECTION3_REPO = "RCC-MSU/collection3"
COLLECTION3_REVISION = "18841ce4c41a94aaed0041342c6a7cb0c59cfcfe"

WIKINEURAL_REPO = "Babelscape/wikineural"
WIKINEURAL_REVISION = "74c9b9dca034bb1606a6769457983904bd976803"
# WikiNEuRal has its own tag order (ORG before LOC) and an extra MISC class
WIKINEURAL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def tag_names(ds):
    """ner_tags as lists of label strings, the format seqeval and CRF work with."""
    return [[LABELS[i] for i in tags] for tags in ds["ner_tags"]]


def parse_conll(lines):
    """Parse `token<TAB>tag` lines into sentences separated by blank lines."""
    sentences = []
    tokens, tags = [], []
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("<DOCSTART>"):
            if tokens:
                sentences.append({"tokens": tokens, "ner_tags": tags})
                tokens, tags = [], []
            continue
        token, tag = line.split("\t")
        tokens.append(token)
        tags.append(tag)
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


def load_collection3():
    splits = {}
    for split, filename in [("train", "train"), ("validation", "valid"), ("test", "test")]:
        path = hf_hub_download(
            COLLECTION3_REPO,
            f"data/{filename}.txt.gz",
            repo_type="dataset",
            revision=COLLECTION3_REVISION,
        )
        with gzip.open(path, "rt", encoding="utf-8") as f:
            sentences = parse_conll(f)
        for s in sentences:
            s["ner_tags"] = [LABEL2ID[t] for t in s["ner_tags"]]
        splits[split] = Dataset.from_list(sentences, features=FEATURES)
    return DatasetDict(splits)


def _wikineural_to_our_labels(example):
    tags = [WIKINEURAL_LABELS[t] for t in example["ner_tags"]]
    # MISC has no counterpart in Collection3, so it becomes O
    return {"ner_tags": [LABEL2ID["O" if t.endswith("MISC") else t] for t in tags]}


def _load_wikineural_raw(split):
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split}")
    return load_dataset(
        WIKINEURAL_REPO,
        data_files={split: f"data/{split}_ru-00000-of-00001.parquet"},
        revision=WIKINEURAL_REVISION,
        split=split,
    )


def load_wikineural_ru(split="test"):
    ds = _load_wikineural_raw(split).map(_wikineural_to_our_labels, remove_columns=["lang"])
    return ds.cast(FEATURES)


def wikineural_misc_spans(split="test"):
    """MISC entities of WikiNEuRal-ru as [(start, end)] per sentence, end exclusive.

    They are O in load_wikineural_ru; kept separately to check which "spurious" predictions
    are actually MISC entities. Sentence order is the same as in load_wikineural_ru.
    """
    return [
        _misc_spans([WIKINEURAL_LABELS[t] for t in tags])
        for tags in _load_wikineural_raw(split)["ner_tags"]
    ]


def _misc_spans(tag_strings):
    return [(s, e + 1) for t, s, e in get_entities(tag_strings) if t == "MISC"]
