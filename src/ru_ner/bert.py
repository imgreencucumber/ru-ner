import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ru_ner.data import LABELS
from ru_ner.metrics import ner_report

MODEL_NAME = "DeepPavlov/rubert-base-cased"
# CrossEntropyLoss in PyTorch skips positions with this label
IGNORE = -100


def align_labels(word_ids, word_labels):
    """Put each word's label on its first subword; other subwords and special tokens get IGNORE."""
    labels, prev = [], None
    for wid in word_ids:
        if wid is None or wid == prev:
            labels.append(IGNORE)
        else:
            labels.append(word_labels[wid])
        prev = wid
    return labels


def tokenize_and_align(batch, tokenizer, max_length=128):
    enc = tokenizer(
        batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length
    )
    enc["labels"] = [
        align_labels(enc.word_ids(i), tags) for i, tags in enumerate(batch["ner_tags"])
    ]
    return enc


def word_predictions(word_ids, pred_ids, n_words):
    """Take the prediction of each word's first subword. Words cut off by truncation get O."""
    tags = ["O"] * n_words
    prev = None
    for wid, p in zip(word_ids, pred_ids, strict=True):
        if wid is not None and wid != prev:
            tags[wid] = LABELS[p]
        prev = wid
    return tags


def compute_metrics(eval_pred):
    """Metrics for Trainer evaluation: subword level, only positions with real labels count."""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    y_true, y_pred = [], []
    for p_row, l_row in zip(preds, labels, strict=True):
        mask = l_row != IGNORE
        y_true.append([LABELS[i] for i in l_row[mask]])
        y_pred.append([LABELS[i] for i in p_row[mask]])
    r = ner_report(y_true, y_pred)["overall"]
    return {"precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}


class BertNER:
    def __init__(self, model_dir, device=None, batch_size=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def predict(self, sentences):
        out = []
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start : start + self.batch_size]
            # No 128-token limit here: at test time every word must get a prediction
            enc = self.tokenizer(
                batch,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            pred = logits.argmax(-1).cpu().tolist()
            for i, tokens in enumerate(batch):
                out.append(word_predictions(enc.word_ids(i), pred[i], len(tokens)))
        return out
