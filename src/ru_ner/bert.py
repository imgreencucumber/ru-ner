"""Training helpers and the PyTorch backend. Decoding lives in ru_ner.inference."""

import torch
from transformers import AutoModelForTokenClassification

from ru_ner.inference import TokenClassifier
from ru_ner.labels import LABELS
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


class BertNER(TokenClassifier):
    """PyTorch backend, GPU if available."""

    def __init__(self, model_dir, device=None, batch_size=32, decoding="viterbi"):
        super().__init__(model_dir, batch_size, decoding)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def _logits(self, inputs):
        inputs = {k: torch.from_numpy(v).to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits.float().cpu().numpy()
