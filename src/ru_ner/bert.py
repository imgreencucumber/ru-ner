import numpy as np
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


def bio_constraints(labels=LABELS):
    """allowed[i, j]: label j may follow label i; start[j]: label j may begin a sentence.

    I-X may only follow B-X or I-X, everything else is allowed.
    """
    n = len(labels)
    allowed = np.ones((n, n), dtype=bool)
    start = np.array([not label.startswith("I-") for label in labels])
    for j, cur in enumerate(labels):
        if cur.startswith("I-"):
            for i, prev in enumerate(labels):
                allowed[i, j] = prev != "O" and prev[2:] == cur[2:]
    return allowed, start


ALLOWED, ALLOWED_START = bio_constraints()


def viterbi_decode(log_probs, allowed=ALLOWED, allowed_start=ALLOWED_START):
    """Label ids with the highest total log-probability among valid BIO sequences.

    log_probs: (n_words, n_labels). Transitions are hard constraints (0 or -inf), not learned.
    """
    n = len(log_probs)
    if n == 0:
        return []
    transitions = np.where(allowed, 0.0, -np.inf)
    score = np.where(allowed_start, log_probs[0], -np.inf)
    backpointers = np.zeros((n, log_probs.shape[1]), dtype=np.int64)
    for t in range(1, n):
        # candidates[i, j]: best score of a path ending in label i at t-1, then label j at t
        candidates = score[:, None] + transitions
        backpointers[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_probs[t]
    path = [int(score.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(backpointers[t, path[-1]]))
    return path[::-1]


def word_log_probs(word_ids, subword_log_probs, n_words):
    """Rows of the first subword of each word. Words cut off by truncation are forced to O."""
    out = np.full((n_words, subword_log_probs.shape[1]), -np.inf)
    out[:, LABELS.index("O")] = 0.0
    prev = None
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid != prev:
            out[wid] = subword_log_probs[pos]
        prev = wid
    return out


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


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class _TokenClassifier:
    """Shared tokenization and decoding. Subclasses only implement `_logits`.

    decoding="viterbi": best valid BIO sequence; "argmax": each word labelled independently.
    """

    def __init__(self, model_dir, batch_size=32, decoding="viterbi"):
        if decoding not in ("viterbi", "argmax"):
            raise ValueError(f"unknown decoding: {decoding}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.batch_size = batch_size
        self.decoding = decoding

    def _logits(self, enc):
        """numpy arrays from the tokenizer -> logits of shape (batch, seq_len, n_labels)."""
        raise NotImplementedError

    def predict(self, sentences):
        # Sentences of similar length go to the same batch, so little compute is spent on padding.
        # Results are put back in the original order.
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        out = [None] * len(sentences)
        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start : start + self.batch_size]
            batch = [sentences[i] for i in batch_idx]
            # No 128-token limit here: at test time every word must get a prediction
            enc = self.tokenizer(
                batch,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="np",
            )
            logits = self._logits(enc)
            if self.decoding == "argmax":
                pred = logits.argmax(-1)
                for i, (idx, tokens) in enumerate(zip(batch_idx, batch, strict=True)):
                    out[idx] = word_predictions(enc.word_ids(i), pred[i].tolist(), len(tokens))
            else:
                log_probs = log_softmax(logits)
                for i, (idx, tokens) in enumerate(zip(batch_idx, batch, strict=True)):
                    word_lp = word_log_probs(enc.word_ids(i), log_probs[i], len(tokens))
                    out[idx] = [LABELS[j] for j in viterbi_decode(word_lp)]
        return out


class BertNER(_TokenClassifier):
    """PyTorch backend, GPU if available."""

    def __init__(self, model_dir, device=None, batch_size=32, decoding="viterbi"):
        super().__init__(model_dir, batch_size, decoding)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def _logits(self, enc):
        inputs = {k: torch.from_numpy(v).to(self.device) for k, v in enc.items()}
        return self.model(**inputs).logits.float().cpu().numpy()


class OnnxBertNER(_TokenClassifier):
    """ONNX Runtime backend on CPU. model_dir is only needed for the tokenizer."""

    def __init__(self, model_dir, onnx_path, batch_size=32, decoding="viterbi", num_threads=None):
        import onnxruntime as ort

        super().__init__(model_dir, batch_size, decoding)
        options = ort.SessionOptions()
        if num_threads:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def _logits(self, enc):
        inputs = {name: enc[name].astype(np.int64) for name in self.input_names}
        return self.session.run(["logits"], inputs)[0]
