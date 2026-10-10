"""Inference without PyTorch: tokenization, ONNX Runtime and BIO-constrained decoding.

Depends only on numpy, tokenizers and onnxruntime, so the service image doesn't need PyTorch.
"""

import re
from pathlib import Path

import numpy as np

from ru_ner.labels import LABELS

# --- decoding --------------------------------------------------------------------------------


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


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


def word_predictions(word_ids, pred_ids, n_words):
    """Take the prediction of each word's first subword. Words cut off by truncation get O."""
    tags = ["O"] * n_words
    prev = None
    for wid, p in zip(word_ids, pred_ids, strict=True):
        if wid is not None and wid != prev:
            tags[wid] = LABELS[p]
        prev = wid
    return tags


def bio_spans(tags):
    """BIO tags -> [(start, end, type)], end exclusive. A stray I-X starts a new entity."""
    spans, start, etype = [], None, None
    for i, tag in enumerate([*tags, "O"]):
        if start is not None and not (tag.startswith("I-") and tag[2:] == etype):
            spans.append((start, i, etype))
            start = None
        if start is None and tag != "O":
            start, etype = i, tag[2:]
    return spans


# --- raw text ----------------------------------------------------------------------------------

# Words and single punctuation marks, the way Collection3 is tokenized ("пресс - служба")
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SENTENCE_END = {".", "!", "?", "…"}


def tokenize(text):
    """Raw text -> [(word, start, end)] with character offsets into the text."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def split_sentences(tokens, max_words=100):
    """Cut a token list into pieces like those the model was trained on: one sentence each,
    at most max_words long (training sentences are rarely longer)."""
    pieces, current = [], []
    for token in tokens:
        current.append(token)
        if token[0] in SENTENCE_END or len(current) >= max_words:
            pieces.append(current)
            current = []
    if current:
        pieces.append(current)
    return pieces


# --- models ----------------------------------------------------------------------------------


class TokenClassifier:
    """Shared tokenization and decoding. Subclasses only implement `_logits`.

    decoding="viterbi": best valid BIO sequence; "argmax": each word labelled independently.
    """

    def __init__(self, model_dir, batch_size=32, decoding="viterbi"):
        from tokenizers import Tokenizer

        if decoding not in ("viterbi", "argmax"):
            raise ValueError(f"unknown decoding: {decoding}")
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        # No 128-token limit here: at test time every word must get a prediction
        self.tokenizer.enable_truncation(512)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.batch_size = batch_size
        self.decoding = decoding

    def _encode(self, batch):
        """Pre-split sentences -> (dict of int64 arrays, word_ids for every sentence)."""
        encodings = self.tokenizer.encode_batch(batch, is_pretokenized=True)
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        return inputs, [e.word_ids for e in encodings]

    def _logits(self, inputs):
        """dict of int64 arrays -> logits of shape (batch, seq_len, n_labels)."""
        raise NotImplementedError

    def predict(self, sentences):
        """Lists of words -> lists of BIO tags."""
        # Sentences of similar length go to the same batch, so little compute is spent on padding.
        # Results are put back in the original order.
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        out = [None] * len(sentences)
        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start : start + self.batch_size]
            batch = [sentences[i] for i in batch_idx]
            inputs, word_ids = self._encode(batch)
            logits = self._logits(inputs)
            if self.decoding == "argmax":
                pred = logits.argmax(-1)
                for i, (idx, tokens) in enumerate(zip(batch_idx, batch, strict=True)):
                    out[idx] = word_predictions(word_ids[i], pred[i].tolist(), len(tokens))
            else:
                log_probs = log_softmax(logits)
                for i, (idx, tokens) in enumerate(zip(batch_idx, batch, strict=True)):
                    word_lp = word_log_probs(word_ids[i], log_probs[i], len(tokens))
                    out[idx] = [LABELS[j] for j in viterbi_decode(word_lp)]
        return out

    def extract_batch(self, texts):
        """Raw texts -> for each text a list of {"text", "type", "start", "end"} entities,
        where start and end are character offsets into that text."""
        pieces, owners = [], []
        for k, text in enumerate(texts):
            for piece in split_sentences(tokenize(text)):
                pieces.append(piece)
                owners.append(k)
        tags = self.predict([[word for word, _, _ in piece] for piece in pieces]) if pieces else []
        results = [[] for _ in texts]
        for k, piece, piece_tags in zip(owners, pieces, tags, strict=True):
            for s, e, etype in bio_spans(piece_tags):
                start, end = piece[s][1], piece[e - 1][2]
                results[k].append(
                    {"text": texts[k][start:end], "type": etype, "start": start, "end": end}
                )
        return results

    def extract(self, text):
        return self.extract_batch([text])[0]


class OnnxBertNER(TokenClassifier):
    """ONNX Runtime backend on CPU. model_dir holds tokenizer.json and, by default, the model."""

    def __init__(
        self,
        model_dir,
        onnx_path=None,
        batch_size=32,
        decoding="viterbi",
        num_threads=None,
    ):
        import onnxruntime as ort

        super().__init__(model_dir, batch_size, decoding)
        options = ort.SessionOptions()
        if num_threads:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(onnx_path or Path(model_dir) / "model_int8.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def _logits(self, inputs):
        return self.session.run(["logits"], {name: inputs[name] for name in self.input_names})[0]
