from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter


def detokenize(tokens):
    """Join tokens with spaces and remember each token's character span."""
    offsets, pos = [], 0
    for t in tokens:
        offsets.append((pos, pos + len(t)))
        pos += len(t) + 1
    return " ".join(tokens), offsets


def spans_to_bio(offsets, spans):
    """Convert character spans (start, stop, type) to BIO tags over our tokens.

    A token belongs to a span if they overlap at all, so a span boundary that falls
    inside a token still marks the whole token.
    """
    tags = ["O"] * len(offsets)
    for start, stop, etype in spans:
        inside = [i for i, (s, e) in enumerate(offsets) if s < stop and e > start]
        for k, i in enumerate(inside):
            tags[i] = ("B-" if k == 0 else "I-") + etype
    return tags


class NatashaNER:
    def __init__(self):
        self.segmenter = Segmenter()
        self.tagger = NewsNERTagger(NewsEmbedding())

    def predict(self, sentences):
        predictions = []
        for tokens in sentences:
            text, offsets = detokenize(tokens)
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.tagger)
            spans = [(s.start, s.stop, s.type) for s in doc.spans]
            predictions.append(spans_to_bio(offsets, spans))
        return predictions
