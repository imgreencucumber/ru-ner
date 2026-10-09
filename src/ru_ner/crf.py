import re

import sklearn_crfsuite


def word_shape(token):
    """Collapsed character classes: 'Медведев' -> 'Xx', 'РФ' -> 'X', '2003г' -> 'dx'."""
    shape = re.sub(r"[A-ZА-ЯЁ]", "X", token)
    shape = re.sub(r"[a-zа-яё]", "x", shape)
    shape = re.sub(r"\d", "d", shape)
    return re.sub(r"(.)\1+", r"\1", shape)


def _token_features(token):
    return {
        "lower": token.lower(),
        "prefix2": token[:2].lower(),
        "suffix2": token[-2:].lower(),
        "suffix3": token[-3:].lower(),
        "shape": word_shape(token),
        "is_title": token.istitle(),
        "is_upper": token.isupper(),
        "is_digit": token.isdigit(),
        "is_latin": bool(re.search(r"[a-zA-Z]", token)),
    }


# Neighbouring tokens get a smaller feature set to keep the model compact
_CONTEXT_FEATURES = ("lower", "suffix3", "shape", "is_title")


def sent2features(tokens):
    base = [_token_features(t) for t in tokens]
    features = []
    for i in range(len(tokens)):
        f = dict(base[i])
        for offset in (-2, -1, 1, 2):
            j = i + offset
            if 0 <= j < len(tokens):
                for name in _CONTEXT_FEATURES:
                    f[f"{offset}:{name}"] = base[j][name]
        if i == 0:
            f["BOS"] = True
        if i == len(tokens) - 1:
            f["EOS"] = True
        features.append(f)
    return features


def train_crf(sentences, tags, c1=0.1, c2=0.1, max_iterations=100):
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )
    crf.fit([sent2features(s) for s in sentences], tags)
    return crf


def predict_crf(crf, sentences):
    return crf.predict([sent2features(s) for s in sentences])
