"""The original notebook implementation of text detection, frozen as a reference.

Copied from src/module_detector.py as it was before the sweet_truth package existed,
changed only to take the spaCy pipeline as an argument instead of loading it per call.
The parity test checks that the package still flags exactly what this does.
"""

import re


def preprocess_text(text, nlp):
    pattern = r"\([^()]*\)|\[[^\[\]]*\]|[.]"
    while re.search(pattern, text):
        text = re.sub(pattern, ",", text)
    text = text.lower().strip()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


def sweeteners_from_list(lst, model, nlp):
    """Return [(ingredient, confidence in percent)] for every predicted sweetener."""
    prep_list = preprocess_text(lst, nlp)
    rx = re.compile(r"\s*(?:\b(?:and|or)\b|[,:])\s*")
    tokens = rx.split(prep_list)
    result = []
    for txt in tokens:
        if model.predict([txt])[0] != 0:
            result.append((txt, max(model.predict_proba([txt])[0]) * 100))
    return result
