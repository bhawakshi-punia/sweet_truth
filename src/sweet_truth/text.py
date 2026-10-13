"""Find the sweeteners in an ingredient list given as text."""

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import spacy
from spacy.language import Language

from sweet_truth.config import Settings
from sweet_truth.ocr import OcrEngine
from sweet_truth.schemas import Item, TextResult

logger = logging.getLogger(__name__)

SWEETENER_LABEL = 1

_OPENING, _CLOSING = "([{", ")]}"
# What separates one ingredient from the next.
_SEPARATOR = re.compile(r"\s*[,:]\s*")
_CONJUNCTIONS = frozenset({"and", "or"})


@dataclass
class Resources:
    """Everything that is expensive to load, loaded once and shared."""

    nlp: Language
    model: Any
    ocr: OcrEngine | None = None
    model_version: str = "unknown"
    # spaCy pipelines are not documented as thread-safe, so calls are serialised.
    nlp_lock: threading.Lock = field(default_factory=threading.Lock)


def load_nlp(name: str = "en_core_web_sm") -> Language:
    # Only lemmas and stop words are used; the parser and entity recogniser are dead weight.
    return spacy.load(name, disable=["parser", "ner"])


def load_resources(settings: Settings, *, with_ocr: bool = True) -> Resources:
    logger.info("Loading model from %s", settings.model_path)
    digest = hashlib.sha256(settings.model_path.read_bytes()).hexdigest()[:12]
    return Resources(
        nlp=load_nlp(settings.spacy_model),
        model=joblib.load(settings.model_path),
        ocr=OcrEngine(settings.tessdata_dir) if with_ocr else None,
        model_version=f"{settings.model_path.stem}@{digest}",
    )


def _bracketed_spans(text: str) -> list[tuple[int, int]]:
    """Find bracketed stretches of `text`, outermost only, as (start, end).

    Any closing bracket closes any opening one: labels read from photos often
    come back as "(472e]" or "(maida }". A bracket with no partner is left alone.
    """
    open_at: list[int] = []
    pairs: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        if char in _OPENING:
            open_at.append(index)
        elif char in _CLOSING and open_at:
            pairs.append((open_at.pop(), index + 1))
    outermost: list[tuple[int, int]] = []
    for start, end in sorted(pairs):
        if not outermost or start >= outermost[-1][1]:
            outermost.append((start, end))
    return outermost


def _strip_noise(text: str) -> tuple[str, list[int]]:
    """Replace bracketed text and full stops with commas.

    Returns the cleaned text and, for each of its characters, the offset of the
    character it came from in `text`.
    """
    chars: list[str] = []
    origin: list[int] = []
    cursor = 0
    for start, end in [*_bracketed_spans(text), (len(text), len(text))]:
        for index in range(cursor, start):
            chars.append("," if text[index] == "." else text[index])
            origin.append(index)
        if start < end:
            chars.append(",")
            origin.append(start)
        cursor = end
    return "".join(chars), origin


def _lowercase(text: str, origin: list[int]) -> tuple[str, list[int]]:
    """Lowercase `text`, keeping `origin` aligned when a character changes length."""
    chars: list[str] = []
    lowered_origin: list[int] = []
    for char, source in zip(text, origin, strict=True):
        lowered = char.lower()
        chars.append(lowered)
        lowered_origin.extend([source] * len(lowered))
    return "".join(chars), lowered_origin


def _segments(nlp: Language, cleaned: str) -> list[tuple[str, int, int]]:
    """Split cleaned text into ingredients: (normalised form, start, end) in `cleaned`."""
    doc = nlp(cleaned)
    # Lemmatise and drop stop words, remembering which token produced each character.
    # "and" / "or" are stop words too, but they also end an ingredient.
    pieces: list[str] = []
    char_token: list[int | None] = []
    kept = [token for token in doc if not token.is_stop or token.lower_ in _CONJUNCTIONS]
    for position, token in enumerate(kept):
        if position:
            pieces.append(" ")
            char_token.append(None)
        if token.lower_ in _CONJUNCTIONS:
            pieces.append(",")
            char_token.append(None)
        else:
            pieces.append(token.lemma_)
            char_token.extend([position] * len(token.lemma_))
    joined = "".join(pieces)

    found: list[tuple[str, int, int]] = []
    cursor = 0
    for separator in [*_SEPARATOR.finditer(joined), None]:
        stop = separator.start() if separator else len(joined)
        tokens = [index for index in char_token[cursor:stop] if index is not None]
        if tokens:
            first, last = kept[tokens[0]], kept[tokens[-1]]
            found.append((joined[cursor:stop], first.idx, last.idx + len(last.text)))
        if separator:
            cursor = separator.end()
    return found


def classify_ingredients(text: str, resources: Resources) -> TextResult:
    """Split `text` into ingredients and predict which of them are sweeteners."""
    cleaned, origin = _strip_noise(text)
    cleaned, origin = _lowercase(cleaned, origin)
    leading = len(cleaned) - len(cleaned.lstrip())
    cleaned = cleaned.strip()
    origin = origin[leading : leading + len(cleaned)]
    with resources.nlp_lock:
        segments = _segments(resources.nlp, cleaned)
    if not segments:
        return TextResult(text=text, items=[], sweeteners=[])

    model = resources.model
    probabilities = model.predict_proba([normalized for normalized, _, _ in segments])
    best = np.argmax(probabilities, axis=1)

    items: list[Item] = []
    for (normalized, start, end), row, column in zip(segments, probabilities, best, strict=True):
        text_start, text_end = origin[start], origin[end - 1] + 1
        items.append(
            Item(
                text=text[text_start:text_end],
                normalized=normalized,
                start=text_start,
                end=text_end,
                is_sweetener=bool(model.classes_[column] == SWEETENER_LABEL),
                confidence=float(row[column]),
            )
        )
    return TextResult(
        text=text, items=items, sweeteners=[item for item in items if item.is_sweetener]
    )
