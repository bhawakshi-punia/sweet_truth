"""Find the sweeteners in an ingredient list given as text."""

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
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

    # spaCy, for v1's lemmatising; not loaded for v2.
    nlp: Language | None
    model: Any
    ocr: OcrEngine | None = None
    model_version: str = "unknown"
    kind: str = "v1"
    # spaCy pipelines are not documented as thread-safe, so calls are serialised.
    nlp_lock: threading.Lock = field(default_factory=threading.Lock)


def load_nlp(name: str = "en_core_web_sm") -> Language:
    # Only lemmas and stop words are used; the parser and entity recogniser are dead weight.
    return spacy.load(name, disable=["parser", "ner"])


def load_resources(settings: Settings, *, with_ocr: bool = True) -> Resources:
    path = settings.model_path
    logger.info("Loading model %s from %s", settings.model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return Resources(
        nlp=load_nlp(settings.spacy_model) if settings.model == "v1" else None,
        model=joblib.load(path),
        ocr=OcrEngine(settings.tessdata_dir) if with_ocr else None,
        model_version=f"{path.stem}@{digest}",
        kind=settings.model,
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


# Top-level boundaries between ingredients when brackets are kept (model v2).
_KEEP_BRACKETS_SEPARATOR = re.compile(r"[,;:]|\b(?:and|or)\b", re.IGNORECASE)
_SPACES = re.compile(r"\s+")
_NEW_WORD = re.compile(r"\s+[A-Za-z]")


def clean_ingredient(text: str) -> str:
    """Tidy an ingredient for model v2: one space between words, no edge punctuation."""
    return _SPACES.sub(" ", text).strip(" .,;:")


def split_ingredients(text: str) -> list[tuple[str, int, int]]:
    """Split a list into ingredients without dropping bracketed text, for model v2.

    Separators inside brackets are ignored, so "Sweeteners (951, 950)" stays one ingredient.
    A word right after a closing bracket also starts a new ingredient.
    Returns (cleaned ingredient, start, end), with offsets into `text`.
    """
    inside = [False] * len(text)
    bracketed = _bracketed_spans(text)
    for start, end in bracketed:
        inside[start:end] = [True] * (end - start)
    boundaries = [
        match.span()
        for match in _KEEP_BRACKETS_SEPARATOR.finditer(text)
        if not inside[match.start()]
    ]
    # Text read from photos often loses the comma after a bracket, as in
    # "Liquid glucose [1%] Permitted emulsifiers": a new word after a bracket starts a new one.
    boundaries += [(end, end) for _, end in bracketed if _NEW_WORD.match(text, end)]
    boundaries.sort()
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    for start, end in [*boundaries, (len(text), len(text))]:
        raw = text[cursor:start]
        cleaned = clean_ingredient(raw)
        if cleaned:
            edge = " .,;:\t\n\r"
            first = cursor + len(raw) - len(raw.lstrip(edge))
            last = cursor + len(raw.rstrip(edge))
            pieces.append((cleaned, first, last))
        cursor = end
    return pieces


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


def explain(pipeline: Any, ingredient: str, top: int = 3) -> list[str]:
    """The parts of an ingredient that most pushed a v2 prediction towards "sweetener".

    Each letter chunk's push is its TF-IDF value times the logistic regression's weight for
    it. The strongest chunks are located in the ingredient and overlapping ones merged, so
    "ugar" and "suga" read as "sugar"; a part a letter or two short of its word becomes the
    word. A "-" marks a cut inside a word: "-ose" is a word ending, "malt-" a word start.
    """
    vectorizer = pipeline.named_steps["vectorizer"]
    weights = pipeline.named_steps["model"].coef_[0]
    cleaned = clean_ingredient(ingredient).lower()
    row = vectorizer.transform([cleaned]).tocoo()
    names = _feature_names(vectorizer)
    pushes = sorted(
        ((value * weights[col], names[col]) for col, value in zip(row.col, row.data, strict=True)),
        reverse=True,
    )
    # Each word is padded with spaces, as the vectoriser saw it.
    padded = " " + " ".join(cleaned.split()) + " "
    spans: list[list[float]] = []  # [start, end, strongest push]
    for push, chunk in pushes[: top * 3]:
        if push <= 0 or len(chunk.strip()) < 3:
            continue
        found = padded.find(chunk)
        if found < 0:
            continue
        start, end = found + (chunk[0] == " "), found + len(chunk) - (chunk[-1] == " ")
        for span in spans:
            if start <= span[1] and end >= span[0]:
                span[0], span[1] = min(span[0], start), max(span[1], end)
                break
        else:
            spans.append([start, end, push])
    reasons = []
    for start, end, _ in sorted(spans, key=lambda span: -span[2])[:top]:
        start, end = int(start), int(end)
        # A part only a letter or two short of its word reads better as the whole word.
        word_start = padded.rfind(" ", 0, start) + 1
        word_end = padded.find(" ", end)
        if 0 < start - word_start <= 2:
            start = word_start
        if 0 < word_end - end <= 2:
            end = word_end
        part = padded[start:end].strip()
        cut_before = padded[start - 1] != " " and padded[start] != " "
        cut_after = padded[end] != " " and padded[end - 1] != " "
        reasons.append(f"{'-' if cut_before else ''}{part}{'-' if cut_after else ''}")
    return reasons


@lru_cache(maxsize=4)
def _feature_names(vectorizer: Any) -> Any:
    return vectorizer.get_feature_names_out()


def _classify_v2(text: str, resources: Resources) -> TextResult:
    pieces = split_ingredients(text)
    if not pieces:
        return TextResult(text=text, items=[], sweeteners=[])
    model = resources.model
    positive = list(model.classes_).index(SWEETENER_LABEL)
    probabilities = model.predict_proba([cleaned for cleaned, _, _ in pieces])[:, positive]
    items = []
    for (cleaned, start, end), probability in zip(pieces, probabilities, strict=True):
        is_sweetener = bool(probability >= 0.5)
        items.append(
            Item(
                text=text[start:end],
                normalized=cleaned,
                start=start,
                end=end,
                is_sweetener=is_sweetener,
                confidence=float(probability if is_sweetener else 1 - probability),
                reasons=explain(model, cleaned) if is_sweetener else [],
            )
        )
    return TextResult(
        text=text, items=items, sweeteners=[item for item in items if item.is_sweetener]
    )


def classify_ingredients(text: str, resources: Resources) -> TextResult:
    """Split `text` into ingredients and predict which of them are sweeteners."""
    if resources.kind == "v2":
        return _classify_v2(text, resources)
    if resources.nlp is None:
        raise RuntimeError("Model v1 needs the spaCy pipeline")
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
