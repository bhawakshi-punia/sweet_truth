"""Ingredient classifiers behind one interface, so models can be compared and swapped."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import spacy

from sweet_truth.text import clean_ingredient, split_ingredients

# Category names used by the spaCy text classifier of model v2.
SWEETENER = "SWEETENER"
OTHER = "OTHER"


class IngredientClassifier(Protocol):
    name: str

    def predict_proba(self, ingredients: Sequence[str]) -> np.ndarray:
        """Probability that each ingredient is a sweetener."""
        ...

    def classify_text(self, text: str) -> list[tuple[str, float]]:
        """Split a whole ingredient list and score each ingredient: (ingredient, probability)."""
        ...


class _SplitsLikeV2:
    """Scores a whole list the model v2 way: split with brackets kept, then classify."""

    def predict_proba(self, ingredients: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def classify_text(self, text: str) -> list[tuple[str, float]]:
        pieces = [piece for piece, _, _ in split_ingredients(text)]
        return list(zip(pieces, self.predict_proba(pieces).tolist(), strict=True))


@dataclass
class SpacyTextcat(_SplitsLikeV2):
    """A trained spaCy pipeline with a text classifier, as produced by `sweet-truth model train`."""

    path: Path
    name: str = "spaCy textcat"
    batch_size: int = 64
    _nlp: spacy.language.Language = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._nlp = spacy.load(self.path)

    def predict_proba(self, ingredients: Sequence[str]) -> np.ndarray:
        texts = [clean_ingredient(text) for text in ingredients]
        docs = self._nlp.pipe(texts, batch_size=self.batch_size)
        return np.array([doc.cats[SWEETENER] for doc in docs], dtype=float)


@dataclass
class SklearnClassifier(_SplitsLikeV2):
    """A fitted scikit-learn pipeline that reads ingredient text, with optional preprocessing."""

    pipeline: Any
    name: str
    preprocess: Callable[[str], str] = clean_ingredient

    def predict_proba(self, ingredients: Sequence[str]) -> np.ndarray:
        texts = [self.preprocess(text) for text in ingredients]
        positive = list(self.pipeline.classes_).index(1)
        return np.asarray(self.pipeline.predict_proba(texts))[:, positive]
