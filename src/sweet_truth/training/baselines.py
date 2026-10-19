"""The models v2 is compared with: v1 (as built in the notebook, as used by the app) and a
character n-gram baseline."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from spacy.language import Language

from sweet_truth.classifiers import SklearnClassifier
from sweet_truth.schemas import Item
from sweet_truth.text import Resources, classify_ingredients, clean_ingredient
from sweet_truth.training.data import Example, IngredientList

_NOTEBOOK_NOISE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|[.]")


def notebook_preprocess(text: str, nlp: Language) -> str:
    """v1's training preprocessing, as in `sweetener_detector_from_text.ipynb`.

    Bracketed text and full stops are deleted, then lower-cased words are lemmatised and stop
    words dropped.
    """
    while _NOTEBOOK_NOISE.search(text):
        text = _NOTEBOOK_NOISE.sub("", text)
    doc = nlp(text.lower().strip())
    return " ".join(token.lemma_ for token in doc if not token.is_stop)


def v1_pipeline() -> Pipeline:
    """v1's architecture with its tuned settings: word TF-IDF + Random Forest."""
    # The notebook seeded NumPy's global generator, which an unseeded forest draws from.
    np.random.seed(101)
    forest = RandomForestClassifier(n_estimators=451, min_samples_split=6, min_samples_leaf=1)
    return Pipeline([("vectorizer", TfidfVectorizer()), ("model", forest)])


def train_v1_on_lists(lists: Sequence[IngredientList], nlp: Language) -> Pipeline:
    """Retrain v1 exactly as the notebook did: one example per whole list.

    The deployed pickle was refitted on all 1,000 lists, hold-out included, so it cannot be
    evaluated fairly on the hold-out lists; this refit on the training lists can.
    """
    pipeline = v1_pipeline()
    pipeline.fit([notebook_preprocess(item.text, nlp) for item in lists], [i.label for i in lists])
    return pipeline


def v1_list_accuracy(pipeline: Pipeline, lists: Sequence[IngredientList], nlp: Language) -> float:
    """v1's original measure: is the whole list correctly called "has a sweetener" or not."""
    predicted = pipeline.predict([notebook_preprocess(item.text, nlp) for item in lists])
    return float(np.mean([p == item.label for p, item in zip(predicted, lists, strict=True)]))


def _sweetener_probability(items: list[Item]) -> float:
    if not items:
        return 0.0
    return max(item.confidence if item.is_sweetener else 1 - item.confidence for item in items)


@dataclass
class V1AsApp:
    """v1 used the way the live app uses it: the list is split and each piece classified.

    An ingredient counts as a sweetener when any piece the app makes of it does.
    """

    pipeline: Pipeline
    nlp: Language
    name: str = "v1 (word TF-IDF + random forest)"

    def __post_init__(self) -> None:
        self._resources = Resources(nlp=self.nlp, model=self.pipeline)

    def predict_proba(self, ingredients: Sequence[str]) -> np.ndarray:
        return np.array(
            [
                _sweetener_probability(classify_ingredients(text, self._resources).items)
                for text in ingredients
            ]
        )

    def classify_text(self, text: str) -> list[tuple[str, float]]:
        items = classify_ingredients(text, self._resources).items
        return [(item.text, _sweetener_probability([item])) for item in items]


# The 2x2 comparison: which matters more, the features or the classifier?
FEATURES = ("words", "chars")
CLASSIFIERS = ("forest", "logistic")
V2_NAME = "v2 (character n-grams + logistic regression)"


def build_pipeline(features: str, classifier: str) -> Pipeline:
    """TF-IDF on whole words (v1's features) or letter chunks of 1 to 6 characters (v2's),
    followed by a random forest (v1's tuned settings) or logistic regression (v2's)."""
    if features == "words":
        vectorizer = TfidfVectorizer()
    elif features == "chars":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 6), sublinear_tf=True)
    else:
        raise ValueError(f"Unknown features {features!r}")
    if classifier == "forest":
        np.random.seed(101)
        model = RandomForestClassifier(n_estimators=451, min_samples_split=6, min_samples_leaf=1)
    elif classifier == "logistic":
        model = LogisticRegression(class_weight="balanced", C=10, solver="liblinear")
    else:
        raise ValueError(f"Unknown classifier {classifier!r}")
    return Pipeline([("vectorizer", vectorizer), ("model", model)])


def fit_ablation(
    examples: Sequence[Example], features: str, classifier: str, nlp: Language | None = None
) -> SklearnClassifier:
    """Train one of the four feature / classifier combinations on single ingredients.

    Word features use v1's preprocessing (brackets dropped, lemmatised), so `nlp` is needed;
    letter-chunk features read the ingredient as written (brackets kept).
    """
    if features == "words":
        if nlp is None:
            raise ValueError("Word features need a spaCy pipeline for lemmatising")

        def preprocess(text: str) -> str:
            return notebook_preprocess(text, nlp)
    else:
        preprocess = clean_ingredient
    pipeline = build_pipeline(features, classifier)
    pipeline.fit([preprocess(e.text) for e in examples], [e.label for e in examples])
    return SklearnClassifier(pipeline, name=f"{features} + {classifier}", preprocess=preprocess)


def fit_v1_on_ingredients(examples: Sequence[Example], nlp: Language) -> SklearnClassifier:
    """v1's architecture trained on single ingredients, for the unseen-ingredient folds."""
    classifier = fit_ablation(examples, "words", "forest", nlp)
    classifier.name = "v1 (word TF-IDF + random forest)"
    return classifier


def fit_v2(examples: Sequence[Example]) -> SklearnClassifier:
    """Model v2: letter chunks of 1 to 6 characters + logistic regression."""
    classifier = fit_ablation(examples, "chars", "logistic")
    classifier.name = V2_NAME
    return classifier
