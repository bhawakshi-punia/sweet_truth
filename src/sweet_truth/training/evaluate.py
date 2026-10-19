"""Comparing v1, the character n-gram baseline and the trained v2 transformers.

Every model is scored the same way:

- hold-out (and validation) lists of the notebook's split, ingredient by ingredient, and as
  whole lists (flagged when any ingredient is), which is v1's original 97% measure;
- unseen-ingredient folds, where test ingredients never appear in training;
- the hand-labelled challenge set of real names that are not in the data;
- the sample photos: are the sweeteners printed on them found, and what else is flagged.
"""

import json
import logging
import pickle
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sweet_truth.classifiers import IngredientClassifier, SpacyTextcat
from sweet_truth.config import REPO_ROOT
from sweet_truth.text import load_nlp
from sweet_truth.training import baselines
from sweet_truth.training.data import (
    Example,
    IngredientList,
    examples_of,
    ingredient_group,
    load_challenge,
    load_lists,
    load_photo_expectations,
    notebook_split,
    unseen_folds,
)
from sweet_truth.training.train import MODELS_DIR, TRANSFORMERS

logger = logging.getLogger(__name__)

REPORTS_DIR = REPO_ROOT / "reports"
IMAGES_DIR = REPO_ROOT / "images"
THRESHOLD = 0.5


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    """Confusion-matrix counts and the measures built on them, sweeteners being positive."""
    y = np.asarray(labels, dtype=int)
    predicted = (np.asarray(probabilities, dtype=float) >= THRESHOLD).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),  # also called sensitivity
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "npv": round(tn / (tn + fn), 4) if tn + fn else 0.0,  # negative predictive value
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(y), 4) if len(y) else 0.0,
    }


def score_examples(
    model: IngredientClassifier, examples: Sequence[Example]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    probabilities = model.predict_proba([e.text for e in examples])
    predictions = [
        {"ingredient": e.text, "label": e.label, "probability": round(float(p), 4)}
        for e, p in zip(examples, probabilities, strict=True)
    ]
    return binary_metrics([e.label for e in examples], probabilities), predictions


def list_accuracy(model: IngredientClassifier, lists: Sequence[IngredientList]) -> float:
    """Share of lists correctly called "has a sweetener" (any ingredient flagged) or not."""
    probabilities = model.predict_proba([e.text for e in examples_of(lists)])
    correct = 0
    start = 0
    for item in lists:
        end = start + len(item.ingredients)
        flagged = bool((probabilities[start:end] >= THRESHOLD).any())
        correct += flagged == bool(item.label)
        start = end
    return round(correct / len(lists), 4)


def _name(text: str) -> str:
    """Compare ingredient names loosely: case, brackets and punctuation ignored."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", ingredient_group(text)).split())


def check_photo(
    model: IngredientClassifier, ingredients_text: str, expected: Sequence[str]
) -> dict[str, Any]:
    """Which printed sweeteners were flagged, which were missed, and what else was flagged."""
    flagged = [piece for piece, p in model.classify_text(ingredients_text) if p >= THRESHOLD]
    flagged_names = {_name(piece) for piece in flagged}
    wanted = {_name(name): name for name in expected}
    return {
        "found": [name for key, name in wanted.items() if key in flagged_names],
        "missed": [name for key, name in wanted.items() if key not in flagged_names],
        "extra": [piece for piece in flagged if _name(piece) not in wanted],
    }


def read_photos(images_dir: Path = IMAGES_DIR) -> dict[str, str]:
    """The ingredient text read from each sample photo listed in photo_sweeteners.csv."""
    from sweet_truth.config import Settings
    from sweet_truth.image import decode_image, ingredients_section, join_words
    from sweet_truth.ocr import OcrEngine

    engine = OcrEngine(Settings().tessdata_dir)
    texts = {}
    try:
        for image in load_photo_expectations():
            words = engine.read_words(decode_image((images_dir / image).read_bytes()))
            ocr_text, _ = join_words(words)
            offset, _ = ingredients_section(ocr_text)
            texts[image] = ocr_text[offset:]
    finally:
        engine.close()
    return texts


def _size_mb(model: Any) -> float:
    if isinstance(model, SpacyTextcat):
        size = sum(f.stat().st_size for f in Path(model.path).rglob("*") if f.is_file())
    else:
        size = len(pickle.dumps(model.pipeline))
    return round(size / 1e6, 1)


def _ms_per_ingredient(model: IngredientClassifier, texts: Sequence[str]) -> float:
    model.predict_proba(texts[:8])  # warm-up
    start = time.perf_counter()
    model.predict_proba(texts)
    return round((time.perf_counter() - start) * 1000 / len(texts), 2)


@dataclass
class Candidate:
    """A model to evaluate, and how to get its version for each unseen-ingredient fold."""

    slug: str
    model: Any
    description: str
    fold_model: Callable[[int, list[Example]], IngredientClassifier | None]


def _transformer_candidates(names: Sequence[str] | None) -> list[Candidate]:
    found = names or sorted(p.name for p in MODELS_DIR.glob("*") if (p / "model-best").is_dir())
    candidates = []
    for name in found:
        best = MODELS_DIR / name / "model-best"
        if not best.is_dir():
            logger.warning("No trained model at %s, skipping %s", best, name)
            continue

        def fold_model(k: int, _train: list[Example], name: str = name) -> SpacyTextcat | None:
            path = MODELS_DIR / name / "folds" / str(k) / "model-best"
            return SpacyTextcat(path, name=f"{name} fold {k}") if path.is_dir() else None

        label = TRANSFORMERS.get(name, name)
        candidates.append(
            Candidate(
                slug=name,
                model=SpacyTextcat(best, name=f"{name} (experiment)"),
                description=f"Experiment: spaCy text classifier on the {label} transformer",
                fold_model=fold_model,
            )
        )
    return candidates


# The data sets every model is scored on, in reading order.
SETS = {
    "train": "Train (ingredients of the 800 training lists)",
    "validation": "Validation (100 lists)",
    "test": "Test (the 100 hold-out lists)",
    "unseen": "Unseen ingredients (5-fold cross-validation)",
    "challenge": "Challenge (46 real names not in the data)",
}


def evaluate_candidate(
    candidate: Candidate,
    split: Any,
    folds: list[tuple[list[Example], list[Example]]],
    challenge: list[Example],
    photo_texts: dict[str, str],
) -> dict[str, Any]:
    model = candidate.model
    logger.info("Evaluating %s", model.name)
    sets: dict[str, dict[str, float] | None] = {}
    sets["train"], _ = score_examples(model, examples_of(split.train))
    sets["validation"], _ = score_examples(model, examples_of(split.validation))
    sets["test"], test_predictions = score_examples(model, examples_of(split.holdout))

    fold_labels: list[int] = []
    fold_probabilities: list[float] = []
    per_fold = []
    for k, (train, test) in enumerate(folds):
        fold_model = candidate.fold_model(k, train)
        if fold_model is None:
            per_fold = []
            break
        probabilities = fold_model.predict_proba([e.text for e in test])
        per_fold.append(binary_metrics([e.label for e in test], probabilities))
        fold_labels += [e.label for e in test]
        fold_probabilities += probabilities.tolist()
    sets["unseen"] = binary_metrics(fold_labels, fold_probabilities) if per_fold else None
    sets["challenge"], challenge_predictions = score_examples(model, challenge)

    expectations = load_photo_expectations()
    photos = {
        image: check_photo(model, text, expectations[image]) for image, text in photo_texts.items()
    }
    return {
        "model": model.name,
        "description": candidate.description,
        "sets": sets,
        "lists_accuracy": {
            "validation": list_accuracy(model, split.validation),
            "test": list_accuracy(model, split.holdout),
        },
        "unseen_per_fold": per_fold or None,
        "challenge_predictions": challenge_predictions,
        "test_errors": [p for p in test_predictions if (p["probability"] >= 0.5) != p["label"]],
        "photos": photos or None,
        "size_mb": _size_mb(model),
        "ms_per_ingredient": _ms_per_ingredient(
            model, [e.text for e in examples_of(split.holdout)]
        ),
    }


def ablation(
    split: Any,
    folds: list[tuple[list[Example], list[Example]]],
    challenge: list[Example],
    nlp: Any,
) -> list[dict[str, Any]]:
    """All four feature / classifier combinations, trained per ingredient in the same way."""
    rows = []
    for features in baselines.FEATURES:
        for classifier in baselines.CLASSIFIERS:
            labels: list[int] = []
            probabilities: list[float] = []
            for train, test in folds:
                fold_model = baselines.fit_ablation(train, features, classifier, nlp)
                labels += [e.label for e in test]
                probabilities += fold_model.predict_proba([e.text for e in test]).tolist()
            model = baselines.fit_ablation(examples_of(split.train), features, classifier, nlp)
            rows.append(
                {
                    "features": features,
                    "classifier": classifier,
                    "unseen": binary_metrics(labels, probabilities),
                    "test": score_examples(model, examples_of(split.holdout))[0],
                    "challenge": score_examples(model, challenge)[0],
                }
            )
    return rows


def _prf(m: dict[str, float] | None) -> str:
    return f"{m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f}" if m else "not trained"


def _summary(reports: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Train P / R / F1 | Test P / R / F1 | Test lists accuracy "
        "| Unseen P / R / F1 | Challenge accuracy | Photos: found / missed / extra "
        "| Size (MB) | ms per ingredient |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        photos = r["photos"] or {}
        found = sum(len(p["found"]) for p in photos.values())
        missed = sum(len(p["missed"]) for p in photos.values())
        extra = sum(len(p["extra"]) for p in photos.values())
        challenge = r["sets"]["challenge"]
        lines.append(
            f"| {r['model']} | {_prf(r['sets']['train'])} | {_prf(r['sets']['test'])} "
            f"| {r['lists_accuracy']['test']:.2f} | {_prf(r['sets']['unseen'])} "
            f"| {challenge['accuracy']:.2f} ({challenge['tp'] + challenge['tn']}/{challenge['n']}) "
            f"| {found} / {missed} / {extra} | {r['size_mb']} | {r['ms_per_ingredient']} |"
        )
    lines += [
        "",
        "## Features vs. classifier (all trained per ingredient)",
        "",
        "| Features | Classifier | Unseen P / R / F1 | Test F1 | Challenge accuracy |",
        "|---|---|---|---|---|",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['features']} | {row['classifier']} | {_prf(row['unseen'])} "
            f"| {row['test']['f1']:.2f} | {row['challenge']['accuracy']:.2f} |"
        )
    return "\n".join(lines)


def run(
    transformers: Sequence[str] | None = None,
    photos: bool = True,
    output: Path = REPORTS_DIR,
) -> str:
    """Evaluate every model; write a JSON report each, ablation.json and the figures.

    Returns a summary table for the terminal.
    """
    from sweet_truth.training.figures import draw_all

    nlp = load_nlp()
    split = notebook_split(load_lists())
    folds = unseen_folds()
    challenge = load_challenge()
    photo_texts = read_photos() if photos else {}

    v1 = baselines.train_v1_on_lists(split.train, nlp)
    candidates = [
        Candidate(
            slug="v1",
            model=baselines.V1AsApp(v1, nlp),
            description="v1 retrained on the 800 training lists as in the notebook, used per "
            "ingredient as in the app",
            fold_model=lambda _k, train: baselines.fit_v1_on_ingredients(train, nlp),
        ),
        Candidate(
            slug="v2",
            model=baselines.fit_v2(examples_of(split.train)),
            description="TF-IDF on letter chunks of 1 to 6 characters + logistic regression, "
            "trained on the ingredients of the 800 training lists",
            fold_model=lambda _k, train: baselines.fit_v2(train),
        ),
        *_transformer_candidates(transformers),
    ]

    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for candidate in candidates:
        report = evaluate_candidate(candidate, split, folds, challenge, photo_texts)
        if candidate.slug == "v1":
            report["notebook_list_accuracy"] = {
                "validation": baselines.v1_list_accuracy(v1, split.validation, nlp),
                "test": baselines.v1_list_accuracy(v1, split.holdout, nlp),
            }
        (output / f"{candidate.slug}.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)

    ablation_rows = ablation(split, folds, challenge, nlp)
    (output / "ablation.json").write_text(json.dumps(ablation_rows, indent=2) + "\n")
    draw_all(reports, output / "figures")

    return _summary(reports, ablation_rows)
