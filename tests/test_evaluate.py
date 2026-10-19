import numpy as np
import pytest

from sweet_truth.training.data import Example, IngredientList
from sweet_truth.training.evaluate import binary_metrics, check_photo, list_accuracy


class Lookup:
    """A stand-in classifier that knows a fixed set of sweeteners."""

    name = "lookup"

    def __init__(self, sweeteners: set[str]) -> None:
        self.sweeteners = {s.lower() for s in sweeteners}

    def predict_proba(self, ingredients):
        return np.array([0.9 if text.lower() in self.sweeteners else 0.1 for text in ingredients])

    def classify_text(self, text):
        pieces = [piece.strip() for piece in text.split(",")]
        return list(zip(pieces, self.predict_proba(pieces).tolist(), strict=True))


def test_binary_metrics_for_the_sweetener_class():
    metrics = binary_metrics([1, 1, 0, 0, 0], [0.9, 0.2, 0.7, 0.1, 0.4])
    assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (1, 1, 1, 2)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == pytest.approx(0.6)


def test_list_accuracy_flags_a_list_when_any_ingredient_is_a_sweetener():
    lists = [
        IngredientList(0, "", (Example("Sugar", 1), Example("Salt", 0))),
        IngredientList(1, "", (Example("Salt", 0), Example("Water", 0))),
        IngredientList(2, "", (Example("Honey", 1),)),
    ]
    assert list_accuracy(Lookup({"sugar"}), lists) == pytest.approx(2 / 3, abs=1e-4)


def test_check_photo_matches_names_loosely():
    model = Lookup({"sugar", "malt extract [44%*]", "salt"})
    result = check_photo(
        model,
        "Malt extract [44%*], SUGAR, Salt, Cocoa",
        ["malt extract", "sugar", "liquid glucose"],
    )
    assert result == {
        "found": ["malt extract", "sugar"],
        "missed": ["liquid glucose"],
        "extra": ["Salt"],
    }
