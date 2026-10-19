"""Training data: the labelled ingredients and the synthetic lists built from them.

The lists in `ingredients_sweeteners_list.csv` were made by joining 3 to 7 rows of
`final_ingredient_dataset.csv` with ", ". Splitting each list back into those rows gives every
ingredient in it an exact label, which is what model v2 learns from.
"""

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from sweet_truth.config import REPO_ROOT
from sweet_truth.text import clean_ingredient

DATA_DIR = REPO_ROOT / "data"
INGREDIENTS_CSV = DATA_DIR / "final_ingredient_dataset.csv"
LISTS_CSV = DATA_DIR / "ingredients_sweeteners_list.csv"
CHALLENGE_CSV = DATA_DIR / "challenge_ingredients.csv"
PHOTOS_CSV = DATA_DIR / "photo_sweeteners.csv"

# The seed and proportions of the original notebook's train / validation / hold-out split.
SPLIT_SEED = 101
_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")


@dataclass(frozen=True)
class Example:
    """One ingredient and whether it is a sweetener (1) or not (0)."""

    text: str
    label: int


@dataclass(frozen=True)
class IngredientList:
    """One synthetic list, as text and as the labelled ingredients it was built from."""

    index: int
    text: str
    ingredients: tuple[Example, ...]

    @property
    def label(self) -> int:
        """1 if any ingredient is a sweetener, as in the CSV's "Sweeteners?" column."""
        return int(any(example.label for example in self.ingredients))


@dataclass(frozen=True)
class Split:
    train: list[IngredientList]
    validation: list[IngredientList]
    holdout: list[IngredientList]


def load_ingredients(path: Path = INGREDIENTS_CSV) -> dict[str, int]:
    """Every labelled ingredient, by its exact name."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["Ingredient"]: int(row["Label"]) for row in csv.DictReader(handle)}


def decompose(list_text: str, labels: dict[str, int]) -> list[str]:
    """Split a synthetic list back into the dataset rows it was joined from.

    Some names contain ", " themselves (e.g. "Edible Vegetable Oil (Soyabean, Palmolein)"), so
    pieces are joined back together until they form a known name.
    """
    parts = list_text.split(", ")
    names: list[str] = []
    start = 0
    while start < len(parts):
        for end in range(start + 1, len(parts) + 1):
            candidate = ", ".join(parts[start:end])
            if candidate in labels:
                names.append(candidate)
                start = end
                break
        else:
            raise ValueError(f"Cannot split into known ingredients: {list_text!r}")
    return names


def load_lists(
    lists_path: Path = LISTS_CSV, ingredients_path: Path = INGREDIENTS_CSV
) -> list[IngredientList]:
    labels = load_ingredients(ingredients_path)
    with lists_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lists = []
    for index, row in enumerate(rows):
        names = decompose(row["Ingredient List"], labels)
        examples = tuple(Example(clean_ingredient(name), labels[name]) for name in names)
        item = IngredientList(index, row["Ingredient List"], examples)
        if item.label != int(row["Sweeteners?"] != "nil"):
            raise ValueError(f"List {index} disagrees with its 'Sweeteners?' column")
        lists.append(item)
    return lists


def notebook_split(lists: Sequence[IngredientList]) -> Split:
    """The 80 / 10 / 10 split of `sweetener_detector_from_text.ipynb`, list for list.

    Without stratification, `train_test_split` shuffles positions only, so splitting the list
    objects gives exactly the rows the notebook's X / y split did.
    """
    train, rest = train_test_split(list(lists), test_size=0.2, random_state=SPLIT_SEED)
    validation, holdout = train_test_split(rest, test_size=0.5, random_state=SPLIT_SEED)
    return Split(train=train, validation=validation, holdout=holdout)


def examples_of(lists: Sequence[IngredientList]) -> list[Example]:
    """Every ingredient occurrence in the lists, in order (repeats included)."""
    return [example for item in lists for example in item.ingredients]


def ingredient_group(name: str) -> str:
    """Names that are the same ingredient: case and bracketed notes ignored.

    "MALTITOL (965)" and "Maltitol" share a group, so a fold can never train on one
    and test on the other.
    """
    while _BRACKETED.search(name):
        name = _BRACKETED.sub(" ", name)
    return " ".join(name.lower().replace(".", " ").split())


def unseen_folds(
    n_folds: int = 5, ingredients_path: Path = INGREDIENTS_CSV
) -> list[tuple[list[Example], list[Example]]]:
    """Cross-validation folds whose test ingredients never appear in their training data.

    Each fold keeps about the same share of sweeteners. Returns (train, test) per fold.
    """
    labels = load_ingredients(ingredients_path)
    names = list(labels)
    examples = [Example(clean_ingredient(name), labels[name]) for name in names]
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED)
    return [
        ([examples[i] for i in train], [examples[i] for i in test])
        for train, test in folds.split(
            names, [labels[n] for n in names], [ingredient_group(n) for n in names]
        )
    ]


def load_challenge(path: Path = CHALLENGE_CSV) -> list[Example]:
    """Real ingredient names that are not in the training data, labelled by hand."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Example(clean_ingredient(row["ingredient"]), int(row["label"]))
            for row in csv.DictReader(handle)
        ]


def load_photo_expectations(path: Path = PHOTOS_CSV) -> dict[str, list[str]]:
    """For each sample photo, the sweeteners printed on it (empty if none)."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["image"]: [name.strip() for name in row["sweeteners"].split(";") if name.strip()]
            for row in csv.DictReader(handle)
        }
