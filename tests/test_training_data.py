import csv

import pytest
from sklearn.model_selection import train_test_split

from sweet_truth.text import split_ingredients
from sweet_truth.training import data


@pytest.fixture(scope="module")
def lists() -> list[data.IngredientList]:
    return data.load_lists()


def test_every_list_splits_into_labelled_ingredients(lists):
    assert len(lists) == 1000
    assert all(3 <= len(item.ingredients) <= 11 for item in lists)
    # load_lists also checks each list's label against its "Sweeteners?" column.
    assert sum(item.label for item in lists) == 491


def test_split_matches_the_original_notebook(lists):
    """Same rows as the notebook's train_test_split(X, y, ..., random_state=101) calls."""
    with data.LISTS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    X = [row["Ingredient List"] for row in rows]
    y = [int(row["Sweeteners?"] != "nil") for row in rows]
    X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    X_val, X_hold, _, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=101)

    split = data.notebook_split(lists)
    assert [item.text for item in split.train] == X_train
    assert [item.text for item in split.validation] == X_val
    assert [item.text for item in split.holdout] == X_hold
    assert (len(split.train), len(split.validation), len(split.holdout)) == (800, 100, 100)


def test_unseen_folds_never_share_an_ingredient():
    folds = data.unseen_folds()
    assert len(folds) == 5
    tested = []
    for train, test in folds:
        train_groups = {data.ingredient_group(example.text) for example in train}
        test_groups = {data.ingredient_group(example.text) for example in test}
        assert not train_groups & test_groups
        assert any(example.label for example in test)  # every fold has sweeteners to find
        tested += test
    assert len(tested) == len(data.load_ingredients())


def test_ingredient_group_ignores_case_and_bracketed_notes():
    assert data.ingredient_group("MALTITOL (965)") == data.ingredient_group("Maltitol")
    assert data.ingredient_group("SWEETENER (955).") == "sweetener"
    assert data.ingredient_group("Gur (Jaggery)") == "gur"


def test_splitter_recovers_the_ingredients_of_a_list(lists):
    """Model v2's splitter, run on a list, gives back the ingredients it was built from.

    The exceptions are dataset rows that hold more than one ingredient, such as
    "fruit and vegetable juices" or "RAISING NTS (...), COCOA SOLIDS (0.7%)", and the row
    "and spices": the splitter separates or trims those, as it would on a real label.
    """
    checked = 0
    for item in lists:
        if any(
            [text for text, _, _ in split_ingredients(example.text)] != [example.text]
            for example in item.ingredients
        ):
            continue
        assert [text for text, _, _ in split_ingredients(item.text)] == [
            example.text for example in item.ingredients
        ]
        checked += 1
    assert checked > 750  # about 1 in 5 lists contains such a row


def test_challenge_ingredients_are_not_in_the_training_data():
    known = {data.ingredient_group(name) for name in data.load_ingredients()}
    challenge = data.load_challenge()
    assert len(challenge) >= 40
    assert {example.label for example in challenge} == {0, 1}
    assert not [e.text for e in challenge if data.ingredient_group(e.text) in known]
