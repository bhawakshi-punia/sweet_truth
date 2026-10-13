import csv
from pathlib import Path

import pytest
import spacy

import legacy_reference
from sweet_truth.text import Resources, classify_ingredients


def test_flags_sweeteners_with_offsets_into_the_original_text(resources: Resources):
    text = "Glucose, Honey, Salt, Vitamins, Minerals"
    result = classify_ingredients(text, resources)

    assert [item.text for item in result.items] == [
        "Glucose",
        "Honey",
        "Salt",
        "Vitamins",
        "Minerals",
    ]
    assert [item.text for item in result.sweeteners] == ["Glucose", "Honey"]
    for item in result.items:
        assert text[item.start : item.end] == item.text
        assert 0.5 <= item.confidence <= 1


def test_bracketed_text_is_left_out_of_the_ingredient(resources: Resources):
    text = "  Sugar (cane), Emulsifiers [481(i), 471 & 472e], Liquid Glucose."
    result = classify_ingredients(text, resources)

    assert [item.text for item in result.items] == ["Sugar", "Emulsifiers", "Liquid Glucose"]
    for item in result.items:
        assert text[item.start : item.end] == item.text


@pytest.mark.parametrize("text", ["", "   ", "...", "()", ", ,"])
def test_text_with_no_ingredients_gives_no_items(resources: Resources, text: str):
    result = classify_ingredients(text, resources)
    assert result.items == []
    assert result.sweeteners == []


def test_matches_the_original_notebook_implementation(resources: Resources, data_dir: Path):
    """Every one of the 1,000 training lists must give the notebook's sweeteners."""
    with (data_dir / "ingredients_sweeteners_list.csv").open(newline="") as handle:
        lists = [row["Ingredient List"] for row in csv.DictReader(handle)]
    assert len(lists) == 1000

    # The notebooks load the full pipeline; the package drops the parser and NER.
    full_nlp = spacy.load("en_core_web_sm")
    for text in lists:
        expected = legacy_reference.sweeteners_from_list(text, resources.model, full_nlp)
        actual = classify_ingredients(text, resources).sweeteners
        assert [item.normalized for item in actual] == [name for name, _ in expected], text
        assert [item.confidence * 100 for item in actual] == pytest.approx(
            [confidence for _, confidence in expected]
        ), text
