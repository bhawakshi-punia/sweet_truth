from pathlib import Path

import cv2
import numpy as np
import pytest

from sweet_truth.errors import InvalidImage, NoTextFound
from sweet_truth.image import boxes_around, detect_from_image, join_words
from sweet_truth.schemas import Box, Word
from sweet_truth.text import Resources


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_finds_sweeteners_and_where_they_are(resources: Resources, images_dir: Path):
    data = (images_dir / "children_nutrition_drink.jpg").read_bytes()
    result = detect_from_image(data, resources)

    assert {"malt extract", "sugar", "liquid glucose"} <= {
        item.normalized for item in result.sweeteners
    }
    assert result.boxes
    for box in result.boxes:
        assert 0 <= box.x1 < box.x2 <= result.image_width
        assert 0 <= box.y1 < box.y2 <= result.image_height
    for item in result.items:
        assert result.ingredients_text[item.start : item.end] == item.text


def test_toothpaste_label(resources: Resources, images_dir: Path):
    result = detect_from_image((images_dir / "toothpaste.jpg").read_bytes(), resources)
    assert {"sorbitol", "sodium saccharin"} <= {item.normalized for item in result.sweeteners}
    assert result.boxes


def test_rejects_a_file_that_is_not_an_image(resources: Resources):
    with pytest.raises(InvalidImage):
        detect_from_image(b"definitely not an image", resources)
    with pytest.raises(InvalidImage):
        detect_from_image(b"", resources)


def test_rejects_an_oversized_image(resources: Resources):
    data = _png(np.full((200, 200), 255, np.uint8))
    with pytest.raises(InvalidImage):
        detect_from_image(data, resources, max_pixels=100)


def test_blank_image_has_no_text(resources: Resources):
    with pytest.raises(NoTextFound):
        detect_from_image(_png(np.full((300, 600), 255, np.uint8)), resources)


def test_photo_without_the_ingredients_heading_checks_all_the_text(resources: Resources):
    image = np.full((200, 1300), 255, np.uint8)
    cv2.putText(image, "Wheat Flour, Sugar, Salt", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 0, 6)
    result = detect_from_image(_png(image), resources)

    assert not result.heading_found
    assert result.ingredients_text == result.ocr_text
    assert [item.normalized for item in result.sweeteners] == ["sugar"]
    assert len(result.boxes) == 1


def test_boxes_only_cover_the_sweetener_itself(resources: Resources, images_dir: Path):
    """Sodium Saccharin is a sweetener; the other "sodium" ingredients must not be boxed."""
    result = detect_from_image((images_dir / "toothpaste.jpg").read_bytes(), resources)
    assert [item.text for item in result.sweeteners] == ["Sorbitol", "Sodium Saccharin"]
    assert len(result.boxes) == 2


def test_join_words_reports_where_each_word_is():
    words = [
        Word(text="Sugar,", left=0, top=0, right=50, bottom=20),
        Word(text="Salt", left=60, top=0, right=90, bottom=20),
    ]
    text, spans = join_words(words)
    assert text == "Sugar, Salt"
    assert [text[start:end] for start, end in spans] == ["Sugar,", "Salt"]


def test_boxes_around_gives_one_box_per_line():
    words = [
        Word(text="Invert", left=400, top=50, right=460, bottom=70),
        Word(text="Sugar", left=470, top=52, right=530, bottom=72),
        Word(text="Syrup", left=20, top=90, right=80, bottom=110),  # wrapped to the next line
    ]
    assert boxes_around(words) == [
        Box(x1=400, y1=50, x2=530, y2=72),
        Box(x1=20, y1=90, x2=80, y2=110),
    ]
    assert boxes_around([]) == []
