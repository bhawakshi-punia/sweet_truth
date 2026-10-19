"""Find the sweeteners in a photo of an ingredient list."""

import logging
import re

import cv2
import numpy as np

from sweet_truth.errors import InvalidImage, NoTextFound
from sweet_truth.schemas import Box, ImageResult, Item, Word
from sweet_truth.text import Resources, classify_ingredients

logger = logging.getLogger(__name__)

_INGREDIENTS_HEADING = re.compile(r"ingredients?[\s:;.\-]*", re.IGNORECASE)


def decode_image(data: bytes, max_pixels: int | None = None) -> np.ndarray:
    """Decode uploaded bytes into a grayscale image.

    The image is deliberately not blurred or thresholded: Tesseract binarises it
    itself, and on the sample labels that read better than a fixed threshold.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE) if buffer.size else None
    if image is None:
        raise InvalidImage("This file could not be read as an image.")
    if max_pixels is not None and image.shape[0] * image.shape[1] > max_pixels:
        raise InvalidImage("This image is too large. Try a smaller photo.")
    return image


def join_words(words: list[Word]) -> tuple[str, list[tuple[int, int]]]:
    """Join words with spaces. Also returns each word's (start, end) in the joined text."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        spans.append((cursor, cursor + len(word.text)))
        cursor += len(word.text) + 1
    return " ".join(word.text for word in words), spans


def _same_line(box: Box, word: Word) -> bool:
    overlap = min(box.y2, word.bottom) - max(box.y1, word.top)
    return overlap > 0.5 * min(box.y2 - box.y1, word.bottom - word.top)


def boxes_around(words: list[Word]) -> list[Box]:
    """Boxes covering consecutive words, one box per line of text."""
    boxes: list[Box] = []
    for word in words:
        if boxes and _same_line(boxes[-1], word) and word.left >= boxes[-1].x1:
            last = boxes[-1]
            boxes[-1] = Box(
                x1=last.x1,
                y1=min(last.y1, word.top),
                x2=max(last.x2, word.right),
                y2=max(last.y2, word.bottom),
            )
        else:
            boxes.append(Box(x1=word.left, y1=word.top, x2=word.right, y2=word.bottom))
    return boxes


def locate(item: Item, offset: int, words: list[Word], spans: list[tuple[int, int]]) -> list[Box]:
    """Find the words an ingredient was read from. `offset` is where its text starts."""
    start, end = offset + item.start, offset + item.end
    covered = [
        word
        for word, (word_start, word_end) in zip(words, spans, strict=True)
        if word_start < end and word_end > start
    ]
    return boxes_around(covered)


def ingredients_section(ocr_text: str) -> tuple[int, bool]:
    """Where the ingredient list starts in text read from a pack, and whether a heading was found.

    The list starts after the word "Ingredients". A photo cropped to just the list has no such
    heading, in which case everything that was read is the list.
    """
    heading = _INGREDIENTS_HEADING.search(ocr_text)
    return (heading.end(), True) if heading else (0, False)


def detect_from_image(
    data: bytes, resources: Resources, max_pixels: int | None = None
) -> ImageResult:
    """Read an image of a pack and predict which listed ingredients are sweeteners."""
    if resources.ocr is None:
        raise RuntimeError("Resources were loaded without an OCR engine")
    image = decode_image(data, max_pixels)
    height, width = image.shape

    words = resources.ocr.read_words(image)
    ocr_text, spans = join_words(words)
    if not ocr_text:
        raise NoTextFound("No text could be read from this image.")

    offset, heading_found = ingredients_section(ocr_text)
    ingredients_text = ocr_text[offset:]

    result = classify_ingredients(ingredients_text, resources)
    return ImageResult(
        ocr_text=ocr_text,
        ingredients_text=ingredients_text,
        heading_found=heading_found,
        items=result.items,
        sweeteners=result.sweeteners,
        boxes=[box for item in result.sweeteners for box in locate(item, offset, words, spans)],
        image_width=width,
        image_height=height,
    )
