"""Text recognition. This is the only module that talks to Tesseract."""

import logging
import os
import threading
from pathlib import Path

import numpy as np
from tesserocr import PSM, RIL, PyTessBaseAPI, iterate_level

from sweet_truth.schemas import Word

logger = logging.getLogger(__name__)

_TESSDATA_CANDIDATES = (
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tessdata",
)


def find_tessdata_dir(configured: Path | None = None, lang: str = "eng") -> Path:
    """Locate the folder holding Tesseract's language data."""
    candidates = [configured, os.environ.get("TESSDATA_PREFIX"), *_TESSDATA_CANDIDATES]
    for candidate in candidates:
        if candidate and (Path(candidate) / f"{lang}.traineddata").is_file():
            return Path(candidate)
    raise FileNotFoundError(
        f"Could not find {lang}.traineddata. Install Tesseract's language data "
        "or set SWEET_TRUTH_TESSDATA_DIR to the folder that contains it."
    )


class OcrEngine:
    """Reads words from images using the Tesseract library in-process."""

    def __init__(self, tessdata_dir: Path | None = None, lang: str = "eng") -> None:
        path = find_tessdata_dir(tessdata_dir, lang)
        logger.info("Loading Tesseract language data from %s", path)
        self._api = PyTessBaseAPI(path=str(path), lang=lang, psm=PSM.AUTO)
        # A Tesseract handle holds one image at a time, so calls are serialised.
        self._lock = threading.Lock()

    def read_words(self, image: np.ndarray) -> list[Word]:
        """Return the words in a single-channel 8-bit image, in reading order."""
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("read_words expects a single-channel uint8 image")
        height, width = image.shape
        words: list[Word] = []
        with self._lock:
            self._api.SetImageBytes(np.ascontiguousarray(image).tobytes(), width, height, 1, width)
            try:
                self._api.Recognize()
                for result in iterate_level(self._api.GetIterator(), RIL.WORD):
                    try:
                        text = result.GetUTF8Text(RIL.WORD).strip()
                    except RuntimeError:  # raised when a position holds no text
                        continue
                    box = result.BoundingBox(RIL.WORD)
                    if text and box:
                        left, top, right, bottom = box
                        words.append(
                            Word(text=text, left=left, top=top, right=right, bottom=bottom)
                        )
            finally:
                self._api.Clear()
        return words

    def close(self) -> None:
        self._api.End()
