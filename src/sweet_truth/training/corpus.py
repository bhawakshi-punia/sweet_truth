"""Training corpora in spaCy's binary format (.spacy), written from the labelled ingredients."""

import logging
import random
from collections.abc import Sequence
from pathlib import Path

import spacy
from spacy.tokens import DocBin

from sweet_truth.classifiers import OTHER, SWEETENER
from sweet_truth.config import REPO_ROOT
from sweet_truth.training.data import (
    SPLIT_SEED,
    Example,
    examples_of,
    ingredient_group,
    load_lists,
    notebook_split,
    unseen_folds,
)

logger = logging.getLogger(__name__)

CORPUS_DIR = REPO_ROOT / "corpus"


def write_docbin(examples: Sequence[Example], path: Path) -> None:
    """Save ingredients with their category, ready for `spacy train`."""
    nlp = spacy.blank("en")
    docs = DocBin()
    for example in examples:
        doc = nlp.make_doc(example.text)
        doc.cats = {SWEETENER: float(example.label), OTHER: float(not example.label)}
        docs.add(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    docs.to_disk(path)
    logger.info("Wrote %d ingredients to %s", len(examples), path)


def _hold_back(examples: Sequence[Example], share: float) -> tuple[list[Example], list[Example]]:
    """Keep a share of ingredient groups aside, so a fold can stop training at the right time
    without ever looking at its test ingredients."""
    groups = sorted({ingredient_group(e.text) for e in examples})
    random.Random(SPLIT_SEED).shuffle(groups)
    held = set(groups[: max(1, round(len(groups) * share))])
    train = [e for e in examples if ingredient_group(e.text) not in held]
    dev = [e for e in examples if ingredient_group(e.text) in held]
    return train, dev


def prepare(corpus_dir: Path = CORPUS_DIR, n_folds: int = 5) -> Path:
    """Write every corpus training needs.

    - main/: the notebook's split, ingredient by ingredient (train from the 800 training
      lists, dev from the 100 validation lists). The 100 hold-out lists are never written.
    - folds/<k>/: unseen-ingredient cross-validation (train, dev and test).
    """
    split = notebook_split(load_lists())
    write_docbin(examples_of(split.train), corpus_dir / "main" / "train.spacy")
    write_docbin(examples_of(split.validation), corpus_dir / "main" / "dev.spacy")
    for k, (train, test) in enumerate(unseen_folds(n_folds)):
        fit, dev = _hold_back(train, share=0.15)
        write_docbin(fit, corpus_dir / "folds" / str(k) / "train.spacy")
        write_docbin(dev, corpus_dir / "folds" / str(k) / "dev.spacy")
        write_docbin(test, corpus_dir / "folds" / str(k) / "test.spacy")
    return corpus_dir
