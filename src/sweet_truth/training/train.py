"""Training model v2: spaCy's trainer, run in-process (no `spacy train` subprocess).

`train_model` is a plain function so it can also be called from a notebook, e.g. on Google
Colab with a GPU.
"""

import logging
from pathlib import Path

from spacy.cli.train import train as spacy_train

from sweet_truth.config import REPO_ROOT
from sweet_truth.training.corpus import CORPUS_DIR

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).parent / "configs" / "textcat_transformer.cfg"
MODELS_DIR = REPO_ROOT / "models" / "v2"

# Short names for the transformers compared. Any Hugging Face model name also works.
TRANSFORMERS = {
    "electra-small": "google/electra-small-discriminator",
    "roberta-base": "roberta-base",
}


def train_model(
    transformer: str = "electra-small",
    corpus: Path = CORPUS_DIR / "main",
    output: Path | None = None,
    gpu_id: int = -1,
    max_steps: int | None = None,
) -> Path:
    """Train one spaCy pipeline and return the folder of its best version.

    `corpus` holds train.spacy and dev.spacy (see `corpus.prepare`). `gpu_id` -1 trains on
    the CPU; 0 uses the first CUDA GPU (e.g. on Colab).
    """
    output = output or MODELS_DIR / transformer
    overrides: dict[str, object] = {
        "paths.train": str(corpus / "train.spacy"),
        "paths.dev": str(corpus / "dev.spacy"),
        "components.transformer.model.name": TRANSFORMERS.get(transformer, transformer),
    }
    if max_steps is not None:
        overrides["training.max_steps"] = max_steps
    logger.info("Training %s on %s into %s", transformer, corpus, output)
    spacy_train(CONFIG, output, use_gpu=gpu_id, overrides=overrides)
    return output / "model-best"


def train_folds(
    transformer: str = "electra-small",
    corpus: Path = CORPUS_DIR / "folds",
    gpu_id: int = -1,
    max_steps: int = 800,
) -> list[Path]:
    """Train one pipeline per unseen-ingredient fold (fewer steps: each fold is small)."""
    folds = sorted(p for p in corpus.iterdir() if p.is_dir())
    return [
        train_model(
            transformer,
            corpus=fold,
            output=MODELS_DIR / transformer / "folds" / fold.name,
            gpu_id=gpu_id,
            max_steps=max_steps,
        )
        for fold in folds
    ]


def export_v2(output: Path | None = None) -> Path:
    """Fit model v2 on the ingredients of all 1,000 lists and save it for the app.

    Like v1's deployed pickle, the final model learns from every labelled list; the
    evaluation numbers come from the version fitted on the 800 training lists only.
    """
    import joblib

    from sweet_truth.config import Settings
    from sweet_truth.training.baselines import fit_v2
    from sweet_truth.training.data import examples_of, load_lists

    output = output or Settings().v2_model_path
    classifier = fit_v2(examples_of(load_lists()))
    joblib.dump(classifier.pipeline, output)
    logger.info("Saved model v2 to %s", output)
    return output
