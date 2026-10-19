"""`sweet-truth model ...`: build the training data, train model v2 and compare it with v1."""

import logging
from pathlib import Path
from typing import Annotated

import typer

model_app = typer.Typer(
    help="Train and evaluate model v2 (needs the train extra: uv sync --extra train).",
    no_args_is_help=True,
)


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@model_app.command()
def prepare() -> None:
    """Write the training corpora (the notebook's split and the unseen-ingredient folds)."""
    from sweet_truth.training.corpus import prepare as prepare_corpora

    _logging()
    typer.echo(f"Corpora written to {prepare_corpora()}")


@model_app.command()
def train(
    transformer: Annotated[
        str, typer.Option(help="electra-small, roberta-base, or any Hugging Face model name.")
    ] = "electra-small",
    folds: Annotated[
        bool, typer.Option(help="Train one model per unseen-ingredient fold instead.")
    ] = False,
    gpu: Annotated[int, typer.Option(help="GPU number, or -1 for the CPU.")] = -1,
    max_steps: Annotated[int | None, typer.Option(help="Override the config's steps.")] = None,
) -> None:
    """Train model v2 with spaCy."""
    from sweet_truth.training.train import train_folds, train_model

    _logging()
    if folds:
        steps = {} if max_steps is None else {"max_steps": max_steps}
        paths = train_folds(transformer, gpu_id=gpu, **steps)
        typer.echo("Fold models: " + ", ".join(str(p) for p in paths))
    else:
        typer.echo(f"Best model: {train_model(transformer, gpu_id=gpu, max_steps=max_steps)}")


@model_app.command()
def export() -> None:
    """Fit model v2 on all 1,000 lists and save it where the app loads it from."""
    from sweet_truth.training.train import export_v2

    _logging()
    typer.echo(f"Model v2 saved to {export_v2()}")


@model_app.command()
def evaluate(
    transformer: Annotated[
        list[str] | None,
        typer.Option(help="Trained transformers to include (default: every one found)."),
    ] = None,
    photos: Annotated[bool, typer.Option(help="Also check the sample photos (OCR).")] = True,
    output: Annotated[Path | None, typer.Option(help="Folder for the reports.")] = None,
) -> None:
    """Compare v1, the character n-gram baseline and the trained transformers."""
    from sweet_truth.training.evaluate import REPORTS_DIR, run

    _logging()
    summary = run(transformers=transformer, photos=photos, output=output or REPORTS_DIR)
    typer.echo(summary)
