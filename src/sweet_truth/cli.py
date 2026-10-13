"""Command line interface: `sweet-truth --help`."""

import logging
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from sweet_truth.config import Settings
from sweet_truth.errors import DetectionError
from sweet_truth.image import detect_from_image
from sweet_truth.schemas import Item
from sweet_truth.text import classify_ingredients, load_resources

app = typer.Typer(help="Detect hidden sweeteners in ingredient lists.", no_args_is_help=True)
console = Console()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _show(sweeteners: list[Item]) -> None:
    if not sweeteners:
        console.print("No sweeteners detected.")
        return
    table = Table("Ingredient", "Confidence")
    for item in sweeteners:
        table.add_row(item.text, f"{item.confidence:.0%}")
    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Use 0.0.0.0 to reach it from your phone.")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Run the web app."""
    from sweet_truth.api import create_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def text(ingredients: Annotated[str, typer.Argument(help="An ingredient list.")]) -> None:
    """Find the sweeteners in a typed ingredient list."""
    resources = load_resources(Settings(), with_ocr=False)
    _show(classify_ingredients(ingredients, resources).sweeteners)


@app.command()
def image(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Find the sweeteners in a photo of an ingredient list."""
    settings = Settings()
    resources = load_resources(settings)
    try:
        result = detect_from_image(path.read_bytes(), resources, settings.max_image_pixels)
    except DetectionError as error:
        console.print(f"[red]{error.message}[/red]")
        raise typer.Exit(code=1) from error
    console.print(f"[dim]{result.ingredients_text}[/dim]")
    _show(result.sweeteners)


@app.command()
def evaluate(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Run the detector over every image in a folder and summarise the results."""
    settings = Settings()
    resources = load_resources(settings)
    table = Table("Image", "Sweeteners", "Boxes", "Problem")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for path in paths:
        try:
            result = detect_from_image(path.read_bytes(), resources, settings.max_image_pixels)
        except DetectionError as error:
            table.add_row(path.name, "", "", error.code)
            continue
        names = ", ".join(item.text for item in result.sweeteners)
        table.add_row(path.name, names, str(len(result.boxes)), "")
    console.print(table)
