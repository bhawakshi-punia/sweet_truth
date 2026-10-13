"""Data models shared by the core, the API and the CLI."""

from pydantic import BaseModel, Field


class Item(BaseModel):
    """One ingredient found in a piece of text."""

    text: str = Field(description="The ingredient as written in the input")
    normalized: str = Field(description="The cleaned, lemmatised form the model saw")
    start: int = Field(description="Offset of the ingredient in the input text")
    end: int = Field(description="End offset (exclusive)")
    is_sweetener: bool
    confidence: float = Field(ge=0, le=1, description="Model probability of the prediction")


class TextResult(BaseModel):
    text: str
    items: list[Item]
    sweeteners: list[Item]


class Word(BaseModel):
    """One word read from an image, with its position in pixels."""

    text: str
    left: int
    top: int
    right: int
    bottom: int


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ImageResult(BaseModel):
    ocr_text: str = Field(description="Everything read from the image")
    ingredients_text: str = Field(description="The ingredient list; item offsets refer to this")
    heading_found: bool = Field(
        description='Whether the word "Ingredients" was found. If not, all the text was checked'
    )
    items: list[Item]
    sweeteners: list[Item]
    boxes: list[Box] = Field(description="Areas of the image where sweeteners appear")
    image_width: int
    image_height: int


class TextRequest(BaseModel):
    text: str


class Health(BaseModel):
    status: str
    model_version: str
