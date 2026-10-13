"""Errors the detector raises for inputs it cannot work with."""


class DetectionError(Exception):
    """Base class. `code` is a stable identifier clients can branch on."""

    code = "detection_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidImage(DetectionError):
    code = "invalid_image"


class NoTextFound(DetectionError):
    code = "no_text_found"
