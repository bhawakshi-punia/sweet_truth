"""Runtime settings, overridable with SWEET_TRUTH_* environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWEET_TRUTH_", protected_namespaces=())

    # The model is a pickle, which runs code when loaded: only ever point this at a trusted file.
    model_path: Path = REPO_ROOT / "models" / "sweet_truth_detector.pkl"
    web_dir: Path = REPO_ROOT / "web"
    spacy_model: str = "en_core_web_sm"
    # Folder holding eng.traineddata. Left unset, common install locations are searched.
    tessdata_dir: Path | None = None

    max_text_chars: int = 5_000
    max_upload_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 40_000_000
