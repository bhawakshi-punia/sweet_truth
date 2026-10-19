"""Runtime settings, overridable with SWEET_TRUTH_* environment variables."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWEET_TRUTH_", protected_namespaces=())

    # Which model the app uses: "v2" (letter chunks + logistic regression) or "v1" (the
    # original word-level random forest). SWEET_TRUTH_MODEL=v1 switches back.
    model: Literal["v1", "v2"] = "v2"
    # Models are pickles, which run code when loaded: only ever point these at trusted files.
    v1_model_path: Path = REPO_ROOT / "models" / "sweet_truth_detector.pkl"
    v2_model_path: Path = REPO_ROOT / "models" / "sweet_truth_v2.joblib"
    web_dir: Path = REPO_ROOT / "web"
    spacy_model: str = "en_core_web_sm"
    # Folder holding eng.traineddata. Left unset, common install locations are searched.
    tessdata_dir: Path | None = None

    # Sites allowed to call the API from a browser, e.g. ["https://bhawakshi-punia.github.io"].
    # Set as JSON in SWEET_TRUTH_CORS_ORIGINS. Empty means same-site only.
    cors_origins: list[str] = []
    # Detector calls allowed per visitor per minute; 0 turns the limit off.
    rate_limit_per_minute: int = 20
    # Behind a hosting proxy (as on Render) the visitor's address arrives in the
    # X-Forwarded-For header. Only trust it when such a proxy is in front of the app.
    trust_forwarded_for: bool = False

    max_text_chars: int = 5_000
    max_upload_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    @property
    def model_path(self) -> Path:
        return self.v2_model_path if self.model == "v2" else self.v1_model_path
