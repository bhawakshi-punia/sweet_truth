from pathlib import Path

import pytest

from sweet_truth.config import Settings
from sweet_truth.text import Resources, load_resources

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def resources(settings: Settings) -> Resources:
    return load_resources(settings)


@pytest.fixture(scope="session")
def images_dir() -> Path:
    return REPO_ROOT / "images"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return REPO_ROOT / "data"
