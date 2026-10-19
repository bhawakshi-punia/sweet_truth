from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sweet_truth.api import create_app
from sweet_truth.config import Settings


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    settings = Settings(max_text_chars=200, max_upload_bytes=1024 * 1024)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # The app uses model v2 unless told otherwise.
    assert response.json()["model_version"].startswith("sweet_truth_v2@")


def test_detect_text(client: TestClient):
    response = client.post("/api/detect/text", json={"text": "Glucose, Honey, Salt"})
    assert response.status_code == 200
    body = response.json()
    assert [item["text"] for item in body["sweeteners"]] == ["Glucose", "Honey"]
    assert len(body["items"]) == 3


def test_detect_text_rejects_long_text(client: TestClient):
    response = client.post("/api/detect/text", json={"text": "sugar, " * 100})
    assert response.status_code == 413


def test_detect_text_requires_text(client: TestClient):
    assert client.post("/api/detect/text", json={}).status_code == 422


def test_detect_image(client: TestClient, images_dir: Path):
    data = (images_dir / "corn_flakes.jpg").read_bytes()
    response = client.post("/api/detect/image", files={"image": ("label.jpg", data)})
    assert response.status_code == 200
    body = response.json()
    assert "sugar" in [item["normalized"].lower() for item in body["sweeteners"]]
    assert body["boxes"]
    assert body["image_width"] > 0


def test_detect_image_rejects_a_non_image(client: TestClient):
    response = client.post("/api/detect/image", files={"image": ("label.jpg", b"not an image")})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_image"


def test_detect_image_rejects_a_large_upload(client: TestClient):
    data = b"0" * (1024 * 1024 + 1)
    response = client.post("/api/detect/image", files={"image": ("label.jpg", data)})
    assert response.status_code == 413


def test_frontend_is_served(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Browsers must re-check for a newer page instead of showing a cached one.
    assert response.headers["cache-control"] == "no-cache"
    assert client.get("/app.js").headers["cache-control"] == "no-cache"


def test_detector_calls_are_rate_limited_per_visitor():
    settings = Settings(rate_limit_per_minute=2, trust_forwarded_for=True)
    with TestClient(create_app(settings)) as limited:
        body = {"text": "Sugar, Salt"}
        visitor_a = {"X-Forwarded-For": "203.0.113.1, 10.0.0.1"}
        visitor_b = {"X-Forwarded-For": "203.0.113.2"}
        for _ in range(2):
            assert limited.post("/api/detect/text", json=body, headers=visitor_a).status_code == 200
        blocked = limited.post("/api/detect/text", json=body, headers=visitor_a)
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) >= 1
        # Another visitor is unaffected, and the health check is never limited.
        assert limited.post("/api/detect/text", json=body, headers=visitor_b).status_code == 200
        assert limited.get("/api/health", headers=visitor_a).status_code == 200


def test_browser_calls_allowed_only_from_configured_sites():
    settings = Settings(cors_origins=["https://bhawakshi-punia.github.io"])
    with TestClient(create_app(settings)) as cross_site:
        allowed = cross_site.get(
            "/api/health", headers={"Origin": "https://bhawakshi-punia.github.io"}
        )
        assert allowed.headers["access-control-allow-origin"] == "https://bhawakshi-punia.github.io"
        other = cross_site.get("/api/health", headers={"Origin": "https://example.com"})
        assert "access-control-allow-origin" not in other.headers
