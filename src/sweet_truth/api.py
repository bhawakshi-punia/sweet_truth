"""The web API, which also serves the frontend."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sweet_truth.config import Settings
from sweet_truth.errors import DetectionError
from sweet_truth.image import detect_from_image
from sweet_truth.ratelimit import RateLimiter
from sweet_truth.schemas import Health, ImageResult, TextRequest, TextResult
from sweet_truth.text import Resources, classify_ingredients, load_resources

logger = logging.getLogger(__name__)


class RevalidatedStaticFiles(StaticFiles):
    """Static files that browsers must re-check before reusing a cached copy.

    Without this, phones keep showing an old page after the frontend changes.
    Unchanged files still cost almost nothing: the server answers "not modified".
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def get_resources(request: Request) -> Resources:
    return request.app.state.resources


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def visitor_address(request: Request, settings: Settings) -> str:
    if settings.trust_forwarded_for:
        # The first address is the visitor's as the proxy received it. A visitor can forge
        # it to dodge the limit, which is acceptable for a courtesy limit like this one.
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    limiter: RateLimiter | None = request.app.state.rate_limiter
    if limiter is None:
        return
    wait = limiter.retry_after(visitor_address(request, request.app.state.settings))
    if wait:
        raise HTTPException(
            429,
            "Too many checks in a short time. Please wait a moment and try again.",
            headers={"Retry-After": str(max(1, round(wait)))},
        )


ResourcesDep = Annotated[Resources, Depends(get_resources)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RateLimited = Depends(check_rate_limit)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.resources = load_resources(settings)
        logger.info("Ready, model %s", app.state.resources.model_version)
        yield
        if app.state.resources.ocr is not None:
            app.state.resources.ocr.close()

    app = FastAPI(title="The Sweet Truth Detector", lifespan=lifespan)
    app.state.settings = settings
    app.state.rate_limiter = (
        RateLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute else None
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            max_age=3600,
        )

    @app.exception_handler(DetectionError)
    def detection_error(request: Request, error: DetectionError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": error.code, "message": error.message},
        )

    # The endpoints below are plain functions on purpose: the work is CPU-bound,
    # so FastAPI runs them in its thread pool instead of blocking the event loop.

    @app.get("/api/health")
    def health(resources: ResourcesDep) -> Health:
        return Health(status="ok", model_version=resources.model_version)

    @app.post("/api/detect/text", dependencies=[RateLimited])
    def detect_text(
        body: TextRequest, resources: ResourcesDep, settings: SettingsDep
    ) -> TextResult:
        if len(body.text) > settings.max_text_chars:
            raise HTTPException(
                413,
                f"Text is limited to {settings.max_text_chars} characters.",
            )
        return classify_ingredients(body.text, resources)

    @app.post("/api/detect/image", dependencies=[RateLimited])
    def detect_image(
        image: Annotated[UploadFile, File()], resources: ResourcesDep, settings: SettingsDep
    ) -> ImageResult:
        data = image.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                413,
                f"Images are limited to {settings.max_upload_bytes // (1024 * 1024)} MB.",
            )
        return detect_from_image(data, resources, settings.max_image_pixels)

    if settings.web_dir.is_dir():
        app.mount("/", RevalidatedStaticFiles(directory=settings.web_dir, html=True), name="web")
    else:
        logger.warning("No frontend found at %s, serving the API only", settings.web_dir)

    return app
