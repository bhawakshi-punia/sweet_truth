# The detector API as run on Hugging Face Spaces (Docker SDK, port 7860).
FROM python:3.12-slim

# Tesseract's English language data. The tesserocr wheel brings the library itself.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /home/user/app

# Dependencies first, so code changes don't reinstall them.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY --chown=user src ./src
COPY --chown=user models ./models
COPY --chown=user web ./web
RUN uv sync --frozen

# The page is served from GitHub Pages and calls the detector from there.
ENV SWEET_TRUTH_CORS_ORIGINS='["https://bhawakshi-punia.github.io"]' \
    SWEET_TRUTH_TRUST_FORWARDED_FOR=true

EXPOSE 7860
CMD ["sweet-truth", "serve", "--host", "0.0.0.0", "--port", "7860"]
