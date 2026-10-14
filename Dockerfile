# The detector API as run on Render (free web service). Works on any Docker host.
FROM python:3.12-slim

# Tesseract's English language data. The tesserocr wheel brings the library itself.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# Run as an ordinary user, not root.
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

# The host says which port to listen on through $PORT.
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "exec sweet-truth serve --host 0.0.0.0 --port \"$PORT\""]
