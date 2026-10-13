FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src ./src
RUN uv sync --frozen && useradd --system --no-create-home app
USER app

EXPOSE 8642
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8642/api/config')"
CMD ["/app/.venv/bin/golddiggers", "--host", "0.0.0.0", "--port", "8642"]
