FROM python:3.12-slim

ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
ARG APP_REVISION=development

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT=120000 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

LABEL org.opencontainers.image.revision=${APP_REVISION}

WORKDIR /app

# Playwright installs native browser dependencies through apt. Keep those
# downloads on Tencent Cloud's Debian mirrors as well.
RUN sed -i \
    -e 's|http://deb.debian.org/debian|https://mirrors.cloud.tencent.com/debian|g' \
    -e 's|https://deb.debian.org/debian|https://mirrors.cloud.tencent.com/debian|g' \
    /etc/apt/sources.list.d/debian.sources

# Dependency and browser layers stay cached while only application code changes.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir app \
    && touch app/__init__.py \
    && python -m pip install --no-cache-dir . \
    && rm -rf app \
    && python -m playwright install --with-deps --only-shell chromium \
    && chmod -R a+rX /ms-playwright

COPY app ./app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
