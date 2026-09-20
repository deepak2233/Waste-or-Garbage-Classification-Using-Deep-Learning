# Inference image. Training happens elsewhere; this serves a finished run.
#
#   docker build -t wasteclf .
#   docker run -p 8000:8000 -v $(pwd)/runs/my-run:/model:ro wasteclf

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgl and libglib are needed by the image decoding stack.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Dependency layer first, so source edits do not invalidate the pip cache.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install ".[serve]"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 wasteclf
USER wasteclf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# Mount a run directory at /model.
CMD ["wasteclf", "serve", "--run", "/model", "--host", "0.0.0.0", "--port", "8000"]
