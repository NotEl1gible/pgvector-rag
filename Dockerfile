# Two stages, and the first one exists for a specific reason worth knowing before you deploy
# anything that uses hnswlib: it publishes NO manylinux wheels. pip gets an sdist and has to
# compile C++14, so `pip install hnswlib` on python:*-slim fails with
#
#     RuntimeError: Unsupported compiler -- at least C++11 support is needed!
#
# Installing build-essential in the runtime image would fix it and add ~300 MB of toolchain to
# every deployed container. So the wheels are built once here and the runtime stage installs
# from them with --no-index, which keeps the final image slim and the build reproducible.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --wheel-dir /wheels -r requirements.txt


# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    RAGX_INDEX_DIR=/app/artifacts

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
# --no-index: nothing is fetched here, so the runtime image cannot quietly resolve a different
# version than the one that was compiled and tested in the builder stage.
RUN pip install --upgrade pip \
    && pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY ragx ./ragx
COPY evals ./evals

# The corpus is generated, not shipped: 8 MB of templated prose that is a pure function of
# ragx/corpus.py and a seed. Generating it at build time keeps the image and the repository
# small and makes the committed hash meaningful.
RUN python -m ragx.cli corpus --write

RUN useradd --create-home --uid 10001 ragx && mkdir -p /app/artifacts \
    && chown -R ragx:ragx /app
USER ragx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status \
    == 200 else 1)"

CMD ["uvicorn", "ragx.api:app", "--host", "0.0.0.0", "--port", "8000"]
