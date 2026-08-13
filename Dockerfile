# CPU only, no torch. fastembed ships ONNX, which is why this image is hundreds of megabytes
# rather than several gigabytes -- and why `pip install -r requirements.txt` is genuinely
# enough for a reader to reproduce the numbers without a GPU.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    RAGX_INDEX_DIR=/app/artifacts

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ragx ./ragx
COPY evals ./evals

# The corpus is generated, not shipped: 8 MB of templated prose that is a pure function of
# ragx/corpus.py and a seed. Generating it at build time keeps the image and the repository
# small and makes the hash check meaningful.
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
