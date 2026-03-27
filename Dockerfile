# Torch-free image: faster-whisper runs on CTranslate2, Silero VAD / Piper / fastembed
# all run on ONNX Runtime. That is what keeps this near 2 GB instead of ~6 GB.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    HF_HOME=/models/hf

WORKDIR /app

# libgomp1 is required by both CTranslate2 and ONNX Runtime; curl is only for HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake every model into the image. A cold container that downloads ~200 MB on its
# first request looks broken, and on a scale-to-zero host it does so repeatedly.
COPY scripts/download_models.py scripts/
COPY app/config.py app/
COPY app/__init__.py app/
RUN python scripts/download_models.py \
 && python -c "from faster_whisper import WhisperModel; \
WhisperModel('base.en', device='cpu', compute_type='int8')"

COPY app/ app/
COPY frontend/ frontend/
COPY eval/ eval/
COPY data/kb/ data/kb/

EXPOSE 8000

# Generous start period: the container ingests the KB and warms all three models
# before it reports healthy, which takes appreciably longer than it serves requests.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker on purpose. Each worker loads its own copy of Whisper, Piper and the
# embedding model, and a WebSocket session is pinned to the worker that accepted it,
# so extra workers multiply memory without helping a single conversation.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
