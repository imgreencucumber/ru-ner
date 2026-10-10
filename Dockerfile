# CPU-only service with the quantized ruBERT. No PyTorch inside: ONNX Runtime + tokenizers.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first: this layer stays cached until pyproject.toml or uv.lock change.
# --no-default-groups installs only the service dependencies, without research and dev groups.
# The uv download cache is mounted only during the build, so it doesn't end up in the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups

COPY models/rubert-ner/onnx/model_int8.onnx models/rubert-ner/onnx/tokenizer.json ./model/
# ONNX Runtime threads per request. More threads = faster single request, but concurrent
# requests then compete for the same cores. On 4 cores 2 threads gave the same single-request
# latency as 4 and almost the best throughput under load (results/service_threads.json).
ENV MODEL_DIR=/app/model \
    NUM_THREADS=2

RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "ru_ner.service:app", "--host", "0.0.0.0", "--port", "8000"]
