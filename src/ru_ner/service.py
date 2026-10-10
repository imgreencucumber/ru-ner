"""HTTP API for the quantized ruBERT NER model.

Run locally: uv run uvicorn ru_ner.service:app
Settings come from environment variables:
    MODEL_DIR    folder with model_int8.onnx and tokenizer.json (default models/rubert-ner/onnx)
    NUM_THREADS  ONNX Runtime threads per request (default: all cores)
"""

import os
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MAX_TEXT_CHARS = 10_000
MAX_BATCH_TEXTS = 64

Text = Annotated[str, Field(max_length=MAX_TEXT_CHARS)]


class Entity(BaseModel):
    text: str
    type: Literal["PER", "LOC", "ORG"]
    start: int = Field(description="offset of the first character in the input text")
    end: int = Field(description="offset after the last character, text[start:end] == entity")


class ExtractRequest(BaseModel):
    text: Text


class ExtractResponse(BaseModel):
    entities: list[Entity]


class BatchRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_TEXTS)


class BatchResponse(BaseModel):
    results: list[list[Entity]]


def load_model():
    from ru_ner.inference import OnnxBertNER

    threads = int(os.getenv("NUM_THREADS", "0")) or None
    return OnnxBertNER(os.getenv("MODEL_DIR", "models/rubert-ner/onnx"), num_threads=threads)


def create_app(ner=None):
    """ner: anything with extract(text) and extract_batch(texts); the real model by default."""

    @asynccontextmanager
    async def lifespan(app):
        # loaded once at startup, not on every request
        app.state.ner = ner or load_model()
        yield

    app = FastAPI(
        title="ru-ner",
        description="Named entities (PER, LOC, ORG) in Russian text, quantized ruBERT on CPU.",
        lifespan=lifespan,
    )

    page = files("ru_ner").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo_page():
        return page

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # Plain `def`, not `async def`: FastAPI runs it in a thread pool, so slow model inference
    # doesn't block the event loop and several requests are served in parallel.
    @app.post("/extract", response_model=ExtractResponse)
    def extract(request: ExtractRequest):
        return {"entities": app.state.ner.extract(request.text)}

    @app.post("/extract/batch", response_model=BatchResponse)
    def extract_batch(request: BatchRequest):
        return {"results": app.state.ner.extract_batch(request.texts)}

    return app


app = create_app()
