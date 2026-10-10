"""Gradio demo for Hugging Face Spaces: highlights PER, LOC and ORG in Russian text.

The quantized ONNX model is downloaded from the Hub; set MODEL_DIR to use a local copy instead.
"""

import os

import gradio as gr
from huggingface_hub import snapshot_download

from ru_ner.inference import OnnxBertNER

MODEL_REPO = "{repo_id}"

model_dir = os.getenv("MODEL_DIR") or os.path.join(
    snapshot_download(MODEL_REPO, allow_patterns=["onnx/*"]), "onnx"
)
ner = OnnxBertNER(model_dir, num_threads=2)

EXAMPLES = [
    "Глава Минфина Антон Силуанов встретился в Москве с представителями МВФ.",
    "Председатель правления «Газпрома» Алексей Миллер выступил на форуме в Санкт-Петербурге.",
    "Сборная Аргентины во главе с Лионелем Месси выиграла чемпионат мира в Катаре.",
]


def highlight(text):
    entities = ner.extract(text)
    return {
        "text": text,
        "entities": [{"entity": e["type"], "start": e["start"], "end": e["end"]} for e in entities],
    }


demo = gr.Interface(
    fn=highlight,
    inputs=gr.Textbox(lines=5, label="Текст", placeholder="Вставьте текст на русском"),
    outputs=gr.HighlightedText(
        label="Сущности",
        color_map={"PER": "#2a78d6", "LOC": "#1baf7a", "ORG": "#eb6834"},
    ),
    examples=EXAMPLES,
    title="NER для русского языка",
    description=(
        "Находит людей (PER), места (LOC) и организации (ORG). ruBERT, дообученный на Collection3, "
        "квантизованный в int8 и запущенный на CPU через ONNX Runtime. "
        "Модель: [{repo_id}](https://huggingface.co/{repo_id}), "
        "код: [GitHub]({github_url})."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
