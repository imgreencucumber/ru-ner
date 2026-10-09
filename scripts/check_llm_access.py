"""Check that LLM API keys work: list available GigaChat models and send one short request
to each model. Costs a few dozen tokens.

Usage: uv run python scripts/check_llm_access.py
"""

import warnings

from ru_ner.llm import GigaChatClient, YandexGPTClient

QUESTION = "Ответь одним словом: столица Франции?"


def check(client):
    try:
        r = client.complete("Ты лаконичный ассистент.", QUESTION)
        print(
            f"  {client.model:20s} OK  {r.text.strip()!r}  "
            f"tokens in/out {r.input_tokens}/{r.output_tokens}  {r.latency_s:.1f}s"
        )
    except Exception as e:  # report and continue with the next model
        print(f"  {client.model:20s} FAILED  {type(e).__name__}: {str(e)[:200]}")


def main():
    # verification is off for GigaChat without a CA bundle, the warning would repeat on every call
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    print("GigaChat")
    available = GigaChatClient().list_models()
    print("  available models:", available)
    for model in available:
        if "embedding" not in model.lower():
            check(GigaChatClient(model=model))

    print("YandexGPT")
    for model in ["yandexgpt-lite", "yandexgpt"]:
        check(YandexGPTClient(model=model))


if __name__ == "__main__":
    main()
