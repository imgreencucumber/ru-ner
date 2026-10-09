"""Thin clients for LLM APIs with one interface: complete(system, user) -> LLMResponse.

Keys are read from environment variables (see .env.example), a local .env file is loaded if present.
"""

import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    # input tokens served from the provider's prompt cache, reported separately by GigaChat
    cached_input_tokens: int = 0


class GigaChatClient:
    """Official gigachat SDK. It exchanges GIGACHAT_CREDENTIALS (the base64 authorization key)
    for an OAuth access token and refreshes the token before it expires (it lives 30 minutes)."""

    def __init__(self, model="GigaChat-2", timeout=60, max_retries=3):
        from gigachat import GigaChat

        ca_bundle = os.getenv("GIGACHAT_CA_BUNDLE") or None
        self.model = model
        self.client = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=model,
            # The API certificate is issued by the Russian Ministry of Digital Development root CA,
            # which is not in the default trust store. If GIGACHAT_CA_BUNDLE points to that root
            # certificate, verification is on. Without it verification is off: a deliberate
            # trade-off for a local experiment, not something to ship.
            verify_ssl_certs=ca_bundle is not None,
            ca_bundle_file=ca_bundle,
            timeout=timeout,
            max_retries=max_retries,
        )

    def list_models(self):
        return [m.id_ for m in self.client.get_models().data]

    def complete(self, system, user, temperature=None):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        start = time.perf_counter()
        r = self.client.chat(payload)
        return LLMResponse(
            text=r.choices[0].message.content,
            input_tokens=r.usage.prompt_tokens,
            output_tokens=r.usage.completion_tokens,
            latency_s=time.perf_counter() - start,
            cached_input_tokens=r.usage.precached_prompt_tokens or 0,
        )


class YandexGPTClient:
    """Yandex Foundation Models REST API, authorized with a service account API key."""

    URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    RETRY_STATUS = (429, 500, 502, 503, 504)

    def __init__(self, model="yandexgpt-lite", timeout=60, max_retries=3):
        folder = os.environ["YANDEX_FOLDER_ID"]
        self.model = model
        self.model_uri = f"gpt://{folder}/{model}/latest"
        self.max_retries = max_retries
        self.http = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}",
                "x-folder-id": folder,
            },
        )

    def complete(self, system, user, temperature=0.0):
        body = {
            "modelUri": self.model_uri,
            "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": "2000"},
            "messages": [{"role": "system", "text": system}, {"role": "user", "text": user}],
        }
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            response = self.http.post(self.URL, json=body)
            if response.status_code not in self.RETRY_STATUS or attempt == self.max_retries:
                break
            time.sleep(2**attempt)
        response.raise_for_status()
        result = response.json()["result"]
        return LLMResponse(
            text=result["alternatives"][0]["message"]["text"],
            input_tokens=int(result["usage"]["inputTextTokens"]),
            output_tokens=int(result["usage"]["completionTokens"]),
            latency_s=time.perf_counter() - start,
        )
