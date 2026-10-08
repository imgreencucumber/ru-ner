# ru-ner

NER для русского языка (PER, LOC, ORG). Сравниваю CRF, дообученный ruBERT
(Hugging Face `transformers`) и LLM по качеству, скорости и стоимости.

Проект в разработке.

## Roadmap

- [x] Данные: Collection3, WikiNEuRal-ru, EDA
- [ ] Baseline: Natasha, CRF
- [ ] Fine-tuning ruBERT
- [ ] Оценка и разбор ошибок
- [ ] Сравнение с LLM (DeepSeek, GigaChat, YandexGPT)
- [ ] ONNX + квантизация
- [ ] FastAPI + Docker, демо

## Запуск

```bash
uv sync
```
