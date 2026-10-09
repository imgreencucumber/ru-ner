# ru-ner

NER для русского языка (PER, LOC, ORG). Сравниваю CRF, дообученный ruBERT
(Hugging Face `transformers`) и LLM по качеству, скорости и стоимости.

Проект в разработке.

## Roadmap

- [x] Данные: Collection3, WikiNEuRal-ru, EDA
- [x] Baseline: Natasha, CRF
- [x] Fine-tuning ruBERT
- [x] Инструменты оценки: bootstrap, новые сущности, типы ошибок
- [ ] ONNX + квантизация
- [ ] Сравнение с LLM (DeepSeek, GigaChat, YandexGPT)
- [ ] Сравнительный разбор ошибок
- [ ] FastAPI + Docker, демо

## Запуск

```bash
uv sync
```
