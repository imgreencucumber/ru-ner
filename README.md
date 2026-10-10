# ru-ner

NER для русского языка (PER, LOC, ORG). Сравниваю CRF, дообученный ruBERT
(Hugging Face `transformers`) и LLM по качеству и скорости, для LLM ещё и по расходу токенов.

Проект в разработке.

## Roadmap

- [x] Данные: Collection3, WikiNEuRal-ru, EDA
- [x] Baseline: Natasha, CRF
- [x] Fine-tuning ruBERT
- [x] Инструменты оценки: bootstrap, новые сущности, типы ошибок
- [x] ONNX + квантизация
- [x] Сравнение с LLM (GigaChat, YandexGPT)
- [x] Сравнительный разбор ошибок
- [x] FastAPI + Docker, нагрузочный тест, CI
- [ ] Публикация: модель на HF Hub, демо, итоговый README

## Запуск

```bash
uv sync
```
