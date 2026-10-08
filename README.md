# ru-ner

NER и поиск персональных данных в русских текстах. Сравниваю CRF, дообученный ruBERT
(Hugging Face `transformers`) и LLM по качеству, скорости и стоимости.

Проект в разработке.

## Roadmap

- [ ] Данные: открытый корпус, BIO-разметка, EDA
- [ ] Baseline: Natasha, CRF
- [ ] Fine-tuning ruBERT
- [ ] Датасет ПДн, предразметка через LLM
- [ ] Сравнение с LLM (DeepSeek, GigaChat, YandexGPT)
- [ ] Дистилляция LLM-разметки в ruBERT
- [ ] ONNX + квантизация
- [ ] FastAPI + Docker, демо

## Запуск

```bash
uv sync
```
