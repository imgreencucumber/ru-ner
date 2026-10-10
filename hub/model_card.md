---
language: ru
license: other
library_name: transformers
pipeline_tag: token-classification
base_model: DeepPavlov/rubert-base-cased
datasets:
- RCC-MSU/collection3
tags:
- ner
- onnx
- russian
metrics:
- f1
- precision
- recall
model-index:
- name: rubert-ner-collection3
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    dataset:
      type: RCC-MSU/collection3
      name: Collection3
      split: test
    metrics:
    - type: f1
      value: 0.976
      name: F1 (entity-level, seqeval)
    - type: precision
      value: 0.972
    - type: recall
      value: 0.980
---

# rubert-ner-collection3

NER для русского языка: PER (люди), LOC (места), ORG (организации).
`DeepPavlov/rubert-base-cased`, дообученный на Collection3 (новости).
Кроме весов PyTorch в репозитории есть квантизованная ONNX-версия для CPU (`onnx/`, 170 МБ).

Код обучения, сравнение с CRF, Natasha и LLM, сервис на FastAPI: {github_url}

## Качество

Тест Collection3: 1922 предложения, 4001 сущность. Метрики на уровне сущностей (`seqeval`):
сущность засчитывается, только если совпали обе границы и тип. В скобках 95% bootstrap-интервал.

| | Precision | Recall | F1 |
|---|---|---|---|
| Все сущности | 0.972 | 0.980 | **0.976** (0.970–0.981) |
| PER | | | 0.998 |
| LOC | | | 0.981 |
| ORG | | | 0.943 |
| Сущности, которых не было в train | | | 0.957 |

ONNX int8 по качеству не отличается от исходной модели: F1 0.977 (0.971–0.982), разница с PyTorch в пределах шума
(парный bootstrap).

Цифры выше получены с декодированием, которое не допускает некорректных BIO-последовательностей
(Витерби с запретом `O → I-X`). Если выбирать метку каждого слова независимо (argmax), F1 0.974.
Стандартный `pipeline` из `transformers` тоже выбирает метки независимо, хотя части слов обрабатывает немного иначе.

Для сравнения на том же тесте: Natasha 0.935, CRF 0.901, GigaChat-3-Ultra с примерами в промпте 0.828.

**Другой домен.** На тесте WikiNEuRal-ru (Википедия) F1 0.799. Разметка WikiNEuRal автоматическая и содержит класс MISC,
которого нет в Collection3, так что цифра занижена: часть «ошибок» приходится на MISC-сущности.

## Использование

С `transformers`:

```python
from transformers import pipeline

ner = pipeline("token-classification", model="{repo_id}", aggregation_strategy="simple")
ner("Глава Минфина Антон Силуанов встретился в Москве с представителями МВФ.")
```

ONNX-версия на CPU, без PyTorch, через пакет из репозитория проекта:

```python
# pip install git+{github_url}
from huggingface_hub import snapshot_download
from ru_ner.inference import OnnxBertNER

path = snapshot_download("{repo_id}", allow_patterns=["onnx/*"])
ner = OnnxBertNER(f"{path}/onnx")
ner.extract("Глава Минфина Антон Силуанов встретился в Москве с представителями МВФ.")
# [{'text': 'Минфина', 'type': 'ORG', 'start': 6, 'end': 13}, ...]
```

## Скорость

Одно предложение за вызов, медиана, Intel Core i3-8100 (4 ядра):

| Вариант | Задержка |
|---|---|
| ONNX int8, CPU | 15.9 мс |
| ONNX fp32, CPU | 25.6 мс |
| PyTorch, CPU | 35.9 мс |
| PyTorch, GTX 1060 | 12.8 мс |

## Обучение

- Данные: Collection3, официальные сплиты train / validation / test (9301 / 2153 / 1922 предложения).
- 4 эпохи, learning rate 3e-5, batch 16, linear warmup 10%, weight decay 0.01, `max_length=128`, seed 42.
- Лучшая эпоха выбиралась по F1 на validation, тест в выборе не участвовал.
- GTX 1060 6GB, 11 минут.
- ONNX: экспорт через `torch.onnx` (dynamo), оптимизация графа ONNX Runtime, динамическая int8-квантизация
  (`per_channel=True`, `reduce_range=True`).

## Ограничения

- Обучена на новостях 2010-х. На текстах другого стиля (соцсети, документы, Википедия) качество ниже.
- Только три типа сущностей. Даты, суммы, должности и т. п. не размечаются.
- Модель повторяет соглашения разметки Collection3: например, кавычки иногда входят в название организации,
  а должность в сущность не входит никогда.
- Около 58% сущностей теста встречались в train, поэтому F1 на тесте частично отражает запоминание.
  F1 на новых сущностях (0.957) — более честная оценка обобщения.

## Лицензия

У базовой модели `DeepPavlov/rubert-base-cased` лицензия не указана, у Collection3 указана `other`,
поэтому и здесь `other`. Перед коммерческим использованием проверьте условия базовой модели и датасета.
Код проекта распространяется по лицензии MIT.
