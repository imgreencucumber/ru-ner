# Деплой в Yandex Cloud Serverless Containers

Контейнер запускается на запрос и останавливается, когда запросов нет, оплата только за время обработки.
Команды для PowerShell. Нужны запущенный Docker и платёжный аккаунт Yandex Cloud.

## 1. CLI и вход

Установка `yc` (официальный скрипт Yandex Cloud), после неё перезапустите терминал:

```powershell
iex (New-Object System.Net.WebClient).DownloadString('https://storage.yandexcloud.net/yandexcloud-yc/install.ps1')
```

Вход и выбор облака и каталога, откроется браузер:

```powershell
yc init
```

## 2. Реестр образов и загрузка образа

```powershell
yc container registry create --name ru-ner
yc container registry configure-docker
$REGISTRY_ID = yc container registry get --name ru-ner --format json --jq .id

docker build -t ru-ner .
docker tag ru-ner cr.yandex/$REGISTRY_ID/ru-ner:v1
docker push cr.yandex/$REGISTRY_ID/ru-ner:v1
```

## 3. Сервисный аккаунт для скачивания образа

Реестр приватный, поэтому контейнеру нужен сервисный аккаунт с правом читать образы.

```powershell
yc iam service-account create --name ru-ner-puller
$SA_ID = yc iam service-account get --name ru-ner-puller --format json --jq .id
$FOLDER_ID = yc config get folder-id
yc resource-manager folder add-access-binding $FOLDER_ID `
  --role container-registry.images.puller --service-account-id $SA_ID
```

## 4. Контейнер

```powershell
yc serverless container create --name ru-ner

yc serverless container revision deploy `
  --container-name ru-ner `
  --image cr.yandex/$REGISTRY_ID/ru-ner:v1 `
  --service-account-id $SA_ID `
  --cores 1 `
  --memory 1GB `
  --concurrency 4 `
  --execution-timeout 30s `
  --environment NUM_THREADS=1
```

- `--memory 1GB`: по умолчанию 128 МБ, а сервису с моделью нужно около 300–400 МБ.
- `--execution-timeout 30s`: по умолчанию 3 секунды, этого не хватит на холодный старт.
- `NUM_THREADS=1`: у экземпляра одно ядро, больше потоков ONNX Runtime только мешают друг другу.
- Порт задавать не нужно: Yandex Cloud передаёт его в переменной `PORT`, сервис её читает.

## 5. Публичный доступ и адрес

```powershell
yc serverless container allow-unauthenticated-invoke --name ru-ner
yc serverless container get --name ru-ner --format json --jq .url
```

По этому адресу открывается демо-страница, документация API — по адресу `/docs`.

## Обновление

Соберите образ с новым тегом (`v2`), загрузите его и повторите `revision deploy` с этим тегом.
