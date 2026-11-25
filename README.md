# ECG-based biometric recognition

Курсовой по Signal Analysis and Representations. Сравниваем классические характеристики ЭКГ и представления 1D-CNN для идентификации, верификации и переноса на участников, которых модель не видела при обучении.

Текущая версия реализует пункты 1–10 плана. Стресс-тест на новых участниках — в [ROBUSTNESS.md](ROBUSTNESS.md). Итоги первого полного сравнения — в [RESULTS.md](RESULTS.md). [Протокол и ТЗ](PROTOCOL.md) фиксирует правила эксперимента. [TODO](TODO.md) показывает, что проверено и что остаётся сделать.

[Короткий отчёт для курса (на английском)](REPORT.md)

## Окружение

Python 3.11 или 3.12. Используйте существующее окружение `marketing` или создайте локальное окружение проекта. Для CPU и Apple Silicon достаточно обычной сборки PyTorch. Для NVIDIA выберите сборку CUDA под драйвер на [официальной странице PyTorch](https://pytorch.org/get-started/locally/) и установите её первой в активном окружении.

```sh
conda activate marketing
python -m pip install -e '.[neural,test]'
python -m pytest -q
```

Флаг `--device auto` выбирает CUDA, затем MPS, затем CPU. Явно заданный недоступный GPU приводит к ошибке. Модели сохраняются на CPU и доступны для загрузки на обеих платформах. Пакет `torch` нужен для обучения CNN и повторного кодирования её запросов с помехами.

## Данные

| Источник | ID и сессия | Скачивание |
| --- | --- | --- |
| [ECG-ID 1.0.0](https://physionet.org/content/ecgiddb/1.0.0/) | Папка `Person_*` и день регистрации из `.hea` | Все оригинальные сигналы отведения I. |
| [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | `patient_id`, `ecg_id` и `recording_date` из таблицы | Все записи участников, у которых в метаданных два или более отдельных `ecg_id`. Используем файлы 100 Гц и отведение I. |
| [Heartprint v3](https://doi.org/10.6084/m9.figshare.20105354.v3) | ID из подпапки участника и сессия из родительской папки | Архив Figshare. Для RAR нужен `bsdtar` либо ручная распаковка. |

Команды запускаются из корня репозитория. Данные каждого источника готовятся отдельно:

```sh
python -m ecg download ecg-id
python -m ecg import-archive heartprint /path/to/Heartprint.rar
python -m ecg import-archive ptb-xl /path/to/ptb-xl-1.0.3.zip

for dataset in ecg-id heartprint ptb-xl; do
  python -m ecg prepare "$dataset"
  python -m ecg split "data/prepared/$dataset" --seed 42
done
```

`import-archive` работает без интернета и оставляет исходные архивы на месте. Для PTB-XL из ZIP извлекаются метаданные и только записи 100 Гц участников с повторными `ecg_id`; общая подготовка использует лишь Lead I. Для Heartprint RAR нужен `bsdtar` (на macOS обычно уже установлен). Если распаковываете вручную, положите дерево сессий Heartprint под `data/raw/heartprint/`, а метаданные и `records100/` PTB-XL — под `data/raw/ptb-xl/`. Вместо локального импорта можно попробовать `python -m ecg download <dataset>`.

Проверенные данные проекта:

| Набор | Подготовленные записи | Участники с повторными записями | Участники с разными сессиями или днями |
| --- | ---: | ---: | ---: |
| ECG-ID | 309 | 89 | 20 |
| Heartprint | 1400 | 199 | 197 |
| PTB-XL | 5041 | 2111 | 2042 |

Для основного эксперимента с несколькими сессиями начните с Heartprint. PTB-XL даёт больше участников и подходит для крупной проверки переноса, но в его `known_val` только 20 участников. ECG-ID удобен для быстрого прогона.

В `data/prepared/<dataset>/` лежат `records.csv`, `signals.npz`, `inventory.json`, `excluded.csv`, `known.csv` и `unseen.csv`. Длина исходной записи сохраняется с точностью до округления при ресемплинге. Все записи привязаны к исходному файлу и сессии. Для PTB-XL `expected_source_records` показывает размер целевого поднабора, поэтому можно увидеть неполную загрузку.

## Базовые модели

```sh
python -m ecg baseline data/prepared/ecg-id --protocol known --out runs/ecg-id-pca-known
python -m ecg baseline data/prepared/ecg-id --protocol unseen --out runs/ecg-id-pca-unseen

python -m ecg compare data/prepared/ecg-id ecg-id:Person_01/rec_1 ecg-id:Person_01/rec_2

python -m ecg train data/prepared/ecg-id --protocol known \
  --out runs/ecg-id-cnn-known --device mps --epochs 20
python -m ecg train data/prepared/ecg-id --protocol unseen \
  --out runs/ecg-id-cnn-unseen --device cuda --epochs 20
```

Примеры MPS и CUDA предназначены для соответствующих машин. Используйте `--device cpu --epochs 1` для быстрого теста. PCA и CNN сохраняют `embeddings.npz` с массивами `record_ids` и `embeddings`. PCA дополнительно сохраняет параметры преобразования. CNN сохраняет веса и историю обучения в `training.json`. Метрики из истории обучения описывают только обучающие записи.

## Оценка качества

После `baseline` или `train` передайте папку соответствующей модели в `evaluate`:

```sh
base=runs/benchmark/heartprint/unseen
python -m ecg baseline data/prepared/heartprint --protocol unseen --out "$base/pca"
python -m ecg train data/prepared/heartprint --protocol unseen --device auto --epochs 20 --out "$base/cnn"
python -m ecg evaluate data/prepared/heartprint --protocol unseen --method pca \
  --model-run "$base/pca" --out "$base/eval-pca"
python -m ecg evaluate data/prepared/heartprint --protocol unseen --method cnn \
  --model-run "$base/cnn" --out "$base/eval-cnn"
python -m ecg evaluate data/prepared/heartprint --protocol unseen --method correlation \
  --out "$base/eval-correlation"
python -m ecg evaluate data/prepared/heartprint --protocol unseen --method dtw \
  --max-participants 10 --out "$base/subset10-dtw"
```

Папки PCA и CNN должны содержать свежий запуск: оценка сверяет протокол и ID записей, на которых обучена модель. Быстрые старые запуски из `runs/smoke/` не содержат этого списка и для оценки не подходят. Прямые методы не требуют `--model-run`.

`metrics.json` содержит отдельно validation и test. `*_identification.csv` показывает, какой эталон выбран, а `*_pairs.csv` — какие заявления личности проверены и какие приняты. По умолчанию берём до 10 разных отрицательных кандидатов на запрос, одинаковых для всех методов при одном seed. Для честного сравнения с DTW на подвыборке передавайте тот же `--max-participants` всем методам. Порог считается по validation; если запросов на validation нет, FAR и FRR не выводятся.

## Шум и аугментации

`train --augment` добавляет случайные помехи только при обучении CNN. `evaluate --query-noise` меняет только тестовые запросы, сохраняя эталоны и порог с валидации. Команда поддерживает PCA и CNN. Для CNN повторное кодирование запросов выполняется на CPU, из сохранённых весов. Без новых флагов поведение не меняется.

```sh
prepared=data/prepared/heartprint
run=runs/robustness/heartprint/cnn-aug
python -m ecg train "$prepared" --protocol unseen --device auto --epochs 20 --augment --out "$run"
python -m ecg evaluate "$prepared" --protocol unseen --method cnn --model-run "$run" \
  --out "$run/clean/clean"
python -m ecg evaluate "$prepared" --protocol unseen --method cnn --model-run "$run" \
  --query-noise band-noise --noise-rms 0.50 --out "$run/band-noise/0.50"
```

Для базового сравнения используйте CNN, обученную с теми же параметрами без `--augment`. Повторите оценку для `drift` и `band-noise` на уровнях 0.25 и 0.50. Выводите каждое условие в отдельную папку.

## Работа в команде

- Работайте в отдельных коротких Git-ветках. Код источников и подготовки находится в `ecg/sources.py` и `ecg/preprocess.py`, протокол в `ecg/splits.py`, модели в `ecg/classical.py` и `ecg/neural.py`.
- Для сопоставления моделей используйте одинаковую версию данных, seed и наборы записей. В описании изменения указывайте команду проверки.
- `data/`, `runs/` и окружение хранятся локально и исключены из Git. Для обмена результатами используйте небольшие таблицы с метриками после их появления.
