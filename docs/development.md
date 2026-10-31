# Разработка и проверяемые результаты

Используйте Python 3.12+ и [uv](https://docs.astral.sh/uv/). Облачные задачи уже изолированы: работайте в существующем checkout, не создавая дополнительный Git worktree без явного запроса.

## Окружение из lockfile

```bash
uv sync --frozen --extra dev --extra research --extra redis --extra observability --extra numba
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright src/
uv run --no-sync pytest
```

`--frozen` не пересоздаёт `uv.lock`; проверка согласованности метаданных и lockfile выполняется через `uv lock --check`. `--no-sync` в следующих командах сохраняет уже выбранный набор extras. Не используйте `uv lock` или обновление зависимостей как часть обычной установки. Изменение диапазонов зависимостей и lockfile — отдельное обозримое изменение с повторной проверкой.

Сборка закрепляет backend в `build-system`; версия uv для CI закреплена в workflow. Чтобы экспортировать переносимое описание окружения PEP 751:

```bash
uv export --frozen --format pylock.toml --no-editable --no-emit-project --output-file pylock.toml
```

Это экспорт зависимостей для потребителей PEP 751, а не второй редактируемый вручную lockfile. Локальный проект исключён: его wheel устанавливается отдельно. Файл следует обновлять из `uv.lock` при изменении зависимостей. Экспорт отражает выбранные extras и маркеры платформы; перед переносом проверьте поддержку целевой платформы потребителем.

## Контракты моделей и проверка сохранённого конверта

`GET /api/v1/models` публикует контракты трёх моделей, включая актуальные JSON Schema. Каталог не загружает необязательные engines; доступность Numba отражает только наличие зависимости в текущем процессе. На всех consumers общей очереди должны оставаться согласованные наборы extras и версии пакета.

Для нового успешного запуска сохраняются прежние manifest и артефакты, а также `envelope.json`. Скопируйте каталог подтверждённой попытки целиком: конверт не включает сами научные массивы. Через API доступны отдельные URL конверта симуляции и анализа:

```bash
curl -o envelope.json http://localhost:8000/api/v1/simulations/TASK_ID/envelope
curl -o analysis_envelope.json http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/envelope
```

Для проверки сохранённого расчёта из Python не требуется HTTP-сервер или ещё доступная запись Redis. Нужны исходный результат и каталог артефактов с сохранёнными относительными путями:

```python
import json
from pathlib import Path

from scientific_api.services.result_envelope_service import read_result_envelope

root = Path("artifacts/runs")
manifest = json.loads((root / "TASK_ID/ATTEMPT_ID/manifest.json").read_text())
envelope = read_result_envelope(manifest["outputs"], artifacts_dir=root)
assert envelope.integrity.status == "verified"
print(envelope.model_dump_json(indent=2))

# Для анализа передайте результат исходной симуляции и идентификатор анализа.
analysis = read_result_envelope(manifest["outputs"], artifacts_dir=root, analysis_id="ANALYSIS_ID")
assert analysis.integrity.status == "verified"
```

Проверяются байты файлов, а не только флаг ранее записанного конверта. При несовпадении возникает `ResultEnvelopeError`; HTTP endpoint переводит её в безопасный ответ с кодом причины. Исторический расчёт без объявления и файла конверта явно возвращает `not_verified`. Для подтверждения нового результата `require_present=True` запрещает такой обход. Эта проверка устанавливает целостность относительно сохранённых метаданных; она не подтверждает авторство и не заменяет физическую валидацию.

При добавлении модели объявите единицы, допущения, область инвариантов, диагностику и требуемые файлы в каталоге; обеспечьте соответствующие scientific/reference tests. JSON Schema не выражает все межполевые физические ограничения: сохраняйте domain-валидацию и проверяйте её через HTTP. Изменение научного поведения требует новой версии модели; изменение метаданных отражается в digest контракта. Конверт сохраняет собственный снимок контракта: последующее чтение проверяет сохранённый digest, а не подменяет снимок текущими метаданными каталога. Старые ответы статуса и `manifest.outputs` не должны меняться ради добавления конверта.

При обновлении остановите старые embedded API и workers до возобновления приёма. Старые процессы не публикуют обязательные конверты; запуск старых и новых исполнителей вперемешку не поддерживается. Сохраняйте существующие тома Redis и артефактов. Обновление не создаёт конверты задним числом и не переписывает ранее сохранённые расчёты.

## Архив жизненного цикла и переносимый RO-Crate

Для новых задач архивирование включено в операции очереди. В Redis архивные записи и lifecycle stream не имеют TTL; частые progress и heartbeat продолжают использовать ограниченный оперативный канал. В памяти архив не исчезает при истечении live TTL, но теряется вместе с процессом. Новых зависимостей для графа и экспорта нет; Redis по-прежнему требует extra `redis`.

Используйте архивные URL, когда старый `/simulations/status/TASK_ID` уже возвращает `404`:

```bash
curl 'http://localhost:8000/api/v1/tasks/TASK_ID/timeline?limit=100'
curl http://localhost:8000/api/v1/provenance/tasks/TASK_ID
curl http://localhost:8000/api/v1/provenance/tasks/TASK_ID/graph
curl -o research-run.zip http://localhost:8000/api/v1/provenance/tasks/TASK_ID/export
python -m zipfile -e research-run.zip extracted-run
```

Следующую страницу timeline запрашивают с `after_id`, равным `next_cursor` предыдущей страницы; учитывайте `has_more`. Это cursor порядка записанных событий, а не запрос физического состояния на заданный шаг. ID артефакта обозначает конкретное происхождение файла и отличается от SHA256 содержимого.

Проверьте экспорт без работающего API, Redis или необязательных библиотек:

```python
import json
from pathlib import Path

from scientific_api.services.result_envelope_service import read_result_envelope

root = Path("extracted-run")
task = json.loads((root / "task.json").read_text())
result = task["record"]
envelope = read_result_envelope(result, artifacts_dir=root, require_present=True)
assert envelope.integrity.status == "verified"
for analysis in task["analyses"]:
    evidence = read_result_envelope(
        result, artifacts_dir=root, analysis_id=analysis["analysis_id"], require_present=True
    )
    assert evidence.integrity.status == "verified"
```

ZIP содержит generic RO-Crate 1.1, граф JSON-LD, архивный снимок задачи, контракт модели и проверенные файлы подтверждённой попытки с зарегистрированными анализами. Не объявляется Workflow Run profile, авторство или лицензия научных данных на основании MIT-лицензии исходного кода. Пределы — `128` файлов, `128 MiB` суммарных несжатых данных и итогового ZIP; отдельные метаданные ограничены `2 MiB`.

Архив связывает manifest SHA256 с фактически подтверждённым результатом. Он не копирует NPZ/PNG/PDF в Redis и не заменяет artifact volume. Сохраняйте вместе AOF/Redis volume, том артефактов и резервные копии. Журнал не обрезается автоматически: оцените рост и согласованную политику retention для связанных задач, анализов и файлов. Не переиспользуйте архивные task IDs после live TTL.

При обновлении остановите приём и все старые consumers общей очереди. Старые workers не записывают lifecycle journal, поэтому смешанное развёртывание не даёт полной истории. Существующие записи могут быть отмечены только как наблюдавшиеся с неполным покрытием; отсутствующие прошедшие события не восстанавливаются. Обновление не переписывает прошлое: checkpoint создаётся только новым worker при включённой настройке.

Создание анализа сохраняет файлы и конверт, затем регистрирует происхождение до ответа `201`. При неоднозначном исходе архивной записи API возвращает `503 ANALYSIS_ARCHIVE_UNAVAILABLE` и сохраняет полные файлы: Redis мог успеть подтвердить запись. Сначала проверьте `/provenance/tasks/TASK_ID`; немедленный повтор POST может создать второй анализ. Неиндексированные файлы при неподтверждённой записи требуют отдельной операционной очистки, без удаления данных уже зарегистрированного анализа.

На одну исходную OU-задачу можно зарегистрировать до `256` анализов сохранённых скоростей. Новый такой анализ сверх предела получает `429 ANALYSIS_ARCHIVE_LIMIT_EXCEEDED` без `Retry-After`: ожидание не освобождает место. Предварительная проверка экономит вычисления, атомарная проверка при регистрации защищает от гонки параллельных запросов. Повторная регистрация того же ID остаётся идемпотентной и не меняет сохранённые данные. Verification и LJ research имеют собственные очереди/пределы. Отдельные пределы графа — `4096` узлов и `16 MiB` ответа; экспорт имеет собственные более строгие ограничения числа файлов и метаданных.

## Checkpoint и восстановление после сбоя

Сохранение включается в окружении вычислительного процесса, без изменения запросов моделей:

```bash
CHECKPOINT_INTERVAL_STEPS=100 uv run --no-sync uvicorn scientific_api.main:app

# Для раздельного worker сохраните прежние Redis URI, namespace и ARTIFACTS_DIR.
EXECUTION_MODE=worker CHECKPOINT_INTERVAL_STEPS=100 \
  REDIS_URI=redis://localhost:6379/0 uv run --no-sync python -m scientific_api.worker
```

По умолчанию `CHECKPOINT_INTERVAL_STEPS=0`. Значение задаёт желаемую частоту; адаптивная схема учитывает размер полного состояния и выборок, сохраняя до `32` зарегистрированных checkpoint и до `512 MiB` на задачу. Начальный шаг `0` и конечный шаг входят в этот бюджет. Память сохранённых траекторий, рабочая память и дисковые файлы checkpoint — разные ограничения; `512 MiB` не означает, что worker потребляет столько RAM.

Compose передаёт эту настройку embedded API и отдельному worker. После согласованного обновления процессов включите сохранение, например командой `CHECKPOINT_INTERVAL_STEPS=100 docker compose up --build`; для раздельных ролей добавьте `-f docker-compose.yml -f compose.worker.yml` перед `up`.

Состояние публикуется атомарно после полного шага и сбора выборки. Оно включает PCG64, массивы и caches интегратора, префиксы выборок и cursor. LJ дополнительно сохраняет силы, U/virial, список соседей и счётчики работы. Конечный checkpoint предшествует terminal RDF, чтобы его расход не удвоился после продолжения. Поддерживаются OU, Langevin, LJ NVE/NVT и NumPy/Numba; выбор backend не меняется при восстановлении.

После потери аренды или кооперативной остановки новый worker получает только последний зарегистрированный checkpoint. Каталог, который успел записаться без подтверждения owner-fenced регистрации, не сканируется и не выбирается автоматически. При отсутствии зарегистрированных состояний остаётся прежний retry с seed. При повреждении или несовместимости используются `CHECKPOINT_INTEGRITY_FAILED` и `REPRODUCIBILITY_CONFLICT`, а не скрытый полный пересчёт.

Посмотрите сохранённые состояния и скачайте выбранный checkpoint:

```bash
curl http://localhost:8000/api/v1/simulations/TASK_ID/checkpoints
curl http://localhost:8000/api/v1/simulations/TASK_ID/checkpoints/CHECKPOINT_ID
curl -o checkpoint.json \
  http://localhost:8000/api/v1/simulations/TASK_ID/checkpoints/CHECKPOINT_ID/artifacts/manifest
curl -o checkpoint-state.npz \
  http://localhost:8000/api/v1/simulations/TASK_ID/checkpoints/CHECKPOINT_ID/artifacts/state
curl -X POST http://localhost:8000/api/v1/simulations/TASK_ID/resume \
  -H 'Content-Type: application/json' -d '{}'
```

Список возвращает зарегистрированные метаданные, а detail/download заново проверяют файлы. Ручное resume продолжает failed/cancelled самостоятельную задачу с прежними параметрами и task ID, создавая новый attempt. `{"checkpoint_id":"CHECKPOINT_ID"}` допускает только последнее зарегистрированное состояние. Повтор уже принятого активного продолжения не создаёт дубликат. Completed задачи и дети ансамбля вручную не переоткрываются; действует предел `32` ручных продолжений. Подробные HTTP-коды — в [справочнике API](api_reference.md).

Fingerprint проверяет state ABI, физический код, версии Python/NumPy/backend/LLVM, CPU/libc и float64/RNG policy. Это консервативная граница: простого наличия того же seed недостаточно. Изменение текста README или API не должно само по себе блокировать продолжение. Роль API не подтверждает совместимость удалённого CPU; проверку перед продолжением выполняет worker. Несовместимость не исправляйте отключением проверки fingerprint в production.

Проверки сравнивают непрерывный запуск с остановленным и восстановленным по научным массивам, выборкам, наблюдаемым и LJ counters. UUID, время создания, manifests и упаковочные байты NPZ/ZIP сравниваются отдельно и могут отличаться. На этом этапе продолжение не переносит работу между произвольными backend или архитектурами CPU и не восстанавливает состояние на незаписанном шаге.

Сохраняйте том артефактов вместе с Redis AOF и его томом; одних JSON-дескрипторов в Redis недостаточно. При обновлении остановите старые workers и запускайте согласованные версии/extras. Не удаляйте зарегистрированные checkpoint, на которые ссылаются ожидающие продолжения. Не подтверждённые остатки после сбоя требуют отдельной операционной очистки, без автоматического признания их корректными checkpoint. RO-Crate остаётся пакетом научных результатов с пределом `128 MiB`: checkpoint lineage входит в метаданные, а state/manifest checkpoint скачиваются явно по отдельным URL.

## Независимые verification workers

Проверка результата использует отдельную очередь и пул. Роли размещаются явно:

| `EXECUTION_MODE` | HTTP | Симуляции | Проверки | LJ research |
| --- | --- | --- | --- | --- |
| `embedded` | да | локальный пул | отдельный пул | отдельный пул |
| `api` | да | нет | нет | нет |
| `worker` | нет | да | нет | нет |
| `verification-worker` | нет | нет | да | нет |
| `research-worker` | нет | нет | нет | да |

Для раздельного запуска используйте общие `REDIS_URI`, `REDIS_NAMESPACE`, `ARTIFACTS_DIR` и согласованные версии/extras. Проверяющий не требует MLflow:

```bash
EXECUTION_MODE=verification-worker REDIS_URI=redis://localhost:6379/0 \
  VERIFICATION_WORKER_COUNT=1 uv run --no-sync python -m scientific_api.verification_worker

# Локальный стек API + simulation/verification/research workers + Redis.
docker compose -f docker-compose.yml -f compose.worker.yml up --build
```

Самостоятельный verifier потребляет только очередь проверок. В embedded три очереди имеют отдельные пулы в одном процессе: планируйте суммарную CPU/память, а не только `WORKER_COUNT`. CLI проверяющего не обслуживает HTTP, завершается при неожиданной остановке своего consumer loop и обрабатывает SIGINT/SIGTERM. Перезапуск обеспечивает supervisor/Compose; при неоднозначном commit уже подтверждённый сертификат не заменяется ошибкой.

Создайте проверку и опрашивайте `check_url` из ответа `202`:

```bash
curl -X POST http://localhost:8000/api/v1/simulations/TASK_ID/verifications \
  -H 'Content-Type: application/json' -d '{"rerun":false}'
curl http://localhost:8000/api/v1/verifications/VERIFICATION_ID
curl -o certificate.json http://localhost:8000/api/v1/verifications/VERIFICATION_ID/certificate
```

`rerun=false` проверяет целостность закреплённого снимка и поддерживаемые независимые наблюдаемые/инварианты. `rerun=true` дополнительно запрашивает ограниченный повтор с прежними параметрами, seed и backend в отдельном временном каталоге. Отсутствие нужной реализации, несовместимость сравнения или превышение бюджета отражаются в checks, без скрытой подмены backend. Это не проверка произвольного другого seed или статистической эквивалентности.

`completed` — состояние исполнения, а не вердикт: прочитайте `passed`/`failed`/`inconclusive`/`unsupported`, отдельные checks и limitations. Исходные задача, envelope, graph и RO-Crate не меняются. Связь с поздними свидетельствами доступна через `/simulations/TASK_ID/verifications`, независимо от истечения live TTL источника. Повтор POST создаёт новое задание; предела `32` сохранённые проверки на источник ожидание не снимает.

Сохраните SHA256 из записи проверки или заголовка `X-Content-SHA256` отдельно от файла. Проверка скачанного свидетельства из checkout:

```bash
uv run --no-sync python scripts/validate_verification_certificate.py certificate.json \
  --sha256 RECORDED_CERTIFICATE_SHA256
```

Команда проверяет канонический JSON, его digest, структуру и согласованность итогового вердикта с checks. Она не выполняет научные проверки повторно, не читает исходные NPZ и не удостоверяет подпись/автора. Сертификат имеет `authenticity="unsigned"` и описывает `recorded_artifacts_at_verification_time`; позднее изменение исходных файлов не изменяет историческое свидетельство. Для оценки сегодняшних байтов запросите новую проверку.

Храните Redis AOF/том и резервные копии: сертификат с SHA256 сохраняется в отдельной подтверждённой записи verification queue, не файлом в source attempt. Память остаётся процессной и недолговечной. Очередь применяет общий lease/max-attempt policy и owner fencing, а приём — ограничения незавершённых работ и сохранённых проверок на источник. Новые зависимости для проверки/JSON-сертификатов не требуются.

## LJ research: условная неопределённость и сравнение запусков

Новая research-очередь получает архивные LJ-задачи и вычисляет отдельный отчёт без повторной симуляции. Базовая установка NumPy достаточна: extra `research` по-прежнему относится к Matplotlib/PNG/PDF, а не к запуску этого worker.

```bash
EXECUTION_MODE=research-worker REDIS_URI=redis://localhost:6379/0 \
  RESEARCH_WORKER_COUNT=1 uv run --no-sync python -m scientific_api.research_worker

curl -X POST http://localhost:8000/api/v1/research/lj/analyses \
  -H 'Content-Type: application/json' \
  -d '{"kind":"timeseries","task_ids":["LJ_TASK_ID"],"observables":["pressure"],"bootstrap_seed":42,"assume_stationary":false}'
curl http://localhost:8000/api/v1/research/analyses/ANALYSIS_ID
curl -o lj-uncertainty.json http://localhost:8000/api/v1/research/analyses/ANALYSIS_ID/report
```

Приём возвращает `202` и `check_url`. `completed` означает готовность отчёта; `availability` и состояния отдельных наблюдаемых определяют доступность статистической оценки. Короткий ряд может успешно получить отчёт с недоступным интервалом и ясной причиной. Отчёт хранится в отдельной терминальной записи очереди с SHA256, возвращаемым также через `X-Content-SHA256`; исходные manifest, envelope, graph, RO-Crate и verification records не переписываются. После source live TTL доступны отчёт и список `/simulations/TASK_ID/research-analyses` по primary anchor.

Для timeseries используйте достаточно длинное сохранённое production-окно на регулярной сетке. Burn-in задаётся физическим временем, без возврата кадров до исходного `equilibration_steps`; дополнительный нерегулярный конечный кадр исключается. IAT имеет конвенцию фактора `g>=1`, ESS равен `n/g`. Основной Student-t CI использует batch-means SE `SD(complete_batch_means)*sqrt(b/n)`; IAT-based `SD*sqrt(g/n)` сохраняется отдельной диагностикой. Центр использует все выбранные кадры, остаток неполного блока исключается только из оценки дисперсии и явно учитывается в metadata. Диагностика не доказывает стационарность или смешивание.

SE и основной интервал требуют явного `assume_stationary=true`, минимум `128` кадров, `20` эффективных отсчётов и `8..50` полных длинных блоков, подходящего хвоста корреляций и отсутствия обнаруженного тренда. Автоматический `b=max(ceil(8*g),ceil(n/50),ceil(n**(1/3)))`; слабая зависимость и приближённая нормальность batch means остаются предположениями. Вторичный moving-block percentile bootstrap помечен exploratory: для его уровня `0.99` нужно минимум `2000` draws, однако нехватка bootstrap-tail resolution не отключает основной Student-t CI. Nominal pointwise CI не покрывает ошибку модели, finite-step/size/cutoff, RDF или все observables одновременно. Частный bootstrap seed сохраняется и не меняет физический RNG. Фиксированная калибровка и её ограничения — в [отчёте валидации](validation/lj-uncertainty.md).

Для replicas запросите как минимум два distinct-seed источника с согласованными параметрами и окнами. Одно среднее целой траектории — одна статистическая единица. Student-t CI требует как минимум восьми приближённо нормальных независимых средних; secondary bootstrap остаётся exploratory. Нельзя удалить failed/cancelled запуски из заранее требуемого набора и назвать остаток тем же исследованием.

Для sensitivity сформируйте `2..8` matched-seed групп, меняя только `time_step` или `cutoff_radius` и сохраняя одинаковые физические окна. Парные SE/интервалы характеризуют наблюдаемый mean contrast, а не ошибку предельного шага или бесконечного cutoff. Это исследование чувствительности, а не автоматическое доказательство сходимости или калибровка вещества.

Первый task ID является primary anchor для списка и лимита `32` сохранённых отчётов; повтор POST создаёт другой анализ. Допустимы до `50` источников, `512 MiB` суммарных исходных артефактов и `1 MiB` закреплённого запроса/готового отчёта. `MAX_RESEARCH_QUEUE_SIZE` ограничивает незавершённые работы, `RESEARCH_WORKER_COUNT` — consumers на процесс. Lease/retry policy используется прежняя. Сохраните Redis AOF/том и source artifacts; память остаётся процессной. Исторический report GET проверяет собственный digest и не анализирует повторно сегодняшние файлы.

Для скачанного отчёта сохраните digest из заголовка `X-Content-SHA256` или записи статуса и проверьте его отдельно:

```bash
uv run --no-sync python scripts/validate_lj_research_report.py lj-uncertainty.json \
  --sha256 RECORDED_REPORT_SHA256
```

Команда проверяет канонический JSON, digest, структуру и согласованность метаданных. Она не пересчитывает статистику, не доказывает стационарность и не удостоверяет авторство неподписанного отчёта.

В Compose исследовательский worker имеет отключённый HTTP healthcheck, как и другие CLI-workers. Для запуска только его роли используйте `up --detach research-worker`; readiness проверяйте прохождением задания и состоянием процесса. `up --wait` с ожиданием HTTP readiness предназначен для API/Redis, а не для процесса без HTTP.

## Запуск и настройки

```bash
uv run --no-sync uvicorn scientific_api.main:app --reload
```

Настройки задаются переменными окружения или локальным `.env`. Секреты не следует сохранять в репозитории или manifest запуска.

| Переменная | Назначение |
| --- | --- |
| `ARTIFACTS_DIR` | каталог manifest и конечных состояний, по умолчанию `./artifacts/runs` |
| `EXECUTION_MODE` | `embedded` по умолчанию; `api` только HTTP; `worker` только симуляции; `verification-worker` только проверки; `research-worker` только LJ research; все раздельные роли требуют Redis |
| `MAX_QUEUE_SIZE` | ограничение принятых незавершённых заданий, по умолчанию `100` |
| `WORKER_COUNT` | число вычислителей на процесс embedded/worker, `1..32`, по умолчанию `2` |
| `VERIFICATION_WORKER_COUNT` | число проверяющих на процесс embedded/verification-worker, `1..8`, по умолчанию `1` |
| `RESEARCH_WORKER_COUNT` | число LJ research consumers на процесс embedded/research-worker, `1..4`, по умолчанию `1` |
| `MAX_RESEARCH_QUEUE_SIZE` | число незавершённых LJ research заданий в отдельной очереди, `1..10000`, по умолчанию `16` |
| `MAX_VERIFICATION_QUEUE_SIZE` | число незавершённых проверок в отдельной очереди, `1..10000`, по умолчанию `32` |
| `VERIFICATION_RERUN_MAX_PARTICLE_STEPS` | бюджет повторного запуска в проверке, `1..100000000`, по умолчанию `1000000` |
| `VERIFICATION_RERUN_MAX_LJ_PAIR_CHECKS` | бюджет проверок пар LJ при повторе, `1..100000000`, по умолчанию `5000000` |
| `CHECKPOINT_INTERVAL_STEPS` | `0..10000`: `0` по умолчанию отключает сохранение; положительный желаемый интервал ограничивается адаптивным бюджетом |
| `ANALYSIS_WORKER_COUNT` | число одновременных анализов сохранённых скоростей на процесс API, `1..8`, по умолчанию `2` |
| `TASK_TTL_SECONDS` | срок хранения статуса, по умолчанию `3600` секунд |
| `TASK_LEASE_SECONDS` | срок аренды симуляции или проверки Redis, по умолчанию `30` секунд; продлевается соответствующим worker |
| `MAX_TASK_ATTEMPTS` | максимальное число попыток Redis для симуляции или проверки, по умолчанию `3` |
| `SHUTDOWN_TIMEOUT_SECONDS` | ожидание остановки вычислителей, по умолчанию `5` секунд |
| `EVENT_POLL_INTERVAL_SECONDS` | интервал чтения обновлений SSE из backend, `(0, 5]`, по умолчанию `0.25` секунды |
| `SSE_MAX_CONNECTIONS` | число открытых SSE-соединений на процесс API, `1..1024`, по умолчанию `32` |
| `MAX_SIMULATION_ITERATIONS` | серверный предел числа шагов, по умолчанию `10000` |
| `MAX_PARTICLE_STEPS` | предел произведения числа частиц и шагов, по умолчанию `100000000` |
| `REDIS_URI` | подключение общей очереди и статусов Redis; нужен extra `redis` |
| `REDIS_NAMESPACE` | префикс ключей Redis, по умолчанию `scientific-api` |
| `MLFLOW_TRACKING_URI` | URI tracking, пустая строка отключает; нужен extra `tracking` |
| `MLFLOW_EXPERIMENT_NAME` | имя эксперимента MLflow |
| `OTEL_ENABLED` | включение OpenTelemetry; нужен extra `observability` |
| `OTEL_SERVICE_NAME` | имя сервиса, по умолчанию `scientific-api` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | базовый URL OTLP HTTP; к нему добавляются `/v1/traces` и `/v1/metrics` |
| `OTEL_EXPORTER_CONSOLE` | вывод телеметрии в консоль для локальной диагностики |

Установка extra сама по себе не включает интеграцию. Например, для Redis нужны и extra, и доступный сервис:

```bash
uv sync --frozen --extra redis
REDIS_URI=redis://localhost:6379/0 uv run --no-sync uvicorn scientific_api.main:app
```

Для консольной телеметрии:

```bash
uv sync --frozen --extra observability
OTEL_ENABLED=true OTEL_EXPORTER_CONSOLE=true uv run --no-sync uvicorn scientific_api.main:app
```

Включение `OTEL_ENABLED` без endpoint и консольного экспортёра создаёт локальные spans, не печатая их. Метрики: `simulation.queue.wait` и `simulation.execution.duration` в секундах, счётчик `simulation.tasks` по статусам, а также `http.server.request.duration`. HTTP-метрики используют шаблон маршрута, метод и статус, чтобы не создавать отдельную метку для каждого task ID.

Запускайте один Uvicorn worker в режиме памяти. При Redis прочитайте [семантику повторной доставки](architecture.md) и обеспечьте общее хранилище артефактов. Для длительных задач согласуйте lifecycle, таймауты и политику восстановления с настройками развёртывания.

## Выбор NumPy или Numba

Минимальная установка исполняет NumPy, включая старые запросы и сохранённые payload без поля `backend`. Установите extra `numba` в API и каждом worker той же очереди, сохранив остальные нужные extras:

```bash
uv sync --frozen --extra redis --extra numba
```

В одиночном запросе укажите `"backend":"numba"`; в запросе ансамбля — `"parameters":{"initial_temperature":300.0,"num_particles":1000,"num_steps":100,"backend":"numba"}` вместе с остальными полями ансамбля. Явный `numpy` и отсутствующее поле эквивалентны. Неизвестный backend отклоняется с `422`, а отсутствие Numba — с `503` до приёма. Ошибка импорта или компиляции на worker завершает исполнение ошибкой: скрытого fallback нет. Общая очередь не маршрутизирует задачи по установленным на worker библиотекам, поэтому неодинаковые возможности её потребителей не поддерживаются.

API проверяет наличие Numba без её импорта и JIT-компиляции. Первая фактическая работа Numba в процессе включает импорт и компиляцию; последующие используют готовый kernel в памяти. Дисковый кеш отключён (`cache=False`), внутренний параллелизм Numba не включается; число одновременно исполняемых заданий по-прежнему задаёт пул workers. Оба backend генерируют шум NumPy PCG64 по одному шагу и используют float64. В production нет массива шума на все шаги.

Сервис вычисляет энергию и температуру в конце, сохраняя проверки отмены между шагами и прежнюю частоту публикации прогресса. `step()` при прямом вызове модели сохраняет возврат наблюдаемых на каждом шаге; `advance()` отделяет продвижение состояния от редукции. Manifest `execution` сохраняет backend, версии реализации и kernel, CPU/dtype, `fastmath`, параллелизм, RNG policy и sampling наблюдаемых. Воспроизводимость проверяйте внутри фиксированного backend и окружения, а между реализациями — с обоснованным численным допуском.

JAX устанавливается только для отдельных benchmark-экспериментов. `backend="jax"` в API пока отсутствует: CPU-измерение JAX не подтверждает пользу GPU. Для будущего GPU-backend нужны фактическое устройство, float64, измерения cold/warm и передачи данных, а также проверка научного результата.

## Отдельные процессы worker

`embedded` совмещает API и вычислители и остаётся режимом быстрого старта. При обновлении сначала выполните порядок остановки старых процессов из раздела ниже. Для нового раздельного запуска установите extra `redis`, подготовьте доступный Redis и общий каталог артефактов. Запустите следующие команды в разных терминалах с одинаковыми `REDIS_URI`, `REDIS_NAMESPACE` и `ARTIFACTS_DIR`:

```bash
uv sync --frozen --extra redis

# HTTP: принимает задания, но не запускает вычислители.
EXECUTION_MODE=api REDIS_URI=redis://localhost:6379/0 \
  uv run --no-sync uvicorn scientific_api.main:app --port 8000

# Отдельный процесс без HTTP-сервера.
EXECUTION_MODE=worker REDIS_URI=redis://localhost:6379/0 WORKER_COUNT=2 \
  uv run --no-sync python -m scientific_api.worker
```

Число `WORKER_COUNT` действует на один simulation-процесс. Несколько процессов делят очередь Redis, но каждый создаёт собственный ограниченный набор вычислителей. Увеличение числа повторов ансамбля не создаёт столько же новых потоков. Работа отдельного worker с собственной памятью вместо Redis запрещена конфигурацией. MLflow-трекинг симуляций настраивайте в embedded/worker: роли api, verification-worker и research-worker его не включают. Verification и research имеют отдельные счётчики и очереди.

`/health/ready` роли `api` проверяет backend, а не наличие доступного worker; роли embedded/worker дополнительно проверяют локальные циклы потребителей. Контролируйте процесс worker отдельно: у него нет HTTP health endpoint; SIGINT/SIGTERM инициируют корректную остановку, а неожиданный отказ цикла завершает процесс, чтобы его мог перезапустить supervisor. Восстановление потерянной аренды продолжает последний зарегистрированный checkpoint, если он есть; иначе начинает задачу с сохранённого seed. Прогресс сам по себе не является checkpoint состояния NumPy. В роли API не запускаются симуляции, но синхронный анализ сохранённых снимков по-прежнему выполняется в её отдельном ограниченном пуле.

## Docker Compose

Требуется Docker с Compose 2.20+. По умолчанию запускаются API и Redis:

```bash
docker compose up --build
```

В новом или уже остановленном развёртывании, чтобы отделить вычисление от HTTP, примените override:

```bash
docker compose -f docker-compose.yml -f compose.worker.yml up --build
```

Он задаёт API роль `api` и добавляет самостоятельные simulation и verification workers с общими Redis и томом артефактов. Унаследованные HTTP-проверки worker-контейнеров отключены: процессы не слушают HTTP и контролируются через их жизненный цикл. Verification-worker не зависит от MLflow.

По умолчанию образ не включает Numba и JAX. Для Numba соберите оба исполняемых образа с `INSTALL_NUMBA=true`:

```bash
INSTALL_NUMBA=true docker compose -f docker-compose.yml -f compose.worker.yml up --build
```

Для embedded-режима используйте `INSTALL_NUMBA=true docker compose up --build`; при самостоятельной сборке — `docker build --build-arg INSTALL_NUMBA=true -t scientific-api-numba .`. Переменная передаёт build argument, а не переключает backend запроса автоматически. JAX в этот образ не добавляется.

### Обновление существующего развёртывания

**Перед приёмом заданий новым API остановите все consumers предыдущей версии в том же Redis namespace.** Приостановите приём новых заданий, по возможности дождитесь завершения принятых расчётов (drain), затем остановите старые embedded API и самостоятельные workers, включая процессы вне Compose. Worker до четвёртого этапа не понимает `langevin-free-v1` и новый формат результата. Версия до выбора backend не принимает поле `backend` в payload. Ещё более старый вычислитель до SSE/ансамблей также не записывает родительские вклады и не учитывает новый протокол отмены. Совместное исполнение старых и новых consumers не поддерживается.

Для Compose используйте прежнее имя проекта (`-p`/`COMPOSE_PROJECT_NAME`, если оно задавалось) и остановку с сохранением томов:

```bash
docker compose down
docker compose -f docker-compose.yml -f compose.worker.yml up --build
```

Не добавляйте `--volumes`/`-v` к остановке: Redis и исходные артефакты должны сохраниться. Одинаковое имя Compose-проекта позволяет новому запуску подключить прежние именованные тома. Если старый stack запускался с дополнительными Compose-файлами, используйте их и при остановке, чтобы остановить все его сервисы.

Новая версия поддерживает старые сохранённые записи и payload заданий после остановки старых процессов. Поддержка этих данных не разрешает одновременно работающих старых и новых consumers в одной очереди. Альтернатива остановке общей очереди — явно задать новый `REDIS_NAMESPACE` во всех новых API/workers: старые задания останутся в старом namespace и автоматически в новый не перенесутся.

API доступен на `http://127.0.0.1:8000`; `API_PORT=8001 docker compose up --build` меняет опубликованный порт. Данные Redis и артефакты хранятся в подключённых томах. Профиль MLflow включается явно:

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000 docker compose --profile tracking up --build
```

Профиль `proxy` добавляет Nginx на `http://127.0.0.1:8080`. Образ приложения включает extras Redis, tracking и observability, однако функции включаются настройками. Сборка устанавливает настоящий wheel; базовые образы закреплены digest. Compose — локальная конфигурация, а доступ к Docker и проверка образов зависят от среды запуска.

Если корпоративный прокси требует дополнительный доверенный CA, передайте полный доверенный bundle через временный BuildKit secret и существующие переменные прокси:

```bash
docker build --secret id=ca_bundle,src=/path/to/trusted-ca-bundle.pem \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  -t scientific-api .
```

Это необязательная настройка сборки: bundle используется только во время установки зависимостей. TLS-проверка и хеши lockfile остаются включёнными; сертификаты и значения прокси не записываются в Dockerfile.

## Проверка научного ядра и артефактов

Научные тесты должны проверять свойства модели: стационарность распределения, ожидаемые энергию и дисперсию, зависимость стандартной ошибки от размера системы и повторение расчёта с фиксированным seed. Для статистических проверок нужны заранее обоснованные допуски; один удачный пример температуры возле заданной не доказывает корректность модели.

Исследовательский CLI использует то же вычислительное ядро, что и API, и сохраняет графики с данными и описанием запуска. Он не передаёт Python-файл инструменту исполнения notebook. Установите extra `research`, затем запустите:

```bash
uv sync --frozen --extra research
uv run --no-sync python scripts/run_paper_pipeline.py \
  --output-dir /tmp/paper-run --seed 42 --num-points 12 \
  --num-particles 2000 --num-steps 100
```

Каталог вывода должен быть новым или пустым. Дополнительные параметры: `--time-step`, `--temperature-start`, `--temperature-end`; полный список — через `--help`. Пайплайн сохраняет `results.json`, `figure_1_temperature.png` со стандартными ошибками, артефакты отдельных расчётов в `runs/` и общий `manifest.json` с SHA256 файлов. Seed точки равен базовому seed плюс её индекс. Совместимый вход `scripts/generate_figure_1.py` принимает те же параметры.

Для воспроизведения сохраняйте manifest, конечные состояния и графики вместе с исходными параметрами. Проверяйте пересчёт температуры и энергии по `final_state.npz`; сравнивайте научные данные, а не только существование PNG. Полная повторяемость требует того же окружения; размер и метаданные изображения могут изменяться между версиями библиотек.

API хранит файлы в `<task_id>/<attempt_id>/` и публикует артефакты подтверждённой попытки. Прямой вызов исследовательского пайплайна не использует аренду очереди и сохраняет стабильные каталоги точек без случайного `attempt_id`; благодаря этому повторные прогоны с теми же параметрами удобнее сравнивать.

## Координаты, поля и анализ переноса

Модель `langevin-free-v1` исполняется через ту же ограниченную очередь NumPy/Numba и сохраняет координаты независимых двумерных частиц. Запросите достаточно длинное наблюдение для диффузионного режима:

```bash
curl -X POST http://localhost:8000/api/v1/simulations/langevin \
  -H 'Content-Type: application/json' \
  -d '{"initial_temperature":1.0,"bath_temperature":1.0,"num_particles":1000,"mass":1.0,"friction":1.0,"time_step":0.02,"num_steps":1000,"seed":42,"acceleration":[0.1,0.0],"sample_stride":10,"diffusion_fit_start_step":500,"diffusion_fit_end_step":1000}'
```

Дождитесь `completed` через `check_url` либо SSE. Физические координаты и скорости продвигаются совместным точным переходом Ланжевена с коррелированным шумом, а не дополнительным шагом Эйлера для координат. `friction` задаёт скорость релаксации `γ`, `acceleration` — ускорение, а не силу; `initial_temperature` и `bath_temperature` независимы. Начальные координаты всех частиц равны `initial_position`, скорости имеют среднее `initial_mean_velocity` и тепловую дисперсию `T_0/m`. Единицы приведённые с `k_B=1`.

Для периодической области добавьте `"boundary":"periodic","box_size":[5.0,5.0]`. При `unbounded` поле `box_size` не задаётся. Периодическое представление конечных `positions` предназначено для просмотра внутри бокса; MSD использует непрерывные координаты и корректно учитывает многократные пересечения границ.

```bash
curl -o manifest.json http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/manifest
curl -o final_state.npz http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/final_state
curl -o trajectory.npz http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/trajectory
curl -o transport.json http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/transport
```

`final_state.npz` содержит массивы `positions`, `unwrapped_positions`, `velocities` формы `(N, 2)`. `trajectory.npz` сохраняет `positions` формы `(frames, N, 2)` именно как unwrapped-координаты, шаги `sample_steps` int64 и времена `times` float64. Кадры включают шаг `0`, кратные `sample_stride` и конечный шаг без дублирования. Domain-проверка до приёма ограничивает массив координат `64 MiB`, число кадров `1001` и работу `N*steps <= 100_000_000`. Увеличение stride снижает объём записи, но не меняет шаг интегратора и поток RNG. Дополнительная память текущего состояния, анализа и сжатия не входит в лимит самой траектории; учитывайте её при выборе `WORKER_COUNT`.

`transport.json` создаётся автоматически без Matplotlib. Он содержит времена, средние перемещения, сырое MSD, стандартную ошибку сырого MSD, несмещённое центрированное MSD, аналитические кривые и результат подгонки. Поскольку поле создаёт дрейф, оценка диффузии использует центрированное MSD. В 2D без начального температурного перехода эталон равен `4D[t-(1-exp(-γt))/γ]`, где `D=T_b/(mγ)`. При коротких временах движение баллистическое; наклон такого участка нельзя объявлять коэффициентом диффузии.

### Окно подгонки и неопределённость

`diffusion_fit_start_step` и `diffusion_fit_end_step` задают включённый диапазон сохранённых кадров. Если начало не задано, оно выбирается по затуханию переходного процесса: `γt >= max(5, log((1+abs(T_0/T_b-1))/0.01))`. Таким образом, сильно нагретое начальное состояние требует более длинной релаксации. Конец по умолчанию равен конечному шагу. Для доступной оценки нужны не менее трёх кадров и относительное смещение аналитической OLS-оценки на том же окне не больше `1%`.

Подгонка центрированного MSD по времени имеет свободный член; `D_hat = slope/4`. В отчёте сохраняются выбранное окно, число точек, ожидаемая аналитическая оценка и её относительное смещение. Неподходящее окно не уничтожает симуляцию: `diffusion.status="unavailable"`, значения оценки/SE равны `null`, а `reason` объясняет отказ. Окончательный шумный наклон может быть отрицательным; его не подменяют положительным числом.

Для стандартной ошибки применяется delete-one-particle jackknife по целым траекториям. Коррелированные временные кадры не считаются независимыми наблюдениями. Эта SE описывает выборочную неопределённость при независимых частицах, не включает остаточное смещение конечного окна или ошибку физической модели и не означает доверительный интервал. При ансамбле дополнительная SD/SE между независимыми seed характеризует другой уровень неопределённости.

`thermal_temperature` вычисляется по скоростям относительно среднего потока с нормировкой `2(N-1)` и массой `m`. `kinetic_energy` включает и поток, и тепловое движение; рост энергии под полем сам по себе не означает нагрев. Теоретическая ошибка тепловой температуры равна `T_v(t)/sqrt(N-1)`, где `T_v(t)=T_b+(T_0-T_b)exp(-2γt)`. Старые формулы `ideal-gas-ou-v1` не подставляются в новый результат автоматически.

### Повторный анализ сохранённой траектории

Следующий пример проверяет SHA256 и ограниченный формат NPZ, затем повторяет анализ без создания симулятора. Запускайте его в каталоге с четырьмя скачанными артефактами и установленным пакетом:

```python
import json
from pathlib import Path

from scientific_api.domain.langevin import LangevinParameters
from scientific_api.research.transport_analysis import (
    analyze_transport,
    load_trajectory_state,
)

manifest = json.loads(Path("manifest.json").read_text())
parameters = LangevinParameters(**manifest["parameters"])
record = next(item for item in manifest["artifacts"] if Path(item["path"]).name == "trajectory.npz")
state = load_trajectory_state(
    Path("trajectory.npz"),
    expected_sha256=record["sha256"],
    expected_particles=parameters.num_particles,
    expected_sample_steps=parameters.sampling_steps(),
    expected_time_step=parameters.time_step,
)
report = analyze_transport(state["positions"], state["sample_steps"], parameters=parameters)
with Path("transport-recomputed.json").open("x") as output:
    json.dump(report, output, indent=2, allow_nan=False)
print(report["diffusion"])
```

Сохраняйте исходный manifest и NPZ вместе с пересчитанным JSON. Если выбираете новое окно анализа, фиксируйте изменённые параметры и происхождение нового отчёта отдельно от исходного результата. Endpoint анализа распределений скоростей `/simulations/{task_id}/analyses` относится только к `ideal-gas-ou-v1`: его нулевая гауссовская/рэлеевская теория не применяется к потоку новой модели.

Для ансамбля задайте `"model":"langevin-free-v1"` и те же параметры без `seed` внутри `parameters`; фактические seed сохраняет координатор. Суммарный бюджет несжатых координат всех реплик — `512 MiB` дополнительно к `64 MiB` на реплику и общему бюджету particle-steps. Это предел объёма выбранных координат, а не полного RSS или размера сжатых файлов. Наличие периодических границ или постоянного ускорения не добавляет межчастичные силы, стены, оседание в равновесном барометрическом профиле или LJ-потенциал. Научные проверки новой модели должны сравнивать средние и ковариации с аналитикой, баллистический и диффузионный пределы, периодические пересечения и восстановление `D`; benchmark прежнего OU-ядра не доказывает ускорение новой модели.

## Леннард — Джонс: взаимодействующие частицы

Для двумерной периодической системы используется отдельный endpoint и версия модели `lennard-jones-2d-v1`:

```bash
curl -X POST http://localhost:8000/api/v1/simulations/lennard-jones \
  -H 'Content-Type: application/json' \
  -d '{"num_particles":64,"box_size":[12.0,12.0],"initial_temperature":1.0,"ensemble":"nve","time_step":0.002,"num_steps":1000,"sample_stride":10,"seed":42,"interaction_backend":"verlet"}'
```

NVE использует velocity Verlet. Для NVT задайте `"ensemble":"nvt","bath_temperature":1.0,"friction":1.0`; применяется BAOAB с точным OU-подшагом скоростей. При NVE температура ванны и трение не принимаются. Начальная решётка исключает перекрытия, скорости после Gaussian-выборки имеют нулевой центр масс и точно заданную кинетическую температуру. Такое начальное состояние не является канонической выборкой. `equilibration_steps` обозначает отброшенную часть ряда для описательной статистики, но не устанавливает равновесие автоматически.

Параметры `epsilon`, `sigma`, `mass`, cutoff, skin и бокс задаются явно в приведённых единицах с `k_B=1`. Шаг ограничен диапазоном `[1e-6,0.01]*sigma*sqrt(mass/epsilon)`, поэтому при изменении масштаба нужно согласовать `time_step`. Начальный шаг решётки должен быть не меньше `0.95*sigma`; `cutoff_radius+neighbor_skin` меньше половины кратчайшей стороны бокса. Близкие столкновения, превышение бюджета или нечисловое состояние завершают задание ошибкой, без сглаживания силы или подмены результата.

`interaction_backend="direct"` — прямой эталон по уникальным парам. `"verlet"` строит соседей через периодические ячейки и обновляет список при непрерывном смещении частицы на `skin/2`. Поле `backend` независимо выбирает NumPy/Numba; optional extra должен присутствовать в API и у всех consumers очереди. Силы и энергии используют согласованную shifted-force обрезку. Сдвиг только энергии оставил бы разрыв силы и не соответствует этой модели.

Дождитесь `completed`, затем скачайте четыре файла:

```bash
curl -o manifest.json http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/manifest
curl -o final_state.npz http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/final_state
curl -o trajectory.npz http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/trajectory
curl -o observables.json http://localhost:8000/api/v1/simulations/TASK_ID/artifacts/observables
```

`final_state.npz` содержит wrapped `positions`, непрерывные `unwrapped_positions` и `velocities`. В `trajectory.npz` сохраняются непрерывные координаты, `sample_steps` и `times`; первые и последние состояния включены независимо от stride. Начальная конфигурация, обновление соседей и политика RNG входят в manifest.

`observables.json` содержит `series` для температуры, кинетической/потенциальной/полной энергии, давления и virial; `production` с числом выбранных кадров, арифметическими средними по сохранённым кадрам и выборочными SD; `energy_drift`; конечный двумерный `rdf`. Принудительно включённый последний кадр может нарушать равномерность интервалов, поэтому это не взвешенное по длительности среднее. Потенциальная/полная энергия и давление могут быть отрицательными. Тепловая температура вычитает средний поток и использует `2(N-1)` степеней свободы. Для давления применяется полный кинетический вклад `(2K+W)/(2A)`.

В NVE проверяйте уменьшение энергетической ошибки при уменьшении шага на одном физическом временном интервале. Диагностическая нормировка — `max(abs(E_initial), epsilon*N)`. В NVT энергия обменивается с термостатом: её изменение не является NVE-ошибкой сохранения. Средние/SD по коррелированным кадрам не превращаются в SEM. Для межповторной статистики используйте независимые seed ансамбля с `"model":"lennard-jones-2d-v1"`; теоретическая ошибка температуры независимого OU-снимка у LJ равна `null`.

RDF рассчитывается для конечного снимка до половины кратчайшей стороны бокса. Нормировка использует `N(N-1)/2` и площади двумерных колец `pi*(r_right²-r_left²)`; трехмерные нормировки и tail corrections сюда не переносятся. Получение похожей RDF-кривой само по себе не доказывает равновесие или свойства конкретной жидкости. Свободная формула `D=T/(m*gamma)` также не используется для LJ.

### Независимая каноническая проверка

Помимо производной потенциала, попарного эталона и сходимости NVE, тест `tests/models/test_lj_canonical.py` сравнивает взаимодействующий NVT с отдельной конфигурационной квадратурой. Эталон `tests/reference/lj2d_canonical.json` относится строго к трём частицам в периодическом боксе `6×6`, `T=epsilon=sigma=mass=1`, `r_c=2.5`, shifted-force. После удаления общей трансляции интегрируются четыре относительные координаты на периодических midpoint-сетках разрешений `64`, `96`, `128`.

```bash
uv run --frozen --no-sync python scripts/validate_lj_reference.py \
  --reference tests/reference/lj2d_canonical.json --output /tmp/lj-canonical-new.json
# Отдельная регенерация конфигурационной квадратуры:
uv run --frozen --no-sync python scripts/validate_lj_reference.py \
  --output /tmp/lj-reference-new.json
```

Сравнение использует независимые seed BAOAB-реплик и межповторную SE. Допуски на дискретизацию квадратуры фиксируются отдельно и не выдаются за строгие математические оценки. Это проверка конкретного малого взаимодействующего случая, а не универсальная валидация фазовой диаграммы или реальной жидкости. Версии модели, размерность, cutoff и термостат должны совпадать с эталоном.

### Повторный анализ LJ-артефактов

Следующий код повторяет анализ сохранённых рядов и координат, без исполнения движка. Работайте в каталоге скачанных файлов:

```python
import json
from pathlib import Path

import numpy as np

from scientific_api.domain.lennard_jones import LennardJonesParameters
from scientific_api.research.lj_analysis import analyze_lj_observables
from scientific_api.research.transport_analysis import load_trajectory_state

manifest = json.loads(Path("manifest.json").read_text())
saved = json.loads(Path("observables.json").read_text())
parameters = LennardJonesParameters(**manifest["parameters"])
record = next(item for item in manifest["artifacts"] if Path(item["path"]).name == "trajectory.npz")
trajectory = load_trajectory_state(
    Path("trajectory.npz"),
    expected_sha256=record["sha256"],
    expected_particles=parameters.num_particles,
    expected_sample_steps=parameters.sampling_steps(),
    expected_time_step=parameters.time_step,
)
series = {name: np.asarray(values, dtype=np.float64) for name, values in saved["series"].items()}
recomputed = analyze_lj_observables(
    trajectory["positions"], trajectory["sample_steps"], series, parameters=parameters
)
with Path("observables-recomputed.json").open("x") as output:
    json.dump(recomputed, output, allow_nan=False, indent=2)
```

Сохраняйте исходные параметры, manifest и ряды вместе. SHA256 удостоверяет целостность файла, а не правильность выбранной физической модели. Ограничения: до `4096` частиц, `20_000_000` particle-steps, `100_000_000` проверок пар сил/построения/RDF, `64 MiB` координат и `1001` кадра на реплику; общий координатный бюджет ансамбля — `512 MiB`. Прямой вариант до приёма проверяет точное число `(steps+2)*N*(N-1)/2`; список соседей отслеживает действительный расход во время исполнения. Одновременную память процессов и накопленное дисковое хранилище планируйте отдельно.

Перед обновлением остановите приём и все старые API/workers данного Redis namespace, обновите их до одной версии и сохраните Redis/artifact volumes. Старый consumer не понимает новую модель. Установленные extras должны поддерживать все backend, принимаемые API.

## Анализ сохранённого снимка через API

Этот сценарий обрабатывает подтверждённый `final_state.npz` уже завершённой симуляции. Он не запускает научное ядро повторно и не изменяет исходное состояние или результат расчёта. Сначала создайте симуляцию обычным `POST /api/v1/simulations/monte-carlo`, затем дождитесь `completed` через возвращённый `check_url`.

JSON-анализ доступен в минимальном окружении без Matplotlib:

```bash
curl -X POST http://localhost:8000/api/v1/simulations/TASK_ID/analyses \
  -H 'Content-Type: application/json' \
  -d '{"bins":32,"formats":[]}'
```

Чтобы получить график и PDF, установите extra `research` и явно запросите оба формата:

```bash
uv sync --frozen --extra research
curl -X POST http://localhost:8000/api/v1/simulations/TASK_ID/analyses \
  -H 'Content-Type: application/json' \
  -d '{"bins":32,"formats":["png","pdf"]}'
```

При добавлении extra к окружению разработки сохраните нужные extras (`dev`, `redis`, `observability` и другие) в команде `uv sync`. HTTP-запрос ждёт завершения ограниченной потоковой обработки и возвращает `201` с URL JSON, manifest и запрошенных форматов. Их можно открыть из ответа в `/docs` (**Scientific Analysis → POST → Try it out → Execute**) либо скачать, подставив `analysis_id`:

```bash
curl -o analysis.json http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/artifacts/analysis
curl -o analysis-manifest.json http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/artifacts/manifest
curl -o velocity_distribution.png http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/artifacts/png
curl -o report.pdf http://localhost:8000/api/v1/simulations/TASK_ID/analyses/ANALYSIS_ID/artifacts/pdf
```

Гистограммы `v_x`, `v_y` и модуля скорости сравниваются с двумерной равновесной теорией при заданной температуре. Теоретическая плотность усредняется по бинам; отсутствующая за диапазоном гистограммы масса учитывается отдельно. Описательные моменты используют `ddof=0`; графики и их согласие с теорией не заменяют статистический тест, не дают p-value и не доказывают достижение равновесия. Текущая модель уже начинает траекторию в равновесном распределении.

Генерация проверяет исходный manifest и checksum конечного состояния, модель, размеры и принадлежность подтверждённому `attempt_id`. Отдельный manifest анализа сохраняет provenance исходного снимка, параметры и версию `velocity-distribution-v1`, окружение и SHA256 результатов. Данные публикуются в `ARTIFACTS_DIR/<task_id>/<attempt_id>/analyses/<analysis_id>/`; исходный manifest и NPZ остаются неизменными. Проверки анализа должны покрывать численные моменты, CDF/гистограммы, отсутствие повторной симуляции, работу JSON без Matplotlib, происхождение и целостность файлов, ограничения параллелизма и выдачу запрошенных форматов.

`ANALYSIS_WORKER_COUNT` ограничивает одновременные генерации в одном процессе API, по умолчанию до двух. При занятости сервер возвращает `429` с `Retry-After: 1`. Этот этап не использует постоянную очередь анализов: соединение ожидает `201`, без промежуточного `202`. Отключение клиента не останавливает поток и не освобождает его слот раньше окончания работы. Если ответ с URL потерян, повторный POST создаст другой анализ; идемпотентности и восстановления запроса пока нет. Учитывайте это при настройке таймаутов клиента и прокси.

Прежние URL создания, конвертов и скачивания анализа ограничены сроком существования исходного статуса `TASK_TTL_SECONDS`. После его истечения зарегистрированное происхождение и файлы остаются доступны через архивные provenance/export-маршруты, если сохранены архив и artifact volume. Автоматической очистки файлов анализа по этому TTL нет. Полный контракт ошибок и форматов — в [справочнике API](api_reference.md). Endpoint создания анализирует один сохранённый снимок и сохраняет синхронный контракт `201`, даже при использовании независимых workers, ансамблей и SSE. Дополнительных GPU-backend он не включает.

## Прогресс, отмена и ансамбли

Чтобы наблюдать расчёт без повторного запуска, используйте SSE; для отмены отправьте отдельный POST:

```bash
curl -N http://localhost:8000/api/v1/simulations/TASK_ID/events
curl -X POST http://localhost:8000/api/v1/simulations/TASK_ID/cancel
```

Клиент должен обрабатывать `progress`, `reset`, `expired` и `error`, сохранять последнее `id` и передавать его как `Last-Event-ID` при reconnect. После `reset` замените локальное состояние актуальным снимком: сервер хранит только последние 256 событий. `attempt` — номер попытки; прогресс новой попытки может начинаться с нуля. Последний вычислительный шаг не равен завершению сохранения файлов. SSE-поток закрывается после терминального состояния; его отключение не отменяет задачу. Учитывайте ограничение соединений и отсутствие буферизации у прокси. Точный формат и лимиты — в [справочнике API](api_reference.md).

Ансамбль использует ту же очередь и существующие процессы:

```bash
curl -X POST http://localhost:8000/api/v1/ensembles \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: temperature-study-001' \
  -d '{"model":"ideal-gas-ou-v1","parameters":{"initial_temperature":300.0,"num_particles":1000,"num_steps":100},"replicas":4,"base_seed":42,"failure_policy":"require_all"}'
curl http://localhost:8000/api/v1/ensembles/ENSEMBLE_ID
curl -X POST http://localhost:8000/api/v1/ensembles/ENSEMBLE_ID/cancel
curl -o ensemble-manifest.json http://localhost:8000/api/v1/ensembles/ENSEMBLE_ID/manifest
```

Приём возвращает `202`, родительский ID и URL статуса; manifest доступен после терминального завершения всех детей. Для `replicas=50` не создаётся 50 новых потоков. Ограничиваются и ёмкость всей группы, и суммарное `replicas * num_particles * num_steps`; при недостатке ёмкости вся группа отклоняется с `429`. `Idempotency-Key` длиной `1..128` символов защищает повтор приёма в течение жизни родительской записи; изменённые параметры с тем же ключом дают `409`.

Перед вычислением сохраняется вся последовательность seed из `SeedSequence.spawn`, а не привязка к worker ID или очередности запуска. Для проверки воспроизводимости скачайте manifest с seed, версией NumPy, входами и подтверждёнными вкладами каждого ребёнка. Родитель сохраняет эти вклады отдельно от временных статусов детей; после своего терминального завершения родитель и ключ идемпотентности также имеют TTL. Исходные NPZ хранятся отдельно и требуют общего файлового хранилища.

Проверяйте агрегацию с явными ошибками и отменами. При `require_all` статистика появляется только после успеха всех реплик; для `M=1` выборочные SD и SE равны `null`, для `M>=2` используются `ddof=1` и `SD/sqrt(M)`. Не смешивайте эту SE с предсказанием модели `T/sqrt(N*M)` или с ошибкой одного снимка `T/sqrt(N)`. Параллельные очереди, SSE и ансамбли не меняют физику OU и не являются GPU-планировщиком. Автоматическое восстановление checkpoint допустимо для попытки дочернего задания; ручное переоткрытие ребёнка запрещено, чтобы не менять уже сохранённый вклад ансамбля.

## Benchmark

### LJ: алгоритм взаимодействий и вычислительный backend

Сравните четыре варианта движка при одной плотности и одном seed:

```bash
uv sync --frozen --extra numba
uv run --frozen --no-sync python scripts/benchmark_lj.py \
  --particles 256 --steps 20 --repeats 5 --seed 42 --density 0.4444444444444444 \
  --variants numpy-direct,numpy-verlet,numba-direct,numba-verlet \
  --require-backends --output /tmp/lj-benchmark.json
```

Каждый вариант измеряется в отдельном свежем процессе. Первый вызов включает необязательный импорт/JIT; прогретый запуск заново создаёт решётку и скорости, вычисляет начальную силу, строит/обновляет список соседей, выполняет интеграцию и конечные наблюдаемые. Сравниваются реальные позиции, скорости, силы и наблюдаемые с NumPy/direct (`rtol=atol=1e-10`), отдельно проверяется точный повтор данного варианта. Длина ограничена коротким интервалом: для длинных хаотических траекторий близость после разного округления не гарантируется.

Этот benchmark исключает HTTP, ожидание очереди, RDF, сохранение траектории, tracking и файловый I/O. Его wall time нельзя называть временем полного API-задания. Отчёт включает cold/warm median/p95, фактические проверки пар и перестроения, параметры, версии, CPU/ограничения потоков и хеш кода. RSS относится ко всему дочернему процессу, включая JIT и проверку. Файл вывода должен быть новым; `--require-backends` превращает пропущенную Numba в ошибку. Ускорение оценивается на целевой нагрузке при фиксированной плотности, cutoff, skin и timestep.

[Сохранённые LJ-измерения и исходные JSON](benchmarks/lj/README.md) показывают ускорение NumPy при использовании списка соседей. Пакетный расчёт геометрии кандидатов снизил цену построения ячеек; при `1024` частицах и `100` шагах полный прогретый вызов Numba/Verlet занял около `114 ms` против `206 ms` у Numba/direct. Для малых задач прямой Numba-вариант остаётся быстрее. Исходные scalar-cell измерения сохранены отдельно, поэтому ускорение и границы его применимости можно проверить.

### Реальное исполнение NumPy и Numba

Сравните production-путь с генерацией случайных чисел и сохранением артефактов, установив extras `numba` и нужные инструменты разработки:

```bash
uv run --frozen --no-sync python scripts/benchmark_runtime.py \
  --particles 100000 --steps 100 --repeats 5 --seed 42 \
  --backends numpy,numba --require-backends --output /tmp/runtime-benchmark.json
```

Файл вывода должен быть новым. Каждый backend выполняется в отдельном свежем дочернем процессе. `cold_run` измеряет первый вызов реального сервиса, включая необязательный импорт и JIT, но исключая старт интерпретатора и общие импорты. Прогретый профиль отдельно измеряет инициализацию, эволюцию с PCG64 и конечные наблюдаемые. Отдельный замер RNG служит диагностикой и не прибавляется к времени эволюции: RNG уже входит в неё.

Полный прогретый вызов сервиса включает RNG, эволюцию, наблюдаемые, сбор сведений об исходниках и окружении, SHA256, сжатие NPZ, manifest и `fsync`. Он исключает HTTP, ожидание очереди, Redis и внешнее tracking: это не клиентская задержка от `202` до результата. Времена сохранения и сбора окружения вложены в полное время и также не складываются с ним повторно. Peak RSS относится ко всему дочернему процессу, включая импорты, проверку результата и NumPy-эталон; его нельзя выдавать за объём памяти одного kernel.

Проверка сравнивает фактически сохранённые конечные состояния и наблюдаемые с тем же NumPy seed при `rtol=atol=1e-11`, а повтор внутри backend проверяет точно. Отчёт содержит холодный запуск, прогретые median/p95, параметры нагрузки, версии, CPU и ограничения нативных потоков. `--require-backends` делает отсутствие запрошенного backend ошибкой; без этого пропуск остаётся явным и не считается измерением. При интерпретации учитывайте цену компиляции для коротких задач и сравнивайте одинаковую нагрузку в одинаковом окружении.

[Сохранённые CPU-измерения и исходные JSON](benchmarks/README.md) показывают небольшие различия полного времени: на одной из нагрузок Numba была чуть медленнее NumPy. Эти результаты сохраняют NumPy вариантом по умолчанию; JAX проверена только на CPU и остаётся экспериментом.

### Микроbenchmark и эксперименты JAX

```bash
uv run --no-sync python scripts/benchmark.py \
  --particles 2000 --steps 100 --repeats 5 --workers 2 --seed 42 \
  --backends numpy,threads,processes --output /tmp/benchmark.json
```

Файл вывода должен быть новым. Все backend получают одинаковые партии входов с фиксированными seed и числом расчётов, включая последовательный NumPy. Отчёт содержит среду выполнения, CPU, состояние GIL, настройки нативных потоков, время прогрева, медиану и p95, а также проверку конечных состояний относительно NumPy. Генерация входов исключена из замера; межпроцессная передача и синхронизация JAX включены. Запуск пулов и компиляция учитываются при прогреве. Эти условия нужны для интерпретации результатов.

Для Numba и JAX установите extra `benchmark` и добавьте `numba,jax` к `--backends`; отсутствующие необязательные библиотеки отмечаются как пропущенные. Их доступность зависит от Python и платформы. Численные проверки используют допуск, а не требование побитового совпадения. Данные benchmark не следует выдавать за физическую валидацию.

Для Python без GIL создайте отдельное окружение с соответствующим интерпретатором, проверьте доступность совместимых wheels и повторите ту же команду. Само наличие такого интерпретатора не доказывает отключение GIL после загрузки расширений; сравнивайте поле состояния GIL в отчётах. В репозитории нет универсального обещания ускорения, GPU-совместимости или поддержки всех сборок без GIL.

## Проверка поставки

Проверяйте wheel вне checkout и без `PYTHONPATH=src`, чтобы случайно не импортировать исходники вместо установленного пакета. CI строит дистрибутив и выполняет такую smoke-проверку. Для локальной сборки:

```bash
uv build
```

Перед отправкой изменений проверьте `git diff` и `git status --short`: локальные артефакты, данные Redis/MLflow, окружения и секреты не должны попадать в коммит. Изменения численной модели требуют новой `model_version`, описания предположений и повторной научной проверки.

## Релизы GitHub

Версия в `pyproject.toml`, `src/scientific_api/__init__.py`, `CITATION.cff` и записи пакета в `uv.lock` должна совпадать. Описание релиза хранится в `docs/releases/v<version>.md`; несовместимые изменения модели, единиц или API требуют повышения major-версии.

Подготовка релиза проходит через pull request в защищённую `main`. После слияния и успешного `Validate scientific API` workflow `Publish GitHub release` проверяет точный SHA, согласованность версий и состояние существующего тега, собирает wheel/sdist и публикует GitHub Release с `SHA256SUMS`. Тег создаётся в рамках этого процесса. Повторный запуск не перемещает теги и не перезаписывает опубликованный релиз. При сбое публикации черновик можно безопасно продолжить после исправления причины.

Для ручного повтора: откройте GitHub Actions → Publish GitHub release → Run workflow и выберите `main`. Проверка успешного CI для того же коммита остаётся обязательной. Workflow использует встроенный `GITHUB_TOKEN` с `Contents: write` и `Actions: read`; личный токен для штатной публикации не требуется.

Релизные файлы перед загрузкой проверяются по версии и SHA256. Для локальной проверки сборки используйте `uv build`; публикацию выполняйте только из проверенной `main`. Этот workflow публикует GitHub Release. Интеграции PyPI и Zenodo настраиваются отдельно.
