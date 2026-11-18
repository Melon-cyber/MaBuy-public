# 🏛️ MaBuy Infrastructure & Reliability Standards (DevOps & Infra)

> **Статус:** ДЕЙСТВУЮЩИЙ СТАНДАРТ (Утверждено директивой CEO от 19.08.2026)  
> **Зона ответственности:** DevOps & Infrastructure, High-Availability, Database Reliability  
> **Целевой SLA/Uptime:** 99.99% (Zero-Downtime 24/7)

---

## 1. ⚡ Market Data Ingestion & Tick Processing

### 1.1. WebSockets + Lock-Free In-Memory RingBuffer
* **Основной транспорт:** Полный отказ от REST-поллинга тиков в пользу постоянных асинхронных соединений **Multiplexed WebSockets** (`wss://fstream.binance.com/stream?streams=...`).
* **Обработка потока:** 
  * Котировки, тики сделок и срезы глубины стакана (L2 Order Book) обрабатываются **исключительно в оперативной памяти (In-Memory RingBuffer / Deque)**.
  * Буфер хранит последние `N` тиков (по умолчанию `maxlen=30000`) в кольцевом формате без оверхеда на дисковый ввод-вывод.
  * Доступ к срезу тиков для инференса ML-моделей — атомарный и потокобезопасный.

### 1.2. 🚫 ЖЕСТКИЙ АРХИТЕКТУРНЫЙ ЗАПРЕТ (Strict Database Guard)
> [!CAUTION]
> **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** производить запись сырых высокочастотных тиков сделок (`trades`) и L2 стаканов глубины (`depth`) в основную базу данных **SQLite (`mabuy_prod.db`)**!

* **Обоснование:** Высокочастотные операции записи (HFT I/O) вызывают блокировки транзакций (`SQLITE_BUSY`), разрастание WAL-журнала и задержки чтения при обработке запросов пользователей бота и API.
* **Долгосрочная стратегия (Tick Archiving):** Для хранения исторических терабайтов тиков и стаканов в roadmap заложено подключение **ClickHouse**. На текущем этапе весь оперативный анализ и микроструктурные фичи рассчитываются строго `in-memory`.

---

## 2. 🗄️ База Данных & Потоковая Репликация (SQLite WAL + Litestream)

### 2.1. Сфера использования SQLite (`mabuy_prod.db`)
SQLite в режиме **WAL (Write-Ahead Logging)** является единым источником истины (Single Source of Truth) исключительно для транзакционного состояния:
1. Пользователи, их роли, профили риска и язык интерфейса (`users`, `user_risk_profiles`, `user_languages`).
2. Подписки и тарифные уровни (`user_subscriptions`, `daily_usage`).
3. Зашифрованные ключи API бирж (`user_api_keys`).
4. Подтвержденные торговые сигналы и статистика исходов (`verified_signals`).
5. Телеметрия качества моделей и проскальзываний (`fine_tuning_telemetry`).

### 2.2. Потоковая репликация WAL через Litestream
* **RPO (Recovery Point Objective):** `< 1 секунды` (вместо 6-часовых интервалов).
* **RTO (Recovery Time Objective):** `~1.5 секунды` при аварийном развертывании.
* **Принцип работы:** Sidecar-контейнер **Litestream** отслеживает появление новых фреймов в `-wal` файле и непрерывно реплицирует их в защищенное S3-совместимое объектное хранилище (Cloudflare R2 / AWS S3 / MinIO).
* **Шифрование:** Все данные шифруются на стороне клиента перед отправкой в удаленный Vault.

---

## 3. 🛡️ Отказоустойчивость, Watchdog & Self-Healing

### 3.1. Watchdog & IPC Heartbeat (`/tmp/daemon_heartbeat`)
Для предотвращения «тихого зависания» (Silent Freeze) торгового демона на сетевых сокетах или дедлоках:
1. Главный цикл `scripts/auto_trader_daemon.py` при каждой итерации обновляет timestamp в shared-файле `/tmp/daemon_heartbeat`.
2. Docker контейнер `mabuy_daemon` снабжается нативным healthcheck-скриптом:
   ```bash
   python -c "import time, os; t=float(open('/tmp/daemon_heartbeat').read().strip()); exit(0 if time.time()-t < 35 else 1)"
   ```
3. Контейнер-супервизор **Autoheal (`willfarrell/autoheal`)** отслеживает статус `unhealthy` и автоматически мягко перезапускает упавший процесс в течение 5 секунд.

### 3.2. Circuit Breaker биржевых соединений
* При 3 последовательных таймаутах сетевого адаптера биржа переводится в статус `CIRCUIT_OPEN` на 30 секунд с переключением на резервные IP/домены (`api1.binance.com`, `api2.binance.com`, `api3.binance.com`).
* Сделки блокируются предохранителем `DataCircuitBreaker` при обнаружении устаревших свечей (`StaleDataException`).

---

## 4. ⏱️ Профилирование и мониторинг задержек (Pipeline Latency Tracing)

Каждая итерация формирования сигнала и исполнения ордера проходит микросекундный аудит через `ExecutionTracer`:
```python
with tracer.step("1_DataIngestion"): ...
with tracer.step("2_FeatureExtraction"): ...
with tracer.step("3_ModelInference"): ...
with tracer.step("4_RiskGuard"): ...
with tracer.step("5_ExchangeExecution"): ...
with tracer.step("6_SQLiteCommit"): ...
```
* **SLA по времени пайплайна:**
  * Инференс и риск-контроль: `< 25 мс`
  * Исполнение лимитного/рыночного ордера на бирже: `< 70 мс`
  * Предупреждение о деградации (Slow Pipeline Warning): при `Total Latency > 100 мс`.

---

## 5. 📋 Сводка регламентов для разработчиков

| Компонент | Разрешено | Запрещено |
|---|---|---|
| **Тики и Стаканы** | RingBuffer в RAM, потоковая агрегация | Прямая запись в `mabuy_prod.db` |
| **Сигналы и Пользователи** | Транзакции в SQLite с WAL-режимом | Хранение только в RAM без персистентности |
| **Репликация БД** | Litestream S3 streaming + 6h Telegram Vault | Ручное копирование незакрытого `.db` файла |
| **Фоновые Демоны** | Обязательный IPC Heartbeat + Autoheal | Бесконечные неконтролируемые `while True: pass` |
