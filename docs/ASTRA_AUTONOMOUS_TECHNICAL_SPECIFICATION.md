```markdown
# ТЕХНИЧЕСКОЕ ЗАДАНИЕ (ТЗ): Полная ликвидация заглушек, сквозная интеграция живой телеметрии и институциональная эксплуатация квант-экосистемы MaBuy V5.2

**Система:** MaBuy V5.2 Renaissance Titan  
**Целевой файл:** `docs/ASTRA_AUTONOMOUS_TECHNICAL_SPECIFICATION.md`  
**Версия ТЗ:** 1.0.0  
**Дата:** 2026-09-12T05:55:00Z  
**Статус:** Mandatory Implementation Specification  
**Класс:** Institutional Quantitative Trading Technology / Research Governance / Operational Due Diligence

---

## 0. Нормативное заявление

Данное ТЗ является обязательной спецификацией полной замены всех mockups, placeholders, stubs, фиктивных UI-значений, hardcoded fallback values и недетерминированных имитаций данными, поступающими из реальных источников MaBuy V5.2.

Принцип реализации:

> **No fabricated telemetry. No silently substituted values. No UI fiction. No operational ambiguity.**

Если реальный источник недоступен, API и UI обязаны явно сообщить о недоступности, возрасте последнего валидного значения, коде ошибки и источнике проблемы. Система не имеет права подменять отсутствие данных демонстрационным числом, статическим массивом, нулевым PnL, тестовой позицией или условным статусом `healthy`.

Допустимый cache разрешён только при соблюдении одновременно всех условий:

1. значение ранее было получено от идентифицированного реального источника;
2. значение содержит `source_timestamp`, `received_at`, `cache_age_ms`;
3. значение находится в пределах заданного TTL;
4. UI маркирует значение как `CACHED`, если оно не является fresh;
5. API не подменяет недоступный источник новым вымышленным значением;
6. все cache-операции подлежат SHA-256 audit trail.

---

# 1. МИССИЯ И ЦЕЛИ ТЗ

## 1.1. Миссия

Создать полностью операционную, проверяемую, академически воспроизводимую и институционально пригодную quant-ecosystem MaBuy V5.2, в которой dashboard, API, risk-control, research layer, торговая телеметрия, execution analytics и аудит построены на сквозных реальных источниках данных.

Система должна быть пригодна для:

- эксплуатации исследовательской и paper/live trading инфраструктуры;
- академической защиты дипломной, магистерской или диссертационной работы;
- подготовки материалов для AIMA DDQ, Operational Due Diligence (`ODD`) и Limited Partner review;
- внутреннего Risk Committee review;
- forensic reconstruction любой отображённой метрики;
- воспроизводимого анализа сигналов, сделок, рисков и решений Astra Research Advisory.

## 1.2. Обязательные целевые результаты

| Область | Обязательный результат |
|---|---|
| Dashboard | Все отображаемые показатели получаются из реальных API, SQLite ledger, JSONL advisory ledger либо live market feeds |
| Equity | Equity, margin, free collateral, positions и Unrealized PnL загружаются из Binance Futures через CCXT/официальный API либо явно маркированный допустимый cache |
| Advisories | Используются реальные записи из `models/research_advisories.jsonl`, минимум 45 валидных записей |
| Risk gates | Hurst, funding, OFI, circuit breaker и статус риска вычисляются или считываются из фактических state stores |
| Equity curve | Формируется из SQLite `paper_trades`, `signal_log`, equity snapshots и execution ledger; запрещён статический Chart.js массив |
| API | Все REST endpoints возвращают typed JSON, provenance metadata, freshness metadata, audit hashes и корректные HTTP error semantics |
| Audit | Каждое критическое состояние имеет SHA-256 hash, timestamp, source и chain linkage |
| Frontend | Frontend каждые 5 секунд обновляет состояние без перезагрузки страницы и без static fallback cards |
| Testing | 100% line and branch coverage для business-critical backend modules; E2E-проверка dashboard и API на VPS |
| Governance | Документация соответствует требованиям академической воспроизводимости и institutional due diligence |

---

# 2. ГРАНИЦЫ СИСТЕМЫ И ИСТОЧНИКИ ИСТИНЫ

## 2.1. Canonical Sources of Truth

| Домен | Источник истины | Допустимый cache TTL | Недопустимая замена |
|---|---|---:|---|
| Binance account equity | Binance USDⓈ-M Futures API через CCXT | 15 секунд | hardcoded equity |
| Positions | Binance Futures `fetch_positions` / account endpoint | 10 секунд | статические position badges |
| Mark prices | Binance Futures mark price feed | 5 секунд | entry price как mark price |
| Funding rate | Binance Futures premium index / funding endpoint | 60 секунд | фиксированный `+0.0100%` |
| Order Flow Imbalance | Реальный L2 order book stream / market data collector | 3 секунды | фиксированный `0.12` |
| Hurst Exponent | Расчёт по market-return series | 60 секунд | фиксированный `0.512` |
| Circuit breaker | `circuit_breaker_state.json` с hash validation | 5 секунд | фиксированный `-1.64%` |
| Research advisories | `models/research_advisories.jsonl` | 30 секунд | статические карточки |
| Equity history | SQLite ledger и equity snapshot series | 60 секунд | hardcoded Chart.js array |
| Realized PnL | Execution/fill ledger, Binance income history и нормализованный trade ledger | 60 секунд | ручная агрегация UI |
| Closed trades | `paper_trades`, execution ledger, exchange fills | 60 секунд | вымышленные closed trade entries |

## 2.2. Источники, запрещённые в production telemetry

Запрещается использовать в production-коде:

- `equity = 39.54`;
- `balance = 40.20` как константу;
- массивы вида `[39.54, ..., 40.28]`;
- mock JSON;
- `TODO`, `pass`, `NotImplementedError` в доступных runtime paths;
- `try/except: return default_value`;
- `or 0`, если `0` не является фактическим значением источника;
- `or []`, если пустой список скрывает failure upstream источника;
- demo/test entries в production JSONL;
- условные UI-плашки `BTC LONG`, `SOL SHORT`, `ETH FLAT`, не подтверждённые live positions;
- фальшивый `healthy`, если источник не прошёл freshness/SHA/schema validation.

---

# 3. ДЕТАЛЬНЫЙ АУДИТ СУЩЕСТВУЮЩИХ ЗАГЛУШЕК И СЛАБЫХ МЕСТ

## 3.1. `main.py`

### Выявленные нарушения

1. Статическая equity:
   ```python
   equity = 39.54
   ```

2. Статический словарь risk gates:
   ```python
   risk_gates = {
       "hurst": 0.512,
       "funding": "+0.0100%",
       "circuit_breaker": "-1.64%",
       "ofi": 0.12,
   }
   ```

3. Вероятное смешение live, paper и cached источников без source attribution.

4. Отсутствие строгого failure contract: при сбое Binance/API система не должна показывать прошлое или демонстрационное значение как live.

### Обязательная замена

`main.py` обязан делегировать получение состояния агрегатору:

```python
dashboard_state = await dashboard_state_service.get_state(
    account_mode="production",
    require_fresh=False,
)
```

Запрещается вычислять dashboard metrics непосредственно в route handler.

---

## 3.2. `quant_dashboard.html`

### Выявленные нарушения

1. Статические research cards за 11 сентября.
2. Статический equity Chart.js dataset.
3. Статические position labels.
4. Невозможность отличить live, cached, unavailable и stale данные.
5. Отсутствие audit hash, source timestamp, refresh age.
6. Отсутствие отображения ошибок источников в UI.

### Обязательная замена

Все dashboard blocks обязаны рендериться только из ответа:

```text
GET /api/v1/dashboard/state
```

Каждый блок отображает:

- источник;
- timestamp источника;
- возраст данных;
- режим: `LIVE`, `CACHED`, `STALE`, `UNAVAILABLE`;
- SHA-256 audit hash;
- semantic validation status;
- human-readable reason при недоступности.

---

## 3.3. Telegram Bot и Signal Logger

### Выявленные слабые места

1. Realized PnL не должен рассчитываться простым суммированием text messages.
2. Closed trades не должны выводиться из неполных локальных сигналов.
3. Telegram не является financial system of record.
4. Лог сигналов без immutable hash chain не пригоден для academic/ODD audit.

### Обязательная архитектура

Telegram Bot является только notification/interaction channel.

System of record:

1. Binance fills/income history — для live execution;
2. SQLite execution ledger — для локального нормализованного учёта;
3. `paper_trades` — для paper mode;
4. `signal_log` — для signal provenance;
5. hash-chained audit events — для forensic reconstruction.

---

# 4. ЦЕЛЕВАЯ АРХИТЕКТУРА

## 4.1. Логическая схема

```text
Binance Futures / CCXT / WebSocket
              │
              ▼
     Market & Account Collectors
              │
              ▼
      Normalization / Validation Layer
              │
      ┌───────┼──────────────────────────┐
      ▼       ▼                          ▼
SQLite Ledger  Redis/Memory Cache   State JSON Stores
      │       │                          │
      └───────┴──────────┬───────────────┘
                          ▼
             Dashboard State Aggregator
                          │
                          ▼
             FastAPI /api/v1/dashboard/*
                          │
                          ▼
       quant_dashboard.html / Chart.js / JS
```

## 4.2. Обязательные сервисы

| Компонент | Ответственность |
|---|---|
| `BinanceAccountProvider` | Баланс, margin, positions, income history, exchange timestamps |
| `MarketDataProvider` | Mark price, funding, L2 order book, trades, volatility inputs |
| `RiskMetricsEngine` | Hurst, OFI, drawdown, leverage, VaR/ES, circuit breaker evaluation |
| `AdvisoryRepository` | Parse, schema validation, pagination, hash validation JSONL advisories |
| `EquityCurveRepository` | Восстановление equity curve из SQLite и snapshot ledger |
| `AuditLedger` | SHA-256 canonical event hashing и hash-chain |
| `DashboardStateService` | Сквозная агрегация, freshness controls, API DTO assembly |
| `ExecutionRouter` | VWAP/TWAP/POV routing, alpha-decay-aware execution, pre-trade risk controls |
| `StatArbEngine` | Cross-exchange lead-lag, cointegration, residual alpha generation |
| `VolSurfaceEngine` | Surface calibration, jump-risk estimation, risk scenario outputs |

---

# 5. МОДЕЛЬ ДАННЫХ И АУДИТ

## 5.1. Стандарт времени

Все timestamps:

- UTC;
- ISO-8601 с milliseconds;
- внутренняя точность не ниже milliseconds;
- exchange timestamps сохраняются отдельно от local receive timestamps.

Пример:

```json
{
  "source_timestamp": "2026-09-12T05:54:58.847Z",
  "received_at": "2026-09-12T05:54:59.021Z",
  "cache_age_ms": 174
}
```

## 5.2. SHA-256 canonical audit hashing

Каждый auditable event должен быть сериализован в canonical JSON:

- UTF-8;
- sorted keys;
- отсутствие пробелов-разделителей;
- Decimal сериализуется как строка;
- timestamp в UTC ISO-8601;
- без runtime-only полей.

Формула:

```text
event_hash = SHA256(
    canonical_json(
        {
            "event_id": ...,
            "event_type": ...,
            "event_timestamp": ...,
            "source": ...,
            "payload": ...,
            "previous_hash": ...
        }
    )
)
```

Для hash chain:

```text
previous_hash[n] = event_hash[n - 1]
```

Нарушение chain integrity является `CRITICAL` operational incident.

## 5.3. SQLite обязательные таблицы

### `equity_snapshots`

```sql
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    account_mode TEXT NOT NULL CHECK(account_mode IN ('paper', 'live')),
    total_equity_usdt TEXT NOT NULL,
    wallet_balance_usdt TEXT NOT NULL,
    available_balance_usdt TEXT NOT NULL,
    unrealized_pnl_usdt TEXT NOT NULL,
    realized_pnl_usdt TEXT NOT NULL,
    margin_balance_usdt TEXT NOT NULL,
    maintenance_margin_usdt TEXT NOT NULL,
    source TEXT NOT NULL,
    source_timestamp TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL UNIQUE,
    previous_hash TEXT,
    created_at TEXT NOT NULL
);
```

### `execution_ledger`

```sql
CREATE TABLE IF NOT EXISTS execution_ledger (
    execution_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL,
    account_mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    position_side TEXT,
    order_id TEXT,
    trade_id TEXT,
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    quote_quantity TEXT NOT NULL,
    commission TEXT NOT NULL,
    commission_asset TEXT,
    realized_pnl_usdt TEXT NOT NULL,
    execution_timestamp TEXT NOT NULL,
    source_payload_sha256 TEXT NOT NULL UNIQUE,
    previous_hash TEXT,
    created_at TEXT NOT NULL
);
```

### `audit_events`

```sql
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT,
    event_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
```

SQLite должна работать в WAL mode:

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```

---

# 6. API SPECIFICATION

## 6.1. Общие правила API

Base path:

```text
/api/v1/dashboard
```

Content type:

```text
application/json; charset=utf-8
```

Все monetary values передаются строками Decimal, а не float:

```json
"total_equity_usdt": "40.20384712"
```

Все ответы содержат:

```json
{
  "request_id": "uuid",
  "generated_at": "2026-09-12T05:55:00.000Z",
  "environment": "production",
  "data": {},
  "health": {},
  "audit": {}
}
```

`environment` обязан быть одним из:

- `production`;
- `paper`;
- `staging`;
- `testnet`.

Запрещается обозначать testnet/paper как production.

---

## 6.2. `GET /api/v1/dashboard/state`

### Назначение

Возвращает единый live/cached dashboard snapshot.

### Query parameters

| Параметр | Тип | Default | Значение |
|---|---|---:|---|
| `include_equity_curve` | boolean | `false` | Включить compact equity curve |
| `equity_limit` | integer | `500` | Количество последних equity points, 1–10000 |
| `advisory_limit` | integer | `10` | Количество последних advisories, 1–50 |
| `require_fresh` | boolean | `false` | Требовать только fresh upstream data |
| `symbols` | string | `BTCUSDT,SOLUSDT,ETHUSDT` | CSV инструментов для asset status |

### Условия HTTP 200

HTTP `200 OK` разрешён только если:

1. schema validation выполнена;
2. hash validation выполнена;
3. все обязательные блоки присутствуют;
4. каждый блок имеет `LIVE` или допустимый `CACHED` status;
5. cache age не превышает configured TTL;
6. отсутствует data fabrication;
7. `health.overall_status` не `CRITICAL`.

### Условия HTTP 503

Возвращать `503 Service Unavailable`, если отсутствует обязательная live/cache телеметрия, превышен TTL, нарушен hash chain или данные не прошли validation.

Пример:

```json
{
  "request_id": "4d612985-0d44-4e2a-8af2-9ab2cc3e3902",
  "generated_at": "2026-09-12T05:55:00.000Z",
  "error": {
    "code": "UPSTREAM_DATA_UNAVAILABLE",
    "message": "Binance Futures account telemetry is unavailable and no valid cache exists.",
    "affected_components": ["account_equity", "positions"],
    "retry_after_seconds": 5
  }
}
```

### Response schema

```json
{
  "request_id": "uuid",
  "generated_at": "2026-09-12T05:55:00.000Z",
  "environment": "production",
  "data": {
    "account": {
      "account_mode": "production",
      "exchange": "binanceusdm",
      "total_equity_usdt": "40.20384712",
      "wallet_balance_usdt": "40.11100000",
      "available_balance_usdt": "32.44000000",
      "margin_balance_usdt": "40.20384712",
      "maintenance_margin_usdt": "1.81000000",
      "unrealized_pnl_usdt": "0.09284712",
      "realized_pnl_usdt_24h": "-0.18420000",
      "source_status": "LIVE",
      "source_timestamp": "2026-09-12T05:54:59.120Z",
      "received_at": "2026-09-12T05:54:59.260Z",
      "cache_age_ms": 0,
      "payload_sha256": "sha256-hex"
    },
    "positions": [
      {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "quantity": "0.00100000",
        "notional_usdt": "58.32100000",
        "entry_price": "58120.00000000",
        "mark_price": "58321.00000000",
        "liquidation_price": "53121.00000000",
        "unrealized_pnl_usdt": "0.20100000",
        "initial_margin_usdt": "5.83210000",
        "maintenance_margin_usdt": "0.23200000",
        "leverage": "10",
        "le_ratio": "0.14400000",
        "margin_ratio": "0.04510000",
        "source_status": "LIVE",
        "source_timestamp": "2026-09-12T05:54:59.120Z",
        "payload_sha256": "sha256-hex"
      }
    ],
    "risk_gates": {
      "overall_status": "GREEN",
      "dynamic_circuit_breaker": {
        "status": "ARMED",
        "threshold_drawdown_pct": "-1.64000000",
        "current_drawdown_pct": "-0.42200000",
        "triggered": false,
        "state_file": "circuit_breaker_state.json",
        "state_sha256": "sha256-hex",
        "source_status": "LIVE"
      },
      "hurst_exponent": {
        "value": "0.51284700",
        "window_observations": 1024,
        "sampling_interval_seconds": 60,
        "method": "DFA",
        "regime": "PERSISTENT",
        "source_status": "LIVE"
      },
      "funding_rate": {
        "symbol": "BTCUSDT",
        "rate": "0.00010000",
        "rate_pct": "0.01000000",
        "next_funding_time": "2026-09-12T08:00:00.000Z",
        "source_status": "LIVE"
      },
      "order_flow_imbalance": {
        "symbol": "BTCUSDT",
        "value": "0.12000000",
        "lookback_ms": 5000,
        "bid_volume": "125.40000000",
        "ask_volume": "98.60000000",
        "source_status": "LIVE"
      }
    },
    "advisories": [],
    "asset_status": [],
    "equity_curve": []
  },
  "health": {
    "overall_status": "GREEN",
    "components": []
  },
  "audit": {
    "dashboard_snapshot_sha256": "sha256-hex",
    "previous_snapshot_sha256": "sha256-hex",
    "audit_chain_valid": true
  }
}
```

### Математические определения

#### Unrealized PnL

Для long:

```text
UPnL = Quantity × (MarkPrice − EntryPrice)
```

Для short:

```text
UPnL = Quantity × (EntryPrice − MarkPrice)
```

#### Equity

```text
Total Equity =
Wallet Balance
+ Unrealized PnL
+ Accrued Funding
− Estimated Fees
```

Если Binance возвращает canonical `totalWalletBalance`, `totalUnrealizedProfit`, `totalMarginBalance`, именно значения exchange являются первичными. Локальное вычисление сохраняется исключительно как reconciliation check.

#### Margin Ratio

```text
Margin Ratio = Maintenance Margin / Margin Balance
```

#### Leverage Exposure Ratio

```text
LE Ratio = Gross Notional Exposure / Total Equity
```

#### Order Flow Imbalance

```text
OFI = (V_bid − V_ask) / (V_bid + V_ask + ε)
```

где `V_bid` и `V_ask` — агрегированные объёмы L2 order book на установленной глубине и временном окне.

---

## 6.3. `GET /api/v1/dashboard/advisories`

### Назначение

Возвращает валидированные Astra Research Advisory entries из:

```text
models/research_advisories.jsonl
```

### Query parameters

| Параметр | Тип | Default |
|---|---|---:|
| `page` | integer | `1` |
| `page_size` | integer | `20` |
| `symbol` | string | optional |
| `recommendation` | string | optional |
| `from` | ISO-8601 | optional |
| `to` | ISO-8601 | optional |
| `include_math` | boolean | `true` |

### Минимальная схема advisory

```json
{
  "advisory_id": "astra-20260912-000001",
  "timestamp_utc": "2026-09-12T05:51:00.000Z",
  "symbol": "BTCUSDT",
  "strategy_family": "cross_exchange_stat_arb",
  "regime": "persistent_trend",
  "recommendation": "REDUCE_LONG_EXPOSURE",
  "confidence": "0.87300000",
  "horizon_seconds": 900,
  "mathematical_justification": {
    "model": "LeadLag-VECM + Hurst DFA + OFI",
    "features": {
      "hurst": "0.61200000",
      "ofi": "-0.24100000",
      "z_score": "2.84000000",
      "half_life_seconds": "186.00000000"
    },
    "equations": [
      "z_t = (spread_t - mu_t) / sigma_t",
      "P(reversion) = sigmoid(beta_0 + beta_1*z_t + beta_2*OFI_t)"
    ]
  },
  "risk_constraints": {
    "max_notional_usdt": "25.00000000",
    "max_leverage": "3.00000000",
    "stop_loss_pct": "-0.75000000"
  },
  "payload_sha256": "sha256-hex",
  "previous_hash": "sha256-hex"
}
```

### Требования

1. JSONL читается построчно.
2. Каждая запись проходит Pydantic/schema validation.
3. Каждая строка имеет SHA-256 hash.
4. Нарушенные строки не скрываются: они фиксируются в `audit_events`.
5. API возвращает `integrity_status`.
6. При количестве валидных entries менее 45 status становится `DEGRADED`; это не компенсируется генерированными advisory entries.

---

## 6.4. `GET /api/v1/dashboard/equity-curve`

### Назначение

Возвращает high-resolution equity curve из SQLite ledger.

### Query parameters

| Параметр | Тип | Default |
|---|---|---:|
| `account_mode` | `paper` / `live` / `all` | `all` |
| `from` | ISO-8601 | начало доступной истории |
| `to` | ISO-8601 | текущее UTC |
| `resolution` | `raw`, `1m`, `5m`, `15m`, `1h`, `1d` | `1m` |
| `max_points` | integer | `5000` |

### Response fields

```json
{
  "series": [
    {
      "timestamp_utc": "2026-09-12T05:50:00.000Z",
      "equity_usdt": "40.20384712",
      "wallet_balance_usdt": "40.11100000",
      "unrealized_pnl_usdt": "0.09284712",
      "realized_pnl_cumulative_usdt": "-0.18420000",
      "account_mode": "paper",
      "source": "sqlite:equity_snapshots",
      "payload_sha256": "sha256-hex"
    }
  ],
  "statistics": {
    "start_equity_usdt": "39.54000000",
    "end_equity_usdt": "40.20384712",
    "return_pct": "1.67993727",
    "max_drawdown_pct": "-1.64000000",
    "annualized_volatility_pct": "0.00000000",
    "observations": 0
  }
}
```

### Принцип построения

1. Использовать `equity_snapshots` как primary source.
2. Если snapshots отсутствуют, реконструировать equity из:
   - `paper_trades`;
   - `signal_log`;
   - `execution_ledger`;
   - initial capital ledger;
   - mark-price history.
3. Реконструированная кривая обязана иметь:
   ```json
   "reconstruction_method": "ledger_replay"
   ```
4. Если исторические mark prices отсутствуют, система возвращает `503` или неполную series с явным `data_gap`; запрещена линейная интерполяция как “реальная equity curve”.

---

# 7. FRONTEND SPECIFICATION

## 7.1. Dashboard refresh engine

Frontend обязан выполнять:

```javascript
setInterval(refreshDashboardState, 5000);
```

При загрузке страницы:

```javascript
document.addEventListener("DOMContentLoaded", refreshDashboardState);
```

Обязательные требования:

- `AbortController` для отмены зависших запросов;
- timeout не более 4 секунд;
- защита от overlapping requests;
- отображение timestamp последнего успешного update;
- отсутствие browser-side synthetic fallback data;
- при API `503` UI показывает `DATA UNAVAILABLE`, а не предыдущее значение как live;
- cached data отображается с age indicator;
- каждое обновление проверяет schema/version.

## 7.2. Equity Chart.js

Запрещён код вида:

```javascript
data: [39.54, 39.72, 40.28]
```

Разрешён только динамический dataset:

```javascript
equityChart.data.labels = state.data.equity_curve.map(p => p.timestamp_utc);
equityChart.data.datasets[0].data = state.data.equity_curve.map(
  p => Number(p.equity_usdt)
);
equityChart.options.scales.y.min = computedMin;
equityChart.options.scales.y.max = computedMax;
equityChart.update("none");
```

Автомасштабирование:

```text
padding = max((max_equity - min_equity) × 0.10, 0.01 USDT)
y_min = min_equity - padding
y_max = max_equity + padding
```

Если точек меньше 2, chart отображает valid single-point state без выдуманной линии.

## 7.3. Astra Advisory cards

Каждая advisory card содержит:

- `advisory_id`;
- UTC timestamp;
- symbol;
- strategy family;
- recommendation;
- confidence;
- horizon;
- regime;
- mathematical justification;
- risk constraints;
- `payload_sha256`;
- кнопку `Copy SHA-256`;
- кнопку expand/collapse;
- integrity status;
- visual label `LIVE RESEARCH LEDGER`.

Кнопка копирования должна использовать:

```javascript
navigator.clipboard.writeText(advisory.payload_sha256);
```

При недоступности clipboard API отображается явная ошибка, а не ложное сообщение “Copied”.

## 7.4. Dynamic asset status indicators

Для `BTCUSDT`, `SOLUSDT`, `ETHUSDT` отображать:

- mark price;
- funding rate;
- current position side;
- notional;
- unrealized PnL;
- OFI;
- Hurst regime;
- data freshness;
- risk gate state.

Если позиция отсутствует, отображать:

```text
NO OPEN POSITION
```

а не `FLAT`, если это термин в UI ошибочно воспринимается как подтверждённая стратегия.

---

# 8. СПЕЦИФИКАЦИЯ QUANT ENGINES

## 8.1. Cross-Exchange Statistical Arbitrage Lead-Lag Engine

### Назначение

Выявление statistically significant lead-lag relationships между ценовыми потоками бирж и инструментов при учёте latency, fees, funding, slippage и execution capacity.

### Минимальные компоненты

1. synchronized market data ingestion;
2. clock-offset monitoring;
3. lead-lag cross-correlation;
4. Granger causality;
5. VECM / cointegration analysis;
6. spread z-score;
7. half-life estimation;
8. transaction-cost model;
9. capacity and liquidity filter;
10. pre-trade risk gate.

### Core definitions

Spread:

```text
S_t = log(P_A,t) − β × log(P_B,t)
```

Z-score:

```text
Z_t = (S_t − μ_t) / σ_t
```

Ornstein-Uhlenbeck half-life:

```text
ΔS_t = θ(μ − S_t)Δt + σdW_t
half_life = ln(2) / θ
```

Signal допускается только если одновременно:

```text
|Z_t| ≥ z_entry
half_life ≤ configured_max_half_life
expected_alpha > fees + slippage + funding_cost + uncertainty_buffer
data_freshness within limit
risk_gates == GREEN
```

## 8.2. Continuous Volatility Surface Calibration

### Назначение

Построение и непрерывное обновление implied/realized volatility representation для risk estimation, scenario analysis и jump-risk modelling.

### Требования

- SVI/SABR-compatible parameterization;
- no-arbitrage checks;
- calendar spread checks;
- butterfly convexity checks;
- robust outlier rejection;
- timestamped calibration parameters;
- calibration residuals;
- confidence diagnostics;
- fallback запрещён: при недостаточном числе valid instruments surface получает `UNAVAILABLE`.

### Risk outputs

- realized volatility;
- implied volatility;
- skew;
- term structure;
- vol-of-vol;
- calibration RMSE;
- tail scenario impact.

## 8.3. Merton Jump-Diffusion Risk Modeler

Модель:

```text
dS_t / S_t = (μ − λκ)dt + σdW_t + (J − 1)dN_t
```

где:

- `λ` — jump intensity;
- `κ = E[J − 1]`;
- `J` — jump-size random variable;
- `N_t` — Poisson jump process.

Обязательные outputs:

- jump intensity;
- estimated jump mean/std;
- jump-adjusted VaR;
- jump-adjusted Expected Shortfall;
- probability of downside jump over horizon;
- scenario PnL under stressed jumps.

Risk engine обязан учитывать model uncertainty. Нельзя представлять Merton output как гарантированный forecast.

## 8.4. Algorithmic Execution Router

Поддерживаемые execution styles:

- VWAP;
- TWAP;
- POV;
- liquidity-seeking passive;
- urgency-constrained marketable limit;
- alpha-decay-aware routing.

### Alpha-decay preservation

Для signal alpha:

```text
Alpha(t) = Alpha_0 × exp(−λt)
```

Execution router обязан минимизировать:

```text
Expected Implementation Shortfall
+ Market Impact
+ Timing Risk
+ Alpha Decay Cost
```

### Обязательные pre-trade проверки

1. circuit breaker не triggered;
2. symbol allowed;
3. position limit не превышен;
4. leverage limit не превышен;
5. daily loss limit не превышен;
6. max order notional соблюдён;
7. stale market data отсутствует;
8. order book liquidity достаточна;
9. duplicate order protection пройдена;
10. account mode явно подтверждён.

При нарушении хотя бы одного условия order должен быть отклонён с immutable audit event.

---

# 9. RISK, SECURITY И OPERATIONAL CONTROLS

## 9.1. API keys и secrets

Запрещается:

- хранить Binance API keys в репозитории;
- включать keys в HTML/JS;
- логировать keys, signatures, secrets, full authorization headers;
- передавать keys в dashboard API.

Допустимое хранение:

- VPS environment variables;
- encrypted secrets manager;
- file permissions `0600`;
- отдельные keys для read-only telemetry и execution;
- IP whitelist;
- withdrawal permission permanently disabled.

## 9.2. Kill switches

Обязательные kill switches:

1. global trading kill switch;
2. per-symbol kill switch;
3. exchange connectivity kill switch;
4. data integrity kill switch;
5. maximum drawdown circuit breaker;
6. volatility shock circuit breaker;
7. repeated order rejection breaker;
8. stale market-data breaker.

Каждое срабатывание:

- сохраняется в `audit_events`;
- имеет SHA-256 hash;
- отображается в dashboard;
- уведомляет Telegram;
- требует explicit authorized reset.

## 9.3. Reconciliation

Каждые 60 секунд:

```text
Local Ledger Equity
vs
Binance Reported Equity
```

Допустимое расхождение:

```text
abs(local_equity - exchange_equity) <= configured_reconciliation_tolerance
```

При превышении tolerance:

- статус `RECONCILIATION_BREAK`;
- новые execution commands блокируются;
- dashboard получает `RED`;
- создаётся incident event.

---

# 10. IMPLEMENTATION PLAN

## Phase 1 — Eradication of static telemetry

1. Удалить hardcoded equity из `main.py`.
2. Удалить static `risk_gates`.
3. Удалить static Chart.js equity data.
4. Удалить статические advisory cards.
5. Удалить static position badges.
6. Добавить CI rule, блокирующий запрещённые паттерны.

Обязательный static scan:

```bash
rg -n \
  '39\.54|40\.28|0\.512|0\.0100%|-1\.64%|0\.12|TODO|NotImplementedError|mock|placeholder|stub' \
  . \
  --glob '!tests/**' \
  --glob '!docs/**'
```

Каждое найденное production-совпадение должно быть классифицировано и устранено либо внесено в explicit approved exception registry с обоснованием. Для telemetry exception registry запрещён.

## Phase 2 — Data providers and normalization

1. Реализовать Binance Futures account provider.
2. Реализовать market data provider.
3. Реализовать strict Decimal normalization.
4. Реализовать freshness/cache layer.
5. Реализовать circuit breaker file validator.
6. Реализовать risk metrics engine.
7. Реализовать JSONL advisory repository.

## Phase 3 — Ledger and audit chain

1. Миграции SQLite.
2. WAL mode.
3. Equity snapshots.
4. Execution reconciliation.
5. SHA-256 event chain.
6. Nightly integrity verification.
7. Exportable audit report.

## Phase 4 — API and frontend

1. Реализовать три обязательных endpoints.
2. Добавить OpenAPI schema.
3. Заменить dashboard rendering engine.
4. Добавить Chart.js live update.
5. Добавить advisory expand/collapse/hash-copy.
6. Добавить explicit source health panel.

## Phase 5 — Quant engine integration

1. Lead-lag/stat-arb engine telemetry.
2. Volatility surface diagnostics.
3. Merton jump-risk outputs.
4. Execution-router status and controls.
5. Risk gate integration.

---

# 11. ACCEPTANCE CRITERIA И ВАЛИДАЦИЯ

## 11.1. Zero-placeholder acceptance

Приёмка считается проваленной, если найден хотя бы один из следующих случаев:

- статическая equity в API/UI;
- статический risk metric, выдаваемый за live;
- test/demo advisory в production feed;
- статическая position indicator;
- скрытый `except` с fabricated default;
- fake chart data;
- dashboard отображает stale/unavailable данные как `LIVE`;
- audit hash отсутствует у критического record;
- hash-chain integrity нарушена;
- front-end содержит static fallback market/account telemetry.

## 11.2. Functional acceptance

Обязательные проверки:

```bash
curl -i http://localhost:8000/api/v1/dashboard/state
curl -i http://localhost:8000/api/v1/dashboard/advisories?page=1&page_size=10
curl -i http://localhost:8000/api/v1/dashboard/equity-curve?resolution=1m
curl -i http://localhost:8000/quant-dashboard
```

Ожидаемые результаты:

1. `200 OK` при наличии валидных live/cached источников.
2. `503` с explicit error contract при недоступности обязательных источников.
3. Отсутствие server-side stack trace в response body.
4. `quant-dashboard` загружается без static mock telemetry.
5. Network inspector подтверждает polling `/api/v1/dashboard/state` каждые 5 секунд.
6. Equity chart получает точки из API.
7. Advisory cards соответствуют JSONL source records.
8. Position cards совпадают с Binance Futures positions.
9. Account equity совпадает с Binance within reconciliation tolerance.
10. Audit chain проходит verification.

## 11.3. Test coverage

Требование: **100% line coverage и 100% branch coverage** для следующих модулей:

- `DashboardStateService`;
- `BinanceAccountProvider`;
- `RiskMetricsEngine`;
- `EquityCurveRepository`;
- `AdvisoryRepository`;
- `AuditLedger`;
- circuit breaker validator;
- API route handlers;
- execution pre-trade risk controls.

Обязательные тестовые категории:

- unit tests;
- schema validation tests;
- cache TTL tests;
- upstream timeout tests;
- malformed JSONL tests;
- SHA mismatch tests;
- broken hash-chain tests;
- Binance payload normalization tests;
- reconciliation mismatch tests;
- API contract tests;
- frontend DOM tests;
- browser E2E tests;
- VPS smoke tests.

Пример CI gate:

```bash
pytest --cov=app --cov-branch --cov-report=term-missing --cov-fail-under=100
```

## 11.4. Academic readiness

Для thesis/dissertation должны быть подготовлены:

1. описание data-generating process;
2. определения всех risk metrics;
3. методология Hurst/DFA;
4. OFI construction methodology;
5. lead-lag and cointegration methodology;
6. volatility surface calibration methodology;
7. Merton jump-diffusion assumptions;
8. reproducibility protocol;
9. SQLite WAL and SHA-256 audit architecture;
10. limitations section;
11. model-risk discussion;
12. validation and out-of-sample protocol.

## 11.5. Institutional DDQ / ODD readiness

Для LP / AIMA DDQ review должны быть документированы:

- legal entity and operational ownership;
- access control;
- key management;
- change management;
- deployment process;
- incident management;
- disaster recovery;
- backup policy;
- market-data dependency map;
- execution counterparty dependency;
- reconciliation methodology;
- valuation policy;
- model governance;
- parameter change approval;
- kill-switch governance;
- audit retention;
- cybersecurity controls;
- concentration and liquidity limits;
- business continuity plan.

---

# 12. DEFINITION OF DONE

MaBuy V5.2 соответствует настоящему ТЗ только при одновременном выполнении всех условий:

1. Нет production telemetry, построенной на hardcoded values.
2. Dashboard получает account/position/risk/advisory/equity data из реальных источников.
3. Каждый critical data object имеет source metadata и SHA-256 audit hash.
4. SQLite используется в WAL mode с immutable audit-chain semantics.
5. API не маскирует upstream failures вымышленными значениями.
6. Dashboard корректно различает `LIVE`, `CACHED`, `STALE`, `UNAVAILABLE`.
7. Research Advisory endpoint обслуживает реальные записи `research_advisories.jsonl`.
8. Equity curve строится из ledger, а не из frontend constant array.
9. HFT/stat-arb/risk engines имеют строгие data, risk и execution contracts.
10. Все critical modules имеют 100% line и branch coverage.
11. На VPS выполнена фактическая HTTP/E2E verification.
12. Подготовлены academic reproducibility package и institutional DDQ/ODD package.

---

# 13. ИТОГОВЫЙ НЕПЕРЕГОВОРНЫЙ ПРИНЦИП

```text
A value displayed by MaBuy must be traceable.
A value used by MaBuy must be validated.
A decision made by MaBuy must be auditable.
A failure encountered by MaBuy must be explicit.
A metric unavailable to MaBuy must never be fabricated by MaBuy.
```

**Любой элемент, не удовлетворяющий данным требованиям, считается заглушкой и подлежит удалению до production acceptance.**
```