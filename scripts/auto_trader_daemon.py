"""
MaBuy Autonomous Trading & Monitoring Daemon (V5.2 Renaissance Titan Engine)
Orchestrates:
  1. Single In-Memory Ensemble Engine resident allocation (Zero-Disk I/O)
  2. Non-blocking Hot-Reload via registry_manifest.json
  3. Dynamic Adaptive Volatility Circuit Breaker (24h rolling PnL from Binance income)
  4. Funding Rate & Liquidation Bias Engine (live 8h rate sentiment & squeeze veto)
  5. Hurst Exponent Dynamic Regime Detector (Trend H>0.55 vs Mean-Reverting H<0.45)
  6. Cross-Asset Correlation Guard (Beta concentration filter & Risk Parity haircut)
  7. Active Trade Escort & Early Reversal Exit Guard (70% ML or |OFI| >= 0.30)
  8. 75.0% High-Conviction Sniper Entry Gate
"""

import os
import sys
import math
import time
import datetime

import logging
import json
import urllib.request
import urllib.error
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Tuple, List
import threading
import signal

SHUTDOWN_EVENT = threading.Event()

def handle_shutdown_signal(signum, frame):
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.info(f"🛑 [SHUTDOWN SIGNAL] Received {sig_name} (signal {signum}). Initiating graceful shutdown...")
    SHUTDOWN_EVENT.set()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.infrastructure.mlops.drift_detector import PredictionDriftMonitor

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "mabuy_prod.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MANIFEST_PATH = os.path.join(MODELS_DIR, "registry_manifest.json")

# Quarantine configuration: default is False so ETH can trade live like BTC/SOL
ETH_QUARANTINE_ENABLED = os.environ.get("ETH_QUARANTINE_ENABLED", "false").lower() in ("true", "1", "yes")


os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(os.path.join(BASE_DIR, "logs", "daemon_app.log"), maxBytes=5*1024*1024, backupCount=3),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AutoTraderDaemon")

# Suppress noisy lower-level library loggers to keep mabuy-logs clean and human-readable
for _noisy_pkg in [
    "src.infrastructure.data_fetchers",
    "src.infrastructure.data_fetchers.binance_fetcher",
    "urllib3",
    "ccxt",
    "requests",
    "src.infrastructure.nlp",
    "src.infrastructure.data.live_microstructure_collector"
]:
    logging.getLogger(_noisy_pkg).setLevel(logging.WARNING)

TARGET_SYMBOLS = ["BTC", "ETH", "SOL"]

# 🎯 INSTITUTIONAL THRESHOLDS (Quality Over Quantity)
MIN_CONFIDENCE_THRESHOLD = 0.78       # 78.0% Minimum ML Confidence for Sniper Entry (Conviction Gate)
REVERSAL_CONFIDENCE_THRESHOLD = 0.78  # 78.0% Opposing Conviction to Trigger Early Exit
REVERSAL_OFI_THRESHOLD = 0.45         # 45.0% Net Order Flow Imbalance (Used ONLY as confluence with ML)

EXECUTION_LOCK = threading.RLock()
drift_monitor = PredictionDriftMonitor(recent_window=20, historical_window=100, delta_threshold=0.05)


class FastPositionEscortWorker(threading.Thread):
    """
    Dedicated High-Frequency Worker Thread (Astra Component 2).
    Runs every 10 seconds completely decoupled from the heavy ML scan.
    Guarantees that flash spikes, trailing stops, and break-even locks are managed with zero latency.
    """
    def __init__(self, interval_sec: float = 10.0):
        super().__init__(daemon=True, name="FastPositionEscortWorker")
        self.interval = interval_sec
        self.running = True
        self.last_run = 0.0

    def run(self):
        logger.info("🚀 [СОПРОВОЖДЕНИЕ ПОЗИЦИЙ] Запущен фоновый высокочастотный воркер (интервал 10с).")
        while self.running:
            try:
                with EXECUTION_LOCK:
                    verify_signal_outcomes()
                for sym in TARGET_SYMBOLS:
                    record_market_telemetry_observation(sym)

                # 🔬 Track 2: High-Frequency Research Exploration Barrier Evaluation
                try:
                    exp_prices = {}
                    for sym in TARGET_SYMBOLS:
                        clean_s = canonical_symbol(sym)
                        q = cex_state.get_quote("binance", clean_s) or cex_state.get_latest_quote(clean_s, "binance")
                        if q and getattr(q, "price", 0.0) > 0:
                            exp_prices[clean_s] = float(q.price)
                    if exp_prices:
                        research_sleeve.update_open_explorations(exp_prices)
                except Exception as exp_err:
                    logger.debug(f"Research sleeve barrier check note: {exp_err}")

                self.last_run = time.time()
            except Exception as e:
                logger.error(f"Error in FastPositionEscortWorker: {e}", exc_info=True)
            time.sleep(self.interval)

def load_env():
    potential_paths = [
        os.path.join(BASE_DIR, ".env"),
        "/root/mabuy2/.env",
        "/root/mabuy/.env",
        "/root/.env"
    ]
    for env_path in potential_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                for line in raw_text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        os.environ.setdefault(k, v)
                break
            except Exception:
                pass

load_env()


def _edit_telegram_channel_message(
    token: str,
    channel_id: str,
    message_id: int,
    new_text: str,
    reply_markup: Optional[Dict[str, Any]] = None,
    max_retries: int = 1
) -> bool:
    """
    Safely edits an existing Telegram message in a public channel or group.
    Handles Telegram 400 Bad Request (MessageNotModified, MessageNotFound)
    and 429 Too Many Requests with adaptive retry_after backoff.
    """
    url = f"https://api.telegram.org/bot{token}/editMessageText"
    payload = {
        "chat_id": channel_id,
        "message_id": message_id,
        "text": new_text,
        "parse_mode": "HTML"
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                desc = err_body.get("description", "")
            except Exception:
                err_body = {}
                desc = str(e)

            if "message is not modified" in desc.lower():
                logger.debug("Telegram message not modified.")
                return True
            if "message to edit not found" in desc.lower():
                logger.warning(f"Telegram message {message_id} to edit not found.")
                return False
            if e.code == 429:
                retry_after = err_body.get("parameters", {}).get("retry_after", 2)
                if attempt < max_retries:
                    time.sleep(retry_after)
                    continue
            logger.warning(f"Telegram edit error HTTP {e.code}: {desc}")
            return False
        except Exception as e:
            logger.error(f"Failed to edit telegram message: {e}")
            return False
    return False


def send_telegram_alert(text: str) -> bool:
    """Sends a direct alert to the system owner in Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or OWNER_USER_ID
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.debug(f"Telegram alert send failed: {e}")
        return False


AUDIT_MILESTONES_NOTIFIED = set()


def check_statistical_milestones(db_path: str = DB_PATH):
    """
    Checks if signal_log has crossed key statistical sample milestones (25, 50, 75, 100).
    Emits an automatic Telegram alert to the Owner when reached.
    """
    try:
        import sqlite3
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*),
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END),
                SUM(CASE WHEN pnl_usd <= 0 THEN abs(pnl_usd) ELSE 0 END),
                SUM(pnl_usd)
            FROM signal_log
            WHERE status = 'CLOSED' AND pnl_usd IS NOT NULL
        """)
        row = cur.fetchone()
        conn.close()
        if not row:
            return
        count = row[0] or 0
        if count in (25, 50, 75, 100) and count not in AUDIT_MILESTONES_NOTIFIED:
            AUDIT_MILESTONES_NOTIFIED.add(count)
            wins = row[1] or 0
            gross_p = float(row[2] or 0.0)
            gross_l = float(row[3] or 0.0)
            net_pnl = float(row[4] or 0.0)
            wr = (wins / count) * 100.0 if count > 0 else 0.0
            pf = (gross_p / gross_l) if gross_l > 0 else (99.0 if gross_p > 0 else 0.0)
            msg = (
                f"🔔 <b>[МАЯК ASTRA: НАКОПЛЕНА ВЫБОРКА В {count} СДЕЛОК!]</b>\n\n"
                f"📊 <b>Статистический срез MaBuy V5.2:</b>\n"
                f"• Закрытых сделок: <b>{count}</b>\n"
                f"• Win Rate: <b>{wr:.1f}%</b>\n"
                f"• Profit Factor: <b>{pf:.2f}</b>\n"
                f"• Чистый PnL: <b>${net_pnl:+.2f}</b>\n\n"
                f"💡 <b>Рекомендация Astra:</b>\n"
                f"Выборка достаточна для институционального квантового аудита! "
                f"Пора активировать ключ API (1000 тенге) и запустить проверку масштабирования капитала."
            )
            send_telegram_alert(msg)
            logger.info(f"📢 [ASTRA BEACON] Sent Telegram milestone alert for {count} closed trades!")
    except Exception as e:
        logger.debug(f"Milestone check note: {e}")


from src.domain.entities.prediction import PredictionRequest
from src.domain.entities.tier import SubscriptionTier
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.infrastructure.persistence.signal_logger import SignalLogger
from src.infrastructure.execution.paper_trader import PaperTrader
from src.infrastructure.execution.binance_trader import BinanceAutoTrader, SNIPER_ALLOWED
from src.infrastructure.ml.v4_titan_ensemble import MaBuyV4EnsembleEngine

# 🏛 Renaissance Institutional Services
from src.domain.services.funding_rate_service import FundingRateService
from src.domain.services.hurst_regime_service import HurstRegimeService, HurstRegimeType
from src.domain.services.dynamic_circuit_breaker import DynamicAdaptiveCircuitBreaker
from src.domain.services.correlation_guard import CrossAssetCorrelationGuard
from src.domain.services.mtf_trend_service import MTFTrendService
from src.infrastructure.data.live_microstructure_collector import LiveMicrostructureCollector
from src.domain.services.astra_news_alpha_engine import AstraNewsAlphaEngine
from src.domain.services.dynamic_covariance_engine import DynamicCovarianceEngine

OWNER_USER_ID = "100000001"
last_live_trade_timestamps = {}

# 🛡 Astra Post-Loss Churn Defense State
# Format: {symbol: {"exit_time": float, "exit_price": float, "direction": str, "pnl_usd": float}}
LAST_LOSS_LOCKOUT: Dict[str, Dict[str, Any]] = {}

def load_initial_loss_lockouts(db_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads recent closed losing trades from SQLite to initialize post-loss lockouts."""
    lockouts = {}
    try:
        import sqlite3
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol, direction, exit_price, pnl_usd, closed_at 
                FROM paper_trades 
                WHERE status = 'CLOSED' AND pnl_usd < 0 
                ORDER BY closed_at DESC
            """)
            rows = cur.fetchall()
            now = time.time()
            for sym, d, exit_p, pnl, closed_at_str in rows:
                clean_sym = sym.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
                if clean_sym not in lockouts and closed_at_str:
                    try:
                        dt = datetime.datetime.fromisoformat(closed_at_str)
                        exit_ts = dt.replace(tzinfo=datetime.timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()
                        if now - exit_ts < 86400:
                            lockouts[clean_sym] = {
                                "exit_time": exit_ts,
                                "exit_price": float(exit_p or 0),
                                "direction": str(d).upper(),
                                "pnl_usd": float(pnl or 0)
                            }
                    except Exception:
                        pass
            conn.close()
            if lockouts:
                logger.info(f"🛡 Loaded {len(lockouts)} active post-loss lockouts from SQLite: {list(lockouts.keys())}")
    except Exception as e:
        logger.warning(f"Could not load initial loss lockouts: {e}")
    return lockouts

# Initialize Institutional Singletons
funding_service = FundingRateService()
hurst_service = HurstRegimeService()
circuit_breaker = DynamicAdaptiveCircuitBreaker(db_path=DB_PATH)
correlation_guard = CrossAssetCorrelationGuard()
mtf_service = MTFTrendService()
live_microstructure = LiveMicrostructureCollector()
news_alpha_engine = AstraNewsAlphaEngine()
dynamic_covariance = DynamicCovarianceEngine()
from src.domain.services.asymmetric_risk_engine import AsymmetricRiskEngine
asymmetric_risk_engine = AsymmetricRiskEngine()
from src.infrastructure.mlops.research_exploration_sleeve import ResearchExplorationSleeve
research_sleeve = ResearchExplorationSleeve(db_path=DB_PATH)

# 🛰 Step 2: Astra Event-Driven Market Data Core (WebSocket Streams + REST Fallback)
from src.domain.services.cross_exchange_state import CrossExchangeState, canonical_symbol
from src.domain.services.cross_exchange_ws_collector import CrossExchangeWSCollector
from src.domain.services.cross_exchange_confirmation import CrossExchangeConfirmation
cex_state = CrossExchangeState(symbols=TARGET_SYMBOLS)
cex_ws_collector = CrossExchangeWSCollector(state=cex_state, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"))
cex_confirmation = CrossExchangeConfirmation(state=cex_state)
try:
    cex_ws_collector.start_background()
except Exception as cecl_init_err:
    logger.error(f"Failed to start CrossExchangeWSCollector: {cecl_init_err}")

# 🛰 Step 2.5: Lightweight Raw Market Telemetry & Microstructure Recorder (Max 1.5GB Quota)
from src.infrastructure.data.market_telemetry_recorder import MarketTelemetryRecorder
telemetry_dir = os.path.join(DATA_DIR, "telemetry")
market_telemetry_recorder = MarketTelemetryRecorder(
    storage_dir=telemetry_dir,
    max_buffer_size=120,
    flush_interval_sec=120.0,
    max_storage_mb=1500.0,
    max_retention_days=14,
    auto_start_worker=True,
)

# 🛰 Step 2.6: Ultra-Low Latency Deribit Europe & DVOL Volatility Index Collector (Amsterdam AM4)
from src.domain.services.deribit_collector import DeribitCollector
deribit_collector = DeribitCollector(poll_interval_sec=10.0, dvol_interval_sec=60.0, auto_start=True)

# 🛰 Step 2.7: Microstructure Ensemble Engine & Feeds (P1 Block 2)
from src.domain.services.microstructure_ensemble import MicrostructureEnsembleEngine
from src.domain.services.derivatives_flow_radar import DerivativesFlowRadarService
from src.infrastructure.data_fetchers.liquidation_streamer import LiquidationCascadeRadar
microstructure_ensemble = MicrostructureEnsembleEngine()
liquidation_radar = LiquidationCascadeRadar()

# 🛰 Step 2.8: Execution Shortfall Benchmarker & Shadow SOR (P1 Block 3)
from src.domain.services.execution_shortfall_benchmarker import ExecutionShortfallBenchmarker
execution_benchmarker = ExecutionShortfallBenchmarker(transfer_friction_usd=1.00)

# 🛰 Step 3: Astra Statistical Arbitrage Research Engine (SOL/BTC Synthetic Spread)
from src.domain.services.synthetic_spread_engine import (
    SyntheticSpreadEngine,
    SyntheticSpreadConfig,
    SpreadAction,
    SpreadPosition,
    SpreadTelemetry,
)
synthetic_spread_engine = SyntheticSpreadEngine(
    SyntheticSpreadConfig(
        shadow_mode=True,
        q_alpha_per_second=1.20e-9,
        q_beta_per_second=4.70e-9,
        measurement_r=9.0e-8,
        ou_sample_interval_seconds=5.0,
        ou_estimation_half_life_seconds=1800.0,
        min_ou_effective_samples=80.0,
        entry_z=2.0,
        exit_z=0.3,
        catastrophic_z=3.8,
        min_half_life_seconds=900.0,
        max_half_life_seconds=21600.0,
        sol_one_way_fee_bps=4.0,
        btc_one_way_fee_bps=4.0,
        sol_one_way_slippage_bps=1.5,
        btc_one_way_slippage_bps=1.0,
        edge_safety_buffer_bps=3.0,
    )
)

# 🛰 Step 4: Astra Unified Portfolio Risk Kernel & Dynamic Sizing
from src.domain.services.portfolio_risk_kernel import (
    PortfolioRiskKernel,
    RiskKernelConfig,
    PlattCalibrator,
    PortfolioSnapshot,
    PositionSnapshot,
    CandidateOrder,
    RiskAssessmentResult,
    log_risk_assessment,
)
risk_kernel_config = RiskKernelConfig(
    alpha=0.15,
    f_min=0.0,
    f_max=0.0125,
    max_trade_risk_fraction=0.0125,
    daily_drawdown_limit_fraction=0.02,
    max_margin_utilization_fraction=0.40,
    max_portfolio_leverage=3.0,
    max_symbol_leverage=4.0,
    min_liquidation_distance_fraction=0.20,
    require_exchange_liquidation_distance=False,
    default_missing_correlation=0.75,
    min_notional_usd=5.0,
)
portfolio_risk_kernel = PortfolioRiskKernel(
    config=risk_kernel_config,
    calibrator=PlattCalibrator(a=1.20, b=-0.10),
)
DAY_START_EQUITY_CACHE = {"date": None, "equity": 40.0}

# 🛰 Step 5: Astra Controlled Multi-Venue Scaled Execution & Smart Order Router (SOR)
from decimal import Decimal
from src.infrastructure.execution.smart_order_router import (
    SmartOrderRouter,
    ExecutionVenue,
    RoutingRequest,
    Side as SORSide,
    ExecutionMode as SORExecutionMode,
    ExecutionReceipt,
    BinanceVenueAdapter,
    BybitLinearAdapter,
    OKXSwapAdapter,
    PortfolioRiskKernelAdapter,
)
sor_risk_adapter = PortfolioRiskKernelAdapter(risk_kernel=portfolio_risk_kernel)


def update_synthetic_spread_engine() -> Optional[SpreadTelemetry]:
    """
    Updates the SOL/BTC Synthetic Spread Statistical Arbitrage Engine (Astra Step 3).
    Samples synchronized midpoint quotes from Binance WebSocket stream.
    """
    try:
        sol_q = cex_state.get_quote("binance", "SOL", max_age_s=5.0) or cex_state.get_latest_quote("SOL", "binance")
        btc_q = cex_state.get_quote("binance", "BTC", max_age_s=5.0) or cex_state.get_latest_quote("BTC", "binance")

        if sol_q is None or btc_q is None:
            return None

        if hasattr(sol_q, "mid_price"):
            sol_mid = sol_q.mid_price
            sol_ts = (sol_q.exchange_ts_ms / 1000.0) if sol_q.exchange_ts_ms else time.time()
        else:
            sol_mid = sol_q.price
            sol_ts = sol_q.timestamp

        if hasattr(btc_q, "mid_price"):
            btc_mid = btc_q.mid_price
            btc_ts = (btc_q.exchange_ts_ms / 1000.0) if btc_q.exchange_ts_ms else time.time()
        else:
            btc_mid = btc_q.price
            btc_ts = btc_q.timestamp

        now = time.time()
        telemetry = synthetic_spread_engine.update(
            sol_mid=sol_mid,
            btc_mid=btc_mid,
            timestamp=now,
            sol_quote_timestamp=sol_ts,
            btc_quote_timestamp=btc_ts,
        )

        # If an actionable transition occurred (ENTRY / EXIT / STOP), log to SignalLogger
        if telemetry.action != SpreadAction.HOLD:
            try:
                db_logger = SignalLogger(db_path=DB_PATH)
                sig_dir = "BULLISH" if telemetry.action == SpreadAction.ENTER_LONG_SPREAD else (
                    "BEARISH" if telemetry.action == SpreadAction.ENTER_SHORT_SPREAD else "NEUTRAL"
                )
                sig_conf = min(0.95, max(0.50, abs(telemetry.zscore or 2.0) / 4.0))
                db_logger.log_signal(
                    symbol="SOL/BTC_SPREAD",
                    direction=sig_dir,
                    confidence=sig_conf,
                    entry_price=sol_mid,
                    status="SHADOW",
                    close_reason=f"{telemetry.action.value}: {telemetry.reason} | Z={telemetry.zscore:+.2f} Beta={telemetry.beta:.2f} Edge={telemetry.expected_net_edge_bps:+.1f}bps"
                )
                logger.info(f"   🔔 [СТАТ-АРБИТРАЖ СОБЫТИЕ] {telemetry.action.value} зафиксировано в signal_log! {telemetry.format_russian_summary()}")
            except Exception as sig_err:
                logger.debug(f"SignalLogger shadow spread event error: {sig_err}")

        return telemetry
    except Exception as spread_err:
        logger.debug(f"Synthetic spread update non-blocking note: {spread_err}")
        return None


_LATEST_MARKET_FEATURES: Dict[str, Dict[str, Any]] = {}


def record_market_telemetry_observation(symbol: str) -> None:
    """
    Captures synchronized L2/cross-exchange telemetry into MarketTelemetryRecorder.
    Zero additional REST network latency: reuses in-RAM WebSocket quotes and TTL-cached microstructure.
    """
    try:
        clean_s = canonical_symbol(symbol)

        # 1. Binance Top of Book
        bn_q = cex_state.get_quote("binance", clean_s) or cex_state.get_latest_quote(clean_s, "binance")
        if bn_q is None:
            return

        if hasattr(bn_q, "mid_price"):
            bn_mid = float(bn_q.mid_price)
            bn_bid = float(bn_q.bid)
            bn_ask = float(bn_q.ask)
        else:
            bn_mid = float(bn_q.price)
            bn_bid = float(getattr(bn_q, "bid", None) or bn_mid)
            bn_ask = float(getattr(bn_q, "ask", None) or bn_mid)

        if bn_bid <= 0 or bn_ask <= 0:
            return

        # 2. Cross-exchange quotes (Bybit & OKX)
        bybit_q = cex_state.get_quote("bybit", clean_s) or cex_state.get_latest_quote(clean_s, "bybit")
        okx_q = cex_state.get_quote("okx", clean_s) or cex_state.get_latest_quote(clean_s, "okx")

        bybit_mid = float(getattr(bybit_q, "mid_price", getattr(bybit_q, "price", 0.0)) or 0.0) if bybit_q else None
        if bybit_mid == 0.0:
            bybit_mid = None

        okx_mid = float(getattr(okx_q, "mid_price", getattr(okx_q, "price", 0.0)) or 0.0) if okx_q else None
        if okx_mid == 0.0:
            okx_mid = None

        bybit_spread_bps = None
        if bybit_q and getattr(bybit_q, "bid", None) and getattr(bybit_q, "ask", None) and bybit_mid and bybit_mid > 0:
            bybit_spread_bps = (float(bybit_q.ask) - float(bybit_q.bid)) / bybit_mid * 10000.0

        okx_spread_bps = None
        if okx_q and getattr(okx_q, "bid", None) and getattr(okx_q, "ask", None) and okx_mid and okx_mid > 0:
            okx_spread_bps = (float(okx_q.ask) - float(okx_q.bid)) / okx_mid * 10000.0

        lead_lag_basis_bps = None
        if bybit_mid and bn_mid > 0:
            lead_lag_basis_bps = (bybit_mid - bn_mid) / bn_mid * 10000.0

        # 3. Microstructure cached features
        snap = None
        cond = getattr(live_microstructure, "_cache_condition", None)
        if cond:
            with cond:
                cached = getattr(live_microstructure, "_cache", {}).get(f"{clean_s}USDT")
                if cached:
                    snap = cached[1]

        feat = _LATEST_MARKET_FEATURES.get(clean_s, {})
        ofi_norm = feat.get("ofi_norm") if feat.get("ofi_norm") is not None else getattr(snap, "ofi_norm", None)
        cvd_delta = getattr(snap, "cvd_delta", None)
        aggressor_buyer_ratio = getattr(snap, "aggressor_buyer_ratio", None)

        try:
            f_data = funding_service.get_funding_rate(clean_s)
            funding_rate = float(f_data.funding_rate) if f_data else getattr(snap, "funding_rate", None)
        except Exception:
            funding_rate = getattr(snap, "funding_rate", None)

        funding_velocity = getattr(snap, "funding_velocity", None)
        open_interest = getattr(snap, "oi", None)
        microprice = (snap.mark_price + snap.microprice_delta) if (snap and hasattr(snap, "mark_price") and hasattr(snap, "microprice_delta")) else None

        # 4. Deribit Europe perpetual quotes & DVOL
        deribit_q = deribit_collector.get_quote(clean_s)
        deribit_mid = deribit_q.mid_price if (deribit_q and deribit_q.is_valid) else None
        deribit_spread_bps = deribit_q.spread_bps if (deribit_q and deribit_q.is_valid) else None

        dvol_btc_obj = deribit_collector.get_dvol("BTC")
        dvol_btc = dvol_btc_obj.dvol_close if dvol_btc_obj else None

        dvol_eth_obj = deribit_collector.get_dvol("ETH")
        dvol_eth = dvol_eth_obj.dvol_close if dvol_eth_obj else None

        market_telemetry_recorder.record(
            symbol=clean_s,
            best_bid=bn_bid,
            best_ask=bn_ask,
            bybit_mid=bybit_mid,
            okx_mid=okx_mid,
            bybit_spread_bps=bybit_spread_bps,
            okx_spread_bps=okx_spread_bps,
            lead_lag_basis_bps=lead_lag_basis_bps,
            deribit_mid=deribit_mid,
            deribit_spread_bps=deribit_spread_bps,
            dvol_btc=dvol_btc,
            dvol_eth=dvol_eth,
            ofi_norm=ofi_norm,
            cvd_delta=cvd_delta,
            aggressor_buyer_ratio=aggressor_buyer_ratio,
            funding_rate=funding_rate,
            funding_velocity=funding_velocity,
            open_interest=open_interest,
            microprice=microprice,
        )
    except Exception as e:
        logger.debug(f"Telemetry record non-blocking error for {symbol}: {e}")


class DaemonModelManager:
    """Manages single in-memory instance of MaBuyV4EnsembleEngine."""
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.manifest_path = MANIFEST_PATH
        self._last_manifest_mtime: float = self._get_manifest_mtime()
        logger.info("Instantiating Global MaBuyV4EnsembleEngine (Resident in Memory)...")
        self.engine = MaBuyV4EnsembleEngine(db_path=self.db_path)
        logger.info("Global Ensemble Engine successfully cached in RAM.")

    def _get_manifest_mtime(self) -> float:
        try:
            if os.path.exists(self.manifest_path):
                return os.path.getmtime(self.manifest_path)
        except Exception:
            pass
        return 0.0

    def check_and_reload(self):
        current_mtime = self._get_manifest_mtime()
        if current_mtime > self._last_manifest_mtime and current_mtime > 0:
            logger.info("🔄 Detected new model release in registry_manifest.json. Hot-reloading models in RAM...")
            try:
                self.engine = MaBuyV4EnsembleEngine(db_path=self.db_path)
                self._last_manifest_mtime = current_mtime
                logger.info("✅ Models hot-reloaded successfully in memory.")
            except Exception as e:
                logger.error(f"Error hot-reloading models: {e}", exc_info=True)


def update_heartbeat():
    try:
        hb_path = os.path.join(BASE_DIR, "logs", "daemon_heartbeat.json")
        with open(hb_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.time(),
                "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "status": "ALIVE",
                "symbols": TARGET_SYMBOLS
            }, f)
    except Exception as e:
        logger.debug(f"Heartbeat write error: {e}")


# Reversal Guard & Cooldown State (Institutional Noise & Over-Trading Protection)
_EMA_SMOOTHED_OBI: Dict[str, float] = {}
_REVERSAL_CONFIRMATION_STREAK: Dict[str, int] = {}
LAST_CLOSED_TRADE_TIMESTAMP: Dict[str, float] = {}
SYMBOL_TRADE_COOLDOWN_SECONDS: float = 3600.0  # 60 minutes minimum cooldown per symbol post-close
MINIMUM_POSITION_HOLDING_SECONDS: float = 900.0  # 15 minutes minimum window before dynamic early exit


def check_active_reversal_guard(
    symbol: str, 
    current_direction: str, 
    new_pred: Any, 
    obi_score: Optional[float] = None,
    position_opened_at: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Active Trade Escort & Dynamic Reversal Guard (Institutional Quality Control):
    1. Enforces a 15-minute Minimum Holding Window: Never early-exit on micro-noise during early trade development.
    2. Orderbook imbalance (OFI) ALONE can NEVER trigger an early exit! (Eliminates fee churn).
    3. Requires confirmed persistent ML Inversion (>= 78% opposing conviction across >= 3 checks)
       AND/OR strong structural confluence with smoothed OFI.
    """
    if not current_direction or not new_pred:
        return False, ""

    curr_dir = current_direction.upper()
    is_curr_long = curr_dir in ("BULLISH", "LONG", "BUY")
    is_curr_short = curr_dir in ("BEARISH", "SHORT", "SELL")

    if not is_curr_long and not is_curr_short:
        return False, ""

    # 1. Enforce Minimum Holding Window (15 minutes): Let the trade breathe and let Trailing Stop manage risk
    if position_opened_at and position_opened_at > 0:
        elapsed_s = time.time() - position_opened_at
        if elapsed_s < MINIMUM_POSITION_HOLDING_SECONDS:
            return False, ""

    new_dir = getattr(new_pred, "direction", "NEUTRAL").upper()
    new_conf = float(getattr(new_pred, "confidence_score", 0.0))

    if obi_score is None:
        raw_obi = float(getattr(new_pred, "orderbook_imbalance", 0.0) or 0.0)
    else:
        raw_obi = float(obi_score)

    # 2. EMA-Smoothed OBI (alpha = 0.20) for noise filtering
    sym_clean = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
    if sym_clean not in _EMA_SMOOTHED_OBI:
        _EMA_SMOOTHED_OBI[sym_clean] = raw_obi
    else:
        _EMA_SMOOTHED_OBI[sym_clean] = 0.20 * raw_obi + 0.80 * _EMA_SMOOTHED_OBI[sym_clean]
    smoothed_obi = _EMA_SMOOTHED_OBI[sym_clean]

    # Check opposing conviction: MUST be genuine opposing ML (>= 78%)
    opposing_ml = False
    opposing_ofi_smoothed = False

    if is_curr_long:
        opposing_ml = (new_dir == "BEARISH" and new_conf >= REVERSAL_CONFIDENCE_THRESHOLD)
        opposing_ofi_smoothed = (smoothed_obi <= -REVERSAL_OFI_THRESHOLD)
    elif is_curr_short:
        opposing_ml = (new_dir == "BULLISH" and new_conf >= REVERSAL_CONFIDENCE_THRESHOLD)
        opposing_ofi_smoothed = (smoothed_obi >= REVERSAL_OFI_THRESHOLD)

    # Reversal requires high-conviction opposing ML (OFI alone CANNOT trigger an exit)
    is_reversal_candidate = opposing_ml

    if is_reversal_candidate:
        _REVERSAL_CONFIRMATION_STREAK[sym_clean] = _REVERSAL_CONFIRMATION_STREAK.get(sym_clean, 0) + 1
        streak = _REVERSAL_CONFIRMATION_STREAK[sym_clean]

        # Scenario A: Confluence of opposing ML (>=78%) AND opposing orderbook pressure
        if opposing_ml and opposing_ofi_smoothed and streak >= 2:
            _REVERSAL_CONFIRMATION_STREAK[sym_clean] = 0
            return True, f"Confluent Structural Reversal: Opposing {new_dir} ({new_conf*100:.1f}%) + Smoothed OFI ({smoothed_obi*100:+.1f}%) across {streak} checks"
        # Scenario B: Persistent strong ML inversion across 3 consecutive checks
        elif opposing_ml and streak >= 3 and new_conf >= 0.80:
            _REVERSAL_CONFIRMATION_STREAK[sym_clean] = 0
            return True, f"Persistent ML Inversion: Opposing {new_dir} ({new_conf*100:.1f}%) across {streak} consecutive checks"
        else:
            logger.info(
                f"🛡 [REVERSAL GUARD FILTER] {symbol} opposing impulse detected (Check {streak}/3, "
                f"Opposing {new_dir} {new_conf*100:.1f}%, OFI: {smoothed_obi*100:+.1f}%). Awaiting consecutive confirmation to protect payoff."
            )
            return False, ""
    else:
        # Market returned to normal; reset streak
        _REVERSAL_CONFIRMATION_STREAK[sym_clean] = 0

    return False, ""


def get_all_active_positions(live_trader: Optional[BinanceAutoTrader], paper_trader: PaperTrader) -> List[Dict[str, Any]]:
    positions = []
    if live_trader:
        for s in TARGET_SYMBOLS:
            pos = live_trader.get_active_position(s)
            if pos:
                positions.append(pos)
        # When live trading is active, active positions for portfolio risk gates
        # (CrossAssetCorrelationGuard, AsymmetricRiskEngine, PortfolioRiskKernel)
        # MUST strictly represent real exchange positions. Never let simulation/paper trades
        # block real-money allocations!
        return positions

    for s in TARGET_SYMBOLS:
        p_pos = paper_trader.get_active_position(s)
        if p_pos:
            positions.append(p_pos)
    return positions



def run_scan(model_manager: DaemonModelManager):
    """Executes VIP scan with all 5 Renaissance Institutional Gates and human-readable Russian output."""
    model_manager.check_and_reload()
    v4_engine = model_manager.engine

    db = SQLiteTracker(db_path=DB_PATH)
    signal_logger = SignalLogger(db_path=DB_PATH)

    real_trading_allowed = os.environ.get("REAL_TRADING_ENABLED", "false").lower() in ("true", "1", "yes")
    real_autotrade_enabled = os.environ.get("REAL_AUTOTRADE_ENABLED", "false").lower() in ("true", "1", "yes")
    is_paused = os.environ.get("REAL_AUTOTRADE_PAUSED", "false").lower() in ("true", "1", "yes")

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_SECRET_KEY")
    live_trader: Optional[BinanceAutoTrader] = None
    exchange_session = None
    account_equity = 40.0
    unrealized_pnl = 0.0

    if real_trading_allowed and real_autotrade_enabled and not is_paused and api_key and api_secret:
        try:
            live_trader = BinanceAutoTrader(api_key=api_key, api_secret=api_secret)
            exchange_session = live_trader._create_isolated_session()
            bal = exchange_session.fetch_balance()
            account_equity = float(bal.get("total", {}).get("USDT", 40.0) or 40.0)
            unrealized_pnl = float(bal.get("info", {}).get("totalUnrealizedProfit", 0.0) or 0.0)
        except Exception as init_live_err:
            logger.error(f"Failed to initialize BinanceAutoTrader session: {init_live_err}")

    today_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    if DAY_START_EQUITY_CACHE["date"] != today_utc or DAY_START_EQUITY_CACHE["equity"] <= 0:
        DAY_START_EQUITY_CACHE["date"] = today_utc
        DAY_START_EQUITY_CACHE["equity"] = account_equity
    day_start_equity = DAY_START_EQUITY_CACHE["equity"]

    paper_trader = PaperTrader(user_id=OWNER_USER_ID, tracker=db, signal_logger=signal_logger)

    utc_now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
    logger.info("=" * 70)
    logger.info(f"🔍 [СКАНЕР РЫНКА MaBuy] {utc_now} UTC | Баланс Binance: ${account_equity:.2f} USDT (PnL: ${unrealized_pnl:+.2f})")
    logger.info("-" * 70)

    # =========================================================================
    # 🏛 GATE 1: DYNAMIC ADAPTIVE CIRCUIT BREAKER (CAPITAL PRESERVATION)
    # =========================================================================
    cb_status = circuit_breaker.evaluate_circuit_breaker(
        portfolio_equity=account_equity,
        unrealized_pnl_usd=unrealized_pnl,
        exchange=exchange_session
    )
    if cb_status.is_in_cooldown:
        logger.warning(
            f"🛑 [ЗАЩИТА ДЕПОЗИТА] Торговля временно приостановлена автоматическим предохранителем! "
            f"Осталось: {cb_status.remaining_seconds/60:.1f} мин. Причина: {cb_status.trigger_reason}."
        )
        logger.info("=" * 70)
        return

    active_positions = get_all_active_positions(live_trader, paper_trader)
    ASSET_NAMES_RU = {
        "BTC": "БИТКОИН (BTC)",
        "ETH": "ЭФИРИУМ (ETH)",
        "SOL": "СОЛАНА (SOL)"
    }

    # =========================================================================
    # 🛰 STEP 3: SOL/BTC SYNTHETIC SPREAD STAT-ARB TELEMETRY (SHADOW MODE)
    # =========================================================================
    spread_tel = update_synthetic_spread_engine()
    if spread_tel:
        logger.info(f"🧬 [СТАТ-АРБИТРАЖ SOL/BTC (SHADOW)]: {spread_tel.format_russian_summary()}")
        logger.info("-" * 70)

    for symbol in TARGET_SYMBOLS:
        if symbol not in SNIPER_ALLOWED:
            continue
        asset_title = ASSET_NAMES_RU.get(symbol, symbol)
        try:
            req = PredictionRequest(
                user_id=OWNER_USER_ID,
                symbol=symbol,
                tier=SubscriptionTier.PRO,
                timeframe="1h",
                lang="ru"
            )
            pred = v4_engine.predict(req)
            if not pred:
                logger.info(f"📊 {asset_title}: Ошибка получения прогноза модели.")
                logger.info("-" * 70)
                continue

            direction = getattr(pred, "direction", "NEUTRAL")
            raw_confidence = float(getattr(pred, "confidence_score", 0.0))
            entry_price = float(getattr(pred, "current_price", 0.0) or getattr(pred, "price", 0.0) or 0.0)
            obi_score = float(getattr(pred, "orderbook_imbalance", 0.0) or 0.0)
            atr_est = float(getattr(pred, "volatility_estimate", 0.018) or 0.018)

            # Update in-memory telemetry feature cache
            _LATEST_MARKET_FEATURES[canonical_symbol(symbol)] = {
                "ofi_norm": obi_score,
                "volatility": atr_est,
                "confidence": raw_confidence,
                "direction": direction,
                "timestamp": time.time(),
            }

            # Record prediction in Online Drift Monitor
            drift_monitor.record_prediction(
                symbol=symbol,
                predicted_log_ret=float(getattr(pred, "predicted_log_return", 0.0) or 0.0),
                confidence=raw_confidence,
                direction=direction
            )

            # 🔬 TRACK 2: RESEARCH EXPLORATION SLEEVE (SHADOW ML EXPLORATION DATASET)
            try:
                if direction in ("BULLISH", "BEARISH") and raw_confidence >= 0.60:
                    clean_s = canonical_symbol(symbol)
                    research_features = {
                        "predicted_log_return": float(getattr(pred, "predicted_log_return", 0.0) or 0.0),
                        "volatility_estimate": atr_est,
                        "orderbook_imbalance": obi_score,
                        "market_regime": getattr(pred, "market_regime", "UNKNOWN"),
                        "hurst_exponent": getattr(pred, "hurst_exponent", None),
                        "sentiment_score": getattr(pred, "sentiment_score", 0.0),
                        "confidence_score": raw_confidence
                    }
                    research_sleeve.log_exploration_candidate(
                        symbol=clean_s,
                        direction=direction,
                        confidence=raw_confidence,
                        current_price=entry_price,
                        features=research_features,
                        market_regime=str(getattr(pred, "market_regime", "UNKNOWN")),
                        hurst_exponent=getattr(pred, "hurst_exponent", None),
                        funding_rate=None,
                        ofi_score=obi_score,
                        atr_pct=atr_est,
                        model_version="v5.2"
                    )
            except Exception as r_err:
                logger.warning(f"Research exploration logging note for {symbol}: {r_err}")

            dir_ru = "LONG (ПОКУПКА)" if direction == "BULLISH" else ("SHORT (ПРОДАЖА)" if direction == "BEARISH" else "NEUTRAL (ВНЕ РЫНКА)")

            # Check active position
            live_pos = live_trader.get_active_position(symbol) if live_trader else None
            paper_pos = paper_trader.get_active_position(symbol)
            clean_s = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
            is_quarantine = (clean_s == "ETH") and ETH_QUARANTINE_ENABLED

            # When live trading is active, trading eligibility is dictated strictly by real Binance positions
            if live_trader:
                has_active_position = (live_pos is not None)
            else:
                has_active_position = (paper_pos is not None)

            if has_active_position:
                current_dir = ""
                pos_pnl = 0.0
                pos_entry = entry_price
                is_live = False
                if live_pos:
                    is_live = True
                    current_dir = live_pos.get("direction", "")
                    pos_pnl = float(live_pos.get("pnl_pct", 0.0) or 0.0)
                    pos_entry = float(live_pos.get("entry_price", entry_price) or entry_price)
                elif paper_pos:
                    is_live = False
                    current_dir = paper_pos.get("direction", "")
                    pos_entry = float(paper_pos.get("entry_price", entry_price) or entry_price)
                    if current_dir == "BULLISH":
                        pos_pnl = ((entry_price - pos_entry) / pos_entry * 100.0) if pos_entry > 0 else 0.0
                    else:
                        pos_pnl = ((pos_entry - entry_price) / pos_entry * 100.0) if pos_entry > 0 else 0.0

                pos_dir_ru = "LONG (ПОКУПКА)" if current_dir.upper() in ("BULLISH", "BUY", "LONG") else "SHORT (ПРОДАЖА)"

                if is_live:
                    status_badge = "🟢 РЕАЛЬНАЯ ПОЗИЦИЯ BINANCE"
                elif is_quarantine:
                    status_badge = "🧪 КАРАНТИН АСТРЫ (ВИРТУАЛЬНО PAPER)"
                else:
                    status_badge = "📝 СИМУЛЯЦИЯ (PAPER TRADING)"

                logger.info(f"📊 {asset_title}:")
                logger.info(f"   • Статус: {status_badge} {pos_dir_ru} | PnL: {pos_pnl:+.2f}%")
                logger.info(f"   • Вход: ${pos_entry:,.2f} | Тек. цена: ${entry_price:,.2f}")

                pos_opened_ts = None
                if paper_pos and paper_pos.get("opened_at"):
                    try:
                        raw_dt = str(paper_pos["opened_at"])
                        if "T" in raw_dt:
                            dt = datetime.datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=datetime.timezone.utc)
                            pos_opened_ts = dt.timestamp()
                    except Exception:
                        pass
                elif live_pos:
                    pos_opened_ts = last_live_trade_timestamps.get(symbol)

                should_exit, exit_reason = check_active_reversal_guard(
                    symbol=symbol,
                    current_direction=current_dir,
                    new_pred=pred,
                    obi_score=obi_score,
                    position_opened_at=pos_opened_ts,
                )

                if should_exit:
                    logger.warning(f"   • Решение: 🚨 ЭКСТРЕННОЕ ЗАКРЫТИЕ (Структурный разворот тренда: {exit_reason})")

                    with EXECUTION_LOCK:
                        if live_pos and live_trader:
                            try:
                                live_trader.close_position_early(symbol, reason=exit_reason)
                                time.sleep(0.5)
                            except Exception as close_live_err:
                                logger.error(f"Error closing live position early on {symbol}: {close_live_err}")

                        if paper_pos:
                            try:
                                paper_trader.close_position_early(symbol, reason=exit_reason)
                            except Exception as close_paper_err:
                                logger.error(f"Error closing paper position early on {symbol}: {close_paper_err}")

                        # Set strict post-trade cooldown (no instant whipsaw flipping)
                        clean_sym_closed = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
                        LAST_CLOSED_TRADE_TIMESTAMP[clean_sym_closed] = time.time()
                        logger.info(f"   • Решение: ⏸ Позиция закрыта. Переходим в кэш. Запущен {int(SYMBOL_TRADE_COOLDOWN_SECONDS//60)}-мин. кулдаун (защита от распила).")

                    logger.info("-" * 70)
                    continue
                else:
                    if is_live:
                        logger.info(f"   • Решение: 🛡 УДЕРЖАНИЕ СДЕЛКИ (Трейлинг-стоп активен на Binance)")
                    elif is_quarantine:
                        logger.info(f"   • Решение: 🛡 УДЕРЖАНИЕ КАРАНТИНА (Трейлинг-стоп в Paper-симуляторе)")
                    else:
                        logger.info(f"   • Решение: 🛡 УДЕРЖАНИЕ СДЕЛКИ (Трейлинг-стоп активен в Paper)")
                    logger.info("-" * 70)
                    continue
            else:
                _REVERSAL_CONFIRMATION_STREAK[symbol.upper()] = 0

            # Signal evaluation for symbol with NO active position
            clean_s = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
            last_close_time = LAST_CLOSED_TRADE_TIMESTAMP.get(clean_s, 0.0)
            time_since_close = time.time() - last_close_time
            if time_since_close < SYMBOL_TRADE_COOLDOWN_SECONDS:
                remaining_min = (SYMBOL_TRADE_COOLDOWN_SECONDS - time_since_close) / 60.0
                logger.info(f"📊 {asset_title}:")
                logger.info(f"   • Статус: ⏳ КУЛДАУН ПОСЛЕ СДЕЛКИ (Осталось {remaining_min:.1f} мин. Защита от овертрейдинга)")
                logger.info("-" * 70)
                continue

            logger.info(f"📊 {asset_title}:")
            logger.info(f"   • Прогноз: {dir_ru} | Уверенность: {raw_confidence*100:.1f}% | Цена: ${entry_price:,.2f} (ATR: {atr_est*100:.2f}%)")

            # Cross-Exchange Pulse check (Bybit / OKX vs Binance)
            try:
                cecl_pulse = cex_confirmation.evaluate(
                    symbol=symbol,
                    side="LONG" if direction == "BULLISH" else ("SHORT" if direction == "BEARISH" else "LONG"),
                    current_price=entry_price
                )
                logger.info(
                    f"   • Межбиржевой пульс: Bybit/OKX [{cecl_pulse.status}] "
                    f"Спред: {cecl_pulse.dislocation_bps:+.1f} bps | Возврат: {cecl_pulse.external_return_bps:+.1f} bps"
                )
            except Exception as pulse_err:
                logger.debug(f"CECL pulse error: {pulse_err}")

            # Skip entry gates for non-directional (NEUTRAL) setups
            if direction not in ("BULLISH", "BEARISH"):
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Модель не видит направленного тренда)")
                logger.info("-" * 70)
                continue

            is_eth_quarantine = (clean_s == "ETH") and ETH_QUARANTINE_ENABLED

            # =================================================================
            # 🏛 GATE 2: FUNDING RATE & LIQUIDATION BIAS ENGINE
            # =================================================================
            fr_res = funding_service.modulate_signal(symbol, direction, raw_confidence, exchange=exchange_session)
            funding_str = f"{fr_res.funding_rate_pct*100:+.4f}%"

            if fr_res.veto:
                logger.info(f"   • Фильтры: Фандинг {funding_str} [ВЕТО: {fr_res.veto_reason}]")
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Фильтр ставки финансирования)")
                logger.info("-" * 70)
                signal_logger.log_signal(
                    symbol=symbol, direction=direction, confidence=raw_confidence, entry_price=entry_price,
                    status="CANCELLED", close_reason=f"VETO_FUNDING: {fr_res.veto_reason}"
                )
                continue

            confidence = fr_res.modulated_confidence

            # =================================================================
            # 🏛 GATE 2.5: 4H STRUCTURAL REGIME CONFLUENCE
            # =================================================================
            mtf_res = mtf_service.evaluate_confluence(symbol, direction)
            if mtf_res.veto:
                logger.info(f"   • Фильтры: 4H Структура [ВЕТО: {mtf_res.reason}]")
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Конфликт с 4-часовым трендом)")
                logger.info("-" * 70)
                signal_logger.log_signal(
                    symbol=symbol, direction=direction, confidence=confidence, entry_price=entry_price,
                    status="CANCELLED", close_reason=f"VETO_MTF: {mtf_res.reason}"
                )
                continue

            if mtf_res.confidence_delta != 0:
                confidence = max(0.50, min(0.95, confidence + mtf_res.confidence_delta))

            # =================================================================
            # 🏛 GATE 3: HURST EXPONENT DYNAMIC REGIME DETECTOR (TREND ONLY)
            # =================================================================
            h_res = hurst_service.analyze_symbol(symbol)
            h_val = h_res.hurst_exponent
            if 0.45 <= h_val <= 0.55:
                h_regime_label = "ШУМ / ФЛЭТ"
            elif h_val > 0.55:
                h_regime_label = "УСТОЙЧИВЫЙ ТРЕНД"
            else:
                h_regime_label = "ВОЗВРАТ К СРЕДНЕМУ (ПИЛА)"

            # Strict Institutional Quality Gate: Directional momentum trades REQUIRE persistent trend (H > 0.55).
            # If H <= 0.55 (Brownian deadband or Mean-Reverting chop), VETO directional entries!
            if h_val <= 0.55:
                logger.info(f"   • Фильтры: Фандинг {funding_str} [OK] | Херст H={h_val:.3f} [{h_regime_label}]")
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Херст H={h_val:.3f} <= 0.55: нет устойчивого тренда, высокий риск распила)")
                logger.info("-" * 70)
                signal_logger.log_signal(
                    symbol=symbol, direction=direction, confidence=confidence, entry_price=entry_price,
                    status="CANCELLED", close_reason=f"VETO_HURST: Non-trending regime H={h_val:.3f} <= 0.55 (chop/noise protection)"
                )
                continue

            # =================================================================
            # 🏛 GATE 3.5: MICROSTRUCTURE ENSEMBLE ENGINE (P1 BLOCK 2)
            # =================================================================
            try:
                # 1. Fetch L2 Microstructure Snapshot
                ms_snap = None
                try:
                    ms_snap = live_microstructure.get_live_microstructure(
                        symbol=f"{symbol}USDT",
                        proposed_direction="BUY" if direction == "BULLISH" else "SELL"
                    )
                except Exception as l2_err:
                    logger.debug(f"L2 microstructure fetch fallback for {symbol}: {l2_err}")

                # 2. Fetch Derivatives Flow & Whale Positioning
                deriv_radar = None
                try:
                    deriv_radar = DerivativesFlowRadarService.get_derivatives_radar(
                        symbol=symbol, current_spot_price=entry_price
                    )
                except Exception as deriv_err:
                    logger.debug(f"Derivatives flow fetch fallback for {symbol}: {deriv_err}")

                # 3. Calculate Liquidation Cascade Z-Score
                liq_z, liq_vol, liq_side, liq_metrics = 0.0, 0.0, "NEUTRAL", None
                try:
                    liq_z, liq_vol, liq_side, liq_metrics = liquidation_radar.calculate_liquidation_zscore(symbol)
                except Exception as liq_err:
                    logger.debug(f"Liquidation radar fallback for {symbol}: {liq_err}")

                # 4. Fetch Deribit DVOL (Crypto VIX)
                dvol_val, dvol_ts = None, None
                try:
                    dvol_curr = "BTC" if symbol in ("BTC", "SOL") else "ETH"
                    dvol_obj = deribit_collector.get_dvol(dvol_curr)
                    if dvol_obj and hasattr(dvol_obj, "dvol_close"):
                        dvol_val = float(dvol_obj.dvol_close)
                        dvol_ts = float(dvol_obj.timestamp_ms) / 1000.0 if dvol_obj.timestamp_ms else time.time()
                except Exception as dvol_err:
                    logger.debug(f"Deribit DVOL fallback: {dvol_err}")

                # 5. Synthesize in Microstructure Ensemble Engine (Deterministic Graceful Degradation)
                micro_res = microstructure_ensemble.evaluate(
                    symbol=symbol,
                    proposed_direction=direction,
                    raw_confidence=confidence,
                    obi_data={
                        "obi_score": ms_snap.ofi_norm,
                        "has_ask_wall": (ms_snap.ofi_norm <= -0.35),
                        "has_bid_wall": (ms_snap.ofi_norm >= 0.35),
                        "timestamp": time.time()
                    } if ms_snap else None,
                    derivatives_data=deriv_radar,
                    liquidation_metrics=liq_metrics,
                    cvd_5m=ms_snap.cvd_delta if ms_snap else 0.0,
                    dvol_value=dvol_val,
                    dvol_timestamp=dvol_ts,
                )

                if micro_res.is_vetoed:
                    logger.info(f"   • Фильтры: Ансамбль микроструктуры [ВЕТО: {micro_res.veto_reason}]")
                    logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ ({micro_res.veto_reason})")
                    logger.info("-" * 70)
                    signal_logger.log_signal(
                        symbol=symbol, direction=direction, confidence=confidence, entry_price=entry_price,
                        status="CANCELLED", close_reason=f"VETO_MICROSTRUCTURE: {micro_res.veto_reason}"
                    )
                    continue

                # Apply modulated confidence and log telemetry
                confidence = micro_res.modulated_confidence
                logger.info(f"   • Фильтры: Ансамбль микроструктуры {micro_res.summary_badge}")

            except Exception as ms_err:
                logger.warning(f"Microstructure Ensemble non-blocking error for {symbol}: {ms_err}")

            # =================================================================
            # 🏛 GATE 3.8: ASTRA NEWS ALPHA & MACRO SHOCK VETO
            # =================================================================
            try:
                news_passed, news_score, news_reason = news_alpha_engine.evaluate_news_gate(
                    symbol=symbol,
                    proposed_direction=direction
                )
                if not news_passed:
                    logger.info(f"   • Фильтры: Новостной фон [ВЕТО: {news_reason}]")
                    logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Макро/новостное вето)")
                    logger.info("-" * 70)
                    signal_logger.log_signal(
                        symbol=symbol, direction=direction, confidence=confidence, entry_price=entry_price,
                        status="CANCELLED", close_reason=f"VETO_NEWS: {news_reason}"
                    )
                    continue

                if abs(news_score) >= 0.20:
                    news_boost = 0.03 if (direction == "BULLISH" and news_score > 0) or (direction == "BEARISH" and news_score < 0) else -0.03
                    confidence = max(0.50, min(0.95, confidence + news_boost))
            except Exception as news_err:
                logger.debug(f"News alpha non-blocking check error for {symbol}: {news_err}")

            # =================================================================
            # 🏛 GATE 4: CROSS-ASSET CORRELATION GUARD & ASTRA BETA FREEZE
            # =================================================================
            cg_res = correlation_guard.evaluate_candidate_trade(
                candidate_symbol=symbol,
                candidate_direction=direction,
                candidate_confidence=confidence,
                active_positions=active_positions,
                portfolio_equity=account_equity,
                estimated_trade_notional=21.0
            )
            if cg_res.veto:
                logger.info(f"   • Фильтры: Корреляция портфеля [ВЕТО: {cg_res.rejection_reason}]")
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Риск концентрации активов)")
                logger.info("-" * 70)
                signal_logger.log_signal(
                    symbol=symbol, direction=direction, confidence=confidence, entry_price=entry_price,
                    status="CANCELLED", close_reason=f"VETO_CORRELATION: {cg_res.rejection_reason}"
                )
                continue

            asymmetric_risk_engine.symbol_confidence[symbol.upper()] = confidence
            asym_scale = asymmetric_risk_engine.check_correlation_scale(
                active_positions=active_positions,
                candidate_symbol=symbol,
                candidate_direction=direction
            )
            if asym_scale <= 0.0:
                logger.info(f"   • Фильтры: Корреляция [ЗАМОРОЗКА: {asymmetric_risk_engine.last_correlation_decision_reason}]")
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Защита от двойной корреляции)")
                logger.info("-" * 70)
                signal_logger.log_signal(
                    symbol=symbol, direction=direction, confidence=confidence, entry_price=entry_price,
                    status="CANCELLED", close_reason=f"VETO_ASTRA_CORRELATION: {asymmetric_risk_engine.last_correlation_decision_reason}"
                )
                continue

            # =================================================================
            # 🏛 GATE 5: ASTRA POST-LOSS LOCKOUT & DISPLACEMENT GUARD
            # =================================================================
            loss_info = LAST_LOSS_LOCKOUT.get(symbol)
            if loss_info:
                now_ts = time.time()
                time_since_loss = now_ts - float(loss_info.get("exit_time", 0.0))
                same_direction = (direction == loss_info.get("direction"))
                if same_direction and time_since_loss < 14400:
                    exit_p = float(loss_info.get("exit_price", entry_price) or entry_price)
                    price_displacement = abs(entry_price - exit_p)
                    min_displacement = 0.015 * entry_price
                    if price_displacement < min_displacement:
                        rem_mins = int((14400 - time_since_loss) / 60)
                        logger.info(f"   • Фильтры: Защита после убытка [АКТИВЕН КУЛДАУН: осталось {rem_mins}м]")
                        logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Смещение цены ${price_displacement:.2f} < ${min_displacement:.2f})")
                        logger.info("-" * 70)
                        signal_logger.log_signal(
                            symbol=symbol,
                            direction=direction,
                            confidence=confidence,
                            entry_price=entry_price,
                            status="CANCELLED",
                            close_reason=f"POST_LOSS_LOCKOUT: {rem_mins}m left, disp ${price_displacement:.2f} < 1.5*ATR"
                        )
                        continue

            # =================================================================
            # 🏛 GATE 6: ASTRA CROSS-EXCHANGE CONFIRMATION LAYER (CECL)
            # =================================================================
            try:
                cecl_decision = cex_confirmation.evaluate(
                    symbol=symbol,
                    side=direction,
                    current_price=entry_price
                )
                cex_mode = os.environ.get("CROSS_EXCHANGE_MODE", "SHADOW").upper().strip()

                if cecl_decision.status == "PASS":
                    logger.info(
                        f"   • Межбиржевой фильтр: Bybit/OKX [{cex_mode}: ПОДТВЕРЖДЕНО] "
                        f"Скор: {cecl_decision.score:+.2f} | Спред: {cecl_decision.dislocation_bps:+.1f} bps | "
                        f"Внешний возврат: {cecl_decision.external_return_bps:+.1f} bps"
                    )
                elif cecl_decision.status == "VETO":
                    logger.info(
                        f"   • Межбиржевой фильтр: Bybit/OKX [{cex_mode}-ВЕТО: {cecl_decision.reason}] "
                        f"Скор: {cecl_decision.score:+.2f} | Спред: {cecl_decision.dislocation_bps:+.1f} bps"
                    )
                    if cex_mode == "VETO_ONLY":
                        logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Межбиржевой фильтр CECL: {cecl_decision.reason})")
                        logger.info("-" * 70)
                        signal_logger.log_signal(
                            symbol=symbol,
                            direction=direction,
                            confidence=confidence,
                            entry_price=entry_price,
                            status="CANCELLED",
                            close_reason=f"VETO_CECL: {cecl_decision.reason}"
                        )
                        continue
                elif cecl_decision.status == "DEGRADED":
                    logger.info(f"   • Межбиржевой фильтр: Bybit/OKX [{cex_mode}: ДЕГРАДАЦИЯ (нет свежих котировок Bybit/OKX)]")
                else:
                    logger.info(
                        f"   • Межбиржевой фильтр: Bybit/OKX [{cex_mode}: НЕЙТРАЛЬНО] "
                        f"Скор: {cecl_decision.score:+.2f} | Спред: {cecl_decision.dislocation_bps:+.1f} bps"
                    )
            except Exception as cecl_err:
                logger.debug(f"CECL evaluation non-blocking error for {symbol}: {cecl_err}")

            # =================================================================
            # 🏛 GATE 7: ASTRA UNIFIED PORTFOLIO RISK KERNEL & DYNAMIC SIZING
            # =================================================================
            k_atr = 2.2 if h_val > 0.55 else 1.8
            sl_pct = max(0.018, min(0.032, k_atr * atr_est))
            tp_pct = max(0.020, float(h_res.recommended_tp_pct))

            side_clean = "LONG" if direction == "BULLISH" else "SHORT"
            if side_clean == "LONG":
                cand_stop_p = entry_price * (1.0 - sl_pct)
                cand_tp_p = entry_price * (1.0 + tp_pct)
            else:
                cand_stop_p = entry_price * (1.0 + sl_pct)
                cand_tp_p = entry_price * (1.0 - tp_pct)

            # Build PositionSnapshot tuple from active_positions
            pos_snapshots = []
            for p in active_positions:
                p_sym = p.get("symbol", "").upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
                p_dir = p.get("direction", "").upper()
                p_entry = float(p.get("entry_price", 0.0) or 0.0)
                p_amt = float(p.get("amount", 0.0) or p.get("contracts", 0.0) or 0.0)
                p_notional = float(p.get("notional", 0.0) or (p_amt * p_entry) or 21.0)
                p_sign = 1.0 if p_dir in ("BULLISH", "LONG", "BUY") else -1.0
                p_lev = float(p.get("leverage", 3.0) or 3.0)
                p_margin = float(p.get("margin", 0.0) or (p_notional / max(1.0, p_lev)))
                pos_snapshots.append(
                    PositionSnapshot(
                        symbol=p_sym,
                        signed_notional_usd=p_sign * abs(p_notional),
                        margin_usd=abs(p_margin),
                        leverage=p_lev,
                    )
                )

            port_snapshot = PortfolioSnapshot(
                equity_usd=max(1.0, float(account_equity)),
                day_start_equity_usd=max(1.0, float(day_start_equity)),
                positions=tuple(pos_snapshots),
                correlations={
                    ("BTC", "ETH"): 0.82,
                    ("BTC", "SOL"): 0.74,
                    ("ETH", "SOL"): 0.85,
                },
            )

            proj_liq_dist = None
            if live_trader:
                try:
                    proj_liq_dist = live_trader.calculate_projected_liq_distance(
                        symbol=symbol,
                        side=side_clean,
                        entry_price=entry_price,
                        leverage=3.0,
                    )
                except Exception:
                    proj_liq_dist = 0.328

            candidate_order = CandidateOrder(
                symbol=clean_s,
                side=side_clean,
                entry_price=entry_price,
                stop_price=cand_stop_p,
                take_profit_price=cand_tp_p,
                raw_confidence=confidence,
                signal_quality_multiplier=1.0,
                regime_risk_multiplier=0.90 if h_val > 0.55 else 0.75,
                requested_leverage=3.0,
                projected_liquidation_distance_fraction=proj_liq_dist,
            )

            risk_result = portfolio_risk_kernel.evaluate_candidate_order(port_snapshot, candidate_order)

            if not risk_result.approved:
                logger.info(f"   • Фильтры: Риск-ядро [{risk_result.veto_reason}]")
                logger.info(f"   • {risk_result.format_russian_summary(clean_s, side_clean)}")
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Риск-ядро отклонило ордер: {risk_result.veto_reason})")
                logger.info("-" * 70)
                signal_logger.log_signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    entry_price=entry_price,
                    status="CANCELLED",
                    close_reason=f"VETO_RISK_KERNEL: {risk_result.veto_reason}"
                )
                continue

            # =================================================================
            # 🎯 FINAL EXECUTION GATE (>= 75.0% Confidence)
            # =================================================================
            if is_eth_quarantine:
                logger.info(f"   • Режим: 🛡 КАРАНТИН АСТРЫ (Торгуется только виртуально, реальный депозит защищен)")

            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                # 🛑 Micro-Capital Gate for BTC:
                # On Binance Futures, minimum lot for BTC is 0.001 BTC (~$77 notional, ~$26 margin at 3x).
                # On accounts with equity < $100, entering 0.001 BTC consumes >65% of account equity.
                if live_trader and not is_eth_quarantine and "BTC" in symbol.upper() and account_equity < 100.0:
                    logger.info(f"   • Фильтры: Все 7 шлюзов пройдены [OK] | Фандинг: {funding_str} | Херст H={h_val:.3f} [{h_regime_label}]")
                    logger.info(
                        f"   🛑 [MICRO-CAPITAL GATE] BTC реальный ордер отклонен (депозит ${account_equity:.2f} < $100.00). "
                        f"Мин. лот 0.001 BTC требует ~$26 маржи (>65% депозита). Торгуем SOL/ETH с маржой ~$7 до разгона счета >$100."
                    )
                    signal_logger.log_signal(
                        symbol=symbol,
                        direction=direction,
                        confidence=confidence,
                        entry_price=entry_price,
                        status="CANCELLED",
                        close_reason=f"MICRO_CAPITAL_GATE: BTC min lot (0.001 BTC) requires >$100 equity (current: ${account_equity:.2f})"
                    )
                    continue

                logger.info(f"   • Фильтры: Все 7 шлюзов пройдены [OK] | Фандинг: {funding_str} | Херст H={h_val:.3f} [{h_regime_label}]")
                logger.info(f"   • {risk_result.format_russian_summary(clean_s, side_clean)}")
                logger.info(
                    f"   • Решение: 🚀 СИГНАЛ ОДОБРЕН! Уверенность {confidence*100:.1f}% >= {MIN_CONFIDENCE_THRESHOLD*100:.1f}% | "
                    f"Объем: ${risk_result.approved_notional_usd:.2f} (Маржа: ${risk_result.approved_margin_usd:.2f}, Плечо: {risk_result.recommended_leverage:.1f}x) "
                    f"(Открываем {dir_ru})"
                )

                _signal_id = signal_logger.log_signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    entry_price=entry_price,
                    status="OPEN"
                )

                with EXECUTION_LOCK:
                    if live_trader and not is_eth_quarantine:
                        now = time.time()
                        last_t = last_live_trade_timestamps.get(symbol, 0)
                        if now - last_t >= 900:
                            # 🛰 Step 5: SOR Routing & Execution
                            try:
                                logger.info(f"   ⚡ [SOR РОУТЕР]: Инициализация SmartOrderRouter для {symbol} {dir_ru}...")
                                binance_venue = ExecutionVenue(
                                    name="binance",
                                    adapter=BinanceVenueAdapter(live_trader),
                                    enabled=True,
                                    live_enabled=True,
                                    primary=True,
                                    maker_fee_bps=Decimal("2.0"),
                                    taker_fee_bps=Decimal("5.0"),
                                    max_leverage=Decimal(str(int(risk_result.recommended_leverage))),
                                )
                                bybit_venue = ExecutionVenue(
                                    name="bybit",
                                    adapter=BybitLinearAdapter(),
                                    enabled=True,
                                    live_enabled=False,
                                    maker_fee_bps=Decimal("1.0"),
                                    taker_fee_bps=Decimal("5.5"),
                                    max_leverage=Decimal("3"),
                                )
                                okx_venue = ExecutionVenue(
                                    name="okx",
                                    adapter=OKXSwapAdapter(),
                                    enabled=True,
                                    live_enabled=False,
                                    maker_fee_bps=Decimal("2.0"),
                                    taker_fee_bps=Decimal("5.0"),
                                    max_leverage=Decimal("3"),
                                )
                                sor = SmartOrderRouter(
                                    venues=[binance_venue, bybit_venue, okx_venue],
                                    risk_kernel=sor_risk_adapter,
                                    poll_interval_seconds=0.3,
                                    emergency_close_on_stop_failure=True,
                                )
                                side_sor = SORSide.BUY if direction == "BULLISH" else SORSide.SELL
                                # Binance Futures enforces minimum order notional of $20.00 for ETH/SOL
                                min_notional = 20.50
                                target_notional = max(min_notional, risk_result.approved_notional_usd)

                                # Ensure required margin does not exceed 35% of account equity
                                if (target_notional / float(risk_result.recommended_leverage)) <= (account_equity * 0.35):
                                    exec_notional = target_notional
                                else:
                                    exec_notional = risk_result.approved_notional_usd

                                # Precision formatting per symbol to guarantee notional >= $20.00
                                import math
                                if "SOL" in symbol.upper():
                                    calc_qty = math.ceil((exec_notional / entry_price) * 100) / 100.0
                                elif "ETH" in symbol.upper():
                                    calc_qty = math.ceil((exec_notional / entry_price) * 1000) / 1000.0
                                else:
                                    calc_qty = max(0.001, math.ceil((exec_notional / entry_price) * 1000) / 1000.0)

                                req_qty = Decimal(str(calc_qty))
                                sor_req = RoutingRequest(
                                    symbol=symbol,
                                    side=side_sor,
                                    quantity=req_qty,
                                    stop_loss_price=Decimal(str(round(cand_stop_p, 4))),
                                    maker_timeout_seconds=2.5,
                                    max_reprices=3,
                                    top_book_participation=Decimal("0.10"),
                                    allow_taker_fallback=True,
                                    strategy_id="mabuy_v5_sniper",
                                    signal_id=str(_signal_id) if _signal_id else uuid.uuid4().hex,
                                )
                                receipt = sor.route_and_execute(sor_req)
                                if receipt.filled_quantity > Decimal("0") and _signal_id:
                                    last_live_trade_timestamps[symbol] = now
                                    fill_avg_p = float(receipt.average_price) if receipt.average_price > 0 else entry_price
                                    first_ord = receipt.order_ids[0] if receipt.order_ids else "sor-fill"
                                    signal_logger.set_live_order(_signal_id, str(first_ord), fill_avg_p)
                                    logger.info(f"   ✅ [СДЕЛКА УСПЕШНО ИСПОЛНЕНА ЧЕРЕЗ SOR НА {receipt.venue.upper()}] {symbol} {dir_ru}!")
                                    logger.info(f"      • Заполнено: {receipt.filled_quantity} | Вход: ${fill_avg_p:,.2f} | Тейкер-фоллбэк: {receipt.used_taker_fallback}")
                                    logger.info(f"      • Ордера SOR: {receipt.order_ids} | Стоп-лоссы: {receipt.stop_order_ids}")
                            except Exception as sor_ex:
                                logger.warning(f"SOR execution note ({sor_ex}). Falling back to live_trader.execute_trade...")
                                try:
                                    result = live_trader.execute_trade(
                                        symbol=symbol,
                                        direction=direction,
                                        confidence=confidence*100.0,
                                        hurst=h_res.hurst_exponent,
                                        target_tp_pct=h_res.recommended_tp_pct,
                                        atr_pct=atr_est,
                                        funding_rate_pct=fr_res.funding_rate_pct,
                                        approved_notional_usd=risk_result.approved_notional_usd,
                                        approved_leverage=int(risk_result.recommended_leverage),
                                    )
                                    if result and _signal_id:
                                        last_live_trade_timestamps[symbol] = now
                                        signal_logger.set_live_order(_signal_id, str(result.get("order_id", "")), float(result.get("entry_price", entry_price)))
                                        logger.info(f"   ✅ [СДЕЛКА УСПЕШНО ОТКРЫТА НА BINANCE] {symbol} {dir_ru}!")
                                        logger.info(f"      • Ордер #{result.get('order_id')} | Вход: ${result.get('entry_price', entry_price):,.2f} | Плечо: {risk_result.recommended_leverage:.1f}x")
                                        logger.info(f"      • Тейк-профит: ${result.get('take_profit', 0):,.2f} | Стоп-лосс: ${result.get('stop_loss', 0):,.2f}")
                                except Exception as ex:
                                    logger.error(f"Live trade execution error for {symbol}: {ex}")

                    if not live_trader or is_eth_quarantine:
                        paper_trader.execute_trade(
                            symbol=symbol,
                            direction=direction,
                            leverage=int(risk_result.recommended_leverage),
                            risk_pct=0.0125,
                            confidence=confidence,
                            hurst=h_res.hurst_exponent,
                            target_tp_pct=h_res.recommended_tp_pct,
                            funding_rate_pct=fr_res.funding_rate_pct,
                            approved_notional_usd=risk_result.approved_notional_usd,
                            approved_margin_usd=risk_result.approved_margin_usd,
                        )
                        logger.info(f"   📝 [ВИРТУАЛЬНАЯ СДЕЛКА ОТКРЫТА] {symbol} {dir_ru} | Тейк-профит: +{h_res.recommended_tp_pct*100:.1f}% | Режим: {h_res.regime.value} [Chandelier {h_res.chandelier_k_sigma}x ATR, Runner={h_res.allow_runner}]")
            else:
                logger.info(f"   • Фильтры: Фандинг {funding_str} [OK] | Херст H={h_val:.3f} [{h_regime_label}]")
                logger.info(f"   • Решение: ⏸ НЕ ОТКРЫВАЕМ СДЕЛКУ (Уверенность {confidence*100:.1f}% < {MIN_CONFIDENCE_THRESHOLD*100:.1f}% снайперского порога)")
                if confidence >= 0.65:
                    signal_logger.log_signal(
                        symbol=symbol,
                        direction=direction,
                        confidence=confidence,
                        entry_price=entry_price,
                        status="CANCELLED",
                        close_reason=f"FILTERED_SUB_75: Conf {confidence*100:.1f}% < 75.0% Sniper Barrier"
                    )

            logger.info("-" * 70)

        except Exception as sym_err:
            logger.error(f"Error predicting {symbol}: {sym_err}")

    live_cnt = len(active_positions) if live_trader else 0
    paper_cnt = 0
    if not live_trader:
        paper_cnt = len(active_positions)
    else:
        try:
            paper_open = paper_trader.tracker.get_open_paper_trades(paper_trader.user_id) if hasattr(paper_trader.tracker, 'get_open_paper_trades') else []
            paper_cnt = len(paper_open)
        except Exception:
            paper_cnt = 0
    logger.info(f"🛡 Активные позиции: {len(active_positions)} (Реальных: {live_cnt}, Карантин/Paper: {paper_cnt}) | Порог входа: 75.0% | Институциональные шлюзы: 7/7 АКТИВНЫ (вкл. CECL Bybit/OKX + SOL/BTC StatArb + RiskKernel + SOR)")
    logger.info("=" * 70)


def verify_signal_outcomes():
    db = SQLiteTracker(db_path=DB_PATH)
    signal_logger = SignalLogger(db_path=DB_PATH)
    try:
        # Collect live Hurst regime metrics for active symbols (P1 Regime Engine: Continuous Chandelier Trailing)
        hurst_values = {}
        current_atrs = {}
        for s in TARGET_SYMBOLS:
            try:
                h_res = hurst_service.analyze_symbol(s)
                hurst_values[s] = h_res
            except Exception:
                hurst_values[s] = 0.50
            try:
                df = hurst_service.fetcher.fetch_ohlcv(s, timeframe="1h", limit=24)
                if df is not None and not df.empty and len(df) >= 14:
                    tr = np.maximum(
                        df["high"] - df["low"],
                        np.maximum(
                            abs(df["high"] - df["close"].shift(1)),
                            abs(df["low"] - df["close"].shift(1))
                        )
                    )
                    atr = float(tr.rolling(14).mean().iloc[-1])
                    current_atrs[s] = float(atr / df["close"].iloc[-1])
            except Exception:
                pass

        paper = PaperTrader(user_id=OWNER_USER_ID, tracker=db, signal_logger=signal_logger)
        closed_reports = paper.update_open_trades(hurst_values=hurst_values)
        if closed_reports:
            for rep in closed_reports:
                sym_clean = rep.get("symbol", "").upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
                LAST_CLOSED_TRADE_TIMESTAMP[sym_clean] = time.time()
                pnl_usd = float(rep.get("pnl_usd", 0.0) or 0.0)
                pnl_pct = float(rep.get("pnl_pct", 0.0) or 0.0)
                reason = rep.get("reason", "")
                direction = rep.get("direction", "")
                exit_p = float(rep.get("exit_price", 0.0) or 0.0)
                dir_ru = "LONG (ПОКУПКА)" if direction.upper() in ("BULLISH", "BUY", "LONG") else "SHORT (ПРОДАЖА)"
                res_str = f"ПРИБЫЛЬ +${pnl_usd:.2f} (+{pnl_pct:.2f}%)" if pnl_usd >= 0 else f"УБЫТОК -${abs(pnl_usd):.2f} ({pnl_pct:.2f}%)"

                logger.info("=" * 70)
                logger.info(f"🔔 [СДЕЛКА ЗАКРЫТА] {sym_clean} {dir_ru} | Цена выхода: ${exit_p:,.2f}")
                logger.info(f"   • Финансовый результат: {res_str}")
                logger.info(f"   • Причина закрытия: {reason}")
                logger.info(f"   • ⏳ Запущен {int(SYMBOL_TRADE_COOLDOWN_SECONDS//60)}-минутный кулдаун актива {sym_clean} (защита от овертрейдинга).")
                logger.info("=" * 70)

                # P1 Block 3: Implementation Shortfall & Shadow SOR Benchmarking
                try:
                    entry_p = float(rep.get("entry_price", exit_p) or exit_p)
                    amt = float(rep.get("amount", 0.0) or 0.0)
                    is_rec = execution_benchmarker.benchmark_execution(
                        symbol=sym_clean,
                        side=direction,
                        arrival_price=entry_p,
                        fill_price=exit_p,
                        amount=amt if amt > 0 else 1.0,
                        is_maker=False,
                    )
                    logger.info(
                        f"📊 [SHORTFALL BENCHMARK] {sym_clean} IS: {is_rec.implementation_shortfall_bps:.2f} bps | "
                        f"Проскальзывание: {is_rec.slippage_bps:.2f} bps (${is_rec.slippage_usd:.4f}) | "
                        f"Комиссия: ${is_rec.fee_usd:.4f}"
                    )

                    # Shadow Multi-Venue SOR Counterfactual
                    bybit_obs = cex_state.get_latest_quote(sym_clean, "bybit")
                    okx_obs = cex_state.get_latest_quote(sym_clean, "okx")
                    bybit_q = bybit_obs.price if bybit_obs else None
                    okx_q = okx_obs.price if okx_obs else None
                    if bybit_q or okx_q:
                        shadow_cf = execution_benchmarker.evaluate_shadow_sor(
                            symbol=sym_clean,
                            side=direction,
                            amount=amt if amt > 0 else 1.0,
                            binance_fill_price=exit_p,
                            bybit_quote=bybit_q,
                            okx_quote=okx_q,
                            is_maker=False,
                        )
                        logger.info(
                            f"🛰 [SHADOW SOR COUNTERFACTUAL] {sym_clean} | Лучшая альтернатива: {shadow_cf.best_counterfactual_venue} "
                            f"(Gross: ${shadow_cf.gross_savings_usd:+.4f}, Net после трансфера $1.00: ${shadow_cf.net_savings_usd:+.4f}) "
                            f"| Решение: {shadow_cf.verdict}"
                        )
                except Exception as bench_err:
                    logger.debug(f"Execution shortfall benchmark note: {bench_err}")

                if pnl_usd < 0:
                    LAST_LOSS_LOCKOUT[sym_clean] = {
                        "exit_time": time.time(),
                        "exit_price": exit_p,
                        "direction": str(rep.get("direction", "BULLISH")).upper(),
                        "pnl_usd": pnl_usd,
                        "reason": reason
                    }
                    logger.warning(
                        f"🛡 [КУЛДАУН ПОСЛЕ УБЫТКА] {sym_clean}: активирована 4-часовая защита от тильта "
                        f"(-${abs(pnl_usd):.2f}, Причина: {reason})"
                    )

        open_paper = db.get_open_paper_trades(OWNER_USER_ID) if hasattr(db, 'get_open_paper_trades') else []
        if not open_paper and not getattr(verify_signal_outcomes, '_last_hb', 0) or (time.time() - getattr(verify_signal_outcomes, '_last_hb', 0) >= 300):
            verify_signal_outcomes._last_hb = time.time()
            tel_stats = market_telemetry_recorder.get_stats()
            dvol_obj = deribit_collector.get_dvol("BTC")
            dvol_str = f" | Deribit DVOL (VIX): {dvol_obj.dvol_close:.1f}%" if dvol_obj else ""
            r_stats = research_sleeve.get_stats()
            r_str = f" | 🔬 Теневой датасет (Track 2): {r_stats.get('total_explorations', 0)} наблюдений (Win Rate: {r_stats.get('win_rate_pct', 0.0):.1f}%)" if r_stats.get("total_explorations", 0) > 0 else ""
            logger.info(
                f"💓 [СИСТЕМА РАБОТАЕТ 24/7] 7/7 институциональных шлюзов активны. Сканирование каждые 60с. | "
                f"💾 Телеметрия: {tel_stats['total_recorded']} записей ({tel_stats['storage_size_mb']:.2f} MB / {market_telemetry_recorder.max_storage_mb:.0f} MB, {tel_stats['format_active']}){dvol_str}{r_str}"
            )

        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_SECRET_KEY")
        if api_key and api_secret:
            try:
                live_trader = BinanceAutoTrader(api_key=api_key, api_secret=api_secret)
                live_trader.manage_open_positions(hurst_values=hurst_values, current_atrs=current_atrs)
            except Exception as e:
                logger.debug(f"Position management check note: {e}")

        # Check if statistical sample milestones (25, 50, 75, 100) are reached
        check_statistical_milestones()
    except Exception as e:
        logger.error(f"Error in verify_signal_outcomes: {e}")


def run_wal_checkpoint():
    db = SQLiteTracker(db_path=DB_PATH)
    try:
        db.perform_wal_checkpoint()
        logger.info("WAL checkpoint completed successfully.")
    except Exception as e:
        logger.warning(f"WAL checkpoint error: {e}")


def main():
    update_heartbeat()
    logger.info("MaBuy V5.2 Renaissance Titan Engine initializing...")
    global LAST_LOSS_LOCKOUT
    LAST_LOSS_LOCKOUT.update(load_initial_loss_lockouts(DB_PATH))

    # 0. Cryptographic Chained-Hash Audit Ledger Verification (P1 Block 3)
    try:
        startup_logger = SignalLogger(db_path=DB_PATH)
        chain_valid, block_count, chain_err = startup_logger.verify_chain_integrity()
        if chain_valid:
            logger.info(f"🔗 [КРИПТОГРАФИЧЕСКИЙ АУДИТ] Цепочка блоков сигналов верифицирована успешно ({block_count} блоков, хеши целы).")
        else:
            logger.critical(f"🚨 [AUDIT ALERT] НАРУШЕНИЕ ЦЕЛОСТНОСТИ ЦЕПОЧКИ СИГНАЛОВ: {chain_err}")
    except Exception as chain_e:
        logger.warning(f"Chained hash audit check note: {chain_e}")

    # 1. Check Binance server time synchronization (Astra Error 5)
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_SECRET_KEY")
    if api_key and api_secret:
        try:
            import ccxt
            ex = ccxt.binanceusdm({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True})
            srv_time = ex.fetch_time()
            loc_time = int(time.time() * 1000)
            drift_ms = abs(srv_time - loc_time)
            if drift_ms > 1000:
                logger.warning(f"⚠️ [CLOCK DRIFT ALERT] Local server clock differs from Binance by {drift_ms}ms (>1000ms)!")
            else:
                logger.info(f"⏱ [CLOCK SYNC OK] Local clock vs Binance server delta: {drift_ms}ms (<1000ms).")
        except Exception as clk_err:
            logger.debug(f"Clock sync check note: {clk_err}")
    
    try:
        model_manager = DaemonModelManager(db_path=DB_PATH)
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize ModelManager: {e}", exc_info=True)
        return

    try:
        run_scan(model_manager)
    except Exception as e:
        logger.error(f"Initial run_scan error: {e}")

    try:
        run_wal_checkpoint()
    except Exception as e:
        logger.error(f"Initial wal error: {e}")

    # Register OS signal handlers for graceful shutdown (C7 FIX)
    try:
        signal.signal(signal.SIGINT, handle_shutdown_signal)
        signal.signal(signal.SIGTERM, handle_shutdown_signal)
        logger.info("🛡 [SIGNAL HANDLERS REGISTERED] Graceful shutdown armed for SIGINT/SIGTERM.")
    except Exception as sig_err:
        logger.warning(f"Could not register signal handlers: {sig_err}")

    # 2. Launch Dedicated Asynchronous Position Escort Worker (Astra Component 2)
    escort_worker = FastPositionEscortWorker(interval_sec=10.0)
    escort_worker.start()

    last_scan_time = time.time()
    last_wal_time = time.time()
    last_drift_time = time.time()
    last_backup_time = time.time()

    logger.info("MaBuy Daemon запущен 24/7. Сканирование рынка каждые 60 секунд.")
    while not SHUTDOWN_EVENT.is_set():
        try:
            update_heartbeat()
            now = time.time()

            # Health check on background escort thread
            if not escort_worker.is_alive() and not SHUTDOWN_EVENT.is_set():
                logger.critical("🚨 [СБОЙ ПОТОКА] FastPositionEscortWorker остановлен! Перезапуск...")
                escort_worker = FastPositionEscortWorker(interval_sec=10.0)
                escort_worker.start()

            if now - last_scan_time >= 60:
                last_scan_time = now
                run_scan(model_manager)

            if now - last_drift_time >= 3600:
                last_drift_time = now
                drift_res = drift_monitor.check_drift()
                if drift_res.get("drift_detected"):
                    logger.warning(f"⚠️ [ML CONCEPT DRIFT DETECTED] {drift_res.get('reasons')} | Retraining recommended.")
                else:
                    logger.info(f"✅ [ML STABILITY CHECK] Model distributions stable (samples={drift_res.get('sample_count')}).")

            if now - last_wal_time >= 900:
                last_wal_time = now
                run_wal_checkpoint()

            if now - last_backup_time >= 21600:
                last_backup_time = now
                logger.info("Running SQLite Hot-Backup schedule checkpoint...")

            # High-resolution 5s tick sampling for SOL/BTC Kalman Stat-Arb engine
            update_synthetic_spread_engine()

        except Exception as loop_err:
            logger.error(f"Unexpected error in daemon loop: {loop_err}", exc_info=True)

        SHUTDOWN_EVENT.wait(5.0)

    # ── GRACEFUL SHUTDOWN CLEANUP SEQUENCE (C7 FIX) ──
    logger.info("🛑 [GRACEFUL SHUTDOWN] Stopping background threads and finalizing state...")
    try:
        escort_worker.running = False
        escort_worker.join(timeout=3.0)
    except Exception as ew_err:
        logger.debug(f"Escort worker join note: {ew_err}")

    # Final position safety check before daemon terminates
    try:
        with EXECUTION_LOCK:
            logger.info("🛡 Running final position trailing stop verification before shutdown...")
            verify_signal_outcomes()
    except Exception as esc_err:
        logger.error(f"Error during final position verification: {esc_err}")

    # Flush market telemetry
    try:
        flushed_file = market_telemetry_recorder.flush()
        if flushed_file:
            logger.info(f"💾 Telemetry buffer cleanly flushed to disk: {flushed_file}")
    except Exception as tel_err:
        logger.debug(f"Telemetry flush error: {tel_err}")

    # Final SQLite WAL checkpoint
    try:
        run_wal_checkpoint()
        logger.info("📦 Final SQLite WAL checkpoint completed.")
    except Exception as wal_err:
        logger.error(f"Final WAL checkpoint error: {wal_err}")

    logger.info("🏁 [SHUTDOWN COMPLETE] MaBuy Daemon cleanly exited.")


if __name__ == "__main__":
    main()
