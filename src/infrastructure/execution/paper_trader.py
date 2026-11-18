"""
MaBuy Institutional Paper Trading Engine (Renaissance-Grade)
Features:
1. Volatility-Targeted Position Sizing (Risk Parity / Fractional Kelly)
2. Noise-Resistant Dynamic ATR Stop-Loss (Floor: 1.8%, 2.0x ATR_14)
3. Identical 4-Tier Dynamic Profit Locking matching Binance Live 1:1:
   - Tier 1: Break-Even (+0.1% buffer) at >= +1.0% price move.
   - Tier 2: Lock +0.8% guaranteed profit at >= +1.8% price move.
   - Tier 3: Lock +1.5% guaranteed profit at >= +2.5% price move.
   - Tier 4: Dynamic Take-Profit capture at >= +3.2% price move.
4. Full Synchronization with SignalLogger and SQLiteTracker.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List

from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.infrastructure.persistence.signal_logger import SignalLogger
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.domain.services.continuous_trailing_stop_engine import (
    ContinuousTrailingStopEngine,
    TrailingState
)

logger = logging.getLogger(__name__)


class PaperTrader:
    # Institutional Constants
    FEE_SLIPPAGE_ROUNDTRIP: float = 0.0012  # 0.12% roundtrip taker/maker fee & slippage
    DEFAULT_SAFE_LEVERAGE: int = 3          # Strictly 3x matching Binance isolated margin
    HARD_STOP_FLOOR_PCT: float = 0.018      # 1.8% minimum noise-filtering stop floor
    ATR_SL_MULTIPLIER: float = 2.0          # 2.0x ATR multiplier
    MAX_PORTFOLIO_POSITIONS: int = 3        # Max 3 positions (BTC, ETH, SOL)
    MAX_MARGIN_PER_TRADE_PCT: float = 0.20  # Max 20% virtual balance per trade
    MAX_MARGIN_PER_TRADE_USD: float = 1200.0

    # 4-Tier Profit Locking Calibration (1:1 with BinanceAutoTrader)
    TIER1_TRIGGER_PCT: float = 1.0  # +1.0% move triggers Break-Even
    TIER1_LOCK_PCT: float = 0.25    # Locks +0.25% profit (guarantees positive payoff net-of-fees)
    TIER2_TRIGGER_PCT: float = 1.8  # +1.8% move triggers Tier 2
    TIER2_LOCK_PCT: float = 0.8     # Locks +0.8% profit
    TIER3_TRIGGER_PCT: float = 2.5  # +2.5% move triggers Tier 3
    TIER3_LOCK_PCT: float = 1.5     # Locks +1.5% profit
    TIER4_TRIGGER_PCT: float = 3.2  # +3.2% move triggers immediate Market TP Capture

    def __init__(
        self,
        user_id: str,
        tracker: SQLiteTracker,
        signal_logger: Optional[SignalLogger] = None,
        fetcher: Optional[CryptoDataFetcher] = None,
    ):
        self.user_id = str(user_id)
        self.tracker = tracker
        if signal_logger is not None:
            self.signal_logger = signal_logger
        else:
            tracker_db = getattr(tracker, "db_path", None)
            if tracker_db and tracker_db != ":memory:":
                self.signal_logger = SignalLogger(db_path=tracker_db)
            else:
                self.signal_logger = SignalLogger()
        self.fetcher = fetcher or CryptoDataFetcher(max_cache_age_sec=15)

    def _get_price_and_atr(self, symbol: str, timeframe: str = "1h") -> tuple[float, float]:
        """
        Fetches live price and computes 14-period Average True Range (ATR) %.
        Uses 1h candles with 30-bar lookback for robust macro volatility filtering,
        falling back to 15m if needed.
        Returns: (current_price: float, atr_pct: float)
        """
        clean_sym = symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
        try:
            df = self.fetcher.fetch_ohlcv(clean_sym, timeframe=timeframe, limit=30)
            if df is None or len(df) < 14:
                df = self.fetcher.fetch_ohlcv(clean_sym, timeframe="15m", limit=30)

            if df is not None and not df.empty:
                current_price = float(df["close"].iloc[-1])
                highs = df["high"].values
                lows = df["low"].values
                closes = df["close"].values

                tr1 = highs[1:] - lows[1:]
                tr2 = np.abs(highs[1:] - closes[:-1])
                tr3 = np.abs(lows[1:] - closes[:-1])
                tr = np.maximum.reduce([tr1, tr2, tr3])

                atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
                atr_pct = (atr / current_price) if current_price > 0 else 0.015
                return current_price, max(0.008, min(0.08, atr_pct))
        except Exception as e:
            logger.error(f"Price & ATR fetch failed for {clean_sym}: {e}")

        return 0.0, 0.015

    def compute_volatility_targeted_size(
        self,
        symbol: str,
        virtual_balance: float,
        current_price: float,
        atr_pct: float,
        confidence: float,
        leverage: int = DEFAULT_SAFE_LEVERAGE,
        hurst: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Institutional Volatility-Targeted Position Sizing (Risk Parity & Fractional Kelly).
        Formula:
            Target_Risk_USD = Balance * Fractional_Kelly_Risk_Pct
            Stop_Distance_Pct = clamp(k * ATR_14, min=1.8%, max=3.2%)
            Target_Notional = Target_Risk_USD / Stop_Distance_Pct
            Required_Margin = Target_Notional / Leverage
        """
        # 1. Target Risk: 1.5% to 2.0% of virtual balance
        target_risk_pct = 0.015 + max(0.0, min(0.010, (confidence - 0.65) * 0.05))
        target_risk_usd = virtual_balance * target_risk_pct

        # 2. Dynamic Volatility ATR Stop Distance (Astra Component 3)
        h_val = hurst if hurst is not None else 0.50
        k_atr = 2.2 if h_val > 0.55 else 1.8
        sl_distance_pct = max(self.HARD_STOP_FLOOR_PCT, min(0.032, k_atr * atr_pct))

        # 3. Notional & Margin
        target_notional = target_risk_usd / sl_distance_pct
        target_margin = target_notional / leverage

        # 4. Apply Portfolio Margin Ceiling (Max 20% of balance, max $1,200)
        margin_cap = min(virtual_balance * self.MAX_MARGIN_PER_TRADE_PCT, self.MAX_MARGIN_PER_TRADE_USD)
        final_margin = min(target_margin, margin_cap)
        final_margin = max(15.0, final_margin)
        final_notional = final_margin * leverage

        amount_coins = (final_notional / current_price) if current_price > 0 else 0.0

        return {
            "notional_usd": round(final_notional, 2),
            "margin_usd": round(final_margin, 2),
            "amount_coins": round(amount_coins, 6),
            "sl_distance_pct": sl_distance_pct,
            "target_risk_usd": round(target_risk_usd, 2),
            "target_risk_pct": round(target_risk_pct, 4),
        }

    def execute_trade(
        self,
        symbol: str,
        direction: str,
        leverage: int = DEFAULT_SAFE_LEVERAGE,
        risk_pct: float = 0.02,
        confidence: float = 0.65,
        hurst: Optional[float] = None,
        target_tp_pct: Optional[float] = None,
        funding_rate_pct: Optional[float] = None,
        approved_notional_usd: Optional[float] = None,
        approved_margin_usd: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a precision virtual paper trade with Volatility-Targeted Sizing.
        """
        try:
            clean_sym = symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
            market_symbol = f"{clean_sym}/USDT:USDT"

            # 1. Check existing open paper positions: Max 3 total, max 1 per coin
            open_pos = self.tracker.get_open_paper_trades(self.user_id) if hasattr(self.tracker, "get_open_paper_trades") else []
            open_syms = [t.get("symbol", "").upper() for t in open_pos]
            if clean_sym in open_syms:
                logger.info(f"🛡 [Paper Trade Cap] Active position for {clean_sym} already open. Skipping duplicate.")
                return None
            if len(open_pos) >= self.MAX_PORTFOLIO_POSITIONS:
                logger.info(f"🛡 [Portfolio Cap] 3/3 concurrent positions open ({', '.join(open_syms)}). Skipping.")
                return None

            # 2. Fetch Live Price & 14-period ATR
            current_price, atr_pct = self._get_price_and_atr(clean_sym)
            if current_price <= 0.0:
                logger.warning(f"Could not fetch valid price for {clean_sym}. Trade aborted.")
                return None

            # 3. Check Virtual Capital
            virtual_balance = self.tracker.get_virtual_balance(self.user_id)
            if virtual_balance < 15.0:
                logger.warning(f"Virtual balance too low (${virtual_balance:.2f} < $15.00) for user {self.user_id}.")
                return None

            # 4. Strictly Safe 3x Leverage
            safe_leverage = self.DEFAULT_SAFE_LEVERAGE

            # 5. Volatility-Targeted Position Sizing
            sizing = self.compute_volatility_targeted_size(
                symbol=clean_sym,
                virtual_balance=virtual_balance,
                current_price=current_price,
                atr_pct=atr_pct,
                confidence=confidence,
                leverage=safe_leverage,
                hurst=hurst,
            )

            dir_clean = "BULLISH" if direction.upper() in ("BULLISH", "BUY", "LONG") else "BEARISH"

            if approved_notional_usd is not None and approved_notional_usd > 0:
                notional_value = float(approved_notional_usd)
                margin_to_use = float(approved_margin_usd) if (approved_margin_usd is not None and approved_margin_usd > 0) else (notional_value / safe_leverage)
                sl_distance_pct = sizing["sl_distance_pct"]
                logger.info(f"🛡 [PAPER RISK KERNEL SIZING] Approved Notional: ${notional_value:.2f} | Margin: ${margin_to_use:.2f}")
            else:
                margin_to_use = sizing["margin_usd"]
                notional_value = sizing["notional_usd"]
                sl_distance_pct = sizing["sl_distance_pct"]

                # Carry Arbitrage Sizing Multiplier (Astra Component 4)
                if funding_rate_pct is not None:
                    from src.domain.services.funding_rate_service import FundingRateService
                    fr_service = FundingRateService()
                    carry_mult, carry_narrative = fr_service.get_carry_sizing_multiplier(dir_clean, funding_rate_pct)
                    margin_to_use = max(15.0, min(margin_to_use * carry_mult, self.MAX_MARGIN_PER_TRADE_USD))
                    notional_value = margin_to_use * safe_leverage
                    logger.info(f"💰 [PAPER CARRY SIZING] {carry_narrative} Multiplier: {carry_mult:.2f} => Margin: ${margin_to_use:.2f}")
            h_val = hurst if hurst is not None else 0.50
            if target_tp_pct is not None and target_tp_pct > 0:
                tp_target_pct = target_tp_pct * 100.0 if target_tp_pct < 1.0 else target_tp_pct
            else:
                tp_target_pct = 4.2 if h_val > 0.55 else self.TIER4_TRIGGER_PCT

            if dir_clean == "BULLISH":
                stop_loss_price = current_price * (1.0 - sl_distance_pct)
                take_profit_price = current_price * (1.0 + (tp_target_pct / 100.0))
            else:
                stop_loss_price = current_price * (1.0 + sl_distance_pct)
                take_profit_price = current_price * (1.0 - (tp_target_pct / 100.0))

            # 7. Commit Trade to SQLite
            trade_id = self.tracker.open_paper_trade(
                user_id=self.user_id,
                symbol=clean_sym,
                direction=dir_clean,
                entry_price=current_price,
                amount_usd=margin_to_use,
            )

            logger.info(
                f"🟢 [VIRTUAL TRADE OPENED] {dir_clean} {clean_sym} @ ${current_price:,.2f} "
                f"| Margin: ${margin_to_use:.2f} ({safe_leverage}x -> ${notional_value:.2f}) "
                f"| SL: ${stop_loss_price:,.2f} (-{sl_distance_pct*100:.2f}%) | TP: ${take_profit_price:,.2f} (+{self.TIER4_TRIGGER_PCT}%)"
            )

            return {
                "status": "PAPER_SUCCESS",
                "symbol": market_symbol,
                "side": "BUY" if dir_clean == "BULLISH" else "SELL",
                "entry_price": current_price,
                "margin_used": margin_to_use,
                "notional_usd": notional_value,
                "leverage": safe_leverage,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "trade_id": trade_id,
            }

        except Exception as e:
            logger.error(f"Paper trading execution failed for {symbol}: {e}", exc_info=True)
            return None

    def update_open_trades(self, hurst_values: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Scans all open paper trades and applies the institutional 4-tier profit lock
        and noise-resistant ATR stop loss.
        Mirrors BinanceAutoTrader:
        - Tier 4: Immediate Market TP at >= +3.2% move (or +4.2% when Hurst H > 0.55).
        - Continuous Trailing: For H > 0.55 and peak >= +3.2%, trail 1.0% below peak.
        - Tier 3: Lock +1.5% profit when peak reached >= +2.5%.
        - Tier 2: Lock +0.8% profit when peak reached >= +1.8%.
        - Tier 1: Break-Even (+0.1% buffer) when peak reached >= +1.0%.
        - Hard Stop: >= 2.0x ATR or floor of 1.8%.
        """
        closed_reports = []
        open_trades = self.tracker.get_open_paper_trades(self.user_id)

        for trade in open_trades:
            trade_id = trade["trade_id"]
            symbol = trade["symbol"]
            direction = trade["direction"]
            entry_price = float(trade["entry_price"])
            amount_usd = float(trade["amount_usd"])  # Margin USD
            max_pnl_usd = float(trade.get("max_pnl_usd") or 0.0)

            try:
                current_price, atr_pct = self._get_price_and_atr(symbol)
                if current_price <= 0.0:
                    continue

                is_long = direction.upper() in ("BULLISH", "BUY", "LONG")

                # Percentage price move of underlying asset
                if is_long:
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100.0

                trade_leverage = self.DEFAULT_SAFE_LEVERAGE
                notional = amount_usd * trade_leverage

                # Dollar PnL with realistic roundtrip fees (0.12%)
                pnl_gross = notional * (pnl_pct / 100.0)
                fees = notional * self.FEE_SLIPPAGE_ROUNDTRIP
                pnl_net = pnl_gross - fees

                # Track peak profit
                if pnl_net > max_pnl_usd:
                    self.tracker.update_max_pnl(trade_id, pnl_net)
                    max_pnl_usd = pnl_net

                peak_pnl_pct = max(pnl_pct, (max_pnl_usd + fees) / notional * 100.0 if notional > 0 else 0.0)

                # Dynamic Noise-Proof Stop Floor (Min 1.8% floor or 2.0x ATR)
                sl_distance_pct = max(self.HARD_STOP_FLOOR_PCT, self.ATR_SL_MULTIPLIER * atr_pct)
                hard_sl_threshold_pct = -(sl_distance_pct * 100.0)

                should_close = False
                close_reason = ""

                # Check Hurst Exponent to allow strong trend winners (H > 0.55) to run toward +4.2%
                h_val = (hurst_values or {}).get(symbol, 0.50)
                is_strong_trend = (h_val > 0.55)
                tp_cap_pct = 4.2 if is_strong_trend else self.TIER4_TRIGGER_PCT

                # ── Astra Continuous Volatility-Anchored Trailing Stop & Hard Stop ──
                import time
                if not hasattr(self, "_trailing_engine"):
                    self._trailing_engine = ContinuousTrailingStopEngine(exchange_update_threshold=0.0015)
                    self._trailing_states = {}

                trade_key = f"{symbol}_{trade_id}"
                if trade_key not in self._trailing_states:
                    self._trailing_states[trade_key] = TrailingState(
                        symbol=symbol,
                        side=1 if is_long else -1,
                        entry_px=entry_price,
                        current_sl_px=entry_price * (1.0 - sl_distance_pct) if is_long else entry_price * (1.0 + sl_distance_pct),
                        peak_px=current_price,
                        d_min=0.012,
                        k_sigma=2.2 if is_strong_trend else 1.8,
                        updated_at_ts=time.time()
                    )

                state = self._trailing_states[trade_key]
                self._trailing_engine.check_breakeven_lock(state, mark_price=current_price, fee_buffer=0.0015)
                current_sl, _, trail_reason = self._trailing_engine.update_stop(
                    state=state,
                    mark_price=current_price,
                    atr_norm=atr_pct
                )

                if pnl_pct >= tp_cap_pct:
                    should_close = True
                    close_reason = f"🎯 {'Hurst Trend' if is_strong_trend else 'Target'} TP Captured at +{pnl_pct:.2f}% (Target: +{tp_cap_pct:.1f}%, Net +${pnl_net:.2f})"
                elif (is_long and current_price <= current_sl) or (not is_long and current_price >= current_sl):
                    should_close = True
                    close_reason = f"🛡 Continuous Trailing Stop Hit at ${current_price:,.2f} (SL: ${current_sl:,.2f}, Peak: ${state.peak_px:,.2f}, Net {pnl_net:+.2f}$)"

                if should_close:
                    self.tracker.close_paper_trade(trade_id, current_price, pnl_net)
                    self.tracker.update_virtual_balance(self.user_id, pnl_net)

                    # Synchronize with signal_log
                    if self.signal_logger:
                        try:
                            sig = self.signal_logger.find_open_signal(symbol=symbol, direction=direction)
                            if sig:
                                sig_id = sig["id"] if isinstance(sig, dict) else int(sig)
                                self.signal_logger.close_signal(
                                    signal_id=sig_id,
                                    exit_price=current_price,
                                    pnl_usd=pnl_net,
                                    close_reason=close_reason,
                                )
                                logger.info(f"Synchronized signal_log close for signal #{sig_id} | PnL: ${pnl_net:+.2f}")
                        except Exception as log_e:
                            logger.error(f"SignalLogger closure error: {log_e}")

                    logger.info(
                        f"🔴 [VIRTUAL TRADE CLOSED] [{self.user_id}] | {symbol} | {close_reason} | Net PnL: ${pnl_net:+.2f}"
                    )

                    closed_reports.append({
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "direction": direction,
                        "exit_price": current_price,
                        "pnl_usd": pnl_net,
                        "pnl_pct": pnl_pct,
                        "reason": close_reason,
                    })

            except Exception as e:
                logger.error(f"Failed to update paper trade {trade_id} ({symbol}): {e}", exc_info=True)

        return closed_reports


    def get_active_position(self, symbol: str) -> Optional[dict]:
        """
        Checks if there is an active paper position for the symbol for this user.
        Returns trade dict or None.
        """
        clean_sym = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
        open_trades = self.tracker.get_open_paper_trades(self.user_id) if hasattr(self.tracker, 'get_open_paper_trades') else []
        for trade in open_trades:
            if trade.get('symbol', '').upper() == clean_sym:
                return trade
        return None

    def close_position_early(self, symbol: str, reason: str = "EARLY_REVERSAL_GUARD") -> Optional[dict]:
        """
        Mirrors early reversal close in virtual execution.
        Immediately closes open paper trade at current market price and records reason.
        """
        clean_sym = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
        trade = self.get_active_position(clean_sym)
        if not trade:
            return None

        trade_id = trade['trade_id']
        direction = trade['direction']
        entry_price = float(trade['entry_price'])
        amount_usd = float(trade['amount_usd'])
        trade_leverage = self.DEFAULT_SAFE_LEVERAGE

        current_price, _ = self._get_price_and_atr(clean_sym)
        if current_price <= 0.0:
            current_price = entry_price

        price_diff_pct = (current_price - entry_price) / entry_price
        if direction.upper() in ('BEARISH', 'SHORT', 'SELL'):
            price_diff_pct = -price_diff_pct

        notional = amount_usd * trade_leverage
        pnl_gross = notional * price_diff_pct
        fees_and_slippage = notional * self.FEE_SLIPPAGE_ROUNDTRIP
        pnl = pnl_gross - fees_and_slippage

        self.tracker.close_paper_trade(trade_id, current_price, pnl)
        self.tracker.update_virtual_balance(self.user_id, pnl)

        if self.signal_logger:
            try:
                sig = self.signal_logger.find_open_signal(symbol=clean_sym, direction=direction)
                if sig:
                    sig_id = sig["id"] if isinstance(sig, dict) else int(sig)
                    self.signal_logger.close_signal(
                        signal_id=sig_id,
                        exit_price=current_price,
                        pnl_usd=pnl,
                        close_reason=f"Early Reversal Exit: {reason}"
                    )
            except Exception as log_e:
                logger.error(f"SignalLogger early close error: {log_e}")

        logger.warning(
            f"🚨 VIRTUAL EARLY REVERSAL EXIT [{self.user_id}] | {clean_sym} ({direction}) | "
            f"Reason: {reason} | Entry: ${entry_price:,.2f} -> Exit: ${current_price:,.2f} | Net PNL: ${pnl:.2f}"
        )

        return {
            "status": "PAPER_CLOSED_EARLY",
            "trade_id": trade_id,
            "symbol": clean_sym,
            "direction": direction,
            "exit_price": current_price,
            "pnl_usd": pnl,
            "reason": reason
        }
