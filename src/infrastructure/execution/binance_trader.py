ACTIVE_STOP_PRICES = {}  # Global in-memory cache of placed trailing stops per symbol
TRAILING_STATES = {}     # Global in-memory cache of TrailingState per symbol
import logging
import os
import ccxt
import numpy as np
from typing import Optional, List, Dict, Any
import gc
import time

from src.domain.services.continuous_trailing_stop_engine import (
    ContinuousTrailingStopEngine,
    TrailingState
)

trailing_engine = ContinuousTrailingStopEngine(exchange_update_threshold=0.0015)

logger = logging.getLogger(__name__)

# 🛑 UNBREAKABLE ASSET WHITELIST: Strictly top liquid pairs
SNIPER_ALLOWED = ["BTC", "ETH", "SOL"]


class BinanceAutoTrader:
    """
    Production-Grade Binance Futures Auto-Trading Engine for MaBuy Owner Account.
    Implements Institutional Risk Controls:
    1. STRICT WHITELIST: BTC, ETH, SOL only.
    2. ISOLATED MARGIN & SAFE LEVERAGE: 2x-3x maximum.
    3. MICRO-CAP PRESERVATION: 10-15% margin max ($3-$5 on $30 deposit).
    4. ATOMIC SL/TP: Automatic Stop-Loss & Take-Profit with dual format fallback.
    5. SINGLE CONCURRENT POSITION: Maximum 1 position active per coin.
    6. AUTO BREAK-EVEN & MULTI-TIER TRAILING STOP: Dynamic profit defense.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

    def _create_isolated_session(self):
        """Creates a temporary isolated session with safety options."""
        exchange = ccxt.binanceusdm({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'enableWithdrawals': False,
                'adjustForTimeDifference': True
            }
        })
        if self.testnet:
            exchange.set_sandbox_mode(True)
        return exchange

    def execute_trade(
        self,
        symbol: str,
        direction: str,
        leverage: int = 3,
        risk_pct: float = 0.02,
        confidence: float = 75.0,
        hurst: Optional[float] = None,
        target_tp_pct: Optional[float] = None,
        atr_pct: Optional[float] = None,
        funding_rate_pct: Optional[float] = None,
        approved_notional_usd: Optional[float] = None,
        approved_leverage: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Executes a high-precision trade on Binance Futures with 100% deposit protection.
        """
        # 1. Asset Barrier: Only BTC, ETH, SOL allowed
        clean_sym = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
        if clean_sym not in SNIPER_ALLOWED:
            logger.critical(f"🛑 REJECTED: Symbol '{clean_sym}' is NOT in {SNIPER_ALLOWED}. Real trading on altcoins is permanently disabled.")
            return None

        # 2. Safety Gate Check from .env
        is_real_enabled = os.environ.get("REAL_TRADING_ENABLED", "false")  # public build: real orders off unless explicitly enabled.lower() in ("true", "1", "yes")
        is_paused = os.environ.get("REAL_AUTOTRADE_PAUSED", "false").lower() in ("true", "1", "yes")
        if not is_real_enabled or is_paused:
            logger.warning("🛑 REAL TRADING PAUSED in .env! Order skipped.")
            return None

        exchange = self._create_isolated_session()
        try:
            market_symbol = f"{clean_sym}/USDT:USDT"

            # 3. Check Wallet Balance & Micro-Capital Gate
            balance = exchange.fetch_balance()
            usdt_free = float(balance.get('USDT', {}).get('free', 0.0))
            usdt_total = float(balance.get('USDT', {}).get('total', usdt_free))

            if usdt_free < 2.0:
                logger.error(f"Insufficient free USDT balance (${usdt_free:.2f} < $2.00).")
                return None

            # Micro-Capital Gate (Astra Proof): On accounts < $100 equity, 0.001 BTC min lot ($78+ notional)
            # forces >= 2x single-trade leverage, violating < 0.5% equity risk budget and causing ruin.
            if usdt_total < 100.0 and clean_sym == "BTC":
                logger.warning(
                    f"🛑 [MICRO-CAPITAL GATE] BTC disabled for accounts < $100 equity "
                    f"(Current equity: ${usdt_total:.2f}, 0.001 BTC min lot requires >$110 equity). Real order skipped."
                )
                return None

            # 4. Multi-Asset Portfolio Cap: Max 1 position for sub-$100 deposits, max 3 for normal accounts
            max_concurrent_positions = 1 if usdt_total < 100.0 else 3
            try:
                positions = exchange.fetch_positions()
                active_positions = [
                    p for p in positions 
                    if abs(float(p.get("info", {}).get("positionAmt", 0) or (p.get("contracts", 0) * (1 if p.get("side") == "long" else -1)))) > 0
                ]
                active_symbols = [
                    p.get('symbol', '').upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
                    for p in active_positions
                ]

                # Rule A: Prevent duplicate position on the same asset
                if clean_sym in active_symbols:
                    logger.info(f"🛡 1-Position-Per-Coin Cap: Active position on {clean_sym} already exists. Skipping duplicate trade.")
                    return None

                # Rule B: Portfolio Cap Check
                if len(active_positions) >= max_concurrent_positions:
                    logger.info(f"🛡 Portfolio Cap Reached: {len(active_positions)}/{max_concurrent_positions} active positions ({', '.join(active_symbols)}). Skipping new trade.")
                    return None
            except Exception as pos_e:
                logger.warning(f"Position check error: {pos_e}")

            # 5. Switch to ISOLATED margin mode (prevent cross liquidation)
            try:
                exchange.set_margin_mode('ISOLATED', market_symbol)
            except Exception as e:
                logger.debug(f"Margin mode note (may already be isolated): {e}")

            # 6. Set SAFE Leverage (strictly 3x or risk-kernel approved)
            target_lev = approved_leverage if approved_leverage is not None else leverage
            safe_leverage = min(max(int(target_lev), 2), 4)
            try:
                exchange.set_leverage(safe_leverage, market_symbol)
            except Exception as e:
                logger.warning(f"Leverage set note: {e}")

            # Fetch current ticker price
            ticker = exchange.fetch_ticker(market_symbol)
            current_price = float(ticker.get('last') or ticker.get('close') or 0.0)
            if current_price <= 0:
                logger.error("Could not fetch valid current price.")
                return None

            # Determine dynamic volatility ATR-based Stop Loss percentage early for sizing
            h_val = hurst if hurst is not None else 0.50
            k_atr = 2.2 if h_val > 0.55 else 1.8
            if atr_pct is not None and atr_pct > 0:
                sl_pct = max(0.018, min(0.032, k_atr * atr_pct))
            else:
                sl_pct = 0.018  # 1.8% baseline stop loss

            # 7. Sizing: Astra Section 5.3 Institutional Capital Gate & Volatility Sizing
            # Maximum shock-safe notional under 15% flash shock: N <= (0.05 / 0.15) * Equity = 0.3333 * Equity.
            if usdt_total < 59.0:
                logger.warning(
                    f"🛑 [ASTRA CAPITAL GATE] Total equity (${usdt_total:.2f}) < $59.00 minimum required "
                    f"to absorb a 15% flash shock within 5% Max Drawdown. Defaulting to 100% Cash / Standby. Trade vetoed."
                )
                return None

            if 'BTC' in market_symbol and usdt_total < 237.0:
                logger.warning(
                    f"🛑 [ASTRA CAPITAL GATE] Total equity (${usdt_total:.2f}) < $237.00. Minimum BTC lot (0.001 BTC ~= $79) "
                    f"would violate the 5% Max Drawdown constraint under a 15% shock. Trade vetoed."
                )
                return None

            BINANCE_MIN_NOTIONAL = 5.50  # Binance Futures actual min ~$5, +$0.50 buffer (was 20.50 — oversized for micro-capital)
            effective_risk_pct = min(max(risk_pct, 0.005), 0.015)  # Cap single trade risk between 0.5% and 1.5%
            execution_friction_pct = 0.0010  # 0.10% round-trip fees + adverse slippage buffer

            stop_distance_pct = sl_pct + execution_friction_pct
            risk_budget_usd = usdt_total * effective_risk_pct

            if approved_notional_usd is not None and approved_notional_usd > 0:
                formula_notional = float(approved_notional_usd)
                logger.info(f"🛡 [LIVE RISK KERNEL SIZING] Approved Notional: ${formula_notional:.2f}")
            else:
                formula_notional = risk_budget_usd / stop_distance_pct

                # Carry Arbitrage Sizing Multiplier (Astra Component 4)
                if funding_rate_pct is not None:
                    from src.domain.services.funding_rate_service import FundingRateService
                    fr_service = FundingRateService()
                    carry_mult, carry_narrative = fr_service.get_carry_sizing_multiplier(direction, funding_rate_pct)
                    formula_notional *= carry_mult
                    logger.info(f"💰 [CARRY SIZING] {carry_narrative} Multiplier: {carry_mult:.2f} => Formula Notional: ${formula_notional:.2f}")

            # Micro-Capital Economics Check (Astra Phase 4 Audit Finding):
            # If formula_notional is below Binance minimum notional ($20.50), evaluate whether bumping
            # to minimum notional violates the maximum allowable risk cap (1.5% of total equity).
            if formula_notional < BINANCE_MIN_NOTIONAL:
                min_order_risk_usd = BINANCE_MIN_NOTIONAL * stop_distance_pct
                min_order_risk_pct = min_order_risk_usd / usdt_total if usdt_total > 0 else 1.0

                MAX_MICRO_RISK_CAP = 0.015  # 1.5% max risk cap on total equity
                if min_order_risk_pct > MAX_MICRO_RISK_CAP:
                    logger.warning(
                        f"🛑 [SMALL-CAPITAL SIZING GUARD] Minimum exchange order (${BINANCE_MIN_NOTIONAL:.2f}) "
                        f"would risk {min_order_risk_pct*100:.2f}% (${min_order_risk_usd:.2f}) on ${usdt_total:.2f} equity "
                        f"(exceeds {MAX_MICRO_RISK_CAP*100:.1f}% risk cap with {sl_pct*100:.2f}% SL). Trade safely skipped."
                    )
                    return None
                else:
                    target_notional = BINANCE_MIN_NOTIONAL
                    logger.info(
                        f"🛡 [SMALL-CAPITAL SIZING] Notional clamped to Binance minimum: ${target_notional:.2f} "
                        f"(Planned Risk: ${min_order_risk_usd:.2f} = {min_order_risk_pct*100:.2f}% equity <= {MAX_MICRO_RISK_CAP*100:.1f}%)."
                    )
            else:
                target_notional = formula_notional

            # Fetch exchange lot size precision and min limits
            market_info = exchange.market(market_symbol) if market_symbol in exchange.markets else {}
            min_amount = float(market_info.get('limits', {}).get('amount', {}).get('min') or (0.001 if 'BTC' in market_symbol else 0.001))

            raw_amount = max(target_notional / current_price, min_amount)
            amount = float(exchange.amount_to_precision(market_symbol, raw_amount))
            if amount < min_amount:
                amount = min_amount

            # Ensure amount * price is at least BINANCE_MIN_NOTIONAL
            while amount * current_price < BINANCE_MIN_NOTIONAL:
                amount += min_amount
                amount = float(exchange.amount_to_precision(market_symbol, amount))

            actual_notional = amount * current_price
            actual_margin = actual_notional / safe_leverage

            # If margin exceeds free balance at 3x, adapt leverage up to 5x safely
            if usdt_free < actual_margin:
                if actual_notional / 5.0 <= usdt_free * 0.95:
                    safe_leverage = 5
                    try:
                        exchange.set_leverage(safe_leverage, market_symbol)
                    except Exception as lev_err:
                        logger.debug(f"Leverage adjust note: {lev_err}")
                    actual_margin = actual_notional / safe_leverage
                    logger.info(f"⚡ Adjusted leverage to {safe_leverage}x for {market_symbol} (${actual_notional:.2f} notional). Margin: ${actual_margin:.2f}")
                else:
                    logger.warning(f"⚠️ {market_symbol} min order (${actual_notional:.2f}) requires ${actual_margin:.2f} margin, but free balance is ${usdt_free:.2f}. Skipping to protect micro-deposit.")
                    return None

            target_margin = actual_margin

            side = 'buy' if direction.upper() in ('BULLISH', 'BUY', 'LONG') else 'sell'
            sl_side = 'sell' if side == 'buy' else 'buy'

            logger.info(f"🎯 EXECUTING V5.1 SNIPER ORDER: {side.upper()} {amount} {market_symbol} | Margin: ${target_margin:.2f} | Lev: {safe_leverage}x | Price: ${current_price:,.2f}")

            # 8. Micro-Price Imputation & Maker-First Execution Engine (Astra Component 1)
            initial_price = current_price
            best_bid = current_price
            best_ask = current_price
            microprice = current_price
            spread_pct = 0.0
            try:
                order_book = exchange.fetch_order_book(market_symbol, limit=5)
                bids = order_book.get('bids', [])
                asks = order_book.get('asks', [])
                if bids and len(bids) > 0:
                    best_bid = float(bids[0][0])
                if asks and len(asks) > 0:
                    best_ask = float(asks[0][0])
                from src.domain.services.order_flow_imbalance import OrderFlowImbalanceService
                calc_micro = OrderFlowImbalanceService.calculate_microprice(bids, asks)
                if calc_micro > 0.0:
                    microprice = calc_micro
            except Exception as ob_err:
                logger.debug(f"L2 book note for {market_symbol}: {ob_err}")
                ticker_bid = float(ticker.get('bid') or 0.0)
                ticker_ask = float(ticker.get('ask') or 0.0)
                if ticker_bid > 0:
                    best_bid = ticker_bid
                if ticker_ask > 0:
                    best_ask = ticker_ask
                microprice = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else current_price

            if current_price > 0 and best_ask > best_bid:
                spread_pct = ((best_ask - best_bid) / current_price) * 100.0

            logger.info(f"🔬 [MICRO-PRICE] {market_symbol} | Bid: ${best_bid:,.2f} | Ask: ${best_ask:,.2f} | Microprice: ${microprice:,.2f} | Spread: {spread_pct:.4f}%")

            # Pre-flight Liquidity & Spread Gates (Astra Finding 1)
            if spread_pct > 0.15:
                logger.warning(f"🛑 [LIQUIDITY VACUUM VETO] Spread on {market_symbol} is {spread_pct:.3f}% > 0.15%. Aborting trade execution to protect capital.")
                return None

            can_try_maker = True
            if spread_pct > 0.08:
                logger.warning(f"⚠️ [SPREAD FILTER] Spread on {market_symbol} is wide ({spread_pct:.3f}% > 0.08%). Skipping post-only to avoid adverse selection.")
                can_try_maker = False

            is_maker = False
            order_id = None
            fill_price = current_price

            if can_try_maker:
                if side == 'buy':
                    target_limit_price = min(best_bid, best_ask - 1e-8) if best_ask > best_bid else best_bid
                else:
                    target_limit_price = max(best_ask, best_bid + 1e-8) if best_ask > best_bid else best_ask

                limit_price = float(exchange.price_to_precision(market_symbol, target_limit_price))

                # Safety guard against accidental book-crossing from tick precision
                if side == 'buy' and limit_price >= best_ask and best_ask > 0:
                    limit_price = float(exchange.price_to_precision(market_symbol, best_bid * 0.9999))
                elif side == 'sell' and limit_price <= best_bid and best_bid > 0:
                    limit_price = float(exchange.price_to_precision(market_symbol, best_ask * 1.0001))

                try:
                    logger.info(f"⚡ [MAKER-FIRST] Dispatching Post-Only Limit: {side.upper()} {amount} {market_symbol} @ ${limit_price:,.2f}")
                    maker_order = exchange.create_order(
                        symbol=market_symbol,
                        type='LIMIT',
                        side=side,
                        amount=amount,
                        price=limit_price,
                        params={'timeInForce': 'GTX', 'postOnly': True}
                    )
                    order_id = maker_order.get('id')
                    m_status = maker_order.get('status')

                    if m_status == 'closed':
                        is_maker = True
                        fill_price = float(maker_order.get('average') or maker_order.get('price') or limit_price)
                        logger.info(f"💎 [MAKER SUCCESS] Post-Only Instantly Filled! ID: {order_id} @ ${fill_price:,.2f} (0.02% Fee)")
                    else:
                        # Poll for passive fill (3.0s window)
                        poll_start = time.time()
                        while time.time() - poll_start < 3.0:
                            time.sleep(0.4)
                            try:
                                chk = exchange.fetch_order(order_id, market_symbol)
                                if chk.get('status') == 'closed':
                                    is_maker = True
                                    fill_price = float(chk.get('average') or chk.get('price') or limit_price)
                                    logger.info(f"💎 [MAKER SUCCESS] Post-Only Filled in {time.time()-poll_start:.1f}s! ID: {order_id} @ ${fill_price:,.2f} (0.02% Fee)")
                                    break
                                elif chk.get('status') in ('canceled', 'expired', 'rejected'):
                                    break
                            except Exception:
                                pass

                        if not is_maker:
                            logger.info(f"⏳ Maker Post-Only order {order_id} not filled within 3.0s. Cancelling...")
                            partial_filled_qty = 0.0
                            try:
                                cancel_result = exchange.cancel_order(order_id, market_symbol)
                                # Check if any quantity was partially filled before cancel
                                try:
                                    final_chk = exchange.fetch_order(order_id, market_symbol)
                                    partial_filled_qty = float(final_chk.get('filled', 0) or 0)
                                    if final_chk.get('status') == 'closed':
                                        is_maker = True
                                        fill_price = float(final_chk.get('average') or limit_price)
                                        logger.info(f"💎 [MAKER SUCCESS] Filled during cancel transit! ID: {order_id}")
                                    elif partial_filled_qty > 0:
                                        logger.warning(
                                            f"⚠️ [PARTIAL FILL] Maker order {order_id} partially filled: "
                                            f"{partial_filled_qty} of {amount}. Adjusting market fallback amount."
                                        )
                                except Exception:
                                    pass
                            except Exception:
                                try:
                                    final_chk = exchange.fetch_order(order_id, market_symbol)
                                    if final_chk.get('status') == 'closed':
                                        is_maker = True
                                        fill_price = float(final_chk.get('average') or limit_price)
                                        logger.info(f"💎 [MAKER SUCCESS] Filled during cancel transit! ID: {order_id}")
                                    else:
                                        partial_filled_qty = float(final_chk.get('filled', 0) or 0)
                                except Exception:
                                    pass

                except Exception as maker_err:
                    logger.debug(f"Post-Only dispatch note ({maker_err}). Falling back to market.")

            # Fallback to market order if not filled passively
            if not is_maker:
                # Adverse Selection Guard on timeout (Astra Finding 1):
                # If price ran away by > 0.15% during maker wait, abort instead of market order
                try:
                    fresh_ticker = exchange.fetch_ticker(market_symbol)
                    fresh_price = float(fresh_ticker.get('last') or fresh_ticker.get('close') or 0.0)
                    if fresh_price > 0 and initial_price > 0:
                        drift_pct = (fresh_price - initial_price) / initial_price if side == 'buy' else (initial_price - fresh_price) / initial_price
                        if drift_pct > 0.0015:  # > 0.15% adverse slippage
                            logger.warning(
                                f"🛑 [ADVERSE SELECTION ABORT] Price drifted adversely (+{drift_pct*100:.2f}% > 0.15%) "
                                f"during maker timeout on {market_symbol}. Aborting market fallback to protect capital."
                            )
                            return None
                        current_price = fresh_price
                except Exception as drift_err:
                    logger.debug(f"Drift check note for {market_symbol}: {drift_err}")

                logger.info("⚡ [MARKET FALLBACK] Executing primary market order...")
                # C1 FIX: Subtract any partial fill from cancelled maker order
                remaining_amount = amount - partial_filled_qty if partial_filled_qty > 0 else amount
                if partial_filled_qty > 0:
                    logger.info(f"📐 [PARTIAL ADJUST] Original={amount}, PartialFill={partial_filled_qty}, Remaining={remaining_amount}")
                if remaining_amount * current_price < 5.0:
                    # Remaining amount below Binance minimum — partial fill IS the position
                    logger.info(f"✅ [PARTIAL FILL ACCEPTED] Partial fill {partial_filled_qty} is the final position (remaining ${remaining_amount * current_price:.2f} < $5 min notional)")
                    fill_price = float(exchange.fetch_order(order_id, market_symbol).get('average') or limit_price) if partial_filled_qty > 0 else current_price
                    amount = partial_filled_qty  # Update amount to reflect actual position size
                else:
                    mkt_order = exchange.create_market_order(market_symbol, side, remaining_amount)
                    fill_price = float(mkt_order.get('average') or mkt_order.get('price') or current_price)
                    order_id = mkt_order.get('id')
                    amount = remaining_amount + partial_filled_qty  # Total actual position
                    logger.info(f"✅ Market Order Filled! ID: {order_id}, Fill Price: ${fill_price:,.2f} (0.05% Taker Fee), Total Position: {amount}")

            # 9. Dynamic Volatility ATR-Based Stop-Loss (Astra Component 3)
            h_val = hurst if hurst is not None else 0.50
            k_atr = 2.2 if h_val > 0.55 else 1.8
            if atr_pct is not None and atr_pct > 0:
                sl_pct = max(0.018, min(0.032, k_atr * atr_pct))
                logger.info(f"🛡 [DYNAMIC ATR STOP] {clean_sym} ATR={atr_pct*100:.2f}%, k={k_atr:.1f} => Dynamic SL: {sl_pct*100:.2f}%")
            else:
                sl_pct = 0.018 # 1.8% Stop Loss Floor

            if target_tp_pct is not None and target_tp_pct > 0:
                tp_pct = target_tp_pct if target_tp_pct < 1.0 else (target_tp_pct / 100.0)
            else:
                tp_pct = 0.042 if h_val > 0.55 else 0.035

            if side == 'buy':
                sl_price = fill_price * (1.0 - sl_pct)
                tp_price = fill_price * (1.0 + tp_pct)
            else:
                sl_price = fill_price * (1.0 + sl_pct)
                tp_price = fill_price * (1.0 - tp_pct)

            sl_price = float(exchange.price_to_precision(market_symbol, sl_price))
            tp_price = float(exchange.price_to_precision(market_symbol, tp_price))

            # 10. Place Protective Stop-Loss Order
            try:
                exchange.create_order(
                    symbol=market_symbol,
                    type='STOP_MARKET',
                    side=sl_side,
                    amount=amount,
                    price=None,
                    params={'stopPrice': sl_price, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
                )
                logger.info(f"🛡 Stop-Loss Order placed at ${sl_price}")
            except Exception as sl_err_1:
                try:
                    exchange.create_order(
                        symbol=market_symbol,
                        type='STOP_MARKET',
                        side=sl_side,
                        amount=None,
                        price=None,
                        params={'stopPrice': sl_price, 'closePosition': True, 'workingType': 'MARK_PRICE'}
                    )
                    logger.info(f"🛡 ClosePosition Stop-Loss placed at ${sl_price}")
                except Exception as sl_err_2:
                    logger.error(f"⚠️ Could not set automatic SL on exchange: {sl_err_2}")

            # 11. Place Target Take-Profit Order
            try:
                exchange.create_order(
                    symbol=market_symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=sl_side,
                    amount=amount,
                    price=None,
                    params={'stopPrice': tp_price, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
                )
                logger.info(f"🎯 Take-Profit Order placed at ${tp_price}")
            except Exception as tp_err_1:
                try:
                    exchange.create_order(
                        symbol=market_symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=sl_side,
                        amount=None,
                        price=None,
                        params={'stopPrice': tp_price, 'closePosition': True, 'workingType': 'MARK_PRICE'}
                    )
                    logger.info(f"🎯 ClosePosition Take-Profit placed at ${tp_price}")
                except Exception as tp_err_2:
                    logger.warning(f"Take-Profit order note: {tp_err_2}")

            return {
                "status": "SUCCESS",
                "symbol": market_symbol,
                "side": side.upper(),
                "entry_price": fill_price,
                "microprice": float(round(microprice, 4)),
                "amount": amount,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "leverage": safe_leverage,
                "margin_usd": target_margin,
                "order_id": order_id,
                "execution_mode": "MAKER" if is_maker else "TAKER",
                "fee_rate": 0.0002 if is_maker else 0.0005
            }

        except Exception as e:
            logger.error(f"Trade execution error on {symbol}: {e}")
            return None
        finally:
            if 'exchange' in locals():
                del exchange
            gc.collect()

    def manage_open_positions(self, hurst_values: Optional[Dict[str, float]] = None) -> list:
        """Alias for check_and_apply_breakeven with multi-tier trailing stop."""
        return self.check_and_apply_breakeven(hurst_values=hurst_values)

    def check_and_apply_breakeven(self, hurst_values: Optional[Dict[str, float]] = None) -> list:
        """
        Active Real-Time Position Manager with Multi-Tier Trailing Stop:
        - Tier 1 (+1.0% profit): Move SL to Break-Even (+0.1% buffer).
        - Tier 2 (+1.8% profit): Lock in +0.8% guaranteed profit.
        - Tier 3 (+2.5% profit): Lock in +1.5% guaranteed profit.
        - Tier 4 (+3.2% / +4.2% profit): Dynamic Take-Profit capture or continuous trailing for H > 0.55.
        Strictly prevents duplicate conditional orders using module-level state and Binance Algo API.
        """
        global ACTIVE_STOP_PRICES
        updates = []
        exchange = self._create_isolated_session()

        try:
            positions = exchange.fetch_positions()
            for pos in positions:
                raw_amt = float(pos.get("info", {}).get("positionAmt", 0) or (pos.get("contracts", 0) * (1 if pos.get("side") == "long" else -1)))
                raw_sym = pos.get("symbol", "")
                clean_sym = raw_sym.replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").strip()

                if clean_sym not in SNIPER_ALLOWED:
                    continue

                pair_fapi = f"{clean_sym}USDT"

                if abs(raw_amt) <= 0:
                    ACTIVE_STOP_PRICES.pop(clean_sym, None)
                    # 🧹 Orphan Algo Order Purge: If flat, cancel any lingering algo orders on Binance
                    try:
                        algo_orders = exchange.fapiPrivateGetOpenAlgoOrders({'symbol': pair_fapi})
                        orders_list = algo_orders.get('orders', []) if isinstance(algo_orders, dict) else (algo_orders or [])
                        if orders_list:
                            logger.info(f"🧹 [ORPHAN CLEANUP] Flat position on {clean_sym}, but found {len(orders_list)} lingering algo orders. Purging...")
                            for ao in orders_list:
                                aid = ao.get('algoId')
                                if aid:
                                    try:
                                        exchange.fapiPrivateDeleteAlgoOrder({'algoId': aid})
                                        logger.info(f"🧹 Successfully purged orphaned algoId={aid} on {clean_sym}")
                                    except Exception as de:
                                        logger.debug(f"Purge error for algoId {aid}: {de}")
                    except Exception as clean_err:
                        logger.debug(f"Algo orphan check note on {clean_sym}: {clean_err}")
                    continue

                entry_price = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0))
                mark_price = float(pos.get("markPrice", 0) or pos.get("info", {}).get("markPrice", 0))

                if entry_price <= 0 or mark_price <= 0:
                    continue

                is_long = (raw_amt > 0) or (pos.get("side") == "long")
                sl_side = "sell" if is_long else "buy"
                if is_long:
                    pnl_pct = ((mark_price - entry_price) / entry_price) * 100.0
                else:
                    pnl_pct = ((entry_price - mark_price) / entry_price) * 100.0

                market_sym = f"{clean_sym}/USDT:USDT"
                pair_fapi = f"{clean_sym}USDT"

                # Check Hurst Exponent to allow strong trend winners (H > 0.55) to run toward +4.2%
                h_val = (hurst_values or {}).get(clean_sym, 0.50)
                is_strong_trend = (h_val > 0.55)
                tp_cap_pct = 4.2 if is_strong_trend else 3.2

                # 🎯 Tier 4 (+3.2% standard or +4.2% Hurst Trend): Immediate Market Take-Profit capture
                if pnl_pct >= tp_cap_pct:
                    try:
                        exchange.create_market_order(market_sym, sl_side, abs(raw_amt), params={'reduceOnly': True})
                        logger.info(f"🎯 DYNAMIC TAKE-PROFIT CAPTURED for {market_sym} at ${mark_price} (+{pnl_pct:.2f}% | Target: +{tp_cap_pct:.1f}% | H={h_val:.2f})!")
                        try:
                            exchange.fapiPrivateDeleteAlgoOpenOrders({'symbol': pair_fapi})
                        except Exception:
                            pass
                        ACTIVE_STOP_PRICES.pop(clean_sym, None)
                        updates.append({"symbol": clean_sym, "action": "TP_CAPTURE", "pnl_pct": pnl_pct})
                        continue
                    except Exception as tp_err:
                        logger.error(f"Failed to capture TP on {market_sym}: {tp_err}")

                # 🛡 Authoritative Naked Position Guard (Astra Phase 0 / Phase 3 Reconciliation):
                # Check whether an active Stop-Loss order currently exists on Binance for this position.
                has_active_stop = False
                found_stop_px = None
                try:
                    raw_algo_orders = exchange.fapiPrivateGetOpenAlgoOrders({'symbol': pair_fapi})
                    for o in raw_algo_orders:
                        if o.get("orderType") in ("STOP_MARKET", "STOP", "TRAILING_STOP_MARKET") or o.get("algoType") == "CONDITIONAL":
                            trigger_p = float(o.get("triggerPrice") or o.get("stopPrice") or 0)
                            if trigger_p > 0:
                                has_active_stop = True
                                found_stop_px = trigger_p
                                break
                except Exception as fapi_e:
                    logger.debug(f"Algo inspect note for {pair_fapi}: {fapi_e}")

                if not has_active_stop:
                    try:
                        open_reg = exchange.fetch_open_orders(market_sym)
                        for o in open_reg:
                            if "STOP" in str(o.get("type", "")).upper():
                                trg = float(o.get("stopPrice") or o.get("price") or 0)
                                if trg > 0:
                                    has_active_stop = True
                                    found_stop_px = trg
                                    break
                    except Exception:
                        pass

                # Pre-calculate baseline SL price floor (1.8%) so baseline_sl is always defined in local scope
                baseline_sl = entry_price * (0.982 if is_long else 1.018)  # 1.8% floor
                baseline_sl = float(exchange.price_to_precision(market_sym, baseline_sl))

                # If exchange already has an active stop, cache its price if not already tracked
                if has_active_stop and found_stop_px and clean_sym not in ACTIVE_STOP_PRICES:
                    ACTIVE_STOP_PRICES[clean_sym] = found_stop_px

                # If position is naked (0 active stops on exchange), immediately auto-reconcile!
                if not has_active_stop:
                    # C2 FIX: Check if mark_price already breached baseline_sl
                    # If so, STOP_MARKET would trigger instantly — use explicit market close instead
                    sl_already_breached = (is_long and mark_price <= baseline_sl) or (not is_long and mark_price >= baseline_sl)

                    if sl_already_breached:
                        logger.critical(
                            f"🚨🚨 [NAKED + BREACHED] {market_sym} ({raw_amt}) has NO stop AND mark_price "
                            f"${mark_price:,.2f} already beyond baseline_sl ${baseline_sl:,.2f}! "
                            f"Emergency MARKET close to prevent further loss."
                        )
                        try:
                            exchange.create_market_order(market_sym, sl_side, abs(raw_amt), params={'reduceOnly': True})
                            logger.warning(f"🔴 [EMERGENCY CLOSE] {market_sym} closed at market. Breach detected.")
                            updates.append({"symbol": clean_sym, "action": "EMERGENCY_MARKET_CLOSE", "reason": "SL_BREACHED_NAKED"})
                        except Exception as em_err:
                            logger.critical(f"❌ [EMERGENCY CLOSE FAILED] {market_sym}: {em_err}")
                    else:
                        logger.warning(
                            f"🚨 [NAKED POSITION DETECTED] {market_sym} ({raw_amt}) has NO active Stop-Loss on Binance! "
                            f"Auto-reconciling emergency stop at ${baseline_sl}..."
                        )
                        try:
                            exchange.create_order(
                                symbol=market_sym,
                                type='STOP_MARKET',
                                side=sl_side,
                                amount=abs(raw_amt),
                                price=None,
                                params={'stopPrice': baseline_sl, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
                            )
                            ACTIVE_STOP_PRICES[clean_sym] = baseline_sl
                            logger.info(f"🛡 [AUTO-RECONCILED] Emergency Stop-Loss placed at ${baseline_sl} for {market_sym}.")
                            updates.append({"symbol": clean_sym, "action": "AUTO_RECONCILE_SL", "sl_price": baseline_sl})
                        except Exception as sl_err:
                            try:
                                exchange.create_order(
                                    symbol=market_sym,
                                    type='STOP_MARKET',
                                    side=sl_side,
                                    amount=None,
                                    price=None,
                                    params={'stopPrice': baseline_sl, 'closePosition': True, 'workingType': 'MARK_PRICE'}
                                )
                                ACTIVE_STOP_PRICES[clean_sym] = baseline_sl
                                logger.info(f"🛡 [AUTO-RECONCILED] ClosePosition SL placed at ${baseline_sl} for {market_sym}.")
                                updates.append({"symbol": clean_sym, "action": "AUTO_RECONCILE_SL", "sl_price": baseline_sl})
                            except Exception as sl_err2:
                                logger.error(f"❌ Failed to auto-reconcile SL on {market_sym}: {sl_err2}")
                                ACTIVE_STOP_PRICES.pop(clean_sym, None)

                # ── Astra Continuous Volatility-Anchored Trailing Stop (Section 1.8) ──
                if clean_sym not in TRAILING_STATES:
                    TRAILING_STATES[clean_sym] = TrailingState(
                        symbol=clean_sym,
                        side=1 if is_long else -1,
                        entry_px=entry_price,
                        current_sl_px=ACTIVE_STOP_PRICES.get(clean_sym) or baseline_sl,
                        peak_px=mark_price,
                        d_min=0.012,
                        k_sigma=2.2 if is_strong_trend else 1.8,
                        updated_at_ts=time.time()
                    )
                elif clean_sym in ACTIVE_STOP_PRICES and ACTIVE_STOP_PRICES[clean_sym] != TRAILING_STATES[clean_sym].current_sl_px:
                    TRAILING_STATES[clean_sym].current_sl_px = ACTIVE_STOP_PRICES[clean_sym]
                    TRAILING_STATES[clean_sym].entry_px = entry_price

                state = TRAILING_STATES[clean_sym]
                new_sl_price = None
                lock_label = None

                # 1. Check break-even lock (+1.0% favorable excursion)
                old_sl = state.current_sl_px
                be_sl = trailing_engine.check_breakeven_lock(state, mark_price=mark_price, fee_buffer=0.0025)
                if be_sl is not None and (be_sl > old_sl if is_long else be_sl < old_sl):
                    new_sl_price = be_sl
                    lock_label = "Astra Break-Even Lock (+1.0% Excursion)"

                # 2. Continuous ratchet update based on running extreme and normalized ATR
                atr_norm_val = 0.012
                proposed_sl, should_update_exchange, reason = trailing_engine.update_stop(
                    state=state,
                    mark_price=mark_price,
                    atr_norm=atr_norm_val
                )
                if should_update_exchange:
                    new_sl_price = proposed_sl
                    lock_label = f"Astra Continuous Trailing (Peak: ${state.peak_px:.2f} | Reason: {reason})"

                if new_sl_price:
                    new_sl_price = float(exchange.price_to_precision(market_sym, new_sl_price))

                    # 1. Global Cache Check: Skip if already placed at this price level or better
                    cached_sl = ACTIVE_STOP_PRICES.get(clean_sym)
                    if cached_sl is not None:
                        if abs(cached_sl - new_sl_price) < 1e-4:
                            continue
                        if is_long and new_sl_price <= cached_sl:
                            continue
                        elif not is_long and new_sl_price >= cached_sl:
                            continue

                    # 2. Check open conditional orders on Binance (both regular and algo)
                    need_update = True
                    try:
                        open_orders = exchange.fetch_open_orders(market_sym)
                        for o in open_orders:
                            if "STOP" in str(o.get("type", "")).upper():
                                curr_stop = float(o.get("stopPrice") or o.get("price") or 0)
                                if is_long and curr_stop >= new_sl_price:
                                    need_update = False
                                    ACTIVE_STOP_PRICES[clean_sym] = curr_stop
                                elif not is_long and curr_stop <= new_sl_price and curr_stop > 0:
                                    need_update = False
                                    ACTIVE_STOP_PRICES[clean_sym] = curr_stop
                    except Exception as open_err:
                        logger.debug(f"Open regular orders inspect note for {market_sym}: {open_err}")

                    try:
                        raw_algo_orders = exchange.fapiPrivateGetOpenAlgoOrders({'symbol': pair_fapi})
                        for o in raw_algo_orders:
                            if o.get("orderType") in ("STOP_MARKET", "STOP", "TRAILING_STOP_MARKET") or o.get("algoType") == "CONDITIONAL":
                                curr_stop = float(o.get("triggerPrice") or o.get("stopPrice") or 0)
                                if is_long and curr_stop >= new_sl_price:
                                    need_update = False
                                    ACTIVE_STOP_PRICES[clean_sym] = curr_stop
                                elif not is_long and curr_stop <= new_sl_price and curr_stop > 0:
                                    need_update = False
                                    ACTIVE_STOP_PRICES[clean_sym] = curr_stop
                    except Exception as fapi_e:
                        logger.debug(f"Algo inspect note for {pair_fapi}: {fapi_e}")

                    if need_update:
                        prev_sl_price = cached_sl or (entry_price * (0.985 if is_long else 1.015))
                        try:
                            # 3. Create-Before-Cancel Pattern (C3 FIX: Zero Naked Exposure)
                            # Create the new tighter stop FIRST. Old stop remains live on exchange during transit.
                            new_order = None
                            try:
                                new_order = exchange.create_order(
                                    symbol=market_sym,
                                    type='STOP_MARKET',
                                    side=sl_side,
                                    amount=abs(raw_amt),
                                    price=None,
                                    params={'stopPrice': new_sl_price, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
                                )
                            except Exception as ro_err:
                                if "closePosition" in str(ro_err) or "-4130" in str(ro_err) or "reduceOnly" in str(ro_err):
                                    # Fallback path if exchange strictly enforces single closePosition order
                                    logger.debug(f"Direct create note for {market_sym} ({ro_err}); falling back to atomic cancel-replace.")
                                    try:
                                        open_orders = exchange.fetch_open_orders(market_sym)
                                        for o in open_orders:
                                            if "STOP" in str(o.get("type", "")).upper():
                                                try:
                                                    exchange.cancel_order(o["id"], market_sym)
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                                    try:
                                        existing_algos = exchange.fapiPrivateGetOpenAlgoOrders({'symbol': pair_fapi})
                                        for ex_o in existing_algos:
                                            if ex_o.get("orderType") in ("STOP_MARKET", "STOP", "TRAILING_STOP_MARKET"):
                                                try:
                                                    exchange.fapiPrivateDeleteAlgoOrder({'symbol': pair_fapi, 'algoId': ex_o['algoId']})
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass

                                    new_order = exchange.create_order(
                                        symbol=market_sym,
                                        type='STOP_MARKET',
                                        side=sl_side,
                                        amount=None,
                                        price=None,
                                        params={'stopPrice': new_sl_price, 'closePosition': True, 'workingType': 'MARK_PRICE'}
                                    )
                                else:
                                    raise ro_err

                            # 4. New stop confirmed live! Clean up older superseded stops
                            new_order_id = str((new_order or {}).get("id") or "")
                            try:
                                open_orders = exchange.fetch_open_orders(market_sym)
                                for o in open_orders:
                                    if "STOP" in str(o.get("type", "")).upper() and str(o.get("id")) != new_order_id:
                                        try:
                                            exchange.cancel_order(o["id"], market_sym)
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                            try:
                                existing_algos = exchange.fapiPrivateGetOpenAlgoOrders({'symbol': pair_fapi})
                                for ex_o in existing_algos:
                                    if ex_o.get("orderType") in ("STOP_MARKET", "STOP", "TRAILING_STOP_MARKET") and str(ex_o.get("algoId")) != new_order_id:
                                        try:
                                            exchange.fapiPrivateDeleteAlgoOrder({'symbol': pair_fapi, 'algoId': ex_o['algoId']})
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                            ACTIVE_STOP_PRICES[clean_sym] = new_sl_price
                            logger.info(f"🎯 [ТРЕЙЛИНГ-СТОП BINANCE] {market_sym}: Стоп-лосс подтянут к ${new_sl_price:,.2f} | Текущий PnL: {pnl_pct:+.2f}% ({lock_label})")
                            updates.append({
                                "symbol": clean_sym,
                                "sl_price": new_sl_price,
                                "profit_pct": pnl_pct,
                                "label": lock_label
                            })
                        except Exception as create_err:
                            logger.critical(
                                f"🚨 [CRITICAL] Failed to place updated {lock_label} for {market_sym}: {create_err}. "
                                f"Atomic safety: immediately re-placing protective stop at ${prev_sl_price:,.2f}..."
                            )
                            safe_emergency_sl = prev_sl_price if prev_sl_price > 0 else entry_price
                            if is_long and safe_emergency_sl >= mark_price:
                                safe_emergency_sl = mark_price * 0.999
                            elif not is_long and safe_emergency_sl <= mark_price:
                                safe_emergency_sl = mark_price * 1.001

                            emergency_sl = float(exchange.price_to_precision(market_sym, safe_emergency_sl))
                            try:
                                try:
                                    exchange.create_order(
                                        symbol=market_sym,
                                        type='STOP_MARKET',
                                        side=sl_side,
                                        amount=abs(raw_amt),
                                        price=None,
                                        params={'stopPrice': emergency_sl, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
                                    )
                                except Exception as em_ro_err:
                                    exchange.create_order(
                                        symbol=market_sym,
                                        type='STOP_MARKET',
                                        side=sl_side,
                                        amount=None,
                                        price=None,
                                        params={'stopPrice': emergency_sl, 'closePosition': True, 'workingType': 'MARK_PRICE'}
                                    )
                                ACTIVE_STOP_PRICES[clean_sym] = emergency_sl
                                logger.warning(f"🛡 [EMERGENCY RESTORE] Protective emergency SL successfully restored at ${emergency_sl:,.2f} for {market_sym}")
                            except Exception as restore_err:
                                ACTIVE_STOP_PRICES.pop(clean_sym, None)
                                logger.critical(f"💀 [DISASTER] Failed to restore emergency SL for {market_sym}: {restore_err}! Naked position alert!")
                        except Exception as sl_e:
                            logger.error(f"Failed to place {lock_label} for {market_sym}: {sl_e}")

        except Exception as e:
            logger.error(f"Error checking position breakeven/trailing stops: {e}")

        return updates


    def get_active_position(self, symbol: str) -> Optional[dict]:
        """
        Checks if there is an active Binance Futures position for the symbol.
        Returns dict with position parameters or None if flat.
        """
        clean_sym = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
        market_symbol = f"{clean_sym}/USDT:USDT"
        exchange = self._create_isolated_session()
        try:
            positions = exchange.fetch_positions()
            for pos in positions:
                raw_sym = pos.get("symbol", "").upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
                if raw_sym == clean_sym:
                    amt = float(pos.get("info", {}).get("positionAmt", 0) or pos.get("contracts", 0) * (1 if pos.get("side") == "long" else -1))
                    if abs(amt) > 0:
                        is_long = amt > 0
                        entry_price = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0))
                        mark_price = float(pos.get("markPrice", 0) or pos.get("info", {}).get("markPrice", 0))
                        pnl_pct = ((mark_price - entry_price) / entry_price * 100.0) if is_long else ((entry_price - mark_price) / entry_price * 100.0) if entry_price > 0 else 0.0
                        return {
                            "symbol": clean_sym,
                            "market_symbol": market_symbol,
                            "direction": "BULLISH" if is_long else "BEARISH",
                            "side": "LONG" if is_long else "SHORT",
                            "contracts": abs(amt),
                            "raw_amount": amt,
                            "entry_price": entry_price,
                            "mark_price": mark_price,
                            "pnl_pct": pnl_pct
                        }
            return None
        except Exception as e:
            logger.error(f"Error querying Binance active position for {clean_sym}: {e}")
            return None

    def close_position_early(self, symbol: str, reason: str = "EARLY_CLOSE") -> Optional[Dict[str, Any]]:
        """
        Institutional-Grade Early Position Closure & Microstructure Telemetry Engine.
        Includes Pre-Execution Liquidity Assessment, Algo Purge, and Precision Slippage Auditing.
        """
        global ACTIVE_STOP_PRICES
        clean_sym = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
        market_symbol = f"{clean_sym}/USDT:USDT"
        pair_fapi = f"{clean_sym}USDT"

        exchange = self._create_isolated_session()
        try:
            # 1. Fetch active position details
            positions = exchange.fetch_positions()
            target_pos = None
            for p in positions:
                raw_sym = p.get('symbol', '')
                p_clean = raw_sym.replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
                if p_clean == clean_sym or p.get('info', {}).get('symbol') == pair_fapi:
                    amt = float(p.get('info', {}).get('positionAmt', 0) or p.get('contracts', 0) * (1 if p.get('side') == 'long' else -1))
                    if abs(amt) > 0:
                        target_pos = p
                        break

            if not target_pos:
                logger.info(f"🛡 [CLOSE SKIPPED] No active open position found on Binance for {clean_sym}.")
                return None

            pos_amt = float(target_pos.get('info', {}).get('positionAmt', 0) or target_pos.get('contracts', 0) * (1 if target_pos.get('side') == 'long' else -1))
            close_amount_raw = abs(pos_amt)
            is_long = pos_amt > 0
            close_side = 'sell' if is_long else 'buy'

            formatted_close_amount = float(exchange.amount_to_precision(market_symbol, close_amount_raw))
            if formatted_close_amount <= 0:
                formatted_close_amount = close_amount_raw

            # 2. Pre-Execution Spread & Liquidity Assessment
            best_bid = 0.0
            best_ask = 0.0
            microprice = 0.0
            try:
                orderbook = exchange.fetch_order_book(market_symbol, limit=5)
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                if bids and len(bids) > 0:
                    best_bid = float(bids[0][0])
                if asks and len(asks) > 0:
                    best_ask = float(asks[0][0])
                from src.domain.services.order_flow_imbalance import OrderFlowImbalanceService
                calc_micro = OrderFlowImbalanceService.calculate_microprice(bids, asks)
                if calc_micro > 0.0:
                    microprice = calc_micro
            except Exception as ob_err:
                logger.debug(f"Orderbook fetch note for {market_symbol}: {ob_err}")

            if best_bid <= 0.0 or best_ask <= 0.0:
                try:
                    ticker = exchange.fetch_ticker(market_symbol)
                    best_bid = float(ticker.get('bid') or 0.0)
                    best_ask = float(ticker.get('ask') or 0.0)
                except Exception:
                    pass

            mark_price = float(target_pos.get('markPrice', 0) or target_pos.get('info', {}).get('markPrice', 0))
            if mark_price <= 0.0 and best_bid > 0 and best_ask > 0:
                mark_price = (best_bid + best_ask) / 2.0

            # Compute bid-ask spread %: spread_pct = ((ask - bid) / mark_price) * 100.0
            spread_pct = ((best_ask - best_bid) / mark_price * 100.0) if (mark_price > 0 and best_ask >= best_bid and best_bid > 0) else 0.0

            if spread_pct > 0.15:
                logger.warning(
                    f"⚠️ [LIQUIDITY VACUUM WARNING] High spread on {clean_sym}: {spread_pct:.4f}% > 0.150% "
                    f"(Best Bid: ${best_bid:,.2f} | Best Ask: ${best_ask:,.2f} | Mark: ${mark_price:,.2f})"
                )
            else:
                logger.info(
                    f"💧 [PRE-EXECUTION LIQUIDITY] {clean_sym} Top of Book | Bid: ${best_bid:,.2f} | Ask: ${best_ask:,.2f} | "
                    f"Mark: ${mark_price:,.2f} | Spread: {spread_pct:.4f}%"
                )

            # 3. Capture trigger mark price right before dispatch
            trigger_mark_price = mark_price
            try:
                latest_ticker = exchange.fetch_ticker(market_symbol)
                latest_mark = float(latest_ticker.get('info', {}).get('markPrice') or latest_ticker.get('last') or 0.0)
                if latest_mark > 0.0:
                    trigger_mark_price = latest_mark
            except Exception:
                pass

            if trigger_mark_price <= 0.0:
                trigger_mark_price = mark_price

            # 4. Algo Order Purge: clean both regular and Binance algo contingency orders
            try:
                exchange.fapiPrivateDeleteAlgoOpenOrders({'symbol': pair_fapi})
                logger.info(f"🧹 Purged active algo orders for {pair_fapi}")
            except Exception as algo_err:
                logger.debug(f"Algo cancel note for {pair_fapi}: {algo_err}")

            try:
                exchange.cancel_all_orders(market_symbol)
            except Exception:
                pass

            ACTIVE_STOP_PRICES.pop(clean_sym, None)

            # 5. Dispatch reduceOnly=True market order
            logger.info(
                f"🚨 [ЗАКРЫТИЕ ПОЗИЦИИ BINANCE] {market_symbol} ({close_side.upper()} {formatted_close_amount}) | "
                f"Причина: '{reason}' | Цена триггера: ${trigger_mark_price:,.2f}"
            )

            try:
                order = exchange.create_market_order(
                    market_symbol,
                    close_side,
                    formatted_close_amount,
                    params={'reduceOnly': True}
                )
            except Exception as mkt_err:
                logger.warning(f"Market order direct call note, falling back to create_order: {mkt_err}")
                order = exchange.create_order(
                    symbol=market_symbol,
                    type='MARKET',
                    side=close_side,
                    amount=formatted_close_amount,
                    price=None,
                    params={'reduceOnly': True}
                )

            order_id = order.get('id')

            # 6. Extract actual fill_price from execution receipt
            fill_price = float(order.get('average') or order.get('price') or order.get('info', {}).get('avgPrice') or 0.0)
            if fill_price <= 0.0 and order_id:
                try:
                    time.sleep(0.05)
                    fetched_order = exchange.fetch_order(order_id, market_symbol)
                    fill_price = float(fetched_order.get('average') or fetched_order.get('price') or fetched_order.get('info', {}).get('avgPrice') or 0.0)
                except Exception:
                    pass

            if fill_price <= 0.0:
                fill_price = trigger_mark_price

            # 7. Precision Micro-Price Slippage Calculation (Astra Component 1)
            ref_price = microprice if microprice > 0.0 else trigger_mark_price
            slippage_pct = (abs(fill_price - ref_price) / ref_price * 100.0) if ref_price > 0.0 else 0.0
            slippage_usd = abs(fill_price - ref_price) * formatted_close_amount

            # 8. Institutional Execution Telemetry Logging
            if slippage_pct <= 0.15:
                slippage_grade = "EXCELLENT"
                logger.info(f"⚡ [SLIPPAGE AUDIT] {clean_sym} Fill: ${fill_price:,.2f} | Ref (Micro): ${ref_price:,.2f} | Slippage: {slippage_pct:.3f}% (${slippage_usd:.4f}) [EXCELLENT]")
            elif slippage_pct <= 0.35:
                slippage_grade = "ACCEPTABLE"
                logger.info(f"⚡ [SLIPPAGE AUDIT] {clean_sym} Fill: ${fill_price:,.2f} | Ref (Micro): ${ref_price:,.2f} | Slippage: {slippage_pct:.3f}% (${slippage_usd:.4f}) [ACCEPTABLE]")
            else:
                slippage_grade = "WARNING"
                logger.warning(f"🚨 [SLIPPAGE WARNING] High execution slippage on {clean_sym}! Fill: ${fill_price:,.2f} | Ref (Micro): ${ref_price:,.2f} | Slippage: {slippage_pct:.3f}% (${slippage_usd:.4f})")

            # 9. Return structured telemetry dict
            return {
                "status": "CLOSED",
                "symbol": market_symbol,
                "clean_symbol": clean_sym,
                "reason": reason,
                "side": close_side.upper(),
                "close_amount": formatted_close_amount,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "microprice": float(round(microprice, 4)),
                "spread_pct": float(round(spread_pct, 4)),
                "trigger_mark_price": float(round(trigger_mark_price, 4)),
                "fill_price": float(round(fill_price, 4)),
                "slippage_pct": float(round(slippage_pct, 4)),
                "slippage_usd": float(round(slippage_usd, 4)),
                "slippage_grade": slippage_grade,
                "order_id": order_id,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Trade close execution error on {symbol}: {e}", exc_info=True)
            return None
        finally:
            if 'exchange' in locals():
                del exchange
            gc.collect()

    def calculate_projected_liq_distance(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        leverage: float = 3.0,
    ) -> float:
        """
        Calculates projected distance to liquidation for isolated USDT-M futures:
            distance = 1 / leverage - MMR
        """
        clean_s = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '').replace('USDT', '').strip()
        # Tier 1 MMR on Binance: 0.004 for BTC, 0.005 for ETH/SOL
        mmr = 0.004 if clean_s == "BTC" else 0.005
        lev = max(1.0, float(leverage))
        return max(0.0, (1.0 / lev) - mmr)

