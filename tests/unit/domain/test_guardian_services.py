import time
import pytest

from src.domain.services.depth_slippage_auditor import DepthIntegratedSlippageAuditor, DepthExecutionAuditResult
from src.domain.services.stratified_regime_tracker import StratifiedRegimeQuotaTracker
from src.domain.services.hardware_kill_switch import HardwareKillSwitchService, KillSwitchState


class TestGuardianServices:
    def test_depth_slippage_auditor_hollow_book_veto(self):
        auditor = DepthIntegratedSlippageAuditor(max_allowed_slippage_pct=0.25)

        # Normal deep book: 100 BTC depth near spread
        bids = [[50000.0, 50.0], [49950.0, 50.0]]
        asks = [[50050.0, 50.0], [50100.0, 50.0]]
        
        # Test small position $5,000 (~0.1 BTC)
        res_normal = auditor.audit_execution_cost(bids, asks, position_size_usd=5000.0, direction="BULLISH")
        assert res_normal.is_liquid_enough is True
        assert res_normal.is_hollow_book_veto is False
        assert res_normal.order_type_recommended == "MARKET_TAKER"

        # Hollow book: only 0.05 BTC available at top levels, then empty gap
        hollow_asks = [[50050.0, 0.05], [52000.0, 1.0]]
        res_hollow = auditor.audit_execution_cost(bids, hollow_asks, position_size_usd=10000.0, direction="BULLISH")
        assert res_hollow.is_hollow_book_veto is True
        assert res_hollow.order_type_recommended == "LIMIT_POST_ONLY"
        assert "HOLLOW BOOK VETO" in res_hollow.veto_reason

    def test_stratified_regime_quota_prevents_overfitting(self):
        tracker = StratifiedRegimeQuotaTracker(custom_quotas={
            "BULL_TREND": 5,
            "BEAR_TREND": 5,
            "SQUEEZE": 5,
            "HIGH_VOLATILITY": 5
        })

        # Record 100 trades solely in SQUEEZE (Overfitting scenario)
        for _ in range(100):
            tracker.record_shadow_trade(regime="SQUEEZE", is_win=True, pnl_usd=50.0)

        eval_1 = tracker.evaluate_live_readiness()
        assert eval_1["is_live_ready"] is False
        assert eval_1["all_quotas_met"] is False
        assert len(eval_1["unmet_reasons"]) == 3 # BULL, BEAR, HIGH_VOL not fulfilled

        # Fulfill all 4 regimes with positive expectancy
        for r in ["BULL_TREND", "BEAR_TREND", "HIGH_VOLATILITY"]:
            for _ in range(5):
                tracker.record_shadow_trade(regime=r, is_win=True, pnl_usd=40.0)

        eval_2 = tracker.evaluate_live_readiness()
        assert eval_2["is_live_ready"] is True
        assert eval_2["all_quotas_met"] is True
        assert eval_2["overall_profit_factor"] > 1.50

    def test_hardware_kill_switch_protections(self):
        kill_switch = HardwareKillSwitchService(max_api_errors_5m=3, max_consecutive_sl=3)

        # 1. Idempotent Order ID Generation
        id1 = kill_switch.generate_idempotent_order_id("BTCUSDT", "BUY", timestamp_ms=1700000001000)
        id2 = kill_switch.generate_idempotent_order_id("BTCUSDT", "BUY", timestamp_ms=1700000002000)
        # Within same 5s bucket => identical ID
        assert id1 == id2

        # 2. 3-Strike API Failure Halt
        kill_switch.record_api_error("502 Bad Gateway", "Binance API down")
        kill_switch.record_api_error("504 Gateway Timeout", "Socket freeze")
        assert kill_switch.evaluate_state().is_trading_allowed is True

        # 3rd error trips the kill-switch
        kill_switch.record_api_error("429 Rate Limit", "IP throttled")
        state = kill_switch.evaluate_state()
        assert state.is_trading_allowed is False
        assert state.state_code == "API_ERROR_HALT"
        assert "CRITICAL API HALT" in state.reason

        # 3. Manual Emergency Halt override
        kill_switch.trigger_manual_emergency_halt("Operator intervention via /emergency_halt")
        manual_state = kill_switch.evaluate_state()
        assert manual_state.is_trading_allowed is False
        assert manual_state.state_code == "MANUAL_EMERGENCY_HALT"
