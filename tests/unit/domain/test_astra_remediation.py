import pytest
import numpy as np

from src.domain.services.correlation_guard import CrossAssetCorrelationGuard
from src.domain.services.hurst_regime_service import HurstRegimeService, HurstRegimeType
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.infrastructure.persistence.signal_logger import SignalLogger
from src.infrastructure.execution.paper_trader import PaperTrader


class TestAstraQuantRemediation:
    def test_correlation_guard_gross_and_aligned_caps(self):
        guard = CrossAssetCorrelationGuard()

        # Equity = $40.0. Max gross = 3.0x = $120.0. Max aligned = 2.2x = $88.0.
        equity = 40.0

        # Scenario 1: First Trade (BTC Long) -> 100% sizing, allowed
        res1 = guard.evaluate_candidate_trade(
            candidate_symbol="BTC",
            candidate_direction="LONG",
            candidate_confidence=0.76,
            active_positions=[],
            portfolio_equity=equity,
            estimated_trade_notional=25.0
        )
        assert res1.allowed is True
        assert res1.margin_haircut_multiplier == 1.00
        assert res1.veto is False

        # Scenario 2: Second Aligned Trade (ETH Long, BTC Long already active with $25 notional)
        active_pos_1 = [{
            "symbol": "BTC/USDT:USDT",
            "direction": "LONG",
            "notional": 25.0,
            "markPrice": 60000.0,
            "info": {"positionAmt": "0.0004"}
        }]
        res2 = guard.evaluate_candidate_trade(
            candidate_symbol="ETH",
            candidate_direction="LONG",
            candidate_confidence=0.76,
            active_positions=active_pos_1,
            portfolio_equity=equity,
            estimated_trade_notional=25.0
        )
        assert res2.allowed is True
        assert res2.margin_haircut_multiplier == 0.75  # 25% haircut for 2nd aligned trade

        # Scenario 3: Third Aligned Trade (SOL Long, BTC and ETH Long active)
        active_pos_2 = active_pos_1 + [{
            "symbol": "ETH/USDT:USDT",
            "direction": "LONG",
            "notional": 25.0,
            "markPrice": 3000.0,
            "info": {"positionAmt": "0.008"}
        }]
        # Subcase 3a: Confidence 76% < 80% threshold -> VETO
        res3_low = guard.evaluate_candidate_trade(
            candidate_symbol="SOL",
            candidate_direction="LONG",
            candidate_confidence=0.76,
            active_positions=active_pos_2,
            portfolio_equity=equity,
            estimated_trade_notional=25.0
        )
        assert res3_low.allowed is False
        assert res3_low.veto is True
        assert "CORRELATION_VETO" in (res3_low.rejection_reason or "")
        assert "requires >= 80% confidence" in (res3_low.rejection_reason or "")

        # Subcase 3b: Confidence 82% >= 80% -> Allowed with 50% haircut
        res3_high = guard.evaluate_candidate_trade(
            candidate_symbol="SOL",
            candidate_direction="LONG",
            candidate_confidence=0.82,
            active_positions=active_pos_2,
            portfolio_equity=equity,
            estimated_trade_notional=25.0
        )
        assert res3_high.allowed is True
        assert res3_high.margin_haircut_multiplier == 0.50

        # Scenario 4: Portfolio Gross Notional Cap Exceeded (> 3.0x equity = $120)
        huge_active = [{
            "symbol": "BTC/USDT:USDT",
            "direction": "SHORT",
            "notional": 105.0,
            "markPrice": 60000.0,
            "info": {"positionAmt": "-0.0017"}
        }]
        res_gross = guard.evaluate_candidate_trade(
            candidate_symbol="ETH",
            candidate_direction="LONG",
            candidate_confidence=0.85,
            active_positions=huge_active,
            portfolio_equity=equity,
            estimated_trade_notional=25.0
        )
        assert res_gross.allowed is False
        assert res_gross.veto is True
        assert "GROSS_NOTIONAL_VETO" in (res_gross.rejection_reason or "")

    def test_hurst_dual_window_momentum_exhaustion(self):
        service = HurstRegimeService()

        # Synthetic prices: strong macro trend (N=100) followed by fast mean-reverting chop (last 36)
        macro_trend = np.linspace(100.0, 200.0, 64) + np.random.normal(0, 0.5, 64)
        t = np.arange(36)
        fast_chop = 200.0 + 3.0 * np.sin(t * 1.2)

        combined_prices = np.concatenate([macro_trend, fast_chop])
        
        H_med = service.calculate_hurst_rs(combined_prices)
        H_fast = service.calculate_hurst_rs(combined_prices[-36:])

        assert H_med > 0.50
        assert H_fast <= H_med

    def test_paper_trader_net_breakeven_floor(self, tmp_path):
        db_file = str(tmp_path / "test_mabuy.db")
        tracker = SQLiteTracker(db_path=db_file)
        signal_logger = SignalLogger(db_path=db_file)

        paper = PaperTrader(user_id="test_user", tracker=tracker, signal_logger=signal_logger)
        assert paper.TIER1_LOCK_PCT == 0.25  # +0.25% Net-of-fees floor
