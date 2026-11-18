import pytest
from src.domain.services.asymmetric_risk_engine import AsymmetricRiskEngine, RiskEngineConfiguration

def test_asymmetric_loss_cutter():
    engine = AsymmetricRiskEngine()
    
    # Normal hold: -0.5% after 1 bar
    should_cut, reason = engine.evaluate_early_loss_cut("BTCUSDT", current_pnl_pct=-0.005, bars_held=1, hurst=0.60, atr_pct=0.015)
    assert not should_cut
    
    # Hard stop: -2.0% (exceeds max 1.8%)
    should_cut, reason = engine.evaluate_early_loss_cut("BTCUSDT", current_pnl_pct=-0.020, bars_held=1, hurst=0.60, atr_pct=0.015)
    assert should_cut
    assert "volatility hard loss cut" in reason
    
    # Early cut: 3 bars held, Hurst 0.50 (stalled non-trending), PnL -1.3% (below tightened -1.2%)
    should_cut, reason = engine.evaluate_early_loss_cut("SOLUSDT", current_pnl_pct=-0.013, bars_held=3, hurst=0.50, atr_pct=0.015)
    assert should_cut
    assert "early loss cut" in reason

def test_dynamic_tp_targets():
    engine = AsymmetricRiskEngine()
    
    # Trending Hurst (H=0.62) -> Runner target in [3.5%, 4.5%]
    tp_trend = engine.get_dynamic_take_profit_target("BTCUSDT", hurst=0.62, atr_pct=0.018)
    assert 0.035 <= tp_trend <= 0.045
    
    # Mean-reverting Hurst (H=0.42) -> Scalp target in [0.8%, 1.5%]
    tp_revert = engine.get_dynamic_take_profit_target("BTCUSDT", hurst=0.42, atr_pct=0.010)
    assert 0.008 <= tp_revert <= 0.015

def test_fee_proof_breakeven():
    engine = AsymmetricRiskEngine()
    
    # Long from $100.0, Maker fee (0.02%), spread 0.03%
    be_long = engine.calculate_fee_proof_breakeven(entry_price=100.0, direction="BULLISH", is_maker=True, spread_pct=0.0003)
    assert be_long > 100.0
    # Expected buffer: 2 * 0.0002 + 0.0003 + 0.0020 = 0.0027 (+0.27%)
    assert abs(be_long - 100.27) < 0.01
    
    # Short from $100.0
    be_short = engine.calculate_fee_proof_breakeven(entry_price=100.0, direction="BEARISH", is_maker=True, spread_pct=0.0003)
    assert be_short < 100.0
    assert abs(be_short - 99.73) < 0.01

def test_correlation_beta_freeze():
    engine = AsymmetricRiskEngine(symbol_confidence={"SOLUSDT": 0.82, "ETHUSDT": 0.74})
    
    # Case 1: No active positions -> 1.0 (Full scale)
    scale = engine.check_correlation_scale([], "BTCUSDT", "BULLISH")
    assert scale == 1.0
    
    # Case 2: One active LONG position (BTC), candidate SOL is BULLISH with 82% confidence -> 0.5 (Half margin)
    active = [{"symbol": "BTCUSDT", "direction": "BULLISH", "status": "OPEN"}]
    scale = engine.check_correlation_scale(active, "SOLUSDT", "BULLISH")
    assert scale == 0.5
    
    # Case 3: One active LONG position (BTC), candidate ETH is BULLISH with 74% (<80%) -> 0.0 (VETO)
    scale = engine.check_correlation_scale(active, "ETHUSDT", "BULLISH")
    assert scale == 0.0
    
    # Case 4: Two active LONG positions (BTC, SOL), candidate ETH is BULLISH -> 0.0 (VETO)
    active_two = [
        {"symbol": "BTCUSDT", "direction": "BULLISH", "status": "OPEN"},
        {"symbol": "SOLUSDT", "direction": "BULLISH", "status": "OPEN"}
    ]
    scale = engine.check_correlation_scale(active_two, "ETHUSDT", "BULLISH")
    assert scale == 0.0
