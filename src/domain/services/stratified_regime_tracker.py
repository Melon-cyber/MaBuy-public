import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RegimePerformance:
    regime: str
    target_quota: int
    trades_count: int = 0
    wins_count: int = 0
    losses_count: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0

    @property
    def win_rate_pct(self) -> float:
        total = self.wins_count + self.losses_count
        return round((self.wins_count / total * 100.0), 2) if total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0.0:
            return 99.0 if self.gross_profit > 0 else 1.0
        return round(self.gross_profit / self.gross_loss, 2)

    @property
    def is_quota_fulfilled(self) -> bool:
        return self.trades_count >= self.target_quota

    @property
    def is_statistically_profitable(self) -> bool:
        return self.is_quota_fulfilled and self.profit_factor >= 1.10 and self.win_rate_pct >= 50.0


class StratifiedRegimeQuotaTracker:
    """
    Guarantees Multi-Regime Forward-Testing Verification before Live Capital deployment.
    Eliminates the 'Regime Overfitting' Survivor Bias by requiring minimum trade quotas
    and profitability across all 4 distinct market regimes:
    1. BULL_TREND (Quota: 20)
    2. BEAR_TREND (Quota: 20)
    3. SQUEEZE / Flat (Quota: 25)
    4. HIGH_VOLATILITY / Storm (Quota: 20)
    """

    DEFAULT_QUOTAS = {
        "BULL_TREND": 20,
        "BEAR_TREND": 20,
        "SQUEEZE": 25,
        "HIGH_VOLATILITY": 20
    }

    def __init__(self, custom_quotas: Optional[Dict[str, int]] = None):
        quotas = custom_quotas or self.DEFAULT_QUOTAS
        self.regimes: Dict[str, RegimePerformance] = {
            r: RegimePerformance(regime=r, target_quota=q)
            for r, q in quotas.items()
        }

    def record_shadow_trade(self, regime: str, is_win: bool, pnl_usd: float) -> None:
        clean_regime = regime.upper()
        if clean_regime not in self.regimes:
            clean_regime = "SQUEEZE"

        perf = self.regimes[clean_regime]
        perf.trades_count += 1
        if is_win:
            perf.wins_count += 1
            perf.gross_profit += max(0.0, pnl_usd)
        else:
            perf.losses_count += 1
            perf.gross_loss += abs(pnl_usd)
        perf.net_pnl += pnl_usd

    def evaluate_live_readiness(self) -> Dict[str, Any]:
        """
        Evaluates whether the system is qualified to switch from Shadow Mode to Live Capital.
        """
        all_quotas_met = all(r.is_quota_fulfilled for r in self.regimes.values())
        all_profitable = all(r.is_statistically_profitable for r in self.regimes.values())
        
        total_trades = sum(r.trades_count for r in self.regimes.values())
        total_profit = sum(r.gross_profit for r in self.regimes.values())
        total_loss = sum(r.gross_loss for r in self.regimes.values())
        overall_pf = round(total_profit / max(1e-6, total_loss), 2)
        overall_wr = round(sum(r.wins_count for r in self.regimes.values()) / max(1, total_trades) * 100.0, 2)

        unmet_reasons = []
        for r_name, r_perf in self.regimes.items():
            if not r_perf.is_quota_fulfilled:
                unmet_reasons.append(f"{r_name}: {r_perf.trades_count}/{r_perf.target_quota} trades (Quota not reached)")
            elif not r_perf.is_statistically_profitable:
                unmet_reasons.append(f"{r_name}: PF={r_perf.profit_factor} < 1.10 or WR={r_perf.win_rate_pct}% < 50%")

        is_ready = bool(all_quotas_met and all_profitable)

        breakdown = {
            r_name: {
                "trades": f"{r_perf.trades_count}/{r_perf.target_quota}",
                "trades_count": r_perf.trades_count,
                "target_quota": r_perf.target_quota,
                "win_rate_pct": r_perf.win_rate_pct,
                "profit_factor": r_perf.profit_factor,
                "net_pnl": round(r_perf.net_pnl, 2),
                "is_quota_fulfilled": r_perf.is_quota_fulfilled,
                "is_qualified": r_perf.is_statistically_profitable,
                "is_statistically_profitable": r_perf.is_statistically_profitable
            }
            for r_name, r_perf in self.regimes.items()
        }

        return {
            "is_live_ready": is_ready,
            "is_qualified_for_live_capital": is_ready,
            "total_trades": total_trades,
            "overall_win_rate_pct": overall_wr,
            "overall_profit_factor": overall_pf,
            "all_quotas_met": all_quotas_met,
            "unmet_reasons": unmet_reasons,
            "regimes_breakdown": breakdown,
            "regime_breakdown": breakdown
        }
