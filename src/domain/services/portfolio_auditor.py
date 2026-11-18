import math
from typing import Dict, List, Any, Tuple


class PortfolioRiskAuditorService:
    """
    Analyzes user cryptocurrency portfolios (e.g. BTC 50%, ETH 30%, SOL 20%)
    and calculates correlation-aware portfolio volatility, diversification metrics,
    Parametric 95% Value at Risk (VaR), CVaR (Expected Shortfall), and actionable AI rebalancing advice.
    """

    ASSET_VOLATILITIES = {
        # Cryptocurrencies (Daily volatility)
        "BTC": 0.022,
        "ETH": 0.031,
        "BNB": 0.025,
        "SOL": 0.045,
        "XRP": 0.038,
        "TON": 0.042,
        "NEAR": 0.048,
        "DOGE": 0.055,
        "ADA": 0.042,
        "AVAX": 0.049,
        "USDT": 0.001,
        "USDC": 0.001,
        # US Equities & ETFs (Daily volatility calibrated to 252 trading days)
        "NVDA": 0.028,
        "AAPL": 0.014,
        "TSLA": 0.031,
        "MSFT": 0.014,
        "GOOGL": 0.016,
        "SPY": 0.010,
        "QQQ": 0.013,
        "GLD": 0.009,
    }

    PAIR_CORRELATIONS = {
        # Crypto-Crypto
        ("BTC", "ETH"): 0.82,
        ("BTC", "SOL"): 0.76,
        ("BTC", "BNB"): 0.74,
        ("BTC", "XRP"): 0.62,
        ("BTC", "TON"): 0.55,
        ("BTC", "NEAR"): 0.72,
        ("BTC", "DOGE"): 0.65,
        ("ETH", "SOL"): 0.79,
        ("ETH", "BNB"): 0.76,
        ("ETH", "NEAR"): 0.78,
        ("ETH", "TON"): 0.58,
        ("SOL", "NEAR"): 0.81,
        ("SOL", "TON"): 0.64,
        # Equities & Macro ETFs
        ("SPY", "QQQ"): 0.92,
        ("AAPL", "MSFT"): 0.76,
        ("AAPL", "QQQ"): 0.88,
        ("AAPL", "SPY"): 0.84,
        ("MSFT", "QQQ"): 0.89,
        ("MSFT", "SPY"): 0.86,
        ("NVDA", "QQQ"): 0.86,
        ("NVDA", "SPY"): 0.78,
        ("TSLA", "QQQ"): 0.72,
        ("TSLA", "SPY"): 0.65,
        ("GOOGL", "QQQ"): 0.84,
        ("GOOGL", "SPY"): 0.82,
        ("GLD", "SPY"): 0.05,
        ("GLD", "QQQ"): 0.02,
        # Cross-Market
        ("BTC", "SPY"): 0.38,
        ("BTC", "QQQ"): 0.45,
        ("BTC", "GLD"): 0.18,
        ("BTC", "NVDA"): 0.48,
        ("BTC", "TSLA"): 0.46,
        ("ETH", "SPY"): 0.40,
        ("ETH", "QQQ"): 0.46,
        ("SOL", "QQQ"): 0.48,
        ("SOL", "NVDA"): 0.50,
    }

    PRESETS = {
        "classic": {"BTC": 60.0, "ETH": 30.0, "SOL": 10.0},
        "degen": {"SOL": 40.0, "NEAR": 30.0, "TON": 30.0},
        "balanced": {"BTC": 40.0, "ETH": 30.0, "SOL": 20.0, "USDT": 10.0},
        "all_weather_2": {"SPY": 30.0, "GLD": 20.0, "QQQ": 20.0, "BTC": 15.0, "USDT": 15.0},
        "tech_crypto_alpha": {"NVDA": 25.0, "MSFT": 20.0, "BTC": 25.0, "SOL": 15.0, "ETH": 15.0},
        "buffett_hedge": {"AAPL": 35.0, "GOOGL": 25.0, "GLD": 20.0, "SPY": 15.0, "BTC": 5.0},
    }

    @classmethod
    def get_correlation(cls, a1: str, a2: str) -> float:
        """Returns pairwise return correlation with fallback for unknown altcoins."""
        s1, s2 = a1.upper(), a2.upper()
        if s1 == s2:
            return 1.0
        if s1 in ("USDT", "USDC") or s2 in ("USDT", "USDC"):
            return 0.02
        if (s1, s2) in cls.PAIR_CORRELATIONS:
            return cls.PAIR_CORRELATIONS[(s1, s2)]
        if (s2, s1) in cls.PAIR_CORRELATIONS:
            return cls.PAIR_CORRELATIONS[(s2, s1)]
        # Default correlation among crypto assets
        return 0.70

    @classmethod
    def get_volatility(cls, asset: str) -> float:
        """Returns daily volatility estimate for asset."""
        return cls.ASSET_VOLATILITIES.get(asset.upper(), 0.045)

    @classmethod
    def audit_portfolio(cls, portfolio_allocations: Dict[str, float], lang: str = "ru") -> Dict[str, Any]:
        """
        Calculates correlation-aware portfolio risk profile.
        portfolio_allocations: e.g. {"BTC": 50, "ETH": 30, "SOL": 20}
        """
        if not portfolio_allocations:
            portfolio_allocations = cls.PRESETS["balanced"]

        # 1. Normalize weights to sum strictly to 1.0
        raw_sum = sum(abs(v) for v in portfolio_allocations.values())
        if raw_sum <= 0:
            raw_sum = 1.0

        weights = {k.upper(): abs(v) / raw_sum for k, v in portfolio_allocations.items() if abs(v) > 0}
        if not weights:
            weights = {"BTC": 1.0}

        asset_list = list(weights.keys())
        n = len(asset_list)

        # 2. Correlation-Aware Covariance Portfolio Variance: sigma_p^2 = sum_i sum_j w_i w_j sigma_i sigma_j rho_ij
        variance_p = 0.0
        variance_stress = 0.0
        weighted_vol_sum = sum(weights[a] * cls.get_volatility(a) for a in asset_list)

        for i in range(n):
            a_i = asset_list[i]
            w_i = weights[a_i]
            vol_i = cls.get_volatility(a_i)
            for j in range(n):
                a_j = asset_list[j]
                w_j = weights[a_j]
                vol_j = cls.get_volatility(a_j)
                rho_ij = cls.get_correlation(a_i, a_j)
                variance_p += w_i * w_j * vol_i * vol_j * rho_ij
                
                # Stress scenario: Correlations spike to 0.95 during liquidity crises
                rho_stress = 1.0 if i == j else (0.05 if (a_i in ("USDT", "USDC") or a_j in ("USDT", "USDC")) else 0.95)
                variance_stress += w_i * w_j * vol_i * vol_j * rho_stress

        # Conservative fallback: If matrix calculation encounters non-PSD numerical artifacts,
        # fallback to weighted sum of volatilities (upper bound rho=1.0) rather than zero.
        if variance_p <= 0:
            portfolio_vol = weighted_vol_sum
        else:
            portfolio_vol = math.sqrt(variance_p)

        stress_vol = math.sqrt(max(variance_stress, portfolio_vol ** 2))
        annual_vol = portfolio_vol * math.sqrt(365) * 100.0

        # 3. Diversification Score based on HHI and Effective N
        hhi = sum(w ** 2 for w in weights.values())
        effective_n = 1.0 / (hhi + 1e-8)
        raw_div_score = (1.0 - hhi) * 100.0 + min(20.0, effective_n * 5.0)
        diversification_score = min(100, max(15, int(raw_div_score)))

        # 4. Parametric Value at Risk (VaR 95%) & Stress-Scenario VaR
        daily_var_95 = portfolio_vol * 1.645 * 100.0
        daily_var_usd_100 = round(daily_var_95, 2)
        stress_var_usd_100 = round(stress_vol * 1.645 * 100.0, 2)
        cvar_95 = portfolio_vol * 2.06 * 100.0

        # 5. Risk Level Classification
        if portfolio_vol < 0.024:
            risk_level = "🛡 КОНСЕРВАТИВНЫЙ (Низкий)" if lang == "ru" else "CONSERVATIVE (Low Risk)"
        elif portfolio_vol < 0.038:
            risk_level = "⚖️ СБАЛАНСИРОВАННЫЙ (Умеренный)" if lang == "ru" else "BALANCED (Moderate Risk)"
        else:
            risk_level = "🔥 АГРЕССИВНЫЙ (Высокий)" if lang == "ru" else "AGGRESSIVE (High Risk)"

        # 6. Actionable AI Rebalancing Advice
        max_asset = max(weights, key=weights.get)
        max_pct = weights[max_asset] * 100.0

        if lang == "ru":
            if max_pct > 65:
                advice = (
                    f"⚠️ <b>Критическая концентрация в #{max_asset} ({max_pct:.1f}%).</b>\n"
                    f"При падении #{max_asset} портфель потеряет до <b>-${daily_var_usd_100:.2f}</b> на каждые $100. "
                    f"Рекомендуем снизить долю #{max_asset} на 20% в пользу BTC или USDT для снижения риска на 35%."
                )
            elif diversification_score < 50:
                advice = (
                    f"⚡ <b>Высокая взаимосвязь активов.</b> Ваши альткоины падают синхронно. "
                    f"Добавьте 15-20% USDT или BTC для создания защитной подушки."
                )
            else:
                advice = (
                    f"✅ <b>Отличная структура портфеля!</b> Базовый стержень: #{max_asset} ({max_pct:.1f}%). "
                    f"Диверсификация защищает от резких ликвидаций на споте."
                )
        else:
            if max_pct > 65:
                advice = (
                    f"⚠️ High concentration in #{max_asset} ({max_pct:.1f}%). "
                    f"Rebalance 20% into BTC/USDT to reduce downside drawdown by ~35%."
                )
            elif diversification_score < 50:
                advice = "High asset correlation. Add 15-20% USDT or BTC for volatility protection."
            else:
                advice = f"Well-structured portfolio. Core holding: #{max_asset} ({max_pct:.1f}%)."

        return {
            "weights": weights,
            "portfolio_volatility": portfolio_vol,
            "portfolio_volatility_pct": round(portfolio_vol * 100.0, 2),
            "annual_volatility_pct": round(annual_vol, 1),
            "diversification_score": diversification_score,
            "risk_level": risk_level,
            "daily_var_95_pct": round(daily_var_95, 2),
            "var_95_usd_per_100": daily_var_usd_100,
            "stress_var_usd_per_100": stress_var_usd_100,
            "stress_volatility_pct": round(stress_vol * 100.0, 2),
            "cvar_95_pct": round(cvar_95, 2),
            "advice": advice,
            "rebalancing_advice": advice,
        }

