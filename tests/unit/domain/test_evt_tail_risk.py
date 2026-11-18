import numpy as np
import pytest

from src.domain.services.evt_tail_risk import EVTTailRiskEngine, EVTTailRiskResult


class TestEVTTailRiskEngine:
    def test_fit_and_evaluate_standard_series(self):
        engine = EVTTailRiskEngine(target_es_pct=3.5, shape_veto_threshold=0.42)

        # Generate 100 historical returns with moderate negative skew
        np.random.seed(42)
        normal_rets = np.random.normal(0.001, 0.02, 90)
        fat_tail_drops = np.array([-0.05, -0.07, -0.06, -0.09, -0.12, -0.08, -0.04, -0.05, -0.06, -0.10])
        all_rets = np.concatenate([normal_rets, fat_tail_drops])

        res = engine.fit_and_evaluate(all_rets.tolist())

        assert isinstance(res, EVTTailRiskResult)
        assert res.var_99_evt > res.var_95_evt
        assert res.es_99_evt >= res.var_99_evt
        assert res.excess_count > 0
        assert res.is_fat_tailed is True
        assert res.position_scale_factor > 0.0

    def test_tail_singularity_veto_on_extreme_outliers(self):
        engine = EVTTailRiskEngine(target_es_pct=3.5, shape_veto_threshold=0.42)

        # Extreme catastrophic crash series with huge power-law tail
        normal_rets = np.random.normal(0.001, 0.01, 70)
        huge_crashes = np.array([-0.20, -0.35, -0.45, -0.60, -0.50, -0.70, -0.30, -0.40])
        crashes_rets = np.concatenate([normal_rets, huge_crashes])

        res = engine.fit_and_evaluate(crashes_rets.tolist())

        # When shape parameter exceeds 0.42, singularity veto must trigger
        if res.shape_xi >= 0.42:
            assert res.is_tail_singularity_veto is True
            assert res.position_scale_factor == 0.0
            assert "EVT VETO" in res.veto_reason
        else:
            # Haircut must be significant
            assert res.position_scale_factor < 1.0
