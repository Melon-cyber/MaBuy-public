import pytest
from src.domain.services.safe_leverage_wizard import SafeLeverageWizardService, SafeLeveragePlan


class TestSafeLeverageWizard:
    def test_generate_wizard_plan_ru_and_en(self):
        wizard = SafeLeverageWizardService()

        # Test RU calculation for $500 balance
        plan_ru = wizard.generate_wizard_plan(
            symbol="BTCUSDT",
            account_balance=500.0,
            direction="BULLISH",
            risk_level="balanced",
            model_probability=0.60,
            lang="ru"
        )

        assert isinstance(plan_ru, SafeLeveragePlan)
        assert plan_ru.account_balance == 500.0
        assert plan_ru.max_safe_leverage >= 1 and plan_ru.max_safe_leverage <= 20
        assert plan_ru.safety_buffer_ratio >= 1.5
        assert "Калькулятор безопасного входа MaBuy" in plan_ru.formatted_message
        assert "Запас до ликвидации" in plan_ru.formatted_message

        # Test EN calculation
        plan_en = wizard.generate_wizard_plan(
            symbol="ETHUSDT",
            account_balance=1000.0,
            direction="BEARISH",
            risk_level="conservative",
            model_probability=0.55,
            lang="en"
        )
        assert "MaBuy Safe-Leverage Position Wizard" in plan_en.formatted_message
        assert plan_en.account_balance == 1000.0
        assert plan_en.direction == "BEARISH"
