import unittest
from src.application.services.billing import BillingService
from src.domain.entities.tier import SubscriptionTier


class TestBillingService(unittest.TestCase):

    def setUp(self):
        # Pass empty API key to test local unit logic without hitting CryptoPay API
        self.billing = BillingService(crypto_pay_api_key="")
        self.user_id = "test_user_77"

    def test_pricing_catalog(self):
        self.assertEqual(self.billing.get_tier_price(SubscriptionTier.FREE), 0.00)
        self.assertEqual(self.billing.get_tier_price(SubscriptionTier.PRO), 29.00)
        self.assertEqual(self.billing.get_tier_price(SubscriptionTier.MAINVEST), 29.00)
        self.assertEqual(self.billing.get_tier_price(SubscriptionTier.ALL_ACCESS), 29.00)

    def test_create_and_fulfill_invoice(self):
        invoice = self.billing.create_invoice(self.user_id, SubscriptionTier.PRO, payment_method="CryptoPay_USDT")

        self.assertEqual(invoice.user_id, self.user_id)
        self.assertEqual(invoice.tier, SubscriptionTier.PRO)
        self.assertEqual(invoice.amount_usd, 29.00)
        self.assertEqual(invoice.status, "PENDING")

        fulfilled_tier = self.billing.fulfill_payment(invoice.invoice_id)
        self.assertEqual(fulfilled_tier, SubscriptionTier.PRO)
        self.assertEqual(invoice.status, "PAID")

    def test_create_bundle_invoice(self):
        invoice = self.billing.create_invoice(self.user_id, SubscriptionTier.ALL_ACCESS, payment_method="CryptoPay_USDT")
        self.assertEqual(invoice.tier, SubscriptionTier.ALL_ACCESS)
        self.assertEqual(invoice.amount_usd, 29.00)


if __name__ == "__main__":
    unittest.main()
