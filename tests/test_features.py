import unittest

from features.structural_features import (
    sentence_count,
    currency_reference_count,
)

from features.behavioral_features import (
    payment_request_flag,
    credential_request_flag,
    identity_document_flag,
    equipment_purchase_flag,
    money_transfer_flag,
    paid_training_flag,
    suspicious_application_link_flag,
    selection_bypass_flag,
    cheque_overpayment_flag,
)


class TestStructuralFeatures(unittest.TestCase):

    def test_sentence_count_ignores_url_periods(self):

        text = """
        Apply at https://jobs.example/apply
        Contact hr@example.com.
        """

        self.assertEqual(
            sentence_count(text),
            2,
        )

    def test_currency_reference(self):

        self.assertEqual(
            currency_reference_count(
                "Salary is PKR 90,000."
            ),
            1,
        )

        self.assertEqual(
            currency_reference_count(
                "Salary will be discussed later."
            ),
            0,
        )


class TestBehavioralFeatures(unittest.TestCase):

    def test_applicant_payment_request(self):

        text = (
            "You must pay the visa processing "
            "fee before the interview."
        )

        self.assertEqual(
            payment_request_flag(text),
            1,
        )

    def test_negated_payment_request(self):

        text = (
            "You should not send money "
            "to an agent."
        )

        self.assertEqual(
            payment_request_flag(text),
            0,
        )

    def test_employer_funded_cost(self):

        text = (
            "The employer pays approved "
            "visa costs directly."
        )

        self.assertEqual(
            payment_request_flag(text),
            0,
        )

    def test_credential_request(self):

        text = (
            "Send your password and OTP "
            "to verify the account."
        )

        self.assertEqual(
            credential_request_flag(text),
            1,
        )

    def test_identity_document_signal(self):

        text = (
            "Bring your CNIC on your "
            "first working day."
        )

        self.assertEqual(
            identity_document_flag(text),
            1,
        )

    def test_equipment_purchase(self):

        text = (
            "Purchase a laptop from our "
            "approved vendor."
        )

        self.assertEqual(
            equipment_purchase_flag(text),
            1,
        )

    def test_money_transfer(self):

        text = (
            "Receive client payments in your "
            "personal bank account and forward "
            "the funds."
        )

        self.assertEqual(
            money_transfer_flag(text),
            1,
        )

    def test_paid_training(self):

        text = (
            "Only certificates bought from our "
            "training partner are accepted; "
            "enrol for EUR 60."
        )

        self.assertEqual(
            paid_training_flag(text),
            1,
        )

    def test_legitimate_training_context(self):

        text = (
            "This is a twelve-week contract with "
            "hourly pay and weekly quality review. "
            "Training uses non-sensitive sample records."
        )

        self.assertEqual(
            paid_training_flag(text),
            0,
        )

    def test_sensitive_application_link(self):

        text = (
            "Apply at https://verify.example/profile "
            "and enter your card details and CVV."
        )

        self.assertEqual(
            suspicious_application_link_flag(
                text
            ),
            1,
        )

    def test_selection_bypass(self):

        text = (
            "No interview or further "
            "application stage is required."
        )

        self.assertEqual(
            selection_bypass_flag(text),
            1,
        )

    def test_cheque_overpayment(self):

        text = (
            "We will issue a cheque for more "
            "than your expenses. Deposit it "
            "and return the unused balance."
        )

        self.assertEqual(
            cheque_overpayment_flag(text),
            1,
        )


if __name__ == "__main__":
    unittest.main()