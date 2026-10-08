"""Scenario boundary, temporal features and reproducible loader integration."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from demo_py.data import load_handbook_days
from demo_py.Generate_Scenario.generate_scenario1 import apply_scenario1, transform_transactions, generate_dataset


class GeneratorTests(unittest.TestCase):
    def test_threshold_and_delayed_features(self):
        original = pd.DataFrame({
            "TRANSACTION_ID": [0, 1, 2, 3],
            "TX_DATETIME": pd.to_datetime(["2018-04-01 12:00", "2018-04-02 12:00",
                                           "2018-04-08 12:00", "2018-04-09 12:00"]),
            "CUSTOMER_ID": [1] * 4, "TERMINAL_ID": [1] * 4,
            "TX_AMOUNT": [220., 220.01, 10., 300.],
            "TX_TIME_SECONDS": [43200, 129600, 648000, 734400],
            "TX_TIME_DAYS": [0, 1, 7, 8],
        })
        labeled = apply_scenario1(original)
        self.assertEqual(labeled.TX_FRAUD.tolist(), [0, 1, 0, 1])
        pd.testing.assert_series_equal(labeled.TX_AMOUNT, original.TX_AMOUNT)
        self.assertNotIn("TX_FRAUD", original)
        transformed = transform_transactions(labeled)
        self.assertEqual(len(transformed.columns), 23)
        self.assertEqual(transformed.CUSTOMER_ID_NB_TX_1DAY_WINDOW.tolist(), [1.] * 4)
        self.assertEqual(transformed.TERMINAL_ID_NB_TX_1DAY_WINDOW.tolist(), [0., 0., 1., 1.])
        self.assertEqual(transformed.TERMINAL_ID_RISK_1DAY_WINDOW.tolist(), [0., 0., 0., 1.])
        self.assertAlmostEqual(transformed.CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW.iloc[2], 115.005)
        prefix = transform_transactions(labeled.iloc[:3])
        pd.testing.assert_frame_equal(transformed.iloc[:3], prefix)

    def test_reproducible_generation_and_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            settings = dict(customers=50, terminals=100, days=9, radius=100.)
            manifest = generate_dataset(a, **settings)
            generate_dataset(b, **settings)
            days, features = load_handbook_days(a)
            self.assertEqual(len(days), 9)
            self.assertEqual(len(features), 12)
            self.assertEqual(sum(len(d[2]) for d in days), manifest["transactions"])
            for path in sorted(a.glob("*.pkl")):
                left, right = pd.read_pickle(path), pd.read_pickle(b / path.name)
                pd.testing.assert_frame_equal(left, right)
                self.assertTrue(left.TX_FRAUD.eq(left.TX_AMOUNT.gt(220)).all())
                self.assertTrue(left.TX_FRAUD_SCENARIO.isin([0, 1]).all())
            with self.assertRaises(FileExistsError):
                generate_dataset(a, **settings)


if __name__ == "__main__":
    unittest.main()
