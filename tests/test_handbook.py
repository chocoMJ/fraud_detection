"""Verify the Handbook adapter without network access."""
from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from demo_py.data import load_days, load_handbook_days, handbook_source_directory
from demo_py.data import FEATURES, convert_day
from demo_py.Convert_CSV import convert_pickles


def source_frame(day="2018-04-01"):
    frame = pd.DataFrame({
        "TRANSACTION_ID": [2, 1], "CUSTOMER_ID": ["02", "A6"],
        "TX_DATETIME": [f"{day} 12:00:00", f"{day} 01:00:00"],
        "TX_FRAUD": [1, 0], "TX_FRAUD_SCENARIO": [3, 0],
        "TERMINAL_ID": [10, 20], "TX_TIME_DAYS": [0, 0],
        "TERMINAL_ID_RISK_1DAY_WINDOW": [1., 0.],
    })
    for feature in FEATURES:
        frame[feature] = [2., 1.]
    return frame


class HandbookTests(unittest.TestCase):
    def test_direct_pickle_matches_csv_and_preserves_calendar_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, csv = root / "source" / "revision", root / "daily"
            source.mkdir(parents=True)
            csv.mkdir()
            for number, day in [(1, "2018-04-01"), (3, "2018-04-03")]:
                frame = source_frame(day)
                frame["TRANSACTION_ID"] += number * 10
                frame.to_pickle(source / f"{day}.pkl")
                converted, _ = convert_day(frame, date.fromisoformat(day))
                converted.to_csv(csv / f"day{number:03d}.csv", index=False)
            self.assertEqual(handbook_source_directory(root / "source"), source)
            actual, features = load_handbook_days(source)
            expected, csv_features = load_days(csv)
            self.assertEqual(features, csv_features)
            self.assertEqual([d[0] for d in actual], [1, 3])
            for left, right in zip(actual, expected):
                pd.testing.assert_frame_equal(left[2], right[2])
            with self.assertRaises(ValueError):
                load_handbook_days(source, ["TX_FRAUD"])
            duplicate = source_frame("2018-04-04")
            duplicate["TRANSACTION_ID"] += 10
            duplicate.to_pickle(source / "2018-04-04.pkl")
            with self.assertRaisesRegex(ValueError, "Duplicate transaction"):
                load_handbook_days(source)

    def test_mapping_order_and_feature_whitelist(self):
        result, mapping = convert_day(source_frame(), date(2018, 4, 1))
        self.assertEqual(result.columns.tolist(), ["CARD_ID", *FEATURES, "y"])
        self.assertEqual(result.CARD_ID.tolist(), ["A6", "02"])
        self.assertEqual(result.y.tolist(), [0, 1])
        self.assertEqual(mapping.TRANSACTION_ID.tolist(), [1, 2])
        self.assertEqual(len(FEATURES), 12)
        self.assertFalse(any("RISK" in feature or "FRAUD" in feature for feature in FEATURES))

    def test_hidden_labels_do_not_change_features(self):
        frame = source_frame()
        original, _ = convert_day(frame, date(2018, 4, 1))
        frame["TX_FRAUD"] = 1 - frame.TX_FRAUD
        frame["TX_FRAUD_SCENARIO"] = 99
        frame["TERMINAL_ID_RISK_1DAY_WINDOW"] = .5
        changed, _ = convert_day(frame, date(2018, 4, 1))
        pd.testing.assert_frame_equal(original.drop(columns="y"), changed.drop(columns="y"))

    def test_reject_invalid_source(self):
        for fault in ["day", "label", "missing", "infinite", "duplicate"]:
            frame = source_frame()
            if fault == "day":
                frame.loc[0, "TX_DATETIME"] = "2018-04-02"
            elif fault == "label":
                frame.loc[0, "TX_FRAUD"] = 3
            elif fault == "missing":
                frame = frame.drop(columns=FEATURES[0])
            elif fault == "infinite":
                frame.loc[0, FEATURES[0]] = np.inf
            else:
                frame["TRANSACTION_ID"] = 1
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                convert_day(frame, date(2018, 4, 1))

    def test_inspection_csv_preserves_all_columns_and_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "2018-04-01.pkl"
            original = source_frame()
            original.to_pickle(source)
            files = convert_pickles(source, root / "csv")
            restored = pd.read_csv(files[0], dtype={"CUSTOMER_ID": "str"})
            pd.testing.assert_frame_equal(original, restored, check_dtype=False)
            self.assertIn("TX_FRAUD_SCENARIO", restored)
            self.assertIn("TERMINAL_ID_RISK_1DAY_WINDOW", restored)
            with self.assertRaises(FileExistsError):
                convert_pickles(source, root / "csv")


if __name__ == "__main__":
    unittest.main()
