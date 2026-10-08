"""Behavioral checks for chronology, query budgets, and evaluation units."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from demo_py.data import load_days
from demo_py.evaluation import evaluate
from demo_py.model import fit_forest, positive_probability
from demo_py.simulation import Config, simulate
from demo_py.strategies import select_cards


def fixture_days(count=12):
    days = []
    for day in range(1, count + 1):
        frame = pd.DataFrame({"CARD_ID": [f"{i:02d}" for i in range(12)],
                              "x1": np.arange(12) + day / 10,
                              "x2": np.arange(12) % 3,
                              "y": [0] * 8 + [1] * 4})
        days.append((day, f"day{day:02d}.csv", frame))
    return days


class BaselineTests(unittest.TestCase):
    def test_uncertainty_alignment_and_no_duplicate_cards(self):
        ids = ["A", "B", "C", "D", "B"]
        scores = [.99, .49, .1, .8, .51]
        selected = select_cards(ids, scores, "U", 3, 1, np.random.default_rng(1))
        self.assertEqual(selected.CARD_ID.tolist(), ["A", "D", "B"])
        self.assertEqual(selected.reason.tolist(), ["risk", "risk", "uncertainty"])

    def test_all_strategies_respect_budget_and_small_pool(self):
        for strategy in ["N", "R", "U"]:
            for budget in [2, 100]:
                selected = select_cards(["A", "A", "B", "C"], [.9, .8, .5, .1],
                                        strategy, budget, 1, np.random.default_rng(42))
                self.assertEqual(len(selected), min(budget, 3))
                self.assertTrue(selected.CARD_ID.is_unique)

    def test_card_truth_uses_all_transactions(self):
        frame = pd.DataFrame({"CARD_ID": ["A", "A", "B"], "y": [0, 1, 0]})
        selected = select_cards(frame.CARD_ID, [.99, .2, .1], "N", 1, 0, np.random.default_rng(1))
        metrics = evaluate(frame, [.99, .2, .1], selected)
        self.assertEqual(metrics["card_precision"], 1)
        self.assertEqual(metrics["found_fraud_cards"], 1)

    def test_today_and_future_labels_cannot_affect_today_selection(self):
        days = fixture_days()
        config = Config(warmup_days=3, window_days=3, budget=4, exploration=1,
                        trees_per_day=2, forest_trees=5)
        for strategy in ["N", "R", "U"]:
            original_metrics, original_selected = simulate(days, ["x1", "x2"], strategy, config)
            changed = [(n, name, f.copy()) for n, name, f in days]
            for n, _, frame in changed:
                if n >= 6:
                    frame["y"] = 1 - frame.y
            _, changed_selected = simulate(changed, ["x1", "x2"], strategy, config)
            cols = ["day", "CARD_ID", "score", "reason"]
            pd.testing.assert_frame_equal(original_selected.loc[original_selected.day <= 6, cols],
                                          changed_selected.loc[changed_selected.day <= 6, cols])
            for _, row in original_metrics.iterrows():
                train_days = list(map(int, row.train_days.split(",")))
                self.assertTrue(all(row.day - 3 <= d < row.day for d in train_days))
            again, selections_again = simulate(days, ["x1", "x2"], strategy, config)
            pd.testing.assert_frame_equal(original_metrics, again)
            pd.testing.assert_frame_equal(original_selected, selections_again)

    def test_only_selected_labels_enter_next_training_window(self):
        config = Config(warmup_days=3, window_days=3, budget=4, exploration=1,
                        trees_per_day=1, forest_trees=3)
        daily, _ = simulate(fixture_days(), ["x1", "x2"], "U", config)
        self.assertEqual(daily.iloc[0].window_labeled_rows, 36)
        self.assertEqual(daily.iloc[1].window_labeled_rows, 28)
        self.assertEqual(daily.iloc[3].window_labeled_rows, 12)

    def test_single_class_and_empty_prediction(self):
        for label in [0, 1]:
            frame = pd.DataFrame({"x1": [1., 2.], "y": [label, label]})
            model = fit_forest(frame, ["x1"], trees=2)
            np.testing.assert_array_equal(positive_probability(model, frame[["x1"]]), [label, label])
            self.assertEqual(len(positive_probability(model, frame.iloc[:0][["x1"]])), 0)

    def test_blocking_removes_confirmed_cards_on_next_day(self):
        days = fixture_days(5)
        for _, _, frame in days:
            frame["y"] = 1
        config = Config(warmup_days=2, budget=100, exploration=5, trees_per_day=1,
                        forest_trees=2, block_cards=True)
        daily, selected = simulate(days, ["x1", "x2"], "U", config)
        self.assertEqual(daily.selected_cards.tolist(), [12, 0, 0])
        self.assertEqual(len(selected), 12)

    def test_day_order_and_string_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            for number in [10, 2, 1]:
                Path(directory, f"day{number}.csv").write_text("CARD_ID,x1,y\n02,1,0\nA6,2,1\n")
            days, features = load_days(Path(directory))
            self.assertEqual([n for n, _, _ in days], [1, 2, 10])
            self.assertEqual(days[0][2].CARD_ID.iloc[0], "02")
            self.assertEqual(features, ["x1"])
            with self.assertRaises(ValueError):
                load_days(Path(directory), ["x1", "y"])


if __name__ == "__main__":
    unittest.main()
