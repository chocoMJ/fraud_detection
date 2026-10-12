"""Terminal histories from previous calendar days and actually revealed labels."""
import pandas as pd

WINDOWS = (1, 7, 30)
KINDS = ("OBSERVED_TX", "LABELED_TX", "KNOWN_FRAUD_TX", "KNOWN_FRAUD_RATE")
HISTORY_FEATURES = [f"TERMINAL_{kind}_{w}DAY" for w in WINDOWS for kind in KINDS]


def model_features(features):
    return list(features) + (HISTORY_FEATURES if "TERMINAL_ID" in features else [])


class TerminalHistory:
    def __init__(self):
        self.days = []

    def transform(self, number, frame):
        # Exclude today entirely; preserve these feature values in training history.
        self.days = [(n, stats) for n, stats in self.days if number - 30 <= n < number]
        output = frame.copy()
        for window in WINDOWS:
            parts = [stats for n, stats in self.days if number - window <= n < number]
            totals = (pd.concat(parts).groupby(level=0).sum() if parts else
                      pd.DataFrame(columns=KINDS[:3]))
            for kind in KINDS[:3]:
                output[f"TERMINAL_{kind}_{window}DAY"] = (
                    output.TERMINAL_ID.map(totals[kind]).fillna(0).astype(float))
            denominator = output[f"TERMINAL_LABELED_TX_{window}DAY"]
            output[f"TERMINAL_KNOWN_FRAUD_RATE_{window}DAY"] = (
                output[f"TERMINAL_KNOWN_FRAUD_TX_{window}DAY"] /
                denominator.where(denominator > 0)).fillna(0)
        return output

    def observe(self, number, observed, revealed):
        # All transactions contribute counts; only investigated transactions supply labels.
        stats = observed.groupby("TERMINAL_ID").size().rename("OBSERVED_TX").to_frame()
        known = revealed.groupby("TERMINAL_ID").y.agg(LABELED_TX="size", KNOWN_FRAUD_TX="sum")
        self.days.append((number, stats.join(known).fillna(0)))
