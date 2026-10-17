"""Terminal histories from previous calendar days and actually revealed labels."""
import pandas as pd

WINDOWS = (1, 7, 30)
KINDS = ("OBSERVED_TX", "LABELED_TX", "KNOWN_FRAUD_TX", "KNOWN_FRAUD_RATE")
HISTORY_FEATURES = [f"TERMINAL_{kind}_{w}DAY" for w in WINDOWS for kind in KINDS]


def model_features(features, include_terminal_history=True):
    """Return numerical X columns; TERMINAL_ID itself is never a model feature."""
    if "TERMINAL_ID" in features:
        raise ValueError("TERMINAL_ID is a grouping key, not a model feature")
    return list(features) + (HISTORY_FEATURES if include_terminal_history else [])


class TerminalHistory:
    def __init__(self):
        self.observed_days = []
        self.label_days = []

    def transform(self, number, frame):
        # Exclude today entirely; preserve these feature values in training history.
        self.observed_days = [(n, stats) for n, stats in self.observed_days
                              if number - 30 <= n < number]
        self.label_days = [(n, stats) for n, stats in self.label_days
                           if number - 30 < n <= number]
        output = frame.copy()
        for window in WINDOWS:
            observed = [stats for n, stats in self.observed_days
                        if number - window <= n < number]
            labels = [stats for n, stats in self.label_days
                      if number - window < n <= number]
            observed_totals = (pd.concat(observed).groupby(level=0).sum() if observed else
                               pd.DataFrame(columns=["OBSERVED_TX"]))
            label_totals = (pd.concat(labels).groupby(level=0).sum() if labels else
                            pd.DataFrame(columns=["LABELED_TX", "KNOWN_FRAUD_TX"]))
            output[f"TERMINAL_OBSERVED_TX_{window}DAY"] = (
                output.TERMINAL_ID.map(observed_totals["OBSERVED_TX"]).fillna(0).astype(float))
            for kind in ("LABELED_TX", "KNOWN_FRAUD_TX"):
                output[f"TERMINAL_{kind}_{window}DAY"] = (
                    output.TERMINAL_ID.map(label_totals[kind]).fillna(0).astype(float))
            denominator = output[f"TERMINAL_LABELED_TX_{window}DAY"]
            output[f"TERMINAL_KNOWN_FRAUD_RATE_{window}DAY"] = (
                output[f"TERMINAL_KNOWN_FRAUD_TX_{window}DAY"] /
                denominator.where(denominator > 0)).fillna(0)
        return output

    def observe(self, number, observed):
        """Record transaction counts on their occurrence day."""
        stats = observed.groupby("TERMINAL_ID").size().rename("OBSERVED_TX").to_frame()
        self.observed_days.append((number, stats))

    def reveal(self, number, revealed):
        """Record label counts on the day those labels become available."""
        if revealed.empty:
            return
        known = revealed.groupby("TERMINAL_ID").y.agg(
            LABELED_TX="size", KNOWN_FRAUD_TX="sum")
        self.label_days.append((number, known))
