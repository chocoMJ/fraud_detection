"""Predict -> select -> reveal -> update, with an explicit calendar-day window."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .evaluation import evaluate
from .model import InitialEnsemble, fit_forest, positive_probability
from .strategies import select_cards
from .terminal_history import TerminalHistory, model_features


@dataclass(frozen=True)
class Config:
    warmup_days: int = 7
    window_days: int = 7
    budget: int = 100
    exploration: int = 5
    trees_per_day: int = 10
    forest_trees: int = 500
    seed: int = 42
    block_cards: bool = False

    def validate(self):
        for name in ("warmup_days", "window_days", "budget", "trees_per_day", "forest_trees"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.exploration <= self.budget or self.seed < 0:
            raise ValueError("Invalid exploration budget or seed")


def simulate(days, features, strategy, config):
    config.validate()
    if strategy not in {"N", "R", "U"}:
        raise ValueError("strategy must be N, R, or U")
    if len(days) <= config.warmup_days:
        raise ValueError("Need at least one evaluation day after warm-up")
    numbers = [n for n, _, _ in days]
    if numbers != sorted(set(numbers)):
        raise ValueError("Days must be unique and chronological")
    terminal_history = TerminalHistory() if "TERMINAL_ID" in features else None
    features = model_features(features)
    history = []
    for n, _, f in days[:config.warmup_days]:
        enriched = terminal_history.transform(n, f) if terminal_history else f.copy()
        history.append((n, enriched))
        if terminal_history:
            terminal_history.observe(n, f, f)
    rng = np.random.default_rng(config.seed)
    banned = set()
    records, selections = [], []
    model = None
    model_kind = None
    last_fit_days = []
    for index, (number, filename, full_frame) in enumerate(days[config.warmup_days:]):
        history = [(n, f) for n, f in history if number - config.window_days <= n < number]
        usable = [(n, f) for n, f in history if not f.empty]
        if usable:
            if model is None:
                model = InitialEnsemble(config.trees_per_day, config.seed).fit([f for _, f in usable], features)
                model_kind = "initial_ensemble"
            else:
                model = fit_forest(pd.concat([f for _, f in usable], ignore_index=True), features,
                                   config.forest_trees, config.seed + index)
                model_kind = "random_forest"
            last_fit_days = [n for n, _ in usable]
        elif model is None:
            raise ValueError("No labeled training rows in the initial window")
        frame = full_frame[~full_frame.CARD_ID.isin(banned)].copy().reset_index(drop=True)
        if terminal_history:
            frame = terminal_history.transform(number, frame)
        scores = (model.predict(frame) if model_kind == "initial_ensemble"
                  else positive_probability(model, frame[features]))
        selected = select_cards(frame.CARD_ID, scores, strategy, config.budget, config.exploration, rng)
        # This is the oracle boundary: only now reveal selected cards' labels.
        revealed = frame[frame.CARD_ID.isin(selected.CARD_ID)].copy()
        row = {"strategy": strategy, "seed": config.seed, "day": number, "file": filename,
               "model": model_kind, "model_reused": not bool(usable),
               "train_days": ",".join(map(str, last_fit_days)),
               "window_labeled_rows": sum(len(f) for _, f in usable),
               "excluded_transactions": len(full_frame) - len(frame),
               "labeled_transactions": len(revealed), **evaluate(frame, scores, selected, budget=config.budget)}
        records.append(row)
        card_truth = frame.groupby("CARD_ID").y.max()
        selected["is_fraud_card"] = selected.CARD_ID.map(card_truth).astype(int)
        selected.insert(0, "day", number)
        selected.insert(0, "seed", config.seed)
        selected.insert(0, "strategy", strategy)
        selections.append(selected)
        history.append((number, revealed))
        if terminal_history:
            terminal_history.observe(number, full_frame, revealed)
        if config.block_cards:
            banned.update(revealed.loc[revealed.y.eq(1), "CARD_ID"])
    return pd.DataFrame(records), pd.concat(selections, ignore_index=True)


def config_dict(config):
    return asdict(config)
