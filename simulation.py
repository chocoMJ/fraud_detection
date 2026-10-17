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
    window_days: int = 30
    budget: int = 100
    exploration: int = 5
    trees_per_day: int = 10
    forest_trees: int = 500
    seed: int = 42
    block_cards: bool = False
    feedback_delay_days: int = 1
    delayed_label_days: int = 14
    n_jobs: int = 1
    learning_mode: str = "combined"
    component_trees: int = 100
    feedback_window_days: int = 30
    delayed_window_days: int = 16
    feedback_weight: float = 0.5
    use_terminal_history: bool = True

    def validate(self):
        for name in ("warmup_days", "window_days", "budget", "trees_per_day", "forest_trees",
                     "feedback_delay_days", "delayed_label_days"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.exploration <= self.budget or self.seed < 0:
            raise ValueError("Invalid exploration budget or seed")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.feedback_delay_days >= self.delayed_label_days:
            raise ValueError("feedback_delay_days must be shorter than delayed_label_days")
        if self.learning_mode not in {"combined", "separated-sliding", "separated-ensemble"}:
            raise ValueError("Invalid learning_mode")
        if min(self.component_trees, self.feedback_window_days, self.delayed_window_days) < 1:
            raise ValueError("Separated-model tree and window settings must be positive")
        if not 0 <= self.feedback_weight <= 1:
            raise ValueError("feedback_weight must be between 0 and 1")


def _forest(frames, features, trees, seed, n_jobs):
    """Train one RF from nonempty labeled batches, or return None."""
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return None
    return fit_forest(pd.concat(frames, ignore_index=True), features, trees, seed, n_jobs)


def _forest_scores(model, frame, features):
    return None if model is None else positive_probability(model, frame[features])


def _combine_scores(feedback_scores, delayed_scores, feedback_weight):
    if feedback_scores is None:
        return delayed_scores
    if delayed_scores is None:
        return feedback_scores
    return feedback_weight * feedback_scores + (1 - feedback_weight) * delayed_scores


def _combined_entries_without_duplicate_feedback(history, number, window_days):
    """Pool labels while replacing feedback with the full delayed batch once it arrives."""
    entries = [item for item in history
               if number - window_days < item[0] <= number and not item[3].empty]
    delayed_source_days = {source_day for _, source_day, source, _ in entries
                           if source == "delayed"}
    return [item for item in entries
            if item[2] != "feedback" or item[1] not in delayed_source_days]


def simulate(days, features, strategy, config):
    config.validate()
    if strategy not in {"N", "R", "U"}:
        raise ValueError("strategy must be N, R, or U")
    if len(days) <= config.warmup_days:
        raise ValueError("Need at least one evaluation day after warm-up")
    numbers = [n for n, _, _ in days]
    if numbers != sorted(set(numbers)):
        raise ValueError("Days must be unique and chronological")
    terminal_flags = ["TERMINAL_ID" in frame.columns for _, _, frame in days]
    if any(terminal_flags) and not all(terminal_flags):
        raise ValueError("TERMINAL_ID must be present on every day or none")
    terminal_history = TerminalHistory() if config.use_terminal_history and all(terminal_flags) else None
    features = model_features(features, include_terminal_history=terminal_history is not None)
    history = []
    for n, _, f in days[:config.warmup_days]:
        enriched = terminal_history.transform(n, f) if terminal_history else f.copy()
        history.append((n, n, "warmup", enriched))
        if terminal_history:
            terminal_history.observe(n, f)
            terminal_history.reveal(n, f)
    pending = []
    # First-time label revelations are tracked separately from model batches.
    # A delayed model batch contains the full source day, including rows that
    # were already revealed through feedback.
    pending_label_reveals = []
    rng = np.random.default_rng(config.seed)
    banned = set()
    pending_bans = []
    records, selections = [], []
    model = None
    model_kind = None
    last_fit_days = []
    feedback_model = None
    delayed_model = None
    delayed_daily_models = {}
    warmup_model = None
    for index, (number, filename, full_frame) in enumerate(days[config.warmup_days:]):
        arrived = [item for item in pending if item[0] <= number]
        pending = [item for item in pending if item[0] > number]
        newly_revealed = [item for item in pending_label_reveals if item[0] <= number]
        pending_label_reveals = [item for item in pending_label_reveals if item[0] > number]
        banned.update(card for arrival_day, cards in pending_bans
                      if arrival_day <= number for card in cards)
        pending_bans = [item for item in pending_bans if item[0] > number]
        history.extend(arrived)
        if terminal_history:
            for arrival_day, _, labels in newly_revealed:
                terminal_history.reveal(arrival_day, labels)
        retention = max(config.window_days, config.feedback_window_days,
                        config.delayed_window_days)
        history = [item for item in history if number - retention < item[0] <= number]
        combined_entries = _combined_entries_without_duplicate_feedback(
            history, number, config.window_days)
        feedback_entries = [item for item in history
                            if item[2] == "feedback"
                            and number - config.feedback_window_days < item[0] <= number
                            and not item[3].empty]
        delayed_entries = [item for item in history
                           if item[2] == "delayed"
                           and number - config.delayed_window_days < item[0] <= number
                           and not item[3].empty]

        if config.learning_mode == "combined":
            if combined_entries:
                if model is None:
                    model = InitialEnsemble(config.trees_per_day, config.seed).fit(
                        [f for _, _, _, f in combined_entries], features)
                    model_kind = "initial_ensemble"
                else:
                    model = _forest([f for _, _, _, f in combined_entries], features,
                                    config.forest_trees, config.seed + index, config.n_jobs)
                    model_kind = "random_forest"
                last_fit_days = sorted({day for day, _, _, _ in combined_entries})
            elif model is None:
                raise ValueError("No labeled training rows in the initial window")
        else:
            if warmup_model is None:
                warmup_model = _forest([f for _, _, source, f in history if source == "warmup"],
                                       features, config.component_trees, config.seed, config.n_jobs)
            feedback_model = _forest([f for _, _, _, f in feedback_entries], features,
                                     config.component_trees, config.seed + 10000 + index,
                                     config.n_jobs)
            if config.learning_mode == "separated-sliding":
                trained = _forest([f for _, _, _, f in delayed_entries], features,
                                  config.component_trees, config.seed + 20000 + index,
                                  config.n_jobs)
                delayed_model = trained if trained is not None else warmup_model
                model_kind = "separated_sliding"
            else:
                for arrival_day, source_day, source, labels in arrived:
                    if source == "delayed" and not labels.empty and source_day not in delayed_daily_models:
                        delayed_daily_models[source_day] = _forest(
                            [labels], features, config.component_trees,
                            config.seed + 30000 + source_day, config.n_jobs)
                active_days = {source_day for _, source_day, _, _ in delayed_entries}
                delayed_daily_models = {day: fitted for day, fitted in delayed_daily_models.items()
                                        if day in active_days}
                model_kind = "separated_ensemble"
            last_fit_days = sorted({day for day, _, _, _ in feedback_entries + delayed_entries})
        warmup_entries = [item for item in history if item[2] == "warmup" and not item[3].empty]
        training_entries = (combined_entries if config.learning_mode == "combined" else
                            feedback_entries + delayed_entries +
                            (warmup_entries if not delayed_entries else []))
        frame = full_frame[~full_frame.CARD_ID.isin(banned)].copy().reset_index(drop=True)
        if terminal_history:
            frame = terminal_history.transform(number, frame)
        feedback_scores = None
        delayed_scores = None
        if config.learning_mode == "combined":
            scores = (model.predict(frame) if model_kind == "initial_ensemble"
                      else positive_probability(model, frame[features]))
        else:
            feedback_scores = _forest_scores(feedback_model, frame, features)
            if config.learning_mode == "separated-sliding":
                delayed_scores = _forest_scores(delayed_model, frame, features)
            elif delayed_daily_models:
                delayed_scores = np.mean([
                    positive_probability(fitted, frame[features])
                    for fitted in delayed_daily_models.values()
                ], axis=0)
            else:
                delayed_scores = _forest_scores(warmup_model, frame, features)
            scores = _combine_scores(feedback_scores, delayed_scores, config.feedback_weight)
            if scores is None:
                raise ValueError("No feedback or delayed model is available")
        selected = select_cards(frame.CARD_ID, scores, strategy, config.budget, config.exploration, rng)
        # Paper definition: feedback is revealed quickly, while the delayed
        # model later receives the complete source day (feedback is a subset
        # of the delayed batch). First-time label counts remain deduplicated.
        selected_mask = frame.CARD_ID.isin(selected.CARD_ID)
        feedback = frame[selected_mask].copy()
        delayed = frame.copy()
        newly_delayed = frame[~selected_mask].copy()
        feedback_arrival = number + config.feedback_delay_days
        delayed_arrival = number + config.delayed_label_days
        pending.append((feedback_arrival, number, "feedback", feedback))
        pending.append((delayed_arrival, number, "delayed", delayed))
        pending_label_reveals.append((feedback_arrival, "feedback", feedback))
        pending_label_reveals.append((delayed_arrival, "delayed", newly_delayed))
        arrived_feedback = sum(len(f) for _, _, source, f in arrived if source == "feedback")
        arrived_delayed = sum(len(f) for _, _, source, f in arrived if source == "delayed")
        new_delayed_labels = sum(len(f) for _, source, f in newly_revealed
                                 if source == "delayed")
        row = {"strategy": strategy, "seed": config.seed, "day": number, "file": filename,
               "model": model_kind, "model_reused": not bool(training_entries),
               "train_days": ",".join(map(str, last_fit_days)),
               "window_labeled_rows": sum(len(f) for _, _, _, f in training_entries),
               "window_warmup_rows": sum(len(f) for _, _, source, f in training_entries if source == "warmup"),
               "window_feedback_rows": sum(len(f) for _, _, source, f in feedback_entries if source == "feedback"),
               "window_delayed_rows": sum(len(f) for _, _, source, f in delayed_entries if source == "delayed"),
               "feedback_model_available": feedback_model is not None,
               "delayed_models": (len(delayed_daily_models)
                                  if config.learning_mode == "separated-ensemble"
                                  else int(delayed_model is not None)),
               "labels_arrived": sum(len(f) for _, _, f in newly_revealed),
               "labeled_transactions": sum(len(f) for _, _, f in newly_revealed),
               "feedback_labels_arrived": arrived_feedback,
               "delayed_labels_arrived": arrived_delayed,
               "new_delayed_labels_arrived": new_delayed_labels,
               "excluded_transactions": len(full_frame) - len(frame),
               "feedback_scheduled_transactions": len(feedback),
               "delayed_scheduled_transactions": len(delayed),
               **evaluate(frame, scores, selected, budget=config.budget)}
        records.append(row)
        card_truth = frame.groupby("CARD_ID").y.max()
        selected["is_fraud_card"] = selected.CARD_ID.map(card_truth).astype(int)
        selected.insert(0, "day", number)
        selected.insert(0, "seed", config.seed)
        selected.insert(0, "strategy", strategy)
        selections.append(selected)
        if terminal_history:
            terminal_history.observe(number, full_frame)
        if config.block_cards:
            fraud_cards = set(feedback.loc[feedback.y.eq(1), "CARD_ID"])
            pending_bans.append((feedback_arrival, fraud_cards))
    return pd.DataFrame(records), pd.concat(selections, ignore_index=True)


def config_dict(config):
    return asdict(config)
