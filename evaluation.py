"""Card-level ranking and investigation outcomes, aggregated per day."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(frame, scores, selected, budget=None):
    """Use the configured investigation budget for NCP; default to selected count."""
    if budget is None:
        budget = len(selected)
    if budget < 0 or budget < len(selected):
        raise ValueError("Investigation budget must be nonnegative and cover selected cards")
    cards = frame.assign(score=np.asarray(scores)).groupby("CARD_ID").agg(
        truth=("y", "max"), score=("score", "max")
    )
    truth = cards["truth"]
    selected_truth = truth.reindex(selected.CARD_ID)
    found = int(selected_truth.sum())
    total_fraud_cards = int(truth.sum())
    max_findable_fraud_cards = min(budget, total_fraud_cards)
    two_classes = truth.nunique() == 2
    return {
        "transactions": len(frame), "cards": len(truth),
        "fraud_transactions": int(frame.y.sum()), "fraud_cards": total_fraud_cards,
        "selected_cards": len(selected),
        "risk_cards": int(selected.reason.eq("risk").sum()),
        "exploration_cards": int(selected.reason.ne("risk").sum()),
        "found_fraud_cards": found,
        "card_precision": found / len(selected) if len(selected) else np.nan,
        "normalized_card_precision": found / max_findable_fraud_cards if max_findable_fraud_cards else np.nan,
        "card_recall": found / total_fraud_cards if total_fraud_cards else np.nan,
        "roc_auc": roc_auc_score(truth, cards["score"]) if two_classes else np.nan,
        "average_precision": average_precision_score(truth, cards["score"]) if total_fraud_cards else np.nan,
    }
