"""Select unique cards using only identifiers and model scores, never labels."""
import numpy as np
import pandas as pd


def select_cards(card_ids, scores, strategy, budget, exploration, rng):
    if strategy not in {"N", "R", "U"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if budget < 1 or not 0 <= exploration <= budget:
        raise ValueError("Require budget >= 1 and 0 <= exploration <= budget")
    scores = np.asarray(scores, dtype=float)
    if len(card_ids) != len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must align with cards and be probabilities")
    rows = pd.DataFrame({"CARD_ID": list(card_ids), "score": scores})
    rows["uncertainty"] = abs(rows.score - 0.5)
    cards = rows.groupby("CARD_ID", sort=True).agg(
        score=("score", "max"), uncertainty=("uncertainty", "min")
    ).reset_index()
    count = min(budget, len(cards))
    explore_count = 0 if strategy == "N" else min(exploration, count)
    ranked = cards.sort_values(["score", "CARD_ID"], ascending=[False, True])
    chosen = ranked.head(count - explore_count).copy()
    chosen["reason"] = "risk"
    candidates = cards[~cards.CARD_ID.isin(chosen.CARD_ID)]
    if explore_count:
        if strategy == "R":
            extra = candidates.iloc[rng.choice(len(candidates), explore_count, replace=False)].copy()
        else:
            extra = candidates.sort_values(["uncertainty", "CARD_ID"]).head(explore_count).copy()
        extra["reason"] = "random" if strategy == "R" else "uncertainty"
        chosen = pd.concat([chosen, extra], ignore_index=True)
    return chosen.reset_index(drop=True)
