"""Initial balanced tree ensemble and subsequent random forests."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def categorical_model(estimator, features):
    """Fit categories only on training rows; unseen terminals encode as all-zero."""
    if "TERMINAL_ID" not in features:
        return estimator
    transform = ColumnTransformer([
        ("terminal", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["TERMINAL_ID"]),
        ("numeric", "passthrough", [f for f in features if f != "TERMINAL_ID"]),
    ], sparse_threshold=1.0)
    return Pipeline([("features", transform), ("classifier", estimator)])


def positive_probability(model, x):
    if len(x) == 0:
        return np.empty(0)
    positions = np.flatnonzero(model.classes_ == 1)
    if len(positions) == 0:
        return np.zeros(len(x))
    return model.predict_proba(x)[:, positions[0]]


class InitialEnsemble:
    """For each seed day, train trees on all fraud plus sampled genuine rows."""

    def __init__(self, trees_per_day=10, seed=42):
        self.trees_per_day = trees_per_day
        self.seed = seed

    def fit(self, frames, features):
        rng = np.random.default_rng(self.seed)
        self.features = features
        self.models = []
        for frame in frames:
            if frame.empty:
                continue
            fraud = np.flatnonzero(frame.y.to_numpy() == 1)
            genuine = np.flatnonzero(frame.y.to_numpy() == 0)
            for _ in range(self.trees_per_day):
                if len(fraud) and len(genuine):
                    normal = rng.choice(genuine, len(fraud), replace=len(genuine) < len(fraud))
                    rows = np.concatenate([fraud, normal])
                    train = frame.iloc[rows]
                else:
                    train = frame
                tree = DecisionTreeClassifier(random_state=int(rng.integers(2**31 - 1)))
                tree = categorical_model(tree, features)
                tree.fit(train[features], train.y)
                self.models.append(tree)
        if not self.models:
            raise ValueError("Initial training data is empty")
        return self

    def predict(self, frame):
        return np.mean([positive_probability(m, frame[self.features]) for m in self.models], axis=0)


def fit_forest(frame, features, trees=500, seed=42):
    model = RandomForestClassifier(
        n_estimators=trees, max_features="sqrt", bootstrap=True,
        random_state=seed, n_jobs=1,
    )
    model = categorical_model(model, features)
    model.fit(frame[features], frame.y)
    return model
