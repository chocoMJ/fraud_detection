"""Generate Handbook transformed data with fraud scenarios 2 and 3 only.

Scenario 1 (the fixed amount threshold) is never applied. The base transactions
and feature transformations use the attributed offline Handbook functions.
"""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import tempfile

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo_py.Generate_Scenario.generate_scenario3 import (
    generate_transactions, transform_transactions,
)

ROOT = Path(__file__).resolve().parents[1]


def apply_scenarios23(frame, customers=5000, terminals=10000):
    """Apply official scenario 2 followed by scenario 3, without scenario 1."""
    if customers < 3 or terminals < 2:
        raise ValueError("Scenarios 2/3 require at least 2 terminals and 3 customers")
    frame = frame.copy()
    frame["TX_FRAUD"] = 0
    frame["TX_FRAUD_SCENARIO"] = 0
    terminal_profiles = pd.DataFrame({"TERMINAL_ID": np.arange(terminals)})
    customer_profiles = pd.DataFrame({"CUSTOMER_ID": np.arange(customers)})
    last_day = int(frame.TX_TIME_DAYS.max())

    # Scenario 2: two terminals are compromised each day for the next 28 days.
    for day in range(last_day):
        compromised = terminal_profiles.sample(n=2, random_state=day).TERMINAL_ID.values
        selected = frame.index[
            (frame.TX_TIME_DAYS >= day)
            & (frame.TX_TIME_DAYS < day + 28)
            & frame.TERMINAL_ID.isin(compromised)
        ]
        frame.loc[selected, "TX_FRAUD"] = 1
        frame.loc[selected, "TX_FRAUD_SCENARIO"] = 2

    # Scenario 3 is applied second, matching the official overlap precedence.
    for day in range(last_day):
        compromised = customer_profiles.sample(n=3, random_state=day).CUSTOMER_ID.values
        candidates = frame.index[
            (frame.TX_TIME_DAYS >= day)
            & (frame.TX_TIME_DAYS < day + 14)
            & frame.CUSTOMER_ID.isin(compromised)
        ].tolist()
        selected = random.Random(day).sample(candidates, k=int(len(candidates) / 3))
        frame.loc[selected, "TX_AMOUNT"] *= 5
        frame.loc[selected, "TX_FRAUD"] = 1
        frame.loc[selected, "TX_FRAUD_SCENARIO"] = 3
    return frame


def generate_dataset(output_dir, customers=5000, terminals=10000, days=183,
                     start_date="2018-04-01", radius=5.0):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output_dir}")
    raw = generate_transactions(customers, terminals, days, start_date, radius)
    frame = transform_transactions(apply_scenarios23(raw, customers, terminals))
    scenarios = set(frame.TX_FRAUD_SCENARIO.unique())
    if (len(frame.columns) != 23 or not frame.TX_FRAUD.isin([0, 1]).all()
            or not scenarios.issubset({0, 2, 3}) or 1 in scenarios
            or not frame.TX_FRAUD.eq(frame.TX_FRAUD_SCENARIO.gt(0)).all()):
        raise ValueError("Generated schema or scenario labels failed validation")
    if not np.isfinite(frame.drop(columns="TX_DATETIME").to_numpy(dtype=float)).all():
        raise ValueError("Generated data has non-finite values")
    if sorted(frame.TX_TIME_DAYS.unique()) != list(range(days)):
        raise ValueError("Some days have no transactions; increase generation size")

    upstream = json.loads((ROOT / "vendor/handbook_source.json").read_text(encoding="utf-8"))
    scenario_counts = {str(int(k)): int(v) for k, v in
                       frame.TX_FRAUD_SCENARIO.value_counts().sort_index().items()}
    manifest = {
        "kind": "handbook_scenarios2_and3_transformed", "upstream": upstream,
        "parameters": {"customers": customers, "terminals": terminals, "days": days,
                       "start_date": str(start_date), "radius": radius},
        "seeds": {"customers": 0, "terminals": 1, "transactions": "customer ID",
                  "fraud_sampling": "calendar day"},
        "fraud_rule": "Scenario 2 followed by scenario 3; scenario 1 is excluded",
        "scenario2_policy": "Select 2 terminals daily; label their transactions in [day, day+28) as fraud",
        "scenario3_policy": "Select 3 customers daily; sample floor(1/3) of transactions in [day, day+14), multiply amount by 5, and label scenario 3",
        "overlap_policy": "Scenario 3 is applied after scenario 2 and therefore owns the scenario label on overlaps",
        "feature_policy": "Official 1/7/30-day features recomputed after fraud injection; terminal risk uses 7-day delay",
        "transactions": len(frame), "fraud_transactions": int(frame.TX_FRAUD.sum()),
        "scenario_counts": scenario_counts, "columns": frame.columns.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__}, "files": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scenario23-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        for day, daily in frame.groupby("TX_TIME_DAYS", sort=True):
            filename = (pd.Timestamp(start_date) + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d.pkl")
            target = staging / filename
            daily.to_pickle(target, protocol=4)
            manifest["files"].append({"file": filename, "transactions": len(daily),
                                      "fraud_transactions": int(daily.TX_FRAUD.sum()),
                                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        staging.rename(output_dir)
    print(f"Saved {days} days, {len(frame):,} transactions, "
          f"{frame.TX_FRAUD.sum():,} frauds: {output_dir.resolve()}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/handbook/scenario23_only")
    parser.add_argument("--customers", type=int, default=5000)
    parser.add_argument("--terminals", type=int, default=10000)
    parser.add_argument("--days", type=int, default=183)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2018, 4, 1))
    parser.add_argument("--radius", type=float, default=5.0)
    generate_dataset(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
