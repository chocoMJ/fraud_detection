"""Generate Handbook transformed data with ONLY scenario 3 (temporary customer compromise).

Uses the attributed official functions in vendor/handbook.py, offline.
Scenarios 1/2 are never applied. Features are computed from the new transactions.
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

from demo_py.vendor.handbook import (
    generate_customer_profiles_table, generate_terminal_profiles_table,
    get_list_terminals_within_radius, generate_transactions_table,
    is_weekend, is_night, get_customer_spending_behaviour_features,
    get_count_risk_rolling_window,
)

ROOT = Path(__file__).resolve().parents[1]


def generate_transactions(customers=5000, terminals=10000, days=183,
                          start_date="2018-04-01", radius=5.0):
    """Official defaults/seeds; explicit iteration avoids pandas groupby API changes."""
    if min(customers, terminals, days) < 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Customers, terminals, days and radius must be positive")
    date.fromisoformat(str(start_date))
    customer_profiles = generate_customer_profiles_table(customers, random_state=0)
    terminal_profiles = generate_terminal_profiles_table(terminals, random_state=1)
    coordinates = terminal_profiles[["x_terminal_id", "y_terminal_id"]].values.astype(float)
    frames = []
    for index, customer in customer_profiles.iterrows():
        customer = customer.copy()
        customer["available_terminals"] = get_list_terminals_within_radius(customer, coordinates, radius)
        frame = generate_transactions_table(customer, start_date=str(start_date), nb_days=days)
        if not frame.empty:
            frames.append(frame)
        if (index + 1) % 500 == 0:
            print(f"Transactions: {index + 1}/{customers} customers", flush=True)
    if not frames:
        raise ValueError("No transactions generated; increase customer/terminal counts or radius")
    frame = pd.concat(frames, ignore_index=True).sort_values("TX_DATETIME").reset_index(drop=True)
    frame.insert(0, "TRANSACTION_ID", np.arange(len(frame)))
    for column in ["CUSTOMER_ID", "TERMINAL_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS"]:
        frame[column] = frame[column].astype("int64")
    return frame


def apply_scenario3(frame, customers=5000):
    """Official scenario 3 only, applied to freshly generated transactions.

    Each day selects three customers from the full profile population. Within
    their next 14 days, floor(one third) of the pooled transactions are sampled
    and their amounts multiplied by five. Repeated selections can compound.
    As upstream, no new compromise starts on the final observed day.
    """
    if customers < 3:
        raise ValueError("Scenario 3 requires at least three customer profiles")
    frame = frame.copy()
    frame["TX_FRAUD"] = 0
    frame["TX_FRAUD_SCENARIO"] = 0
    profiles = pd.DataFrame({"CUSTOMER_ID": np.arange(customers)})
    for day in range(int(frame.TX_TIME_DAYS.max())):
        compromised = profiles.sample(n=3, random_state=day).CUSTOMER_ID.values
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


def transform_transactions(frame):
    """Official 14 derived columns; includes current customer transaction, 7-day terminal delay."""
    frame = frame.copy()
    frame["TX_DURING_WEEKEND"] = frame.TX_DATETIME.apply(is_weekend)
    frame["TX_DURING_NIGHT"] = frame.TX_DATETIME.apply(is_night)
    print("Computing customer features (1, 7, 30 days)", flush=True)
    frame = pd.concat([get_customer_spending_behaviour_features(group.copy())
                       for _, group in frame.groupby("CUSTOMER_ID", sort=True)])
    print("Computing terminal features (7-day label delay)", flush=True)
    frame = pd.concat([get_count_risk_rolling_window(group.copy())
                       for _, group in frame.groupby("TERMINAL_ID", sort=True)])
    return frame.reset_index(drop=True).sort_values(["TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)


def generate_dataset(output_dir, customers=5000, terminals=10000, days=183,
                     start_date="2018-04-01", radius=5.0):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output_dir}")
    if customers < 3:
        raise ValueError("Scenario 3 requires at least three customer profiles")
    frame = transform_transactions(apply_scenario3(generate_transactions(
        customers, terminals, days, start_date, radius), customers=customers))
    if (len(frame.columns) != 23 or not frame.TX_FRAUD.isin([0, 1]).all()
            or not frame.TX_FRAUD_SCENARIO.eq(frame.TX_FRAUD * 3).all()):
        raise ValueError("Generated schema or scenario labels failed validation")
    if not np.isfinite(frame.drop(columns="TX_DATETIME").to_numpy(dtype=float)).all():
        raise ValueError("Generated data has non-finite values")
    # Reject tiny settings with empty days: the experiment loader requires nonempty daily files.
    if sorted(frame.TX_TIME_DAYS.unique()) != list(range(days)):
        raise ValueError("Some days have no transactions; increase customer/terminal counts or radius")
    upstream = json.loads((ROOT / "vendor/handbook_source.json").read_text(encoding="utf-8"))
    manifest = {
        "kind": "handbook_scenario3_only_transformed", "upstream": upstream,
        "parameters": {"customers": customers, "terminals": terminals, "days": days,
                       "start_date": str(start_date), "radius": radius},
        "seeds": {"customers": 0, "terminals": 1, "transactions": "customer ID (official behavior)"},
        "fraud_rule": "Scenario 3 only: select 3 customers daily; sample floor(1/3) of their pooled transactions over [day, day+14); multiply amounts by 5 and label fraud. No amount threshold and no scenario 2.",
        "scenario3_policy": {"customer_sampling_seed": "day", "transaction_sampling_seed": "day",
                             "new_compromise_days": "0 through last observed day minus 1 (official loop)",
                             "repeated_selection": "amount multipliers compound, as in official code"},
        "feature_policy": "Official 1/7/30-day features; terminal risk uses 7-day delay; night hour <= 6",
        "transactions": len(frame), "fraud_transactions": int(frame.TX_FRAUD.sum()),
        "columns": frame.columns.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "files": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scenario3-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        for day, daily in frame.groupby("TX_TIME_DAYS", sort=True):
            filename = (pd.Timestamp(start_date) + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d.pkl")
            target = staging / filename
            daily.to_pickle(target, protocol=4)
            manifest["files"].append({"file": filename, "transactions": len(daily),
                                      "fraud_transactions": int(daily.TX_FRAUD.sum()),
                                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output_dir)
    print(f"Saved {days} days, {len(frame):,} transactions, {frame.TX_FRAUD.sum():,} frauds: {output_dir.resolve()}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/handbook/scenario3_only")
    parser.add_argument("--customers", type=int, default=5000)
    parser.add_argument("--terminals", type=int, default=10000)
    parser.add_argument("--days", type=int, default=183)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2018, 4, 1))
    parser.add_argument("--radius", type=float, default=5.0)
    args = parser.parse_args()
    generate_dataset(**vars(args))


if __name__ == "__main__":
    main()
