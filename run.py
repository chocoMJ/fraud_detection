"""Run from any directory: python /path/to/demo_py/run.py --strategies N R U."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import sklearn

from demo_py.data import FEATURES, load_days, load_handbook_days, handbook_source_directory
from demo_py.terminal_history import model_features
from demo_py.simulation import Config, config_dict, simulate


def main():
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--input-format", choices=["pkl", "csv"], default="pkl",
                        help="Read Handbook pickles directly or previously converted model-input CSV files")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-days", type=int, help="Use only the first N days, including warm-up, for a smoke test")
    parser.add_argument("--strategies", nargs="+", choices=["N", "R", "U"], default=["N", "R", "U"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--no-terminal-features", action="store_true",
                        help="Use only the original 12 numeric features for baseline comparison")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--exploration", type=int, default=5)
    parser.add_argument("--trees-per-day", type=int, default=10)
    parser.add_argument("--forest-trees", type=int, default=500)
    parser.add_argument("--block-cards", action="store_true")
    args = parser.parse_args()
    if args.no_terminal_features:
        if args.features is not None:
            parser.error("Use either --features or --no-terminal-features")
        args.features = list(FEATURES)

    input_format = args.input_format
    
    if args.data_dir is None:
        args.data_dir = root / ("data/handbook/scenario1_only" if input_format == "pkl" else "data/handbook/daily")
    if args.output_dir is None:
        output_name = "scenario1_only" if args.data_dir.resolve() == (root / "data/handbook/scenario1_only").resolve() else "handbook"
        if not args.no_terminal_features and (args.features is None or "TERMINAL_ID" in args.features):
            output_name += "_terminal_history"
        args.output_dir = root / "results" / output_name
    if input_format == "pkl":
        args.data_dir = handbook_source_directory(args.data_dir)
        days, features = load_handbook_days(args.data_dir, args.features)
    else:
        days, features = load_days(args.data_dir, args.features)
    if args.max_days is not None:
        if args.max_days <= args.warmup_days:
            parser.error("--max-days must exceed --warmup-days")
        days = days[:args.max_days]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, configs = [], []
    for seed in dict.fromkeys(args.seeds):
        config = Config(args.warmup_days, args.window_days, args.budget, args.exploration,
                        args.trees_per_day, args.forest_trees, seed, args.block_cards)
        configs.append(config_dict(config))
        for strategy in dict.fromkeys(args.strategies):
            daily, selected = simulate(days, features, strategy, config)
            daily.to_csv(args.output_dir / f"daily_{strategy}_seed{seed}.csv", index=False)
            selected.to_csv(args.output_dir / f"selected_{strategy}_seed{seed}.csv", index=False)
            metrics.append(daily)
            print(f"{strategy}, seed={seed}: {len(daily)} evaluation days completed", flush=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    summary = all_metrics.groupby(["strategy", "seed"]).agg(
        evaluation_days=("day", "size"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_average_precision=("average_precision", "mean"),
        mean_card_precision=("card_precision", "mean"),
        mean_normalized_card_precision=("normalized_card_precision", "mean"),
        mean_card_recall=("card_recall", "mean"),
        investigated_card_days=("selected_cards", "sum"),
        found_fraud_card_days=("found_fraud_cards", "sum"),
    ).reset_index()
    summary = summary.rename(columns={
        "mean_roc_auc": "mean_roc_auc (ROC-AUC)",
        "mean_average_precision": "mean_average_precision (AP)",
        "mean_card_precision": "mean_card_precision (CP)",
        "mean_normalized_card_precision": "mean_normalized_card_precision (NCP)",
        "mean_card_recall": "mean_card_recall (Recall)",
    })
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "protocol": "Full-label warmup; initial balanced ensemble on first evaluation day; subsequent RF; selected labels available next day",
        "evaluation": {"unit": "customer_id_per_day", "card_score": "max_transaction_probability",
                       "ncp_denominator": "min(configured_budget, daily_fraud_cards)",
                       "card_label": "any_fraud_transaction", "ranking_population": "all_eligible_cards",
                       "summary_aggregation": "daily_mean_excluding_nan"},
        "data_dir": str(args.data_dir.resolve()), "input_features": features,
        "features": model_features(features),
        "terminal_features": {
            "enabled": "TERMINAL_ID" in features,
            "encoding": "one_hot_training_categories_only; unseen=all_zero",
            "windows_calendar_days": [1, 7, 30],
            "availability": "previous days only; full warmup labels then selected-card labels",
            "observed_counts": "all incoming transactions including blocked cards",
            "fraud_rate_denominator": "labeled transactions; zero when none; labeled count supplied separately",
            "training_history": "features frozen at original prediction time",
            "official_terminal_risk_columns": "not used",
        },
        "dataset": "handbook", "max_days": args.max_days,
        "input_format": input_format,
        "configs": configs, "strategies": list(dict.fromkeys(args.strategies)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "input_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() for _, name, _ in days},
    }
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        manifest_text = manifest_path.read_text(encoding="utf-8")
        metadata["dataset_manifest"] = json.loads(manifest_text)
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
