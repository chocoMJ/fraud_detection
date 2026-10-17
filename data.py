"""Load chronological daily files without inferring identifiers as numbers."""
from pathlib import Path
from datetime import date
import re

import numpy as np
import pandas as pd

FEATURES = ["TX_AMOUNT", "TX_DURING_WEEKEND", "TX_DURING_NIGHT"] + [
    f"CUSTOMER_ID_{kind}_{window}DAY_WINDOW"
    for window in (1, 7, 30) for kind in ("NB_TX", "AVG_AMOUNT")
] + [f"TERMINAL_ID_NB_TX_{window}DAY_WINDOW" for window in (1, 7, 30)]


"""
input : pickle의 path
기능 : pickle을 읽어서 DataFrame 반환
"""
def read_official_pickle(path):
    """Read a trusted local Handbook pickle, including older pandas formats."""
    try:
        return pd.read_pickle(path)
    except (ModuleNotFoundError, AttributeError) as error:
        if "pandas.core.indexes.numeric" not in str(error):
            raise
        # Old pandas pickles refer to index classes removed in pandas 2.
        from pandas.compat.pickle_compat import Unpickler

        class LegacyIndexUnpickler(Unpickler):
            def find_class(self, module, name):
                if module == "pandas.core.indexes.numeric" and name in {"Int64Index", "UInt64Index", "Float64Index"}:
                    return pd.Index
                return super().find_class(module, name)

        with Path(path).open("rb") as stream:
            return LegacyIndexUnpickler(stream).load()

def convert_day(frame, expected_date):
    required = {"CUSTOMER_ID", "TERMINAL_ID", "TX_FRAUD", "TX_DATETIME", "TRANSACTION_ID", *FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Handbook columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Empty source day: {expected_date}")
    timestamps = pd.to_datetime(frame.TX_DATETIME, errors="raise")
    if timestamps.isna().any() or not timestamps.dt.date.eq(expected_date).all():
        raise ValueError("Source rows do not match their filename date")
    if frame.TRANSACTION_ID.isna().any() or frame.TRANSACTION_ID.duplicated().any():
        raise ValueError("Missing or duplicate TRANSACTION_ID")
    if frame.CUSTOMER_ID.isna().any() or not frame.TX_FRAUD.isin([0, 1]).all():
        raise ValueError("Invalid customer identifier or fraud label")
    ordered = frame.assign(TX_DATETIME=timestamps).sort_values(["TX_DATETIME", "TRANSACTION_ID"])
    output = ordered[["CUSTOMER_ID", *FEATURES, "TERMINAL_ID", "TX_FRAUD"]].rename(
        columns={"CUSTOMER_ID": "CARD_ID", "TX_FRAUD": "y"}
    ).reset_index(drop=True)
    output["CARD_ID"] = output.CARD_ID.astype("string")
    output["TERMINAL_ID"] = output.TERMINAL_ID.astype("string")
    if output.TERMINAL_ID.isna().any() or output.TERMINAL_ID.str.strip().eq("").any():
        raise ValueError("Missing terminal identifier")
    if output.CARD_ID.str.strip().eq("").any():
        raise ValueError("Empty customer identifier")
    output[FEATURES] = output[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(output[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite Handbook features")
    output["y"] = output.y.astype(int)
    return output, ordered[["TRANSACTION_ID", "TX_DATETIME"]].reset_index(drop=True)


def handbook_source_directory(directory):
    """Resolve either a directory of pickles or a single cached revision."""
    directory = Path(directory)
    if list(directory.glob("*.pkl")):
        return directory
    revisions = sorted(p for p in directory.iterdir()
                       if p.is_dir() and list(p.glob("*.pkl"))) if directory.exists() else []
    if len(revisions) != 1:
        raise ValueError(f"Specify --data-dir pointing to one directory of Handbook .pkl files: {directory}")
    return revisions[0]


def load_handbook_days(directory, features=None):
    """Read trusted official transformed YYYY-MM-DD.pkl files directly in memory."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.pkl"))
    if not paths:
        raise ValueError(f"No Handbook .pkl files in {directory}")
    columns = list(features) if features is not None else list(FEATURES)
    if not columns or len(set(columns)) != len(columns) or not set(columns).issubset(FEATURES):
        raise ValueError("Handbook model features must be a unique subset of the 12 numeric features")
    days, seen_ids = [], set()
    first_date = None
    for path in paths:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.pkl", path.name):
            raise ValueError(f"Invalid Handbook filename: {path.name}; expected YYYY-MM-DD.pkl")
        day = date.fromisoformat(path.stem)
        frame, mapping = convert_day(read_official_pickle(path), day)
        ids = set(mapping.TRANSACTION_ID)
        if seen_ids & ids:
            raise ValueError(f"Duplicate transaction identifiers across days: {path.name}")
        seen_ids.update(ids)
        if first_date is None:
            first_date = day
        days.append(((day - first_date).days + 1, path.name,
                     frame[["CARD_ID", *columns, "TERMINAL_ID", "y"]]))
    return days, columns


def load_days(directory: Path, features=None):
    paths = list(Path(directory).glob("day*.csv"))
    if not paths:
        raise ValueError(f"No day<number>.csv files in {directory}")
    numbered = []
    for path in paths:
        match = re.fullmatch(r"day(\d+)\.csv", path.name)
        if not match:
            raise ValueError(f"Invalid daily filename: {path.name}")
        numbered.append((int(match.group(1)), path))
    numbered.sort()
    if len({n for n, _ in numbered}) != len(numbered):
        raise ValueError("Duplicate day numbers")
    days = []
    columns = list(features) if features else None
    for number, path in numbered:
        frame = pd.read_csv(path, dtype={"CARD_ID": "string", "TERMINAL_ID": "string"})
        if not {"CARD_ID", "y"}.issubset(frame.columns):
            raise ValueError(f"{path.name}: requires CARD_ID and y")
        if columns is None:
            columns = [c for c in frame if c not in {"CARD_ID", "TERMINAL_ID", "y"}]
        if (not columns or len(set(columns)) != len(columns)
                or set(columns) & {"CARD_ID", "TERMINAL_ID", "y"}):
            raise ValueError("Features must be unique and exclude identifiers and y")
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"{path.name}: missing features")
        metadata = {"TERMINAL_ID"} if "TERMINAL_ID" in frame.columns else set()
        if features is None and set(frame.columns) != set(columns) | {"CARD_ID", "y"} | metadata:
            raise ValueError(f"{path.name}: inconsistent columns")
        if frame.CARD_ID.isna().any() or frame.CARD_ID.str.strip().eq("").any():
            raise ValueError(f"{path.name}: missing CARD_ID")
        if not frame.y.isin([0, 1]).all():
            raise ValueError(f"{path.name}: y must be 0 or 1")
        frame = frame[["CARD_ID", *columns, *metadata, "y"]].copy()
        if "TERMINAL_ID" in metadata:
            if frame.TERMINAL_ID.isna().any() or frame.TERMINAL_ID.str.strip().eq("").any():
                raise ValueError(f"{path.name}: missing TERMINAL_ID")
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: features must be finite numeric values")
        frame["y"] = frame.y.astype(int)
        days.append((number, path.name, frame))
    return days, columns
