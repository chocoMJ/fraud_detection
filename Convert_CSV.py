import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_py.data import read_official_pickle


"""
input_path : 변환할 파일이 있는 폴더
output_dir : CSV저장할 폴더

"""
def convert_pickles(input_path, output_dir, overwrite=False):
    """Preserve all columns and row order; CSV does not preserve Python dtypes."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    paths = ([input_path] if input_path.is_file() and input_path.suffix == ".pkl"
             else sorted(input_path.glob("*.pkl")) if input_path.is_dir() else [])
    if not paths:
        raise ValueError(f"No .pkl files found: {input_path}")
    targets = [output_dir / f"{path.stem}.csv" for path in paths]
    if not overwrite:
        for target in targets:
            if target.exists():
                raise FileExistsError(f"CSV already exists: {target}; use --overwrite to replace")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, target in zip(paths, targets):
        frame = read_official_pickle(source)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"Expected a pandas DataFrame: {source}")
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        print(f"{source.name} -> {target} ({len(frame):,} rows, {len(frame.columns)} columns)")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Local .pkl file or directory containing .pkl files")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data/handbook/inspection_csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_pickles(args.input_path, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
