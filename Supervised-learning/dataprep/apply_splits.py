"""
Slice `model_matrix.parquet` into expanding **train / val / test** sets based
on a YAML split plan **with rigorous range validation**.

Split-plan format
-----------------
```yaml
train:
  start: 2015-01-01
  end:   2022-12-31
val:
  start: 2023-01-01
  end:   2024-06-30
test:
  start: 2024-07-01
  end:   2025-07-12
```

Validation rules implemented (look for the numbered comments in the code)
----------------------------
1. **Internal validity** - each split has `start < end`.
2. **Sequential ordering** - `train.end < val.start < val.end < test.start < test.end`.
3. **Within data bounds** - all split dates lie inside the min/max dates of the
   model matrix.

Usage
-----
```bash
python apply_splits.py \
    --model-matrix data/model_matrix.parquet \
    --split-plan   configs/split_plan.yaml   \
    --out-dir      data/splits
```
"""
from pathlib import Path
import argparse
import sys
import yaml
import joblib

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def _read_split_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        splits_raw = yaml.safe_load(fh)
    required = {"train", "val", "test"}
    if set(splits_raw) != required:
        raise ValueError(f"YAML must define exactly {required} keys")

    # Convert to Timestamp tuples
    splits = {
        k: (pd.Timestamp(v["start"]), pd.Timestamp(v["end"])) for k, v in splits_raw.items()
    }

    # 1. start < end check
    for k, (s, e) in splits.items():
        if s >= e:
            raise ValueError(f"Invalid range for {k}: start {s} >= end {e}")

    # 2. sequential ordering check
    if not (splits["train"][1] < splits["val"][0] < splits["val"][1] < splits["test"][0] < splits["test"][1]):
        raise ValueError("Ranges must satisfy train.end < val.start < val.end < test.start < test.end")

    return splits


def _split(split: tuple[pd.Timestamp, pd.Timestamp], df: pd.DataFrame) -> pd.DataFrame:
    start, end = split
    return df.loc[(slice(start, end), slice(None)), :]


def _fit(df: pd.DataFrame) -> ColumnTransformer:
    SCALABLE = ["log_ret", "z_log_ret", "vol_scaled"]
    ct = ColumnTransformer(
        transformers=[
            (f"std_{c}", StandardScaler(), [c]) for c in SCALABLE
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
        force_int_remainder_cols=False     # Future proof this otherwise throws a warning
    )
    ct.set_output(transform='pandas') # returns df and respects dtypes
    ct.fit(df)

    return ct


def _transform(df: pd.DataFrame, ct: ColumnTransformer) -> pd.DataFrame:
    cols_original = df.columns
    out = ct.transform(df)
    # Ensure original column oder (scikit-learn puts scaled cols first)
    return out.reindex(columns=cols_original)
    

def slice_and_save(matrix_path: Path, split_yaml: Path, out_dir: Path):
    df = pd.read_parquet(matrix_path, engine="pyarrow")
    splits = _read_split_yaml(split_yaml)

    # 3. within data bounds check
    idx_times = df.index.get_level_values(0)
    data_start, data_end = idx_times.min(), idx_times.max()
    for name, (s, e) in splits.items():
        if s < data_start or e > data_end:
            raise ValueError(
                f"{name} range [{s.date()} - {e.date()}] is outside data bounds [{data_start.date()} - {data_end.date()}]"
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(splits)
    train = _split(splits['train'], df)
    val = _split(splits['val'], df)
    test = _split(splits['test'], df)

    print('fit to train data')
    ct = _fit(train)

    print('transform train')
    train_scaled = _transform(train, ct)
    print('transform validation')
    val_scaled = _transform(val, ct)
    print('transform test')
    test_scaled = _transform(test, ct)

    # Save the fitted transformer
    transformer_path = out_dir / "column_transformer.pkl"
    joblib.dump(ct, transformer_path)
    print(f"Saved ColumnTransformer to: {transformer_path}")

    for name, df_scaled in zip(['train_scaled', 'val_scaled', 'test_scaled'], [train_scaled, val_scaled, test_scaled]):
        outfile = out_dir / f"{name}_matrix.parquet"
        df_scaled.to_parquet(outfile, engine="pyarrow", compression="snappy")
        print(f"Saved {name:5s} in: {outfile} | rows={df_scaled.shape[0]:,}")

    for name, df in [("train",train_scaled),("val",val_scaled),("test",test_scaled)]:
        pos = df.loc[df["ticker_type"]=="asset","target"].mean()
        print(f"{name}: positive rate = {pos:.3%}")


def main():
    p = argparse.ArgumentParser(description="Apply validated train/val/test splits to model matrix")
    p.add_argument("--model-matrix", type=Path, required=True)
    p.add_argument("--split-plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    try:
        slice_and_save(args.model_matrix, args.split_plan, args.out_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
