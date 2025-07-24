import yaml
import json
import numpy as np
import joblib
import argparse
import pathlib
import pandas as pd

from models import MODEL_REGISTRY
from helpers.flat_sampler import build_flat_samples
from metrics.classification import all_metrics
from backtest.vectorized import run_backtest


def log_prevalence(name, y):
    pos = y.mean()
    neg = 1 - pos
    print(f"[{name}] positives = {pos:.3%}  |  negatives = {neg:.3%}  "
          f"|  ratio N_pos : N_neg = {pos/(neg or 1):.3f}")


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    lookback = cfg["lookback"]
    features = cfg.get("feature_cols", ["log_ret", "z_log_ret", "vol_scaled"])
    paths = cfg["data"]
    X_train, y_train = build_flat_samples(paths["train"], lookback, features)
    X_val, y_val = build_flat_samples(paths["val"], lookback, features)
    X_test, y_test = build_flat_samples(paths["test"], lookback, features)

    log_prevalence("TRAIN", y_train)
    log_prevalence("VAL",   y_val)
    log_prevalence("TEST",  y_test)

    # imbalance weight
    pos_rate = y_train.mean()
    w = None
    if cfg.get("class_weight") == "balanced":
        w = np.where(y_train == 1, (1 - pos_rate) / pos_rate, 1)

    # model
    ModelCls = MODEL_REGISTRY[cfg["model"]]
    model = ModelCls(**cfg.get("model_params", {}))
    model.fit(X_train, y_train, sample_weight=w)

    # metrics
    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = model.predict_proba(X)
        m = all_metrics(y, prob, k=cfg.get("top_k", 0.02))
        print(f"{split} metrics:", json.dumps(m, indent=2))

        # back-test (assets only rows correspond 1-to-1 with prob order)
        df_split = pd.read_parquet(paths[f"{split}_df"], engine="pyarrow")
        pnl = run_backtest(df_split, prob, lookback)
        print(f"{split} Sharpe: {pnl.mean()/pnl.std(ddof=0)*np.sqrt(252):.2f}")

    # save artefact
    out_dir = pathlib.Path(cfg["artifact_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(out_dir / "model.joblib")
    except NotImplementedError:
        print("[WARN] save() not implemented for this model – skipping disk dump")

    print("model saved in: ", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    args = ap.parse_args()
    main(args.config)
