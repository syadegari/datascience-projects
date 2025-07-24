import numpy as np
from dataprep.window_dataset import WindowDataset


def build_flat_samples(
    parquet_path: str, lookback: int, feature_cols, mask_suffix="_mask", flatten=True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : np.ndarray  [n_samples, lookback * n_tickers * (P+mask?)]
    y : np.ndarray  [n_samples, ]
    dates, tickers  (optional metadata)
    """
    ds = WindowDataset(
        parquet_path=parquet_path,
        lookback=lookback,
        feature_cols=feature_cols,
        mask_suffix=mask_suffix,
        flatten=flatten,  # Make it already 2-D (L, T*P)
        device="cpu",
    )

    n = len(ds)
    L, T, P = ds.lookback, ds.n_tickers, ds.n_feats
    F = T * P if flatten else T * P  # identical here, but future-proof

    X = np.empty((n, L * F), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)

    for i in range(n):
        sample = ds[i]
        X[i] = sample["features"].reshape(-1)  # flatten (L,T*P) → (L*F)
        y[i] = sample["target"]

    return X, y
