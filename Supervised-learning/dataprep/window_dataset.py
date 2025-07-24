import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Hierarchical time-window dataset.

    Entity axis order
    -----------------
        slot 0 : focal asset (targeted ticker)
        slot 1..K : aux tickers in deterministic alphabetical order
                    (sector-ETFs + macro futures / indices).

    Features inside each entity follow:
        [val1, mask1, val2, mask2, ...]  exactly as stored in Parquet.
    """
    def __init__(self,
                 parquet_path: str,
                 lookback: int = 40,
                 feature_cols: list[str] | None = None,
                 mask_suffix: str = "_mask",
                 target_col: str = "target",
                 device: str | torch.device | None = None,
                 flatten: bool = False):
        super().__init__()
        if feature_cols is None:
            feature_cols = ["log_ret", "z_log_ret", "vol_scaled"]

        self.device = torch.device(device) if device else torch.device("cpu")
        self.lookback = lookback
        self.flatten = flatten

        # 1. Load DataFrame
        df: pd.DataFrame = pd.read_parquet(parquet_path, engine="pyarrow")

        # 2. Column list per ticker (interleave value & mask)
        per_tkr_cols: list[str] = []
        for base in feature_cols:
            per_tkr_cols.append(base)
            mask_col = base + mask_suffix
            if mask_col in df.columns:
                per_tkr_cols.append(mask_col)

        # 3. Build ticker-type mapping (asset / sector_etf / macro)
        types: dict[str, str] = (
            df.groupby(level="Ticker")["ticker_type"].first().to_dict()
        )

        self.tickers: list[str] = sorted(types.keys())           # deterministic
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        self.assets = [t for t, tp in types.items() if tp == "asset"]
        self.aux_tickers  = [t for t in self.tickers if t not in self.assets]

        # 4. Build per-ticker numpy blocks
        # TODO: Add Close, will be needed for backtesting later
        self.blocks, self.targets, self.dates = [], [], []
        for tkr in self.tickers:
            sub = df.xs(tkr, level="Ticker")[per_tkr_cols + [target_col]]
            self.blocks.append(sub[per_tkr_cols].to_numpy("float32"))
            self.targets.append(sub[target_col].to_numpy("float32"))
            self.dates.append(sub.index.astype("datetime64[ns]").values)

        self.n_feats = len(per_tkr_cols)
        self.n_aux = len(self.aux_tickers)
        self.entity_order = ["<ASSET>"] + self.aux_tickers
        self.n_tickers = len(self.entity_order) # focal asset + aux tickers

        # 5. Pre-compute valid (asset_idx, t_pos) pairs
        samples: list[tuple[int, int]] = []
        for asset_tkr in self.assets:
            a_idx = self.ticker_to_idx[asset_tkr]
            y     = self.targets[a_idx]
            for t_pos in range(self.lookback - 1, len(y)):
                if not np.isnan(y[t_pos]):
                    samples.append((a_idx, t_pos))
        self.samples = np.asarray(samples, dtype=np.int64)

        print(f"[Dataset] Entity axis order: {self.entity_order}")
        short_hist_aux = sum(len(self.blocks[self.ticker_to_idx[t]]) < self.lookback
                             for t in self.aux_tickers)
        if short_hist_aux:
            print(f"[Dataset] {short_hist_aux}/{len(self.aux_tickers)} aux tickers "
                  "require padding for early dates.")

    def __len__(self):
        return len(self.samples)

    # 
    def _pad_if_short(self, slice_: np.ndarray) -> np.ndarray:
        """Top-pad to lookback length, setting mask bits = 1."""
        if slice_.shape[0] == self.lookback:
            return slice_
        pad_len = self.lookback - slice_.shape[0]
        pad = np.zeros((pad_len, self.n_feats), dtype=np.float32)
        pad[:, 1::2] = 1.0                       # set every mask column to 1
        return np.vstack([pad, slice_])          # older rows first

    # 
    def __getitem__(self, idx: int):
        asset_idx, t_pos = self.samples[idx]

        # focal asset slice
        a_block = self.blocks[asset_idx]
        asset_slice = a_block[t_pos - self.lookback + 1 : t_pos + 1]
        asset_slice = self._pad_if_short(asset_slice)

        # aux slices
        aux_slices = []
        for tkr in self.aux_tickers:
            b = self.blocks[self.ticker_to_idx[tkr]]
            sl = b[t_pos - self.lookback + 1 : t_pos + 1]
            aux_slices.append(self._pad_if_short(sl))

        window = np.stack([asset_slice, *aux_slices], axis=1)   # [L, E, F]

        if self.flatten:
            window = window.reshape(self.lookback, -1)          # [L, ExF]

        return {
            "features": torch.as_tensor(window, device=self.device),
            "target": torch.as_tensor(self.targets[asset_idx][t_pos],
                                        device=self.device),
            "ticker": self.tickers[asset_idx],
            "date": torch.as_tensor(
                        self.dates[asset_idx][t_pos].astype("int64"),
                        device=self.device),
        }
