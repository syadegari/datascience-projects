import pathlib
import joblib

import torch
import torch.nn as nn
from torch.utills.data import DataLoader
from sklearn import metrics

from .nn_blocks import TimesNet
from dataprep.window_dataset import WindowDataset
from .base import BaseModel


class TimesNetWrapper(BaseModel):
    """
    fit and predict_proba API compatible with run_experiment.py
    """

    is_torch_model = True

    def __init__(
        self,
        lookback: int,
        feature_cols,
        batch_size: int = 256,
        epochs: int = 20,
        patience: int = 3,
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.lookback = lookback
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.device = torch.device(device)
        self.net = None
        self.best_state = None

        def _build_net(self, n_feat):
            return TimesNet(n_feat, self.lookback).to(self.device)

        def fit(self, parquet_train, parquet_val, sample_weight=None):
            ds_train = WindowDataset(parquet_train, self.lookback, self.feature_cols, flatten=False)
            ds_val = WindowDataset(parquet_val, self.lookback, self.feature_cols, flatten=False)

            n_feat = ds_train[0]["features"].shape[-1]
            self.net = self._build_net(n_feat)
            opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
            loss_fn = nn.BCEWithLogitsLoss()

            best_auc, patience_left = 0.0, self.patience
            dl_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
            for batch in dl_train:
                x = batch["features"].to(self.device)
                y = batch["target"].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(x), y)
                loss.backward()
                opt.step()

            self.net.eval()
            dl_val = DataLoader(ds_val, batch_size=self.batch_size)
            y_prob, y_true = [], []
            with torch.no_grad():
                for batch in dl_val:
                    x = batch["feature"].to(self.device)
                    logits = self.net(x)
                    y_prob.append(torch.sigmoid(logits).cpu().numpy())
                    y_true.append(batch["target"].cpu().numpy())
            y_prob = torch.cat([torch.from_numpy(p) for p in y_prob]).numpy()
                    
