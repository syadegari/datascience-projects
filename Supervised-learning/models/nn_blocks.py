import torch
import torch.nn as nn

import math


class TimesBlock(nn.Module):
    """One period discovery block from the TimesNet paper."""
    def __init__(self, n_feat: int, n_blocks: int=3, d_model: int, d_period: int=48, dropout: float=0.1) -> None:
        super().__init__()
        self.d_period = d_period
        self.proj_in = nn.Conv1d(d_model, d_model, 1)
        self.freq_att = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=d_period, groups=d_model, padding=d_period//2)
            for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.proj_in(x.transpose(1, 2))
        y = self.freq_att(y)
        y = y.transpose(1, 2)
        return self.norm(x + y)


class TimesNet(nn.Module):
    """
    Tiny multi-block classifier:
     input [B, L, F] -> TimesBlock x N -> GAP -> FC -> logit
    """ 
    def __init__(self, n_feat: int, n_blocks: int=3, d_model: int=64, lookback: int=40, dropout: float=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.blocks = nn.ModuleList(
            TimesBlock(d_model, d_period=lookback, dropout=dropout)
            for _ in range(n_blocks)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # GAP over time
            nn.Flatten(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):       # x[B, L, F]
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(-1)    # logit

