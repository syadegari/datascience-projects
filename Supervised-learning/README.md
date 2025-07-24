# Supervised‑Learning for Short‑Horizon Equity Trading  
A modular research stack for predicting short horizon stock moves and running fast vectorised back‑tests.

<div align="center">
  <!-- Optional shields.io badges -->
  <!-- <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue?logo=python"> -->
  <!-- <img alt="License" src="https://img.shields.io/github/license/syadegari/datascience-projects"> -->
</div>

---

## ️Problem Statement



Given daily Close and Volume data for a basket of **equities** plus auxiliary tickers  (sector ETFs & macro futures), we want to **classify whether each stock will rise by ≥ _c_% within the next _N_ trading days**.  Predicted probabilities are then converted to fractional long positions and evaluated with back‑tests and classical portfolio‑risk metrics (Sharpe, Calmar, Max‑DD).

---

## Key Features

| Module | Highlights |
| ------ | ---------- |
| **`dataprep/`** | *End‑to‑end pipeline*<br>  yfinance download or cached parquet<br> Feature engineering – `log_ret`, `z_log_ret`, `log‑scaled volume`<br> Binary / regression target generation (`target_generation.py`)<br> Merge & NA‑policy with masks (`merge_features_labels.py`)<br> Time‑based train/val/test slicing with YAML plan + scaling |
| **`helpers/`** | Flat sample builder for tree/linear models |
| **`dataprep/window_dataset.py`** | Hierarchical `[lookback × entity × feature]` WindowDataset<br>– deterministic ticker order <br>– automatic mask padding for short histories |
| **`models/`** | Unified `BaseModel` API<br> `SkLogistic` (scikit‑learn)<br> `LGBMClassifier` (LightGBM)<br> `TimesNetWrapper` (PyTorch TimesNet mini‑classifier) |
| **`metrics/`** | Classification (AUC, PRAUC, Brier, Precision@k%)<br>Portfolio risk (Sharpe, Calmar, Max‑DD) |
| **Notebook** | `times_net_demo.ipynb` – interactive walk‑through & visualisations for a pytorch based training of [TimesNet](https://arxiv.org/abs/2210.02186) model  |
| **`run_experiment.py`** | YAML‑driven experiments: ingest parquet splits → train model → print metrics → run back‑test → save artefacts |

---


