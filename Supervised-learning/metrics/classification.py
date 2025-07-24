from sklearn import metrics
import numpy as np


def all_metrics(y_true, y_prob, k=0.02):
    out = {
        "auc": metrics.roc_auc_score(y_true, y_prob),
        "prauc": metrics.average_precision_score(y_true, y_prob),
        "brier": metrics.brier_score_loss(y_true, y_prob),
    }

    threshod = np.quantile(y_prob, 1 - k)
    pred_k = (y_prob >= threshod).astype(int)
    out[f"precision@{int(k * 100)}%"] = metrics.precision_score(y_true, pred_k)

    return out
