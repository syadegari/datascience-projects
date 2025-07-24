from sklearn.linear_model import LogisticRegression
from .base import BaseModel
import joblib
import pathlib


class SkLogistic(BaseModel):
    def __init__(self, **kwargs) -> None:
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    # ---------- persistence ----------
    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        return obj