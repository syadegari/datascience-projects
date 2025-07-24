import pathlib
import joblib
import lightgbm as lgb
from .base import BaseModel


class LGBMClassifier(BaseModel):
    """
    Thin wrapper so LightGBM fits our BaseModel API
    (fit / predict_proba / save / load).
    """

    def __init__(self, params=None, num_boost_round: int = 500):
        # sensible defaults; override via YAML model_params
        self.params = params or {
            "objective": "binary",
            "metric":    "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
        }
        self.num_boost_round = num_boost_round
        self.model: lgb.Booster | None = None

    #
    def fit(self, X, y, sample_weight=None):
        dtrain = lgb.Dataset(X, y, weight=sample_weight)
        self.model = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=self.num_boost_round,
#            verbose_eval=False,
        )

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict(X)             # already probability

    #
    def save(self, path):
        """Persist booster with joblib; auto-create parent dirs."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        return obj
