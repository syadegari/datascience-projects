import numpy as np


class BaseModel:
    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError
