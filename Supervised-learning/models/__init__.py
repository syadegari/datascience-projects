from .sklearn_wrappers import SkLogistic
from .lightgbm_wrapper import LGBMClassifier

MODEL_REGISTRY = {"logreg": SkLogistic, "lightgbm": LGBMClassifier}
