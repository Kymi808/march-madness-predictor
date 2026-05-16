"""Abstract base class for all March Madness prediction models."""

from abc import ABC, abstractmethod
import hashlib
import hmac
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    classification_report,
)

from config import RANDOM_SEED, CV_FOLDS

# pickle.load is unsafe on untrusted input. We append an HMAC-SHA256 of the
# pickle payload to every saved file and verify it on load. The key is read
# from MARCH_MADNESS_MODEL_KEY (defaults to a constant for local dev — set a
# real per-user secret in your environment for any shared model store).
_HMAC_KEY = os.environ.get("MARCH_MADNESS_MODEL_KEY", "march-madness-local-dev").encode()
_HMAC_SIZE = 32  # SHA-256 digest length


class BaseMarchMadnessModel(ABC):
    """Abstract base class providing a standard sklearn-compatible interface
    for every model used in the March Madness prediction pipeline."""

    def __init__(self, name: str = "BaseModel", random_seed: int = RANDOM_SEED):
        self.name = name
        self.random_seed = random_seed
        self.model = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> "BaseMarchMadnessModel":
        """Fit the model on training data. Must set self.is_fitted = True."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates with shape (n_samples, 2)."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray | None:
        """Return feature importances as a 1-D array aligned with the input
        feature columns, or None if the model does not support importances."""
        ...

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model with cross-validation and held-out metrics.

        Returns a dict with accuracy, log_loss, roc_auc, cv_scores, and
        a full classification_report string.
        """
        self._check_fitted()

        preds = self.predict(X)
        probas = self.predict_proba(X)

        metrics: dict = {
            "accuracy": accuracy_score(y, preds),
            "log_loss": log_loss(y, probas[:, 1]),
            "roc_auc": roc_auc_score(y, probas[:, 1]),
            "classification_report": classification_report(y, preds),
        }

        # Cross-validated accuracy when an sklearn-compatible estimator exists
        if self.model is not None and hasattr(self.model, "predict"):
            cv = StratifiedKFold(
                n_splits=CV_FOLDS, shuffle=True, random_state=self.random_seed
            )
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
            metrics["cv_mean_accuracy"] = cv_scores.mean()
            metrics["cv_std_accuracy"] = cv_scores.std()
            metrics["cv_scores"] = cv_scores

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialize the entire model wrapper to *path* with an HMAC tag."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(self)
        tag = hmac.new(_HMAC_KEY, payload, hashlib.sha256).digest()
        with open(path, "wb") as f:
            f.write(tag)
            f.write(payload)

    @classmethod
    def load(cls, path: str | Path) -> "BaseMarchMadnessModel":
        """Deserialize a model wrapper previously saved with :meth:`save`.

        Verifies an HMAC tag before unpickling, so an attacker cannot trigger
        arbitrary code execution by swapping in a crafted file.
        """
        with open(path, "rb") as f:
            blob = f.read()
        if len(blob) < _HMAC_SIZE:
            raise ValueError(f"Model file {path} is too small to contain HMAC tag")
        tag, payload = blob[:_HMAC_SIZE], blob[_HMAC_SIZE:]
        expected = hmac.new(_HMAC_KEY, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError(
                f"HMAC verification failed for {path}. Refusing to unpickle "
                "potentially untrusted model file."
            )
        model = pickle.loads(payload)
        if not isinstance(model, BaseMarchMadnessModel):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, "
                "expected a BaseMarchMadnessModel subclass."
            )
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} has not been trained yet. Call train() first."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self.is_fitted})"
