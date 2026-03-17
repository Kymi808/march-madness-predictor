"""Abstract base class for all March Madness prediction models."""

from abc import ABC, abstractmethod
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
        """Serialize the entire model wrapper to *path* using pickle."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaseMarchMadnessModel":
        """Deserialize a model wrapper previously saved with :meth:`save`."""
        with open(path, "rb") as f:
            model = pickle.load(f)
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
