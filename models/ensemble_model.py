"""Ensemble model combining multiple BaseMarchMadnessModel instances."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report

from config import RANDOM_SEED, CV_FOLDS
from models.base_model import BaseMarchMadnessModel


class EnsembleModel(BaseMarchMadnessModel):
    """Combines predictions from several trained
    :class:`BaseMarchMadnessModel` instances.

    Supported strategies
    --------------------
    * **weighted_average** (default) -- probability-weighted average across
      sub-models.  Weights can be uniform or optimised via
      :meth:`optimize_weights`.
    * **voting** -- hard majority vote.
    * **stacking** -- a logistic-regression meta-learner trained on the
      sub-model probability outputs.
    """

    STRATEGIES = ("weighted_average", "voting", "stacking")

    def __init__(
        self,
        models: Sequence[BaseMarchMadnessModel],
        strategy: Literal["weighted_average", "voting", "stacking"] = "weighted_average",
        weights: np.ndarray | Sequence[float] | None = None,
        random_seed: int = RANDOM_SEED,
        cv_folds: int = CV_FOLDS,
    ):
        super().__init__(name="EnsembleModel", random_seed=random_seed)

        if not models:
            raise ValueError("At least one sub-model is required.")
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {self.STRATEGIES}."
            )

        self.models = list(models)
        self.strategy = strategy
        self.cv_folds = cv_folds

        # Weights (only meaningful for weighted_average)
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)

        # Meta-learner used when strategy == "stacking"
        self.meta_learner: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """For an ensemble the sub-models are assumed to be *already trained*.

        If the strategy is ``stacking``, this method trains the meta-learner
        on the stacked probability outputs of the sub-models.

        If the strategy is ``weighted_average``, this method optimises the
        blending weights on the supplied data.
        """
        for m in self.models:
            if not m.is_fitted:
                raise RuntimeError(
                    f"Sub-model {m.name!r} is not fitted. "
                    "Train all sub-models before creating the ensemble."
                )

        if self.strategy == "stacking":
            meta_X = self._build_meta_features(X)
            self.meta_learner = LogisticRegression(
                random_state=self.random_seed, max_iter=5000
            )
            self.meta_learner.fit(meta_X, y)
        elif self.strategy == "weighted_average":
            self.optimize_weights(X, y)

        self.model = self  # satisfy base-class convention
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.strategy == "weighted_average":
            return self._predict_weighted_average(X)
        elif self.strategy == "voting":
            return self._predict_voting(X)
        elif self.strategy == "stacking":
            return self._predict_stacking(X)
        raise ValueError(f"Unknown strategy {self.strategy!r}.")

    def _predict_weighted_average(self, X: np.ndarray) -> np.ndarray:
        probas = np.array([m.predict_proba(X) for m in self.models])  # (K, N, 2)
        weighted = np.tensordot(self.weights, probas, axes=([0], [0]))  # (N, 2)
        return weighted

    def _predict_voting(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])  # (K, N)
        # Majority vote encoded as pseudo-probabilities
        vote_1 = preds.mean(axis=0)  # fraction voting class 1
        return np.column_stack([1 - vote_1, vote_1])

    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        if self.meta_learner is None:
            raise RuntimeError(
                "Meta-learner has not been trained. Call train() first."
            )
        meta_X = self._build_meta_features(X)
        return self.meta_learner.predict_proba(meta_X)

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------

    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Find blending weights that minimise log-loss on the validation set
        using ``scipy.optimize.minimize`` with the SLSQP solver (constrained
        so weights sum to 1 and are non-negative)."""

        probas = np.array([m.predict_proba(X_val)[:, 1] for m in self.models])  # (K, N)

        def objective(w: np.ndarray) -> float:
            blended = w @ probas  # (N,)
            blended = np.clip(blended, 1e-15, 1 - 1e-15)
            return log_loss(y_val, blended)

        n = len(self.models)
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        self.weights = result.x
        return self.weights

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)

        metrics: dict = {
            "accuracy": accuracy_score(y, preds),
            "log_loss": log_loss(y, probas[:, 1]),
            "roc_auc": roc_auc_score(y, probas[:, 1]),
            "classification_report": classification_report(y, preds),
            "strategy": self.strategy,
            "weights": self.weights.tolist(),
        }

        # Cross-validation (re-blend per fold, sub-models stay fixed)
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )
        fold_accuracies = []
        for _, val_idx in cv.split(X, y):
            fold_preds = self.predict(X[val_idx])
            fold_accuracies.append(accuracy_score(y[val_idx], fold_preds))

        cv_scores = np.array(fold_accuracies)
        metrics["cv_mean_accuracy"] = cv_scores.mean()
        metrics["cv_std_accuracy"] = cv_scores.std()
        metrics["cv_scores"] = cv_scores
        return metrics

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> np.ndarray | None:
        """Weight-averaged feature importance across sub-models that support it."""
        importances = []
        weights_used = []
        for m, w in zip(self.models, self.weights):
            fi = m.get_feature_importance()
            if fi is not None:
                importances.append(fi)
                weights_used.append(w)

        if not importances:
            return None

        weights_arr = np.array(weights_used)
        weights_arr /= weights_arr.sum()
        stacked = np.array(importances)
        return (weights_arr[:, None] * stacked).sum(axis=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Stack class-1 probabilities from each sub-model as meta-features."""
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
