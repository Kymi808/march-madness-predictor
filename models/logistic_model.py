"""Logistic regression model with GridSearchCV hyperparameter tuning."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import RANDOM_SEED, CV_FOLDS
from models.base_model import BaseMarchMadnessModel


class LogisticModel(BaseMarchMadnessModel):
    """Logistic regression wrapped in a pipeline with standard scaling and
    hyperparameter search via :class:`~sklearn.model_selection.GridSearchCV`."""

    DEFAULT_PARAM_GRID = {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["saga"],
        "classifier__max_iter": [5000],
    }

    def __init__(
        self,
        param_grid: dict | None = None,
        random_seed: int = RANDOM_SEED,
        cv_folds: int = CV_FOLDS,
    ):
        super().__init__(name="LogisticModel", random_seed=random_seed)
        self.cv_folds = cv_folds
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.best_params_: dict | None = None

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=self.random_seed)),
        ])

    def train(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )
        grid = GridSearchCV(
            estimator=self._pipeline,
            param_grid=self.param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X, y)
        self.model = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray | None:
        self._check_fitted()
        classifier = self.model.named_steps["classifier"]
        return np.abs(classifier.coef_).flatten()
