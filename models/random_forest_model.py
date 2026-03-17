"""Random forest model with tunable hyperparameters."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import RANDOM_SEED, CV_FOLDS
from models.base_model import BaseMarchMadnessModel


class RandomForestModel(BaseMarchMadnessModel):
    """Random forest classifier with optional grid-search tuning for
    ``n_estimators``, ``max_depth``, ``min_samples_split``, and related
    hyperparameters."""

    DEFAULT_PARAM_GRID = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    def __init__(
        self,
        param_grid: dict | None = None,
        random_seed: int = RANDOM_SEED,
        cv_folds: int = CV_FOLDS,
        tune: bool = True,
        **rf_kwargs,
    ):
        super().__init__(name="RandomForestModel", random_seed=random_seed)
        self.cv_folds = cv_folds
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.tune = tune
        self.rf_kwargs = rf_kwargs
        self.best_params_: dict | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        base_estimator = RandomForestClassifier(
            random_state=self.random_seed, n_jobs=-1, **self.rf_kwargs
        )

        if self.tune:
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            )
            grid = GridSearchCV(
                estimator=base_estimator,
                param_grid=self.param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X, y)
            self.model = grid.best_estimator_
            self.best_params_ = grid.best_params_
        else:
            base_estimator.fit(X, y)
            self.model = base_estimator

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
        return self.model.feature_importances_
