"""XGBoost model with hyperparameter tuning via GridSearchCV."""

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from config import RANDOM_SEED, CV_FOLDS
from models.base_model import BaseMarchMadnessModel


class XGBoostModel(BaseMarchMadnessModel):
    """XGBClassifier with grid-search tuning over ``learning_rate``,
    ``max_depth``, ``n_estimators``, and other common knobs."""

    DEFAULT_PARAM_GRID = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "n_estimators": [100, 300, 500],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    def __init__(
        self,
        param_grid: dict | None = None,
        random_seed: int = RANDOM_SEED,
        cv_folds: int = CV_FOLDS,
        tune: bool = True,
        **xgb_kwargs,
    ):
        super().__init__(name="XGBoostModel", random_seed=random_seed)
        self.cv_folds = cv_folds
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.tune = tune
        self.xgb_kwargs = xgb_kwargs
        self.best_params_: dict | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        base_estimator = XGBClassifier(
            random_state=self.random_seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            **self.xgb_kwargs,
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
