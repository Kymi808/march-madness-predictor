"""Unified feature-engineering pipeline.

Combines all feature extractors, builds team and matchup feature vectors,
constructs training datasets, and provides scaling, selection, and
dimensionality-reduction utilities.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.schema import TeamData
from config import RANDOM_SEED, FEATURE_GROUPS

from features.team_features import TeamFeatureExtractor
from features.advanced_features import AdvancedFeatureExtractor
from features.sentiment_features import SentimentFeatureExtractor
from features.momentum_features import MomentumFeatureExtractor


class FeatureEngineeringPipeline:
    """End-to-end feature engineering for March Madness matchup prediction.

    Orchestrates the four specialised extractors, merges their outputs into
    a single vector, and provides scaling / selection / PCA utilities.

    Parameters
    ----------
    use_pca : bool
        If ``True``, apply PCA after scaling.  Default ``False``.
    pca_variance : float
        Fraction of variance to retain when ``use_pca`` is ``True``.
    n_select_features : int | None
        If set, apply feature selection (mutual information + RFE) to keep
        only the top *n* features.  ``None`` keeps all features.
    """

    def __init__(
        self,
        use_pca: bool = False,
        pca_variance: float = 0.95,
        n_select_features: int | None = None,
    ) -> None:
        # Extractors
        self._team_ext = TeamFeatureExtractor()
        self._adv_ext = AdvancedFeatureExtractor()
        self._sent_ext = SentimentFeatureExtractor()
        self._mom_ext = MomentumFeatureExtractor()

        # Scaling / selection / PCA
        self._scaler: StandardScaler = StandardScaler()
        self._pca: PCA | None = None
        self._selector: SelectKBest | RFE | None = None

        self._use_pca = use_pca
        self._pca_variance = pca_variance
        self._n_select_features = n_select_features

        # Set after fitting
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names (available after first extraction)."""
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Single-team features
    # ------------------------------------------------------------------

    def build_team_features(self, team: TeamData) -> np.ndarray:
        """Extract all features for a single team and return a 1-D array.

        Parameters
        ----------
        team:
            A fully populated :class:`TeamData`.

        Returns
        -------
        np.ndarray
            Shape ``(n_features,)``.
        """
        merged = self._extract_all(team)

        # Lazily record feature names on first call.
        if not self._feature_names:
            self._feature_names = list(merged.keys())

        return np.array(list(merged.values()), dtype=np.float64)

    # ------------------------------------------------------------------
    # Matchup features
    # ------------------------------------------------------------------

    def build_matchup_features(
        self,
        team_a: TeamData,
        team_b: TeamData,
    ) -> np.ndarray:
        """Build a feature vector for the *team_a* vs *team_b* matchup.

        The resulting vector contains:
        1. Team A's features
        2. Team B's features
        3. Differential features (A minus B)

        Parameters
        ----------
        team_a, team_b:
            Fully populated :class:`TeamData` instances.

        Returns
        -------
        np.ndarray
            Shape ``(3 * n_team_features,)``.
        """
        feats_a = self.build_team_features(team_a)
        feats_b = self.build_team_features(team_b)
        diff = feats_a - feats_b
        return np.concatenate([feats_a, feats_b, diff])

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        teams: dict[str, TeamData],
        historical_results: Sequence[tuple[str, str, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a full training dataset from historical matchups.

        Parameters
        ----------
        teams:
            Mapping of team name -> :class:`TeamData`.
        historical_results:
            Iterable of ``(team_a_name, team_b_name, label)`` where
            *label* is 1 if team_a won and 0 if team_b won.

        Returns
        -------
        X : np.ndarray
            Shape ``(n_matchups, n_features)``.  Scaled and optionally
            reduced.
        y : np.ndarray
            Shape ``(n_matchups,)`` with values in {0, 1}.
        """
        rows: list[np.ndarray] = []
        labels: list[int] = []

        for name_a, name_b, label in historical_results:
            team_a = teams.get(name_a)
            team_b = teams.get(name_b)
            if team_a is None or team_b is None:
                continue
            row = self.build_matchup_features(team_a, team_b)
            rows.append(row)
            labels.append(label)

        if not rows:
            return np.empty((0, 0)), np.empty((0,))

        X = np.vstack(rows)
        y = np.array(labels, dtype=np.int32)

        # Fit and apply transforms
        X = self._fit_transform(X, y)
        return X, y

    # ------------------------------------------------------------------
    # Transform (for inference on new matchups after fitting)
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler, selector, and PCA to new data.

        Parameters
        ----------
        X : np.ndarray
            Raw matchup feature matrix, shape ``(n, n_features)``.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.

        Raises
        ------
        RuntimeError
            If :meth:`build_dataset` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline has not been fitted. Call build_dataset() first."
            )
        X = self._scaler.transform(X)
        if self._selector is not None:
            X = self._selector.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        return X

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_all(self, team: TeamData) -> dict[str, float]:
        """Run all extractors and merge into a single ordered dict."""
        merged: dict[str, float] = {}

        # Basic team stats (already 0-1 normalized).
        team_feats = self._team_ext.extract(team)
        for k, v in team_feats.items():
            merged[f"team_{k}"] = v

        # Advanced metrics.
        adv_feats = self._adv_ext.extract(team)
        for k, v in adv_feats.items():
            merged[f"adv_{k}"] = v

        # Sentiment features.
        sent_feats = self._sent_ext.extract(team)
        for k, v in sent_feats.items():
            merged[f"sent_{k}"] = v

        # Momentum features.
        mom_feats = self._mom_ext.extract(team)
        for k, v in mom_feats.items():
            merged[f"mom_{k}"] = v

        return merged

    def _fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit scaler, optional selector, optional PCA, then transform X."""

        # 1. Standard scaling
        X = self._scaler.fit_transform(X)

        # 2. Feature selection (mutual information first pass, then RFE)
        if self._n_select_features is not None and self._n_select_features < X.shape[1]:
            # First pass: mutual information to pre-filter to 2x target.
            mi_k = min(self._n_select_features * 2, X.shape[1])
            mi_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=mi_k,
            )
            X_mi = mi_selector.fit_transform(X, y)

            # Second pass: recursive feature elimination to reach target.
            estimator = LogisticRegression(
                max_iter=500,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=self._n_select_features,
                step=1,
            )
            X = rfe.fit_transform(X_mi, y)

            # Store a combined selector for inference.  We wrap both steps
            # into a simple object that replays the two transforms.
            self._selector = _CombinedSelector(mi_selector, rfe)
        else:
            self._selector = None

        # 3. PCA
        if self._use_pca:
            self._pca = PCA(
                n_components=self._pca_variance,
                random_state=RANDOM_SEED,
            )
            X = self._pca.fit_transform(X)
        else:
            self._pca = None

        self._is_fitted = True
        return X


# ── Helper class ──────────────────────────────────────────────────────────


class _CombinedSelector:
    """Thin wrapper that replays a two-stage feature selection pipeline."""

    def __init__(self, first, second) -> None:
        self._first = first
        self._second = second

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._first.transform(X)
        X = self._second.transform(X)
        return X
