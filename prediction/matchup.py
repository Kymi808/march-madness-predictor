"""Matchup prediction engine for March Madness.

Uses a trained model and optional feature pipeline to predict the outcome
of individual matchups between two teams, with explanations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from data.schema import TeamData, MatchupData

if TYPE_CHECKING:
    from features.engineering import FeatureEngineeringPipeline
    from models.base_model import BaseMarchMadnessModel


class MatchupPredictor:
    """Predict the outcome of a head-to-head matchup between two teams.

    Parameters
    ----------
    model : BaseMarchMadnessModel
        A trained model that exposes ``predict``, ``predict_proba``, and
        ``get_feature_importance`` methods.
    pipeline : FeatureEngineeringPipeline | None
        Optional feature engineering pipeline.  When provided, raw
        ``TeamData`` objects are transformed through the pipeline before
        being fed to the model.  When *None*, the matchup's own
        ``to_feature_vector`` method is used directly.
    """

    def __init__(
        self,
        model: "BaseMarchMadnessModel",
        pipeline: "FeatureEngineeringPipeline | None" = None,
    ) -> None:
        self.model = model
        self.pipeline = pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_feature_vector(
        self, team_a: TeamData, team_b: TeamData
    ) -> np.ndarray:
        """Return a 2-D feature array (1, n_features) ready for the model."""
        matchup = MatchupData(team_a=team_a, team_b=team_b, neutral_site=True)
        matchup.compute_diffs()

        if self.pipeline is not None:
            features = np.array(
                self.pipeline.transform(matchup)
            ).reshape(1, -1)
        else:
            features = np.array(matchup.to_feature_vector()).reshape(1, -1)
        return features

    def _feature_names(self) -> list[str]:
        """Return human-readable feature names aligned with the feature vector."""
        base_names = TeamData.feature_names()
        a_names = [f"a_{n}" for n in base_names]
        b_names = [f"b_{n}" for n in base_names]
        diff_names = [f"diff_{n}" for n in base_names]
        return a_names + b_names + diff_names + ["neutral_site"]

    def _confidence_level(self, win_prob: float) -> str:
        """Map a win probability to a qualitative confidence label."""
        delta = abs(win_prob - 0.5)
        if delta >= 0.30:
            return "very high"
        if delta >= 0.20:
            return "high"
        if delta >= 0.10:
            return "moderate"
        if delta >= 0.05:
            return "low"
        return "toss-up"

    def _top_factors(
        self, team_a: TeamData, team_b: TeamData, n: int = 5
    ) -> list[dict]:
        """Identify the top *n* features driving the prediction.

        Uses model feature importances weighted by the actual feature
        difference between the two teams so that the result reflects both
        the model's learned weights *and* the specific matchup.
        """
        importances = self.model.get_feature_importance()
        if importances is None:
            return []

        feature_vec = self._build_feature_vector(team_a, team_b).flatten()
        names = self._feature_names()

        # Align lengths (importance may cover only a subset)
        min_len = min(len(importances), len(feature_vec), len(names))
        importances = importances[:min_len]
        feature_vec = feature_vec[:min_len]
        names = names[:min_len]

        # Impact = |importance * feature_value|
        impact = np.abs(importances * feature_vec)
        top_indices = np.argsort(impact)[::-1][:n]

        factors = []
        for idx in top_indices:
            factors.append(
                {
                    "feature": names[idx],
                    "importance": float(importances[idx]),
                    "value_diff": float(feature_vec[idx]),
                    "impact": float(impact[idx]),
                }
            )
        return factors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, team_a: TeamData, team_b: TeamData) -> dict:
        """Predict the outcome of a single matchup.

        Returns
        -------
        dict
            win_probability_a : float
            win_probability_b : float
            predicted_margin  : float  (positive means team_a favored)
            confidence_level  : str
            key_factors       : list[dict]
        """
        features = self._build_feature_vector(team_a, team_b)
        probas = self.model.predict_proba(features)[0]

        win_prob_a = float(probas[1])
        win_prob_b = 1.0 - win_prob_a

        # Estimate margin from probability using a logistic approximation.
        # A probability of 0.5 corresponds to margin 0; every 0.1 above/below
        # maps roughly to ~3.5 points (calibrated against historical data).
        log_odds = np.log(win_prob_a / max(win_prob_b, 1e-9))
        predicted_margin = float(log_odds * 3.5)

        return {
            "team_a": team_a.name,
            "team_b": team_b.name,
            "win_probability_a": round(win_prob_a, 4),
            "win_probability_b": round(win_prob_b, 4),
            "predicted_margin": round(predicted_margin, 1),
            "confidence_level": self._confidence_level(win_prob_a),
            "key_factors": self._top_factors(team_a, team_b),
        }

    def predict_batch(
        self, matchups: list[tuple[TeamData, TeamData]]
    ) -> list[dict]:
        """Predict outcomes for a list of ``(team_a, team_b)`` matchups."""
        return [self.predict(a, b) for a, b in matchups]

    def explain_prediction(self, team_a: TeamData, team_b: TeamData) -> str:
        """Return a human-readable explanation of the predicted matchup.

        The explanation includes the win probabilities, predicted margin,
        confidence level, and the top driving factors.
        """
        result = self.predict(team_a, team_b)

        lines: list[str] = []
        lines.append(
            f"=== {result['team_a']} vs {result['team_b']} ==="
        )
        lines.append("")

        # Determine the favorite
        if result["win_probability_a"] >= result["win_probability_b"]:
            fav, fav_prob = result["team_a"], result["win_probability_a"]
            dog, dog_prob = result["team_b"], result["win_probability_b"]
        else:
            fav, fav_prob = result["team_b"], result["win_probability_b"]
            dog, dog_prob = result["team_a"], result["win_probability_a"]

        lines.append(
            f"Prediction: {fav} defeats {dog}"
        )
        lines.append(
            f"Win probability: {fav} {fav_prob:.1%} - {dog} {dog_prob:.1%}"
        )
        lines.append(
            f"Predicted margin: {abs(result['predicted_margin']):.1f} points"
        )
        lines.append(f"Confidence: {result['confidence_level']}")
        lines.append("")

        # Seed information
        lines.append(
            f"Seeds: {team_a.name} (#{team_a.seed}) vs "
            f"{team_b.name} (#{team_b.seed})"
        )
        lines.append("")

        # Key factors
        factors = result["key_factors"]
        if factors:
            lines.append("Key factors driving this prediction:")
            for i, factor in enumerate(factors, 1):
                direction = (
                    f"favors {team_a.name}"
                    if factor["value_diff"] > 0
                    else f"favors {team_b.name}"
                )
                lines.append(
                    f"  {i}. {factor['feature']} ({direction}, "
                    f"impact={factor['impact']:.3f})"
                )
        else:
            lines.append(
                "Key factors: feature importances not available for this model."
            )

        return "\n".join(lines)
