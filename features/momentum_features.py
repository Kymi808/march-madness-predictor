"""Momentum and recent-form feature extraction.

Captures how a team is performing *right now* rather than over the full
season: recent game windows, streaks, conference-tournament boosts, rest,
late-season trajectory, and clutch performance.
"""

from __future__ import annotations

import math

from data.schema import TeamData
from config import FEATURE_GROUPS


# ── Constants ──────────────────────────────────────────────────────────────
# Diminishing-returns curve for win streaks: impact = 1 - exp(-k * streak).
_STREAK_DECAY_K = 0.25

# Conference tournament result -> boost mapping.
_CONF_TOURNEY_BOOST: dict[str, float] = {
    "champion":   1.0,
    "runner-up":  0.65,
    "semifinal":  0.35,
    "quarterfinal": 0.15,
}

# Ideal rest days before a tournament game.
_IDEAL_REST_DAYS = 4
_MAX_REST_DAYS = 14   # Beyond this the rust factor kicks in.


class MomentumFeatureExtractor:
    """Extracts momentum / recent-form features from :class:`TeamData`."""

    FEATURE_NAMES: list[str] = [
        "last_5_win_pct",
        "last_10_win_pct",
        "last_5_margin",
        "last_10_margin",
        "win_streak_impact",
        "conf_tourney_boost",
        "rest_days_factor",
        "late_season_trajectory",
        "clutch_rating",
        "momentum_composite",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, team: TeamData) -> dict[str, float]:
        """Return momentum features for *team*.

        Parameters
        ----------
        team:
            A fully populated :class:`TeamData` instance.

        Returns
        -------
        dict[str, float]
            Feature name -> value.
        """
        features: dict[str, float] = {}

        # ── Recent form (last 5 / last 10) ────────────────────────────
        features["last_5_win_pct"] = self._last_n_win_pct(team, n=5)
        features["last_10_win_pct"] = self._last_n_win_pct(team, n=10)
        features["last_5_margin"] = team.last_5_margin
        features["last_10_margin"] = team.last_10_margin

        # ── Win streak (diminishing returns) ──────────────────────────
        features["win_streak_impact"] = self._win_streak_impact(team)

        # ── Conference tournament performance ─────────────────────────
        features["conf_tourney_boost"] = self._conf_tourney_boost(team)

        # ── Rest days ─────────────────────────────────────────────────
        features["rest_days_factor"] = self._rest_days_factor(team)

        # ── Late-season trajectory ────────────────────────────────────
        features["late_season_trajectory"] = self._late_season_trajectory(team)

        # ── Clutch performance ────────────────────────────────────────
        features["clutch_rating"] = self._clutch_rating(team)

        # ── Composite momentum score ──────────────────────────────────
        features["momentum_composite"] = self._momentum_composite(features)

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_n_win_pct(team: TeamData, n: int) -> float:
        """Win percentage over the last *n* games.

        TeamData stores ``last_10_wins`` / ``last_10_losses``.  For n=5 we
        approximate from the last-5 margin: positive margin -> likely above
        .500, negative -> below.  When exact fields are available for
        last_10 we use those directly.
        """
        if n == 10:
            total = team.last_10_wins + team.last_10_losses
            return team.last_10_wins / total if total > 0 else 0.5
        # n == 5: approximate from margin using a sigmoid.
        margin = team.last_5_margin
        return 1.0 / (1.0 + math.exp(-0.25 * margin))

    @staticmethod
    def _win_streak_impact(team: TeamData) -> float:
        """Map win/loss streak to a bounded impact score.

        Uses ``1 - exp(-k * |streak|)`` so that long streaks have
        diminishing additional impact.  Positive for win streaks, negative
        for losing streaks.  Result is in roughly [-1, 1].
        """
        streak = team.win_streak  # Negative value = losing streak.
        magnitude = 1.0 - math.exp(-_STREAK_DECAY_K * abs(streak))
        return magnitude if streak >= 0 else -magnitude

    @staticmethod
    def _conf_tourney_boost(team: TeamData) -> float:
        """Boost based on how far the team went in its conference tournament.

        Returns a value in [0, 1].
        """
        result = team.conference_tournament_result.strip().lower()
        return _CONF_TOURNEY_BOOST.get(result, 0.0)

    @staticmethod
    def _rest_days_factor(team: TeamData) -> float:
        """Score for rest heading into the tournament.

        Too little rest (fatigue) and too much rest (rust) are both
        penalised.  The ideal is ~4 days.  Result is in [0, 1].
        """
        days = team.days_since_last_game
        if days <= 0:
            return 0.5  # Unknown / same-day -> neutral.

        # Gaussian-like curve centred on ideal rest.
        sigma = 3.0
        factor = math.exp(-0.5 * ((days - _IDEAL_REST_DAYS) / sigma) ** 2)
        return factor

    @staticmethod
    def _late_season_trajectory(team: TeamData) -> float:
        """Measure whether a team is improving or declining late in the season.

        Compares last-5 margin to last-10 margin.  If the last 5 games are
        better than the last 10 (which includes those 5), the team is on an
        upswing.  Result is a signed value (positive = improving).
        """
        # Difference between short-window and long-window margins.
        return team.last_5_margin - team.last_10_margin

    @staticmethod
    def _clutch_rating(team: TeamData) -> float:
        """Rate a team's performance in close / clutch situations.

        Combines close-game win% and overtime record into a single 0-1
        score.
        """
        close_total = team.close_game_wins + team.close_game_losses
        close_pct = team.close_game_wins / close_total if close_total > 0 else 0.5

        ot_total = team.overtime_record_wins + team.overtime_record_losses
        ot_pct = team.overtime_record_wins / ot_total if ot_total > 0 else 0.5

        # Weight close-game performance more heavily (larger sample).
        return 0.75 * close_pct + 0.25 * ot_pct

    @staticmethod
    def _momentum_composite(features: dict[str, float]) -> float:
        """Blend momentum sub-features into a single score.

        Result is intentionally un-bounded so the pipeline scaler can
        normalise it alongside other feature groups.
        """
        return (
            0.20 * features["last_10_win_pct"]
            + 0.15 * features["last_5_win_pct"]
            + 0.10 * (features["last_10_margin"] / 10.0)   # Rough scaling
            + 0.10 * (features["last_5_margin"] / 10.0)
            + 0.15 * features["win_streak_impact"]
            + 0.10 * features["conf_tourney_boost"]
            + 0.05 * features["rest_days_factor"]
            + 0.05 * features["late_season_trajectory"] / 5.0
            + 0.10 * features["clutch_rating"]
        )
