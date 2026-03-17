"""Basic team statistics feature extraction.

Extracts win percentages, scoring margins, and offensive/defensive balance
from raw TeamData, then normalizes all values to the 0-1 range.
"""

from __future__ import annotations

from data.schema import TeamData


# Normalization boundaries derived from historical D1 ranges.
_NORM_RANGES: dict[str, tuple[float, float]] = {
    "win_pct":               (0.0, 1.0),
    "conference_win_pct":    (0.0, 1.0),
    "road_win_pct":          (0.0, 1.0),
    "quad1_win_pct":         (0.0, 1.0),
    "quad2_win_pct":         (0.0, 1.0),
    "quad3_win_pct":         (0.0, 1.0),
    "quad4_win_pct":         (0.0, 1.0),
    "scoring_margin":        (-30.0, 30.0),
    "points_per_game":       (50.0, 100.0),
    "points_allowed_pg":     (50.0, 100.0),
    "offensive_share":       (0.0, 1.0),
    "defensive_share":       (0.0, 1.0),
    "field_goal_pct":        (0.30, 0.60),
    "three_point_pct":       (0.20, 0.45),
    "free_throw_pct":        (0.55, 0.85),
    "assists_pg":            (8.0, 22.0),
    "turnovers_pg":          (8.0, 20.0),
    "assist_turnover_ratio": (0.4, 2.0),
    "seed":                  (1.0, 16.0),
}


def _safe_pct(wins: int, losses: int) -> float:
    """Return win percentage, or 0.0 when there are no games."""
    total = wins + losses
    return wins / total if total > 0 else 0.0


def _normalize(value: float, low: float, high: float) -> float:
    """Clamp *value* into [low, high] then scale to 0-1."""
    if high == low:
        return 0.5
    clamped = max(low, min(high, value))
    return (clamped - low) / (high - low)


class TeamFeatureExtractor:
    """Extracts and normalizes basic team statistics features.

    Every feature returned by :meth:`extract` is scaled to the 0-1 range so
    that downstream models receive comparable magnitudes without requiring a
    separate scaling step for this feature group.
    """

    # Ordered list of feature names produced by this extractor.
    FEATURE_NAMES: list[str] = list(_NORM_RANGES.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, team: TeamData) -> dict[str, float]:
        """Return a dict of normalized basic-stat features for *team*.

        Parameters
        ----------
        team:
            A fully populated :class:`TeamData` instance.

        Returns
        -------
        dict[str, float]
            Feature name -> value in [0, 1].
        """
        raw = self._extract_raw(team)
        return self._normalize_all(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_raw(team: TeamData) -> dict[str, float]:
        """Compute raw (un-normalized) feature values."""
        scoring_margin = team.points_per_game - team.points_allowed_pg

        # Offensive / defensive balance: what fraction of the team's quality
        # comes from offence vs. defence.  We measure "quality" as distance
        # from a league-average 70 ppg baseline.
        _BASELINE = 70.0
        off_quality = max(team.points_per_game - _BASELINE, 0.0)
        def_quality = max(_BASELINE - team.points_allowed_pg, 0.0)
        total_quality = off_quality + def_quality
        offensive_share = off_quality / total_quality if total_quality > 0 else 0.5
        defensive_share = 1.0 - offensive_share

        to_pg = team.turnovers_pg if team.turnovers_pg > 0 else 1.0
        assist_turnover_ratio = team.assists_pg / to_pg

        return {
            "win_pct":               team.win_pct(),
            "conference_win_pct":    team.conference_win_pct(),
            "road_win_pct":          _safe_pct(team.road_wins, team.road_losses),
            "quad1_win_pct":         _safe_pct(team.quad1_wins, team.quad1_losses),
            "quad2_win_pct":         _safe_pct(team.quad2_wins, team.quad2_losses),
            "quad3_win_pct":         _safe_pct(team.quad3_wins, team.quad3_losses),
            "quad4_win_pct":         _safe_pct(team.quad4_wins, team.quad4_losses),
            "scoring_margin":        scoring_margin,
            "points_per_game":       team.points_per_game,
            "points_allowed_pg":     team.points_allowed_pg,
            "offensive_share":       offensive_share,
            "defensive_share":       defensive_share,
            "field_goal_pct":        team.field_goal_pct,
            "three_point_pct":       team.three_point_pct,
            "free_throw_pct":        team.free_throw_pct,
            "assists_pg":            team.assists_pg,
            "turnovers_pg":          team.turnovers_pg,
            "assist_turnover_ratio": assist_turnover_ratio,
            "seed":                  float(team.seed),
        }

    @staticmethod
    def _normalize_all(raw: dict[str, float]) -> dict[str, float]:
        """Normalize every raw feature to [0, 1]."""
        normalized: dict[str, float] = {}
        for name, value in raw.items():
            lo, hi = _NORM_RANGES[name]
            normalized[name] = _normalize(value, lo, hi)
        return normalized
