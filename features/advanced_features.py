"""Advanced basketball analytics feature extraction.

Computes Four Factors (offensive and defensive), luck adjustments,
tempo-adjusted metrics, quality-win ratios, and a composite power rating.
"""

from __future__ import annotations

import math

from data.schema import TeamData
from config import FEATURE_GROUPS


# ── Constants ──────────────────────────────────────────────────────────────
# League-average baselines (approximate D1 values) used for centering.
_AVG_EFG = 0.500
_AVG_TO_PCT = 18.0
_AVG_ORB_PCT = 28.0
_AVG_FT_RATE = 0.30
_AVG_TEMPO = 67.5
_AVG_EFFICIENCY = 100.0

# Weights inside the Four Factors composite (Dean Oliver proportions).
_FF_WEIGHTS = {
    "efg":     0.40,
    "to":      0.25,
    "orb":     0.20,
    "ft_rate": 0.15,
}


class AdvancedFeatureExtractor:
    """Extracts KenPom-style advanced metrics from :class:`TeamData`.

    All features are returned as raw floats (not normalized) so that the
    pipeline can apply a global scaler after combining feature groups.
    """

    FEATURE_NAMES: list[str] = [
        # Offensive Four Factors
        "off_efg_pct",
        "off_to_pct",
        "off_orb_pct",
        "off_ft_rate",
        "off_four_factors_composite",
        # Defensive Four Factors
        "def_opp_efg_pct",
        "def_opp_to_pct",
        "def_opp_orb_pct",
        "def_opp_ft_rate",
        "def_four_factors_composite",
        # Efficiency
        "adj_offensive_efficiency",
        "adj_defensive_efficiency",
        "adj_net_efficiency",
        # Luck
        "luck_factor",
        # Tempo-adjusted
        "tempo_adjusted_margin",
        "pace_factor",
        # Quality wins
        "quality_wins_ratio",
        "bad_losses_penalty",
        # Composite
        "composite_power_rating",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, team: TeamData) -> dict[str, float]:
        """Return advanced features for *team*.

        Parameters
        ----------
        team:
            A fully populated :class:`TeamData` instance.

        Returns
        -------
        dict[str, float]
            Feature name -> raw value.
        """
        features: dict[str, float] = {}

        # ── Offensive Four Factors ────────────────────────────────────
        features["off_efg_pct"] = team.effective_fg_pct
        features["off_to_pct"] = team.turnover_pct
        features["off_orb_pct"] = team.offensive_rebound_pct
        features["off_ft_rate"] = team.free_throw_rate
        features["off_four_factors_composite"] = self._four_factors_composite(
            efg=team.effective_fg_pct,
            to_pct=team.turnover_pct,
            orb_pct=team.offensive_rebound_pct,
            ft_rate=team.free_throw_rate,
            offensive=True,
        )

        # ── Defensive Four Factors ────────────────────────────────────
        features["def_opp_efg_pct"] = team.opponent_effective_fg_pct
        features["def_opp_to_pct"] = team.opponent_turnover_pct
        features["def_opp_orb_pct"] = team.opponent_offensive_rebound_pct
        features["def_opp_ft_rate"] = team.opponent_free_throw_rate
        features["def_four_factors_composite"] = self._four_factors_composite(
            efg=team.opponent_effective_fg_pct,
            to_pct=team.opponent_turnover_pct,
            orb_pct=team.opponent_offensive_rebound_pct,
            ft_rate=team.opponent_free_throw_rate,
            offensive=False,
        )

        # ── Efficiency ────────────────────────────────────────────────
        features["adj_offensive_efficiency"] = team.adj_offensive_efficiency
        features["adj_defensive_efficiency"] = team.adj_defensive_efficiency
        features["adj_net_efficiency"] = team.adj_net_efficiency

        # ── Luck factor ───────────────────────────────────────────────
        features["luck_factor"] = self._luck_factor(team)

        # ── Tempo-adjusted metrics ────────────────────────────────────
        features["tempo_adjusted_margin"] = self._tempo_adjusted_margin(team)
        features["pace_factor"] = team.adj_tempo / _AVG_TEMPO if _AVG_TEMPO else 1.0

        # ── Quality wins / bad losses ─────────────────────────────────
        features["quality_wins_ratio"] = self._quality_wins_ratio(team)
        features["bad_losses_penalty"] = self._bad_losses_penalty(team)

        # ── Composite power rating ────────────────────────────────────
        features["composite_power_rating"] = self._composite_power_rating(features)

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _four_factors_composite(
        efg: float,
        to_pct: float,
        orb_pct: float,
        ft_rate: float,
        offensive: bool,
    ) -> float:
        """Weighted composite of the Four Factors.

        For offence, higher eFG / ORB / FTRate and *lower* TO% are better.
        For defence, lower opponent eFG / ORB / FTRate and *higher* forced
        TO% are better.  The signs are flipped accordingly so that a higher
        composite always means "better".
        """
        efg_z = (efg - _AVG_EFG) / _AVG_EFG
        to_z = (to_pct - _AVG_TO_PCT) / _AVG_TO_PCT
        orb_z = (orb_pct - _AVG_ORB_PCT) / _AVG_ORB_PCT
        ft_z = (ft_rate - _AVG_FT_RATE) / _AVG_FT_RATE if _AVG_FT_RATE else 0.0

        if offensive:
            # Good offence: high eFG, low TO%, high ORB%, high FTRate
            score = (
                _FF_WEIGHTS["efg"] * efg_z
                - _FF_WEIGHTS["to"] * to_z
                + _FF_WEIGHTS["orb"] * orb_z
                + _FF_WEIGHTS["ft_rate"] * ft_z
            )
        else:
            # Good defence: low opponent eFG, high forced TO%, low opp ORB%, low opp FTRate
            score = (
                - _FF_WEIGHTS["efg"] * efg_z
                + _FF_WEIGHTS["to"] * to_z
                - _FF_WEIGHTS["orb"] * orb_z
                - _FF_WEIGHTS["ft_rate"] * ft_z
            )
        return score

    @staticmethod
    def _luck_factor(team: TeamData) -> float:
        """Estimate how 'lucky' a team has been.

        Luck = actual win% minus expected win% from adjusted efficiency
        (Pythagorean expectation).  Positive means the team has won more
        games than their efficiency profile would predict.
        """
        actual_wp = team.win_pct()

        oe = team.adj_offensive_efficiency
        de = team.adj_defensive_efficiency
        if oe <= 0 and de <= 0:
            return 0.0

        # Pythagorean expectation using an exponent of 10 (common for
        # tempo-free college basketball data).
        _EXP = 10.0
        try:
            expected_wp = (oe ** _EXP) / (oe ** _EXP + de ** _EXP)
        except (OverflowError, ZeroDivisionError):
            expected_wp = 0.5

        return actual_wp - expected_wp

    @staticmethod
    def _tempo_adjusted_margin(team: TeamData) -> float:
        """Scoring margin adjusted to a league-average tempo."""
        if team.adj_tempo <= 0:
            return team.scoring_margin
        pace_ratio = _AVG_TEMPO / team.adj_tempo
        return team.scoring_margin * pace_ratio

    @staticmethod
    def _quality_wins_ratio(team: TeamData) -> float:
        """Fraction of total wins that are Quad 1 or Quad 2 wins."""
        quality = team.quad1_wins + team.quad2_wins
        total = team.wins
        return quality / total if total > 0 else 0.0

    @staticmethod
    def _bad_losses_penalty(team: TeamData) -> float:
        """Penalty score for losses against weaker opponents.

        Quad 3 losses are penalised lightly; Quad 4 losses are penalised
        more heavily.  The result is a non-negative value where 0 means no
        bad losses and higher means worse.
        """
        total_games = team.wins + team.losses
        if total_games == 0:
            return 0.0
        penalty = (team.quad3_losses * 1.0 + team.quad4_losses * 2.0) / total_games
        return penalty

    @staticmethod
    def _composite_power_rating(features: dict[str, float]) -> float:
        """Combine advanced sub-features into a single power number.

        The composite blends:
          - Net efficiency (dominant signal)
          - Four Factors composites
          - Quality wins ratio (reward)
          - Bad losses penalty (punishment)
          - Luck discount (partially regress lucky teams toward mean)
        """
        net_eff = features.get("adj_net_efficiency", 0.0)
        off_ff = features.get("off_four_factors_composite", 0.0)
        def_ff = features.get("def_four_factors_composite", 0.0)
        quality = features.get("quality_wins_ratio", 0.0)
        bad = features.get("bad_losses_penalty", 0.0)
        luck = features.get("luck_factor", 0.0)

        # Scale each component so they contribute meaningfully.
        rating = (
            0.40 * net_eff
            + 0.15 * (off_ff * 100)   # Scale up from small z-score range
            + 0.15 * (def_ff * 100)
            + 0.10 * (quality * 30)    # Quality ratio -> ~0-30 range
            - 0.10 * (bad * 30)        # Penalty -> ~0-30 range
            - 0.10 * (luck * 20)       # Partially discount luck
        )
        return rating
