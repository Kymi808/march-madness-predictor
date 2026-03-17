"""Sentiment and public-perception feature extraction.

Combines social media sentiment signals, expert vs. public pick divergence,
social volume metrics, and a composite hype / contrarian indicator.
"""

from __future__ import annotations

import math

from data.schema import TeamData
from config import FEATURE_GROUPS


# ── Weights for the composite sentiment score ─────────────────────────────
_SENTIMENT_WEIGHTS = {
    "twitter":  0.35,
    "reddit":   0.30,
    "news":     0.35,
}

# Sentiment features group weight from config (used to gate extraction).
_GROUP_WEIGHT = FEATURE_GROUPS.get("sentiment", 0.05)

# Volume thresholds for normalization (relative scale).
_MIN_VOLUME = 0.0
_MAX_VOLUME = 100.0   # Assume social_media_volume is pre-scaled to ~0-100

# Hype index sub-weights.
_HYPE_SENTIMENT_W = 0.35
_HYPE_VOLUME_W = 0.30
_HYPE_TRAJECTORY_W = 0.35


class SentimentFeatureExtractor:
    """Derives sentiment-based features from :class:`TeamData`.

    Features capture the *qualitative* perception of a team — how much
    attention it receives, whether the crowd is bullish or bearish, and
    whether popular opinion diverges from the statistical picture.
    """

    FEATURE_NAMES: list[str] = [
        "composite_sentiment",
        "twitter_sentiment",
        "reddit_sentiment",
        "news_sentiment",
        "public_expert_divergence",
        "normalized_volume",
        "hype_index",
        "contrarian_indicator",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, team: TeamData, statistical_win_prob: float = 0.5) -> dict[str, float]:
        """Return sentiment features for *team*.

        Parameters
        ----------
        team:
            A fully populated :class:`TeamData` instance.
        statistical_win_prob:
            The model's current statistical estimate of this team's
            probability of advancing.  Used to compute the contrarian
            indicator (divergence between public sentiment and stats).

        Returns
        -------
        dict[str, float]
            Feature name -> value.
        """
        features: dict[str, float] = {}

        # ── Raw sentiments (pass-through, already in -1..1) ───────────
        features["twitter_sentiment"] = team.twitter_sentiment
        features["reddit_sentiment"] = team.reddit_sentiment
        features["news_sentiment"] = team.news_sentiment

        # ── Composite sentiment ───────────────────────────────────────
        features["composite_sentiment"] = self._composite_sentiment(team)

        # ── Public vs expert pick divergence ──────────────────────────
        features["public_expert_divergence"] = self._public_expert_divergence(team)

        # ── Normalized social media volume ────────────────────────────
        features["normalized_volume"] = self._normalized_volume(team)

        # ── Hype index ────────────────────────────────────────────────
        features["hype_index"] = self._hype_index(team, features)

        # ── Contrarian indicator ──────────────────────────────────────
        features["contrarian_indicator"] = self._contrarian_indicator(
            team, statistical_win_prob,
        )

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_sentiment(team: TeamData) -> float:
        """Weighted average of Twitter, Reddit, and news sentiment.

        Result is in the range [-1, 1].
        """
        return (
            _SENTIMENT_WEIGHTS["twitter"] * team.twitter_sentiment
            + _SENTIMENT_WEIGHTS["reddit"] * team.reddit_sentiment
            + _SENTIMENT_WEIGHTS["news"] * team.news_sentiment
        )

    @staticmethod
    def _public_expert_divergence(team: TeamData) -> float:
        """Signed difference: public pick% minus expert pick%.

        Positive means the public is *more* bullish than experts; negative
        means experts are higher on the team than the public.  The value is
        in roughly [-1, 1] (pick percentages are 0-1).
        """
        return team.public_pick_pct - team.expert_pick_pct

    @staticmethod
    def _normalized_volume(team: TeamData) -> float:
        """Normalize social_media_volume to [0, 1]."""
        vol = team.social_media_volume
        if vol <= _MIN_VOLUME:
            return 0.0
        if vol >= _MAX_VOLUME:
            return 1.0
        return (vol - _MIN_VOLUME) / (_MAX_VOLUME - _MIN_VOLUME)

    @classmethod
    def _hype_index(cls, team: TeamData, features: dict[str, float]) -> float:
        """Composite hype score combining sentiment, volume, and trajectory.

        The hype index captures how much *buzz* surrounds a team and whether
        that buzz is trending upward.  It is designed so that a team with
        strong positive sentiment, high volume, and an improving ranking
        trajectory scores close to 1.0, while a forgotten / declining team
        scores close to 0.0.

        Result is clamped to [0, 1].
        """
        # Shift composite sentiment from [-1, 1] to [0, 1].
        sent_01 = (features.get("composite_sentiment", 0.0) + 1.0) / 2.0

        vol_01 = features.get("normalized_volume", 0.0)

        # ranking_trajectory: positive = improving.  Map via sigmoid to [0, 1].
        traj_raw = team.ranking_trajectory
        traj_01 = 1.0 / (1.0 + math.exp(-traj_raw)) if abs(traj_raw) < 500 else (1.0 if traj_raw > 0 else 0.0)

        hype = (
            _HYPE_SENTIMENT_W * sent_01
            + _HYPE_VOLUME_W * vol_01
            + _HYPE_TRAJECTORY_W * traj_01
        )
        return max(0.0, min(1.0, hype))

    @staticmethod
    def _contrarian_indicator(team: TeamData, statistical_win_prob: float) -> float:
        """Measure divergence between public sentiment and statistical prediction.

        A large *positive* value means the public is much more optimistic
        than the model (potential "fade the public" signal).  A large
        *negative* value means the public is lower on the team than the
        stats suggest (potential value pick).

        Calculation:
            sentiment_prob = (composite_sentiment + 1) / 2   -> [0, 1]
            contrarian = sentiment_prob - statistical_win_prob

        Result is in roughly [-1, 1].
        """
        composite = (
            _SENTIMENT_WEIGHTS["twitter"] * team.twitter_sentiment
            + _SENTIMENT_WEIGHTS["reddit"] * team.reddit_sentiment
            + _SENTIMENT_WEIGHTS["news"] * team.news_sentiment
        )
        sentiment_prob = (composite + 1.0) / 2.0
        return sentiment_prob - statistical_win_prob
