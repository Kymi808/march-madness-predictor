"""
Odds Scraper — retrieves betting odds, spreads, totals, and public-betting
percentages for college basketball via the-odds-api.com.

Requires an API key, which can be passed to the constructor or set via the
THE_ODDS_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# the-odds-api.com endpoints
# ---------------------------------------------------------------------------
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_ncaab"

# Retry / rate-limit defaults
MAX_RETRIES = 3
BACKOFF_BASE = 2.0
REQUEST_DELAY = 1.0


class OddsScraper:
    """Scraper for college basketball betting lines and odds."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        request_delay: float = REQUEST_DELAY,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        if not self.api_key:
            logger.warning(
                "No API key for the-odds-api.com — most methods will fail. "
                "Set THE_ODDS_API_KEY or pass api_key to the constructor."
            )

        self.request_delay = request_delay
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "march-madness-predictor/1.0",
        })
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET *url*, return parsed JSON, with retries and backoff."""
        params = params or {}
        params.setdefault("apiKey", self.api_key)

        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                logger.debug("GET %s (attempt %d)", url, attempt)
                resp = self._session.get(url, params=params, timeout=30)
                self._last_request_time = time.monotonic()

                # Log quota usage from response headers
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                if remaining is not None:
                    logger.debug("Odds API quota: %s used, %s remaining", used, remaining)

                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", BACKOFF_BASE ** attempt))
                    logger.warning("Odds API rate limited — sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    url, attempt, self.max_retries, exc, wait,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(wait)

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts")

    # ------------------------------------------------------------------
    # Odds conversion utilities
    # ------------------------------------------------------------------
    @staticmethod
    def american_to_decimal(american: int | float) -> float:
        """Convert American odds (+150, -110, etc.) to decimal odds."""
        if american > 0:
            return round(american / 100 + 1, 4)
        else:
            return round(100 / abs(american) + 1, 4)

    @staticmethod
    def fractional_to_decimal(numerator: float, denominator: float) -> float:
        """Convert fractional odds (e.g., 3/1) to decimal odds."""
        if denominator == 0:
            return 0.0
        return round(numerator / denominator + 1, 4)

    @staticmethod
    def decimal_to_implied_probability(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability (0-1)."""
        if decimal_odds <= 0:
            return 0.0
        return round(1 / decimal_odds, 6)

    @staticmethod
    def american_to_implied_probability(american: int | float) -> float:
        """Convert American odds directly to implied probability (0-1)."""
        if american > 0:
            return round(100 / (american + 100), 6)
        else:
            return round(abs(american) / (abs(american) + 100), 6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scrape_championship_odds(self) -> list[dict]:
        """Fetch current NCAA championship futures odds.

        Returns a list of dicts per team:
            team_name, bookmaker, american_odds, decimal_odds,
            implied_probability
        """
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
        params = {
            "regions": "us",
            "markets": "outrights",
            "oddsFormat": "american",
        }

        try:
            data = self._get_json(url, params=params)
        except Exception as exc:
            logger.error("Failed to fetch championship odds: %s", exc)
            return []

        rows: list[dict] = []
        events = data if isinstance(data, list) else data.get("data", [])

        for event in events:
            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("title", bookmaker.get("key", ""))
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "outrights":
                        continue
                    for outcome in market.get("outcomes", []):
                        american = outcome.get("price", 0)
                        dec = self.american_to_decimal(american)
                        row = {
                            "team_name": outcome.get("name", ""),
                            "bookmaker": bk_name,
                            "american_odds": american,
                            "decimal_odds": dec,
                            "implied_probability": self.decimal_to_implied_probability(dec),
                        }
                        rows.append(row)

        logger.info("Scraped %d championship-odds entries", len(rows))
        return rows

    def scrape_game_lines(self, date: str) -> list[dict]:
        """Fetch point spreads and totals for games on *date* (YYYY-MM-DD).

        Returns a list of dicts per game/bookmaker:
            game_id, home_team, away_team, bookmaker,
            spread_home, spread_away, total,
            home_ml, away_ml (moneyline in American),
            home_ml_decimal, away_ml_decimal,
            home_ml_implied_prob, away_ml_implied_prob
        """
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
        params = {
            "regions": "us",
            "markets": "spreads,totals,h2h",
            "oddsFormat": "american",
            "commenceTimeFrom": f"{date}T00:00:00Z",
            "commenceTimeTo": f"{date}T23:59:59Z",
        }

        try:
            data = self._get_json(url, params=params)
        except Exception as exc:
            logger.error("Failed to fetch game lines for %s: %s", date, exc)
            return []

        rows: list[dict] = []
        events = data if isinstance(data, list) else data.get("data", [])

        for event in events:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            game_id = event.get("id", "")

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("title", bookmaker.get("key", ""))
                row: dict[str, Any] = {
                    "game_id": game_id,
                    "date": date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bk_name,
                }

                for market in bookmaker.get("markets", []):
                    mkey = market.get("key", "")
                    outcomes = {o.get("name", ""): o for o in market.get("outcomes", [])}

                    if mkey == "spreads":
                        home_sp = outcomes.get(home_team, {})
                        away_sp = outcomes.get(away_team, {})
                        row["spread_home"] = home_sp.get("point")
                        row["spread_away"] = away_sp.get("point")

                    elif mkey == "totals":
                        over = outcomes.get("Over", {})
                        row["total"] = over.get("point")

                    elif mkey == "h2h":
                        home_ml = outcomes.get(home_team, {}).get("price", 0)
                        away_ml = outcomes.get(away_team, {}).get("price", 0)
                        row["home_ml"] = home_ml
                        row["away_ml"] = away_ml
                        row["home_ml_decimal"] = self.american_to_decimal(home_ml) if home_ml else None
                        row["away_ml_decimal"] = self.american_to_decimal(away_ml) if away_ml else None
                        row["home_ml_implied_prob"] = (
                            self.american_to_implied_probability(home_ml) if home_ml else None
                        )
                        row["away_ml_implied_prob"] = (
                            self.american_to_implied_probability(away_ml) if away_ml else None
                        )

                rows.append(row)

        logger.info("Scraped %d game-line entries for %s", len(rows), date)
        return rows

    def scrape_public_betting(self, team_name: str) -> dict:
        """Fetch public betting percentages for *team_name*.

        Note: the-odds-api.com does not natively provide public betting %.
        This method fetches the latest odds for games involving *team_name*
        and derives a rough consensus from line movement across books.

        Returns a dict with:
            team_name, games (list of per-game dicts with consensus info)
        """
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
        params = {
            "regions": "us",
            "markets": "spreads,h2h",
            "oddsFormat": "american",
        }

        try:
            data = self._get_json(url, params=params)
        except Exception as exc:
            logger.error("Failed to fetch odds for public betting: %s", exc)
            return {"team_name": team_name, "games": []}

        events = data if isinstance(data, list) else data.get("data", [])
        team_lower = team_name.strip().lower()
        games: list[dict] = []

        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if team_lower not in home.lower() and team_lower not in away.lower():
                continue

            spreads: list[float] = []
            moneylines: list[float] = []

            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    outcomes = {o.get("name", ""): o for o in market.get("outcomes", [])}
                    team_outcome = outcomes.get(home if team_lower in home.lower() else away, {})

                    if market.get("key") == "spreads" and team_outcome.get("point") is not None:
                        spreads.append(float(team_outcome["point"]))
                    if market.get("key") == "h2h" and team_outcome.get("price") is not None:
                        moneylines.append(float(team_outcome["price"]))

            avg_spread = round(sum(spreads) / len(spreads), 2) if spreads else None
            avg_ml = round(sum(moneylines) / len(moneylines), 1) if moneylines else None
            implied_prob = (
                self.american_to_implied_probability(avg_ml) if avg_ml is not None else None
            )

            games.append({
                "opponent": away if team_lower in home.lower() else home,
                "is_home": team_lower in home.lower(),
                "consensus_spread": avg_spread,
                "consensus_moneyline": avg_ml,
                "implied_win_probability": implied_prob,
                "num_bookmakers": len(event.get("bookmakers", [])),
            })

        logger.info("Public betting: found %d games for '%s'", len(games), team_name)
        return {"team_name": team_name, "games": games}

    # ------------------------------------------------------------------
    # Mapping helper
    # ------------------------------------------------------------------
    @staticmethod
    def to_team_data_fields(raw: dict) -> dict:
        """Map a raw odds dict to TeamData-compatible field names."""
        mapping = {
            "team_name": "team_name",
            "american_odds": "championship_odds_american",
            "decimal_odds": "championship_odds_decimal",
            "implied_probability": "championship_implied_prob",
            "consensus_spread": "consensus_spread",
            "consensus_moneyline": "consensus_moneyline",
            "implied_win_probability": "implied_win_prob",
            "bookmaker": "odds_bookmaker",
            "date": "odds_date",
            "home_team": "home_team",
            "away_team": "away_team",
            "spread_home": "spread_home",
            "total": "total",
        }
        return {mapping[k]: v for k, v in raw.items() if k in mapping}
