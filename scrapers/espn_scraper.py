"""
ESPN Scraper — retrieves team stats, BPI, schedules, and rosters from ESPN.

Prefers ESPN's public JSON API endpoints where available, falling back to
HTML scraping when necessary.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ESPN API base URLs
# ---------------------------------------------------------------------------
API_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
STATS_URL = f"{API_BASE}/statistics"
BPI_URL = "https://site.web.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/powerindex"
SCOREBOARD_URL = f"{API_BASE}/scoreboard"
TEAMS_URL = f"{API_BASE}/teams"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

MAX_RETRIES = 3
BACKOFF_BASE = 2.0
REQUEST_DELAY = 1.0  # seconds between requests


class ESPNScraper:
    """Scraper for ESPN college basketball data."""

    def __init__(
        self,
        *,
        request_delay: float = REQUEST_DELAY,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.request_delay = request_delay
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._last_request_time: float = 0.0

        # Lazy-loaded team-name-to-id lookup
        self._team_id_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict:
        """GET *url* and return parsed JSON, with retries and backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                logger.debug("GET %s (attempt %d)", url, attempt)
                resp = self._session.get(url, params=params, timeout=30)
                self._last_request_time = time.monotonic()
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

    def _resolve_team_id(self, team_name: str) -> str | None:
        """Resolve a human-readable team name to an ESPN numeric team ID."""
        if not self._team_id_cache:
            self._build_team_id_cache()
        normalized = team_name.strip().lower()
        return self._team_id_cache.get(normalized)

    def _build_team_id_cache(self) -> None:
        """Fetch the ESPN teams list and populate the name -> id cache."""
        page = 1
        while True:
            data = self._get_json(TEAMS_URL, params={"page": page, "limit": 100})
            teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            if not teams:
                break
            for entry in teams:
                team = entry.get("team", {})
                tid = str(team.get("id", ""))
                for key in ("displayName", "shortDisplayName", "abbreviation", "name"):
                    val = team.get(key, "").strip().lower()
                    if val:
                        self._team_id_cache[val] = tid
            page += 1
            # ESPN typically returns all teams on the first page
            if page > 10:
                break
        logger.info("Cached %d ESPN team name variants.", len(self._team_id_cache))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scrape_team_stats(self, season: int) -> list[dict]:
        """Fetch team-level statistics for *season* from ESPN's stats API.

        Returns a list of dicts with offensive / defensive / overall stats.
        """
        params = {"season": season, "limit": 400}
        data = self._get_json(STATS_URL, params=params)

        rows: list[dict] = []
        categories = data.get("categories", data.get("resultSets", []))
        if isinstance(categories, list):
            # Structure varies by endpoint version — try the common shape
            for cat in categories:
                cat_name = cat.get("name", cat.get("displayName", ""))
                for team_entry in cat.get("leaders", cat.get("athletes", [])):
                    team_info = team_entry.get("team", {})
                    row = {
                        "team_id": str(team_info.get("id", "")),
                        "team_name": team_info.get("displayName", ""),
                        "stat_category": cat_name,
                        "stat_value": team_entry.get("value"),
                        "stat_display": team_entry.get("displayValue"),
                        "season": season,
                    }
                    rows.append(row)

        # If the above shape yielded nothing, try the flat team-stats shape
        if not rows:
            for team_block in data.get("teams", []):
                team_info = team_block.get("team", team_block)
                stats = team_block.get("statistics", team_block.get("stats", []))
                base = {
                    "team_id": str(team_info.get("id", "")),
                    "team_name": team_info.get("displayName", team_info.get("name", "")),
                    "season": season,
                }
                if isinstance(stats, list):
                    for stat in stats:
                        row = {
                            **base,
                            "stat_category": stat.get("name", ""),
                            "stat_value": stat.get("value"),
                            "stat_display": stat.get("displayValue"),
                        }
                        rows.append(row)
                elif isinstance(stats, dict):
                    for k, v in stats.items():
                        rows.append({**base, "stat_category": k, "stat_value": v})

        logger.info("Scraped %d stat entries for season %d", len(rows), season)
        return rows

    def scrape_bpi(self, season: int) -> list[dict]:
        """Fetch the Basketball Power Index rankings for *season*.

        Returns a list of dicts with keys:
            team_name, team_id, bpi_rank, bpi_value, off_bpi, def_bpi,
            strength_of_record, season
        """
        params = {"season": season, "limit": 400, "page": 1}
        all_rows: list[dict] = []

        while True:
            data = self._get_json(BPI_URL, params=params)
            teams = data.get("teams", [])
            if not teams:
                break

            for entry in teams:
                team = entry.get("team", {})
                stats = {
                    s.get("name", s.get("shortDisplayName", "")).lower(): s.get("value")
                    for s in entry.get("statistics", [])
                }
                row = {
                    "team_id": str(team.get("id", "")),
                    "team_name": team.get("displayName", ""),
                    "bpi_rank": entry.get("rank"),
                    "bpi_value": stats.get("bpi"),
                    "off_bpi": stats.get("obpi", stats.get("offbpi")),
                    "def_bpi": stats.get("dbpi", stats.get("defbpi")),
                    "strength_of_record": stats.get("sor", stats.get("strengthofrecord")),
                    "season": season,
                }
                all_rows.append(row)

            # Pagination
            pagination = data.get("pagination", {})
            if params["page"] >= pagination.get("pages", 1):
                break
            params["page"] += 1

        logger.info("Scraped %d BPI entries for season %d", len(all_rows), season)
        return all_rows

    def scrape_schedule(self, team_name: str, season: int) -> list[dict]:
        """Fetch game-by-game schedule/results for *team_name* in *season*.

        Returns a list of dicts per game:
            date, opponent, home_away, result, team_score, opp_score, location
        """
        team_id = self._resolve_team_id(team_name)
        if team_id is None:
            logger.error("Could not resolve team ID for '%s'", team_name)
            return []

        url = f"{TEAMS_URL}/{team_id}/schedule"
        data = self._get_json(url, params={"season": season})

        rows: list[dict] = []
        for event in data.get("events", []):
            competitions = event.get("competitions", [{}])
            comp = competitions[0] if competitions else {}
            competitors = comp.get("competitors", [])

            team_entry = None
            opp_entry = None
            for c in competitors:
                if str(c.get("id")) == team_id:
                    team_entry = c
                else:
                    opp_entry = c

            if team_entry is None or opp_entry is None:
                continue

            team_score = _safe_int(team_entry.get("score", {}).get("value",
                                   team_entry.get("score")))
            opp_score = _safe_int(opp_entry.get("score", {}).get("value",
                                  opp_entry.get("score")))

            row = {
                "date": event.get("date", ""),
                "opponent": (opp_entry.get("team", {}).get("displayName", "")),
                "home_away": team_entry.get("homeAway", ""),
                "result": "W" if (team_score is not None and opp_score is not None
                                  and team_score > opp_score) else "L",
                "team_score": team_score,
                "opp_score": opp_score,
                "location": comp.get("venue", {}).get("fullName", ""),
                "season": season,
            }
            rows.append(row)

        logger.info("Scraped %d games for %s (%d)", len(rows), team_name, season)
        return rows

    def scrape_roster(self, team_name: str, season: int) -> list[dict]:
        """Fetch the roster for *team_name* in *season*.

        Returns a list of dicts per player:
            player_name, jersey, position, height, weight, year, team_name
        """
        team_id = self._resolve_team_id(team_name)
        if team_id is None:
            logger.error("Could not resolve team ID for '%s'", team_name)
            return []

        url = f"{TEAMS_URL}/{team_id}/roster"
        data = self._get_json(url, params={"season": season})

        rows: list[dict] = []
        for athlete in data.get("athletes", []):
            row = {
                "player_name": athlete.get("displayName", ""),
                "jersey": athlete.get("jersey", ""),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "height": athlete.get("displayHeight", ""),
                "weight": athlete.get("displayWeight", ""),
                "year": athlete.get("experience", {}).get("displayValue", ""),
                "team_name": team_name,
                "season": season,
            }
            rows.append(row)

        logger.info("Scraped %d roster entries for %s (%d)", len(rows), team_name, season)
        return rows

    # ------------------------------------------------------------------
    # Mapping helper
    # ------------------------------------------------------------------
    @staticmethod
    def to_team_data_fields(raw: dict) -> dict:
        """Map a raw scraped dict to TeamData-compatible field names."""
        mapping = {
            "team_name": "team_name",
            "team_id": "espn_team_id",
            "bpi_rank": "bpi_rank",
            "bpi_value": "bpi",
            "off_bpi": "off_bpi",
            "def_bpi": "def_bpi",
            "strength_of_record": "strength_of_record",
            "season": "season",
            "stat_category": "stat_category",
            "stat_value": "stat_value",
        }
        return {mapping[k]: v for k, v in raw.items() if k in mapping}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _safe_int(value: Any) -> int | None:
    """Convert *value* to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None
