"""
KenPom Scraper — scrapes KenPom.com efficiency ratings and four-factors data.

KenPom requires a paid subscription for full access.  If credentials are
provided (via constructor or environment variables KENPOM_EMAIL / KENPOM_PASSWORD),
the scraper will authenticate before fetching pages.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://kenpom.com"
LOGIN_URL = f"{BASE_URL}/handler_login.php"
RATINGS_URL = f"{BASE_URL}/index.php"
FOUR_FACTORS_URL = f"{BASE_URL}/stats.php"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Retry / rate-limit defaults
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds
REQUEST_DELAY = 2.0  # polite delay between requests


class KenPomScraper:
    """Scraper for KenPom college basketball analytics."""

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        *,
        request_delay: float = REQUEST_DELAY,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.email = email or os.getenv("KENPOM_EMAIL")
        self.password = password or os.getenv("KENPOM_PASSWORD")
        self.request_delay = request_delay
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._last_request_time: float = 0.0
        self._logged_in = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def _login(self) -> None:
        """Authenticate with KenPom if credentials are available."""
        if self._logged_in:
            return
        if not self.email or not self.password:
            logger.info("No KenPom credentials provided; proceeding without login.")
            return

        logger.info("Logging in to KenPom as %s ...", self.email)
        payload = {"email": self.email, "password": self.password}
        resp = self._session.post(LOGIN_URL, data=payload, timeout=30)
        resp.raise_for_status()

        if "Incorrect password" in resp.text or "not found" in resp.text:
            raise RuntimeError("KenPom login failed — check credentials.")

        self._logged_in = True
        logger.info("KenPom login successful.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        """Enforce a minimum delay between outgoing requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def _fetch(self, url: str, params: dict[str, Any] | None = None) -> BeautifulSoup:
        """GET *url* with retries, exponential backoff, and rate limiting."""
        self._login()

        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                logger.debug("GET %s (attempt %d)", url, attempt)
                resp = self._session.get(url, params=params, timeout=30)
                self._last_request_time = time.monotonic()
                resp.raise_for_status()
                return BeautifulSoup(resp.text, "html.parser")
            except requests.RequestException as exc:
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    url, attempt, self.max_retries, exc, wait,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(wait)

        # Should never reach here, but just in case:
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scrape_ratings(self, season: int) -> list[dict]:
        """Scrape the main KenPom efficiency ratings table for *season*.

        Returns a list of dicts, one per team, with keys such as:
            team_name, rank, wins, losses, adj_efficiency_margin,
            adj_offensive_efficiency, adj_defensive_efficiency,
            adj_tempo, strength_of_schedule
        """
        soup = self._fetch(RATINGS_URL, params={"y": season})
        table = soup.find("table", id="ratings-table")
        if table is None:
            logger.error("Ratings table not found for season %d", season)
            return []

        rows: list[dict] = []
        tbody = table.find("tbody")
        if tbody is None:
            return []

        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 8:
                continue
            try:
                record = cells[2].get_text(strip=True).split("-")
                wins = int(record[0]) if len(record) >= 1 else 0
                losses = int(record[1]) if len(record) >= 2 else 0

                row = {
                    "rank": int(cells[0].get_text(strip=True)),
                    "team_name": cells[1].get_text(strip=True),
                    "wins": wins,
                    "losses": losses,
                    "adj_efficiency_margin": _to_float(cells[3]),
                    "adj_offensive_efficiency": _to_float(cells[4]),
                    "adj_defensive_efficiency": _to_float(cells[5]),
                    "adj_tempo": _to_float(cells[6]),
                    "strength_of_schedule": _to_float(cells[7]),
                    "season": season,
                }
                rows.append(row)
            except (ValueError, IndexError) as exc:
                logger.debug("Skipping malformed row: %s", exc)

        logger.info("Scraped %d team ratings for season %d", len(rows), season)
        return rows

    def scrape_four_factors(self, season: int) -> list[dict]:
        """Scrape KenPom four-factors stats for *season*.

        Returns a list of dicts with keys:
            team_name, off_efg_pct, off_turnover_pct, off_reb_pct,
            off_ft_rate, def_efg_pct, def_turnover_pct, def_reb_pct,
            def_ft_rate
        """
        soup = self._fetch(FOUR_FACTORS_URL, params={"y": season})
        table = soup.find("table", id="stats-table")
        if table is None:
            # Fall back to first table on the page
            table = soup.find("table")
        if table is None:
            logger.error("Four-factors table not found for season %d", season)
            return []

        rows: list[dict] = []
        tbody = table.find("tbody")
        if tbody is None:
            return []

        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 10:
                continue
            try:
                row = {
                    "team_name": cells[0].get_text(strip=True),
                    "off_efg_pct": _to_float(cells[1]),
                    "off_turnover_pct": _to_float(cells[2]),
                    "off_reb_pct": _to_float(cells[3]),
                    "off_ft_rate": _to_float(cells[4]),
                    "def_efg_pct": _to_float(cells[5]),
                    "def_turnover_pct": _to_float(cells[6]),
                    "def_reb_pct": _to_float(cells[7]),
                    "def_ft_rate": _to_float(cells[8]),
                    "season": season,
                }
                rows.append(row)
            except (ValueError, IndexError) as exc:
                logger.debug("Skipping malformed four-factors row: %s", exc)

        logger.info("Scraped %d four-factors rows for season %d", len(rows), season)
        return rows

    # ------------------------------------------------------------------
    # Mapping helper
    # ------------------------------------------------------------------
    @staticmethod
    def to_team_data_fields(raw: dict) -> dict:
        """Map a raw scraped dict to TeamData-compatible field names.

        Keys that do not map are silently dropped.
        """
        mapping = {
            "team_name": "team_name",
            "rank": "kenpom_rank",
            "wins": "wins",
            "losses": "losses",
            "adj_efficiency_margin": "adj_efficiency_margin",
            "adj_offensive_efficiency": "adj_offensive_efficiency",
            "adj_defensive_efficiency": "adj_defensive_efficiency",
            "adj_tempo": "adj_tempo",
            "strength_of_schedule": "strength_of_schedule",
            "off_efg_pct": "off_efg_pct",
            "off_turnover_pct": "off_turnover_pct",
            "off_reb_pct": "off_reb_pct",
            "off_ft_rate": "off_ft_rate",
            "def_efg_pct": "def_efg_pct",
            "def_turnover_pct": "def_turnover_pct",
            "def_reb_pct": "def_reb_pct",
            "def_ft_rate": "def_ft_rate",
            "season": "season",
        }
        return {mapping[k]: v for k, v in raw.items() if k in mapping}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _to_float(cell) -> float | None:
    """Extract text from a BeautifulSoup cell and convert to float."""
    text = cell.get_text(strip=True)
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None
