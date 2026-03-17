"""
Sentiment Scraper — gathers public-sentiment signals from Twitter/X, Reddit,
and Google News RSS for college basketball teams.

Sentiment scoring uses both VADER (rule-based) and TextBlob (pattern-based)
to provide complementary measures.  Results include aggregate scores, volume
counts, and representative sample texts.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from xml.etree import ElementTree

import requests

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
except ImportError:  # graceful degradation
    VaderAnalyzer = None  # type: ignore[assignment,misc]

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
BACKOFF_BASE = 2.0
REQUEST_DELAY = 1.5  # seconds
MAX_SAMPLE_TEXTS = 10

REDDIT_SEARCH_URL = "https://oauth.reddit.com/r/{subreddit}/search.json"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


class SentimentScraper:
    """Aggregates public sentiment from Twitter/X, Reddit, and news sources."""

    def __init__(
        self,
        *,
        twitter_bearer_token: str | None = None,
        reddit_client_id: str | None = None,
        reddit_client_secret: str | None = None,
        reddit_user_agent: str = "march-madness-predictor/1.0",
        request_delay: float = REQUEST_DELAY,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        # Twitter / X
        self.twitter_bearer_token = twitter_bearer_token or os.getenv("TWITTER_BEARER_TOKEN")

        # Reddit
        self.reddit_client_id = reddit_client_id or os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = reddit_client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = reddit_user_agent

        self.request_delay = request_delay
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self.reddit_user_agent,
        })
        self._last_request_time: float = 0.0

        # Reddit OAuth token (lazily obtained)
        self._reddit_token: str | None = None
        self._reddit_token_expiry: float = 0.0

        # Sentiment analyzers
        self._vader = VaderAnalyzer() if VaderAnalyzer is not None else None
        if self._vader is None:
            logger.warning("vaderSentiment not installed — VADER scores will be unavailable.")
        if TextBlob is None:
            logger.warning("textblob not installed — TextBlob scores will be unavailable.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def _get(self, url: str, *, headers: dict | None = None,
             params: dict[str, Any] | None = None) -> requests.Response:
        """GET with retries and exponential backoff."""
        merged_headers = {**self._session.headers, **(headers or {})}
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                logger.debug("GET %s (attempt %d)", url, attempt)
                resp = self._session.get(url, headers=merged_headers, params=params, timeout=30)
                self._last_request_time = time.monotonic()

                # Handle rate-limit responses gracefully
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", BACKOFF_BASE ** attempt))
                    logger.warning("Rate limited by %s — sleeping %.1fs", url, retry_after)
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
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
    # Sentiment scoring
    # ------------------------------------------------------------------
    def _score_texts(self, texts: list[str]) -> dict:
        """Run VADER and TextBlob over *texts* and return aggregate scores."""
        vader_scores: list[float] = []
        textblob_polarities: list[float] = []
        textblob_subjectivities: list[float] = []

        for text in texts:
            if self._vader is not None:
                vs = self._vader.polarity_scores(text)
                vader_scores.append(vs["compound"])
            if TextBlob is not None:
                blob = TextBlob(text)
                textblob_polarities.append(blob.sentiment.polarity)
                textblob_subjectivities.append(blob.sentiment.subjectivity)

        def _avg(lst: list[float]) -> float | None:
            return round(sum(lst) / len(lst), 4) if lst else None

        return {
            "volume": len(texts),
            "vader_compound_avg": _avg(vader_scores),
            "vader_positive_pct": (
                round(sum(1 for s in vader_scores if s > 0.05) / len(vader_scores), 4)
                if vader_scores else None
            ),
            "vader_negative_pct": (
                round(sum(1 for s in vader_scores if s < -0.05) / len(vader_scores), 4)
                if vader_scores else None
            ),
            "textblob_polarity_avg": _avg(textblob_polarities),
            "textblob_subjectivity_avg": _avg(textblob_subjectivities),
            "sample_texts": texts[:MAX_SAMPLE_TEXTS],
        }

    # ------------------------------------------------------------------
    # Twitter / X
    # ------------------------------------------------------------------
    def scrape_twitter(self, team_name: str, days: int = 7) -> dict:
        """Search recent tweets mentioning *team_name* and score sentiment.

        Requires a valid Twitter API v2 bearer token.

        Returns a dict with sentiment scores, volume, and sample texts.
        """
        if not self.twitter_bearer_token:
            logger.error("Twitter bearer token not provided — cannot scrape Twitter.")
            return self._empty_result(team_name, "twitter")

        query = f'"{team_name}" college basketball -is:retweet lang:en'
        start_time = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        url = "https://api.twitter.com/2/tweets/search/recent"
        params: dict[str, Any] = {
            "query": query,
            "start_time": start_time,
            "max_results": 100,
            "tweet.fields": "created_at,public_metrics,text",
        }
        headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}

        texts: list[str] = []
        next_token: str | None = None

        # Paginate up to 5 pages (500 tweets max)
        for _ in range(5):
            if next_token:
                params["next_token"] = next_token
            try:
                resp = self._get(url, headers=headers, params=params)
                data = resp.json()
            except Exception as exc:
                logger.error("Twitter API error: %s", exc)
                break

            for tweet in data.get("data", []):
                texts.append(tweet.get("text", ""))

            next_token = data.get("meta", {}).get("next_token")
            if not next_token:
                break

        result = self._score_texts(texts)
        result["source"] = "twitter"
        result["team_name"] = team_name
        result["days_back"] = days
        logger.info("Twitter: scored %d tweets for '%s'", len(texts), team_name)
        return result

    # ------------------------------------------------------------------
    # Reddit
    # ------------------------------------------------------------------
    def _ensure_reddit_token(self) -> None:
        """Obtain or refresh an OAuth2 token for Reddit's API."""
        if self._reddit_token and time.monotonic() < self._reddit_token_expiry:
            return
        if not self.reddit_client_id or not self.reddit_client_secret:
            raise RuntimeError("Reddit client ID/secret not provided.")

        resp = requests.post(
            REDDIT_TOKEN_URL,
            auth=(self.reddit_client_id, self.reddit_client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": self.reddit_user_agent},
            timeout=30,
        )
        resp.raise_for_status()
        token_data = resp.json()
        self._reddit_token = token_data["access_token"]
        self._reddit_token_expiry = time.monotonic() + token_data.get("expires_in", 3600) - 60
        logger.info("Obtained Reddit OAuth token.")

    def scrape_reddit(
        self,
        team_name: str,
        subreddit: str = "CollegeBasketball",
    ) -> dict:
        """Search a subreddit for posts mentioning *team_name* and score sentiment.

        Returns a dict with sentiment scores, volume, and sample texts.
        """
        if not self.reddit_client_id or not self.reddit_client_secret:
            logger.error("Reddit credentials not provided — cannot scrape Reddit.")
            return self._empty_result(team_name, "reddit")

        try:
            self._ensure_reddit_token()
        except Exception as exc:
            logger.error("Failed to obtain Reddit token: %s", exc)
            return self._empty_result(team_name, "reddit")

        url = REDDIT_SEARCH_URL.format(subreddit=subreddit)
        headers = {"Authorization": f"Bearer {self._reddit_token}"}
        params: dict[str, Any] = {
            "q": team_name,
            "restrict_sr": "on",
            "sort": "new",
            "limit": 100,
            "t": "week",
        }

        texts: list[str] = []
        after: str | None = None

        # Paginate up to 3 pages
        for _ in range(3):
            if after:
                params["after"] = after
            try:
                resp = self._get(url, headers=headers, params=params)
                data = resp.json()
            except Exception as exc:
                logger.error("Reddit API error: %s", exc)
                break

            posts = data.get("data", {}).get("children", [])
            for post in posts:
                pd = post.get("data", {})
                title = pd.get("title", "")
                selftext = pd.get("selftext", "")
                texts.append(f"{title} {selftext}".strip())

            after = data.get("data", {}).get("after")
            if not after:
                break

        result = self._score_texts(texts)
        result["source"] = "reddit"
        result["subreddit"] = subreddit
        result["team_name"] = team_name
        logger.info("Reddit: scored %d posts for '%s' in r/%s", len(texts), team_name, subreddit)
        return result

    # ------------------------------------------------------------------
    # Google News RSS
    # ------------------------------------------------------------------
    def scrape_news(self, team_name: str) -> dict:
        """Fetch recent news headlines via Google News RSS and score sentiment.

        Returns a dict with sentiment scores, volume, and sample texts.
        """
        query = f"{team_name} college basketball"
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}

        try:
            resp = self._get(GOOGLE_NEWS_RSS, params=params)
        except Exception as exc:
            logger.error("Google News RSS error: %s", exc)
            return self._empty_result(team_name, "news")

        texts: list[str] = []
        try:
            root = ElementTree.fromstring(resp.content)
            for item in root.iter("item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    texts.append(title_el.text)
                desc_el = item.find("description")
                if desc_el is not None and desc_el.text:
                    texts.append(desc_el.text)
        except ElementTree.ParseError as exc:
            logger.error("Failed to parse Google News RSS XML: %s", exc)

        result = self._score_texts(texts)
        result["source"] = "news"
        result["team_name"] = team_name
        logger.info("News: scored %d items for '%s'", len(texts), team_name)
        return result

    # ------------------------------------------------------------------
    # Mapping helper
    # ------------------------------------------------------------------
    @staticmethod
    def to_team_data_fields(raw: dict) -> dict:
        """Map a raw sentiment result dict to TeamData-compatible field names."""
        mapping = {
            "team_name": "team_name",
            "source": "sentiment_source",
            "volume": "sentiment_volume",
            "vader_compound_avg": "sentiment_vader_compound",
            "vader_positive_pct": "sentiment_positive_pct",
            "vader_negative_pct": "sentiment_negative_pct",
            "textblob_polarity_avg": "sentiment_polarity",
            "textblob_subjectivity_avg": "sentiment_subjectivity",
        }
        return {mapping[k]: v for k, v in raw.items() if k in mapping}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_result(team_name: str, source: str) -> dict:
        """Return a skeleton result when scraping is not possible."""
        return {
            "team_name": team_name,
            "source": source,
            "volume": 0,
            "vader_compound_avg": None,
            "vader_positive_pct": None,
            "vader_negative_pct": None,
            "textblob_polarity_avg": None,
            "textblob_subjectivity_avg": None,
            "sample_texts": [],
        }
