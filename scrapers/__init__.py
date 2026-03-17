"""
March Madness Predictor — Scrapers Package

Exports all scraper classes for convenient importing:
    from scrapers import KenPomScraper, ESPNScraper, SentimentScraper, OddsScraper
"""

from scrapers.kenpom_scraper import KenPomScraper
from scrapers.espn_scraper import ESPNScraper
from scrapers.sentiment_scraper import SentimentScraper
from scrapers.odds_scraper import OddsScraper

__all__ = [
    "KenPomScraper",
    "ESPNScraper",
    "SentimentScraper",
    "OddsScraper",
]
