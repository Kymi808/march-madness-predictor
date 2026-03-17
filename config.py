"""Configuration for March Madness Predictor."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys (set in .env)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
ESPN_API_KEY = os.getenv("ESPN_API_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Feature weights for ensemble
FEATURE_GROUPS = {
    "efficiency": 0.25,
    "rankings": 0.15,
    "momentum": 0.10,
    "experience": 0.10,
    "sentiment": 0.05,
    "betting": 0.10,
    "historical": 0.10,
    "matchup": 0.15,
}

# Model hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Tournament structure
NUM_TEAMS = 68
FIRST_FOUR_TEAMS = 4
ROUNDS = ["First Four", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]

CURRENT_SEASON = 2026
