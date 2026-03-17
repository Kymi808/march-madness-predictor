# March Madness Predictor 2026

A comprehensive machine learning system for predicting the 2026 NCAA March Madness tournament. Uses multiple models, 100+ features per team, sentiment analysis, and Monte Carlo simulation to generate bracket predictions.

## Features

- **5 ML Models**: Logistic Regression, Random Forest, XGBoost, Neural Network, and a weighted Ensemble
- **100+ Features Per Team**: Efficiency metrics, Four Factors, rankings, momentum, coaching, player experience, sentiment, betting lines, clutch stats
- **Sentiment Analysis**: Twitter/X, Reddit, and news sentiment scoring with VADER and TextBlob
- **Live Data Scrapers**: KenPom, ESPN, betting odds, social media
- **Monte Carlo Simulation**: 10,000+ tournament simulations for robust probability estimates
- **Full 68-Team Bracket**: All teams with seeds, regions, and First Four matchups

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Predict a single matchup
python predict.py --team-a "Duke" --team-b "Vermont"

# Simulate the full bracket
python simulate_bracket.py --simulations 10000

# Train models on historical data
python train.py --model all --historical-seasons 2015-2025

# Show likely upsets
python simulate_bracket.py --show-upsets
```

## Project Structure

```
march-madness-predictor/
├── config.py                  # Configuration and hyperparameters
├── train.py                   # Model training CLI
├── predict.py                 # Prediction CLI
├── simulate_bracket.py        # Bracket simulation CLI
├── data/
│   ├── schema.py              # TeamData, MatchupData, TournamentBracket
│   └── teams_2026.py          # All 68 teams with full feature data
├── models/
│   ├── base_model.py          # Abstract base model class
│   ├── logistic_model.py      # Logistic Regression
│   ├── random_forest_model.py # Random Forest
│   ├── xgboost_model.py       # XGBoost
│   ├── neural_net_model.py    # PyTorch Neural Network
│   └── ensemble_model.py      # Weighted Ensemble
├── features/
│   ├── team_features.py       # Basic team stat features
│   ├── advanced_features.py   # KenPom-style advanced metrics
│   ├── sentiment_features.py  # Social media & news sentiment
│   ├── momentum_features.py   # Form, streaks, trajectory
│   └── engineering.py         # Full feature pipeline
├── scrapers/
│   ├── kenpom_scraper.py      # KenPom efficiency ratings
│   ├── espn_scraper.py        # ESPN stats and BPI
│   ├── sentiment_scraper.py   # Twitter, Reddit, news sentiment
│   └── odds_scraper.py        # Betting odds and lines
├── prediction/
│   ├── matchup.py             # Head-to-head matchup predictor
│   ├── bracket.py             # Bracket simulator (Monte Carlo)
│   └── tournament.py          # Tournament orchestrator
└── utils/
    ├── validation.py          # Cross-validation, backtesting
    └── visualization.py       # Bracket plots, charts
```

## Data Model

Each team has 100+ features across these categories:

| Category | Features | Examples |
|----------|----------|---------|
| Record | 18 | Overall W/L, Quad 1-4 records, last 10 |
| Offense | 8 | PPG, FG%, 3PT%, FT%, assists, turnovers |
| Defense | 6 | Points allowed, opponent FG%, blocks, steals |
| Advanced | 12 | Adj. efficiency, Four Factors, tempo |
| Rankings | 8 | KenPom, NET, AP, BPI, Sagarin |
| Schedule | 4 | SOS, non-conference SOS, conference strength |
| Momentum | 6 | Win streak, last 5/10 margin, conf tourney |
| Historical | 7 | Tournament appearances, upsets, avg seed |
| Players | 8 | Experience, top scorer, injuries, NBA prospects |
| Coaching | 6 | Tournament wins, Final Fours, tenure |
| Sentiment | 8 | Twitter, Reddit, news, public picks, hype |
| Betting | 6 | Championship odds, spread, ATS record |
| Clutch | 6 | Close games, OT record, half margins |

## Models

### Logistic Regression
Baseline model with L2 regularization. Fast, interpretable, good calibration.

### Random Forest
500 trees with tuned depth and feature sampling. Handles non-linear relationships.

### XGBoost
Gradient boosted trees with learning rate scheduling. Best single-model performance historically.

### Neural Network
PyTorch model with batch normalization, dropout, and early stopping. Captures complex feature interactions.

### Ensemble
Combines all models with optimized weights (scipy.optimize). Supports weighted averaging, stacking, and voting.

## Environment Variables

Create a `.env` file for API access:

```
TWITTER_BEARER_TOKEN=your_token
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
ODDS_API_KEY=your_key
```

## License

MIT
