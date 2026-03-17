#!/usr/bin/env python
"""Training CLI for March Madness Predictor.

Usage examples::

    # Train all models on 2019-2025 data, targeting 2026
    python train.py --season 2026 --model all --historical-seasons 2019-2025

    # Train only XGBoost, save to a specific path
    python train.py --model xgboost --output saved_models/xgb_2026.pkl

    # Train logistic regression with custom data range
    python train.py --model logistic --historical-seasons 2015-2024
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

from config import (
    CURRENT_SEASON,
    MODEL_DIR,
    RANDOM_SEED,
)
from data import get_all_teams
from data.schema import TeamData, MatchupData
from features import FeatureEngineeringPipeline
from models.base_model import BaseMarchMadnessModel
from models.logistic_model import LogisticModel
from utils.validation import cross_validate_model, compare_models


# ------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------

def _build_model(name: str) -> BaseMarchMadnessModel:
    """Instantiate a model by short name."""
    name = name.lower()
    if name == "logistic":
        return LogisticModel()
    elif name == "rf":
        from models.random_forest_model import RandomForestModel
        return RandomForestModel()
    elif name == "xgboost":
        from models.xgboost_model import XGBoostModel
        return XGBoostModel()
    elif name == "neural_net":
        from models.neural_net_model import NeuralNetModel
        return NeuralNetModel()
    elif name == "ensemble":
        from models.ensemble_model import EnsembleModel
        return EnsembleModel()
    else:
        raise click.BadParameter(
            f"Unknown model '{name}'. Choose from: logistic, rf, xgboost, neural_net, ensemble"
        )


ALL_MODEL_NAMES = ["logistic", "rf", "xgboost", "neural_net", "ensemble"]


def _parse_season_range(value: str) -> list[int]:
    """Parse a range string like '2019-2025' into a list of ints."""
    if "-" in value:
        parts = value.split("-", maxsplit=1)
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))
    return [int(s.strip()) for s in value.split(",")]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

@click.command()
@click.option(
    "--season",
    default=CURRENT_SEASON,
    type=int,
    show_default=True,
    help="Target season to predict.",
)
@click.option(
    "--model",
    "model_name",
    default="ensemble",
    type=click.Choice(
        ["logistic", "rf", "xgboost", "neural_net", "ensemble", "all"],
        case_sensitive=False,
    ),
    show_default=True,
    help="Model to train (or 'all' for every model).",
)
@click.option(
    "--historical-seasons",
    "historical_seasons",
    default=None,
    type=str,
    help="Season range for training data, e.g. '2019-2025' or '2018,2019,2021'.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Custom path to save the trained model (pickle). "
         "Defaults to saved_models/<model>_<season>.pkl.",
)
def train(
    season: int,
    model_name: str,
    historical_seasons: str | None,
    output_path: str | None,
) -> None:
    """Train March Madness prediction model(s) on historical data."""

    click.secho(f"\n{'='*60}", fg="cyan")
    click.secho("  March Madness Predictor — Training Pipeline", fg="cyan", bold=True)
    click.secho(f"{'='*60}\n", fg="cyan")

    # Determine training seasons ----------------------------------
    if historical_seasons is not None:
        train_seasons = _parse_season_range(historical_seasons)
    else:
        # Default: last 7 completed seasons before the target
        train_seasons = list(range(season - 7, season))

    click.echo(f"Target season : {season}")
    click.echo(f"Training on   : {train_seasons}")
    click.echo()

    # Load and engineer features ----------------------------------
    click.secho("[1/4] Loading historical data and engineering features ...", fg="yellow")
    X_parts, y_parts = [], []
    for s in train_seasons:
        try:
            pipeline = FeatureEngineeringPipeline(season=s)
            X_s, y_s = pipeline.build_matchup_dataset()
            X_parts.append(X_s)
            y_parts.append(y_s)
            click.echo(f"      Season {s}: {len(y_s)} matchups loaded")
        except Exception as exc:
            click.secho(f"      Season {s}: skipped ({exc})", fg="red")

    if not X_parts:
        click.secho("No training data could be loaded. Exiting.", fg="red", bold=True)
        sys.exit(1)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    click.echo(f"      Total training samples: {len(y)}")
    click.echo()

    # Determine which models to train -----------------------------
    if model_name.lower() == "all":
        names = ALL_MODEL_NAMES
    else:
        names = [model_name.lower()]

    trained_models: list[BaseMarchMadnessModel] = []

    for name in names:
        click.secho(f"[2/4] Training {name} ...", fg="yellow")
        mdl = _build_model(name)

        try:
            mdl.train(X, y)
        except Exception as exc:
            click.secho(f"      FAILED: {exc}", fg="red")
            continue

        trained_models.append(mdl)
        click.secho(f"      {mdl.name} trained successfully.", fg="green")

        # Evaluate ------------------------------------------------
        click.secho(f"[3/4] Evaluating {mdl.name} ...", fg="yellow")
        try:
            cv_results = cross_validate_model(mdl, X, y)
            click.echo(f"      CV Accuracy : {cv_results['mean_accuracy']:.4f} "
                        f"(+/- {cv_results['std_accuracy']:.4f})")
            click.echo(f"      CV Log Loss : {cv_results['mean_log_loss']:.4f} "
                        f"(+/- {cv_results['std_log_loss']:.4f})")
            click.echo(f"      CV AUC      : {cv_results['mean_auc']:.4f} "
                        f"(+/- {cv_results['std_auc']:.4f})")
        except Exception as exc:
            click.secho(f"      Evaluation error: {exc}", fg="red")

        # Save ----------------------------------------------------
        click.secho(f"[4/4] Saving {mdl.name} ...", fg="yellow")
        if output_path and len(names) == 1:
            save_path = Path(output_path)
        else:
            save_path = MODEL_DIR / f"{name}_{season}.pkl"

        try:
            mdl.save(save_path)
            click.secho(f"      Saved to {save_path}", fg="green")
        except Exception as exc:
            click.secho(f"      Save error: {exc}", fg="red")

        click.echo()

    # Multi-model comparison table --------------------------------
    if len(trained_models) > 1:
        click.secho("Model Comparison Summary", fg="cyan", bold=True)
        click.secho("-" * 50, fg="cyan")
        try:
            comparison_df = compare_models(trained_models, X, y)
            click.echo(comparison_df.to_string(index=False))
        except Exception as exc:
            click.secho(f"Comparison error: {exc}", fg="red")
        click.echo()

    click.secho("Training complete.", fg="green", bold=True)


if __name__ == "__main__":
    train()
