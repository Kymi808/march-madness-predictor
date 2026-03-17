#!/usr/bin/env python
"""Prediction CLI for March Madness Predictor.

Usage examples::

    # Single matchup prediction
    python predict.py --team-a "Duke" --team-b "UConn" --model-path saved_models/ensemble_2026.pkl

    # Predict the full bracket
    python predict.py --bracket --model-path saved_models/ensemble_2026.pkl

    # Monte Carlo bracket simulation
    python predict.py --bracket --simulations 50000 --output results/bracket_2026.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

from config import CURRENT_SEASON, ROUNDS, OUTPUT_DIR
from data import get_all_teams, get_team_by_name
from data.schema import TeamData, MatchupData, TournamentBracket
from features import FeatureEngineeringPipeline
from models.base_model import BaseMarchMadnessModel


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _predict_matchup(
    model: BaseMarchMadnessModel,
    team_a: TeamData,
    team_b: TeamData,
) -> dict:
    """Predict a single matchup and return probabilities + winner."""
    matchup = MatchupData(team_a=team_a, team_b=team_b, neutral_site=True)
    matchup.compute_diffs()
    features = np.array([matchup.to_feature_vector()])
    probas = model.predict_proba(features)[0]
    winner = team_a if probas[1] >= 0.5 else team_b
    return {
        "team_a": team_a.name,
        "team_b": team_b.name,
        "prob_a_wins": float(probas[1]),
        "prob_b_wins": float(probas[0]),
        "predicted_winner": winner.name,
        "confidence": float(max(probas)),
    }


def _simulate_bracket_once(
    model: BaseMarchMadnessModel,
    bracket: TournamentBracket,
    stochastic: bool = True,
) -> dict:
    """Simulate one pass through the full tournament bracket.

    Parameters
    ----------
    model : BaseMarchMadnessModel
        Fitted model.
    bracket : TournamentBracket
        Organised bracket with teams sorted into regions.
    stochastic : bool
        If True, sample winners proportional to predicted probabilities.
        If False, always pick the higher-probability team.

    Returns
    -------
    dict
        Round-by-round list of winning team names.
    """
    results: dict[str, list[str]] = {}
    region_winners: dict[str, TeamData] = {}

    for region_name in ["East", "West", "South", "Midwest"]:
        # Round of 64 matchups in this region
        matchups = bracket.get_matchups_round1(region_name)
        current_teams: list[TeamData] = []

        for team_a, team_b in matchups:
            pred = _predict_matchup(model, team_a, team_b)
            if stochastic:
                winner_name = np.random.choice(
                    [pred["team_a"], pred["team_b"]],
                    p=[pred["prob_a_wins"], pred["prob_b_wins"]],
                )
                winner = team_a if winner_name == team_a.name else team_b
            else:
                winner = team_a if pred["prob_a_wins"] >= 0.5 else team_b
            current_teams.append(winner)
            results.setdefault("Round of 64", []).append(winner.name)

        # Subsequent regional rounds
        for round_name in ["Round of 32", "Sweet 16", "Elite 8"]:
            next_teams: list[TeamData] = []
            for i in range(0, len(current_teams), 2):
                if i + 1 >= len(current_teams):
                    next_teams.append(current_teams[i])
                    results.setdefault(round_name, []).append(current_teams[i].name)
                    continue
                pred = _predict_matchup(model, current_teams[i], current_teams[i + 1])
                if stochastic:
                    winner_name = np.random.choice(
                        [pred["team_a"], pred["team_b"]],
                        p=[pred["prob_a_wins"], pred["prob_b_wins"]],
                    )
                    winner = current_teams[i] if winner_name == current_teams[i].name else current_teams[i + 1]
                else:
                    winner = current_teams[i] if pred["prob_a_wins"] >= 0.5 else current_teams[i + 1]
                next_teams.append(winner)
                results.setdefault(round_name, []).append(winner.name)
            current_teams = next_teams

        if current_teams:
            region_winners[region_name] = current_teams[0]

    # Final Four
    ff_matchups = [
        (region_winners.get("East"), region_winners.get("West")),
        (region_winners.get("South"), region_winners.get("Midwest")),
    ]
    finalists: list[TeamData] = []
    for team_a, team_b in ff_matchups:
        if team_a is None or team_b is None:
            if team_a:
                finalists.append(team_a)
            elif team_b:
                finalists.append(team_b)
            continue
        pred = _predict_matchup(model, team_a, team_b)
        if stochastic:
            winner_name = np.random.choice(
                [pred["team_a"], pred["team_b"]],
                p=[pred["prob_a_wins"], pred["prob_b_wins"]],
            )
            winner = team_a if winner_name == team_a.name else team_b
        else:
            winner = team_a if pred["prob_a_wins"] >= 0.5 else team_b
        finalists.append(winner)
        results.setdefault("Final Four", []).append(winner.name)

    # Championship
    if len(finalists) >= 2:
        pred = _predict_matchup(model, finalists[0], finalists[1])
        if stochastic:
            winner_name = np.random.choice(
                [pred["team_a"], pred["team_b"]],
                p=[pred["prob_a_wins"], pred["prob_b_wins"]],
            )
            champion = finalists[0] if winner_name == finalists[0].name else finalists[1]
        else:
            champion = finalists[0] if pred["prob_a_wins"] >= 0.5 else finalists[1]
        results["Championship"] = [champion.name]
    elif finalists:
        results["Championship"] = [finalists[0].name]

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

@click.command()
@click.option(
    "--model",
    "model_name",
    default="ensemble",
    type=click.Choice(
        ["logistic", "rf", "xgboost", "neural_net", "ensemble"],
        case_sensitive=False,
    ),
    show_default=True,
    help="Model type (used when --model-path is not provided).",
)
@click.option(
    "--model-path",
    "model_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a trained model pickle file.",
)
@click.option("--team-a", default=None, type=str, help="First team name for a single matchup.")
@click.option("--team-b", default=None, type=str, help="Second team name for a single matchup.")
@click.option("--bracket", "run_bracket", is_flag=True, default=False, help="Predict the full tournament bracket.")
@click.option(
    "--simulations",
    default=10000,
    type=int,
    show_default=True,
    help="Number of Monte Carlo simulations for bracket prediction.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="File path to save prediction results (JSON).",
)
def predict(
    model_name: str,
    model_path: str | None,
    team_a: str | None,
    team_b: str | None,
    run_bracket: bool,
    simulations: int,
    output_path: str | None,
) -> None:
    """Generate March Madness predictions from a trained model."""

    click.secho(f"\n{'='*60}", fg="cyan")
    click.secho("  March Madness Predictor — Prediction Engine", fg="cyan", bold=True)
    click.secho(f"{'='*60}\n", fg="cyan")

    # Load model --------------------------------------------------
    if model_path:
        click.echo(f"Loading model from {model_path} ...")
        model = BaseMarchMadnessModel.load(model_path)
    else:
        from config import MODEL_DIR
        default_path = MODEL_DIR / f"{model_name.lower()}_{CURRENT_SEASON}.pkl"
        if not default_path.exists():
            click.secho(
                f"No trained model found at {default_path}. "
                "Train a model first with train.py or provide --model-path.",
                fg="red",
            )
            sys.exit(1)
        click.echo(f"Loading model from {default_path} ...")
        model = BaseMarchMadnessModel.load(default_path)

    click.secho(f"Model loaded: {model.name}\n", fg="green")

    # Single matchup prediction -----------------------------------
    if team_a and team_b:
        click.secho(f"Predicting: {team_a} vs {team_b}", fg="yellow", bold=True)
        team_a_data = get_team_by_name(team_a)
        team_b_data = get_team_by_name(team_b)

        if team_a_data is None:
            click.secho(f"Team not found: {team_a}", fg="red")
            sys.exit(1)
        if team_b_data is None:
            click.secho(f"Team not found: {team_b}", fg="red")
            sys.exit(1)

        result = _predict_matchup(model, team_a_data, team_b_data)

        click.echo()
        click.echo(f"  {result['team_a']:>25s}  {result['prob_a_wins']:6.1%}")
        click.echo(f"  {result['team_b']:>25s}  {result['prob_b_wins']:6.1%}")
        click.echo()
        click.secho(
            f"  Predicted winner: {result['predicted_winner']} "
            f"(confidence {result['confidence']:.1%})",
            fg="green", bold=True,
        )

        if output_path:
            _save_results(result, output_path)
        return

    # Full bracket prediction -------------------------------------
    if run_bracket:
        click.secho("Predicting full tournament bracket ...", fg="yellow", bold=True)
        click.echo(f"Running {simulations:,} Monte Carlo simulations ...\n")

        all_teams = get_all_teams()
        bracket = TournamentBracket(teams=all_teams)
        bracket.organize()

        # Deterministic "most likely" bracket
        deterministic = _simulate_bracket_once(model, bracket, stochastic=False)

        # Monte Carlo simulations for probability estimates
        championship_counts: dict[str, int] = {}
        final_four_counts: dict[str, int] = {}

        for _ in range(simulations):
            sim = _simulate_bracket_once(model, bracket, stochastic=True)
            champ = sim.get("Championship", [""])[0]
            championship_counts[champ] = championship_counts.get(champ, 0) + 1
            for team in sim.get("Final Four", []):
                final_four_counts[team] = final_four_counts.get(team, 0) + 1

        # Print deterministic bracket
        click.secho("Most Likely Bracket:", fg="cyan", bold=True)
        for round_name in ROUNDS:
            if round_name in deterministic:
                winners = deterministic[round_name]
                click.echo(f"\n  {round_name}:")
                for w in winners:
                    click.echo(f"    - {w}")

        # Print championship odds
        click.echo()
        click.secho("Championship Odds (Monte Carlo):", fg="cyan", bold=True)
        sorted_champs = sorted(
            championship_counts.items(), key=lambda x: x[1], reverse=True
        )
        for team_name, count in sorted_champs[:20]:
            pct = count / simulations
            click.echo(f"  {team_name:>25s}  {pct:6.1%}")

        # Build combined output
        output_data = {
            "most_likely_bracket": deterministic,
            "championship_odds": {
                name: count / simulations
                for name, count in sorted_champs
            },
            "final_four_odds": {
                name: count / simulations
                for name, count in sorted(
                    final_four_counts.items(), key=lambda x: x[1], reverse=True
                )
            },
            "simulations": simulations,
        }

        if output_path:
            _save_results(output_data, output_path)
        return

    # No action specified -----------------------------------------
    click.secho(
        "Specify --team-a/--team-b for a matchup or --bracket for a full bracket prediction.",
        fg="yellow",
    )
    sys.exit(1)


def _save_results(data: dict, path: str) -> None:
    """Write results dict to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    click.secho(f"Results saved to {out}", fg="green")


if __name__ == "__main__":
    predict()
