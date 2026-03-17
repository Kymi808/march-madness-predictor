#!/usr/bin/env python
"""Bracket simulation CLI for March Madness Predictor.

Runs large-scale Monte Carlo simulations of the full tournament to estimate
championship and round-advancement probabilities for every team.

Usage examples::

    # Default: 10,000 simulations with the ensemble model
    python simulate_bracket.py

    # 50,000 simulations, show upsets, save results
    python simulate_bracket.py --simulations 50000 --show-upsets --output results/sim_2026.json

    # Use a specific model
    python simulate_bracket.py --model xgboost --simulations 25000
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

from config import CURRENT_SEASON, ROUNDS, MODEL_DIR, OUTPUT_DIR
from data import get_all_teams
from data.schema import TeamData, MatchupData, TournamentBracket
from models.base_model import BaseMarchMadnessModel


# ------------------------------------------------------------------
# Simulation engine
# ------------------------------------------------------------------

def _predict_game(
    model: BaseMarchMadnessModel,
    team_a: TeamData,
    team_b: TeamData,
) -> tuple[TeamData, float]:
    """Return the stochastic winner and the probability of team_a winning."""
    matchup = MatchupData(team_a=team_a, team_b=team_b, neutral_site=True)
    matchup.compute_diffs()
    features = np.array([matchup.to_feature_vector()])
    probas = model.predict_proba(features)[0]
    prob_a = float(probas[1])

    winner = team_a if np.random.random() < prob_a else team_b
    return winner, prob_a


def _simulate_tournament(
    model: BaseMarchMadnessModel,
    bracket: TournamentBracket,
) -> dict:
    """Run one full stochastic tournament simulation.

    Returns
    -------
    dict
        ``round_winners`` — mapping of round name to list of winning team names.
        ``champion`` — name of the tournament winner.
        ``upsets`` — list of dicts describing upsets (lower seed beat higher seed).
    """
    round_winners: dict[str, list[str]] = {}
    upsets: list[dict] = []
    region_champions: dict[str, TeamData] = {}

    for region_name in ["East", "West", "South", "Midwest"]:
        matchups = bracket.get_matchups_round1(region_name)
        current_round_teams: list[TeamData] = []

        # Round of 64
        for team_a, team_b in matchups:
            winner, prob_a = _predict_game(model, team_a, team_b)
            current_round_teams.append(winner)
            round_winners.setdefault("Round of 64", []).append(winner.name)

            # Detect upset: lower seed (higher number) beats higher seed
            if winner.seed > min(team_a.seed, team_b.seed):
                loser = team_b if winner is team_a else team_a
                upsets.append({
                    "round": "Round of 64",
                    "region": region_name,
                    "winner": winner.name,
                    "winner_seed": winner.seed,
                    "loser": loser.name,
                    "loser_seed": loser.seed,
                })

        # Subsequent regional rounds
        for round_name in ["Round of 32", "Sweet 16", "Elite 8"]:
            next_round_teams: list[TeamData] = []
            for i in range(0, len(current_round_teams), 2):
                if i + 1 >= len(current_round_teams):
                    next_round_teams.append(current_round_teams[i])
                    round_winners.setdefault(round_name, []).append(
                        current_round_teams[i].name
                    )
                    continue

                t_a, t_b = current_round_teams[i], current_round_teams[i + 1]
                winner, _ = _predict_game(model, t_a, t_b)
                next_round_teams.append(winner)
                round_winners.setdefault(round_name, []).append(winner.name)

                loser = t_b if winner is t_a else t_a
                if winner.seed > loser.seed:
                    upsets.append({
                        "round": round_name,
                        "region": region_name,
                        "winner": winner.name,
                        "winner_seed": winner.seed,
                        "loser": loser.name,
                        "loser_seed": loser.seed,
                    })

            current_round_teams = next_round_teams

        if current_round_teams:
            region_champions[region_name] = current_round_teams[0]

    # Final Four
    ff_pairs = [
        ("East", "West"),
        ("South", "Midwest"),
    ]
    finalists: list[TeamData] = []
    for r1, r2 in ff_pairs:
        t_a = region_champions.get(r1)
        t_b = region_champions.get(r2)
        if t_a is None and t_b is None:
            continue
        if t_a is None:
            finalists.append(t_b)
            round_winners.setdefault("Final Four", []).append(t_b.name)
            continue
        if t_b is None:
            finalists.append(t_a)
            round_winners.setdefault("Final Four", []).append(t_a.name)
            continue

        winner, _ = _predict_game(model, t_a, t_b)
        finalists.append(winner)
        round_winners.setdefault("Final Four", []).append(winner.name)

        loser = t_b if winner is t_a else t_a
        if winner.seed > loser.seed:
            upsets.append({
                "round": "Final Four",
                "region": "National",
                "winner": winner.name,
                "winner_seed": winner.seed,
                "loser": loser.name,
                "loser_seed": loser.seed,
            })

    # Championship
    champion_name = ""
    if len(finalists) >= 2:
        winner, _ = _predict_game(model, finalists[0], finalists[1])
        champion_name = winner.name
        round_winners["Championship"] = [champion_name]

        loser = finalists[1] if winner is finalists[0] else finalists[0]
        if winner.seed > loser.seed:
            upsets.append({
                "round": "Championship",
                "region": "National",
                "winner": winner.name,
                "winner_seed": winner.seed,
                "loser": loser.name,
                "loser_seed": loser.seed,
            })
    elif len(finalists) == 1:
        champion_name = finalists[0].name
        round_winners["Championship"] = [champion_name]

    return {
        "round_winners": round_winners,
        "champion": champion_name,
        "upsets": upsets,
    }


# ------------------------------------------------------------------
# Pretty-print helpers
# ------------------------------------------------------------------

def _print_most_likely_bracket(
    round_advance: dict[str, Counter],
    n_sims: int,
) -> None:
    """Print the single most-likely winner for each slot."""
    click.secho("\nMost Likely Bracket:", fg="cyan", bold=True)
    click.secho("=" * 50, fg="cyan")

    for round_name in ROUNDS:
        if round_name not in round_advance:
            continue
        counter = round_advance[round_name]
        # Each position in the round: we just show the top teams
        most_common = counter.most_common()
        if not most_common:
            continue

        click.secho(f"\n  {round_name}:", bold=True)
        # Show top teams for this round
        display_count = min(len(most_common), 16) if round_name in ("Round of 64",) else len(most_common)
        for team_name, count in most_common[:display_count]:
            pct = count / n_sims
            bar_len = int(pct * 30)
            bar = "#" * bar_len
            click.echo(f"    {team_name:>25s}  {pct:6.1%}  {bar}")


def _print_championship_odds(champ_counts: Counter, n_sims: int) -> None:
    """Print championship odds for all teams, sorted by probability."""
    click.echo()
    click.secho("Championship Odds:", fg="cyan", bold=True)
    click.secho("=" * 50, fg="cyan")

    for rank, (team_name, count) in enumerate(champ_counts.most_common(), start=1):
        pct = count / n_sims
        bar_len = int(pct * 40)
        bar = "#" * bar_len
        click.echo(f"  {rank:3d}. {team_name:>25s}  {pct:6.1%}  {bar}")


def _print_upsets(upset_counts: Counter, n_sims: int, top_n: int = 20) -> None:
    """Print the most frequent upsets across all simulations."""
    click.echo()
    click.secho("Most Common Upsets:", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")

    for desc, count in upset_counts.most_common(top_n):
        pct = count / n_sims
        click.echo(f"  {pct:5.1%}  {desc}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

@click.command()
@click.option(
    "--simulations",
    default=10000,
    type=int,
    show_default=True,
    help="Number of Monte Carlo simulations to run.",
)
@click.option(
    "--model",
    "model_name",
    default="ensemble",
    type=click.Choice(
        ["logistic", "rf", "xgboost", "neural_net", "ensemble"],
        case_sensitive=False,
    ),
    show_default=True,
    help="Model to use for predictions.",
)
@click.option(
    "--model-path",
    "model_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a trained model pickle file. Overrides --model.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="File path to save simulation results (JSON).",
)
@click.option(
    "--show-upsets",
    is_flag=True,
    default=False,
    help="Show the most frequent upsets across simulations.",
)
def simulate_bracket(
    simulations: int,
    model_name: str,
    model_path: str | None,
    output_path: str | None,
    show_upsets: bool,
) -> None:
    """Run Monte Carlo simulations of the full March Madness bracket."""

    click.secho(f"\n{'='*60}", fg="cyan")
    click.secho("  March Madness Predictor — Bracket Simulator", fg="cyan", bold=True)
    click.secho(f"{'='*60}\n", fg="cyan")

    # Load model --------------------------------------------------
    if model_path:
        click.echo(f"Loading model from {model_path} ...")
        model = BaseMarchMadnessModel.load(model_path)
    else:
        default_path = MODEL_DIR / f"{model_name.lower()}_{CURRENT_SEASON}.pkl"
        if not default_path.exists():
            click.secho(
                f"No trained model found at {default_path}. "
                "Train one first with train.py or provide --model-path.",
                fg="red",
            )
            sys.exit(1)
        click.echo(f"Loading model from {default_path} ...")
        model = BaseMarchMadnessModel.load(default_path)

    click.secho(f"Model: {model.name}", fg="green")
    click.echo(f"Simulations: {simulations:,}\n")

    # Build bracket -----------------------------------------------
    all_teams = get_all_teams()
    bracket = TournamentBracket(teams=all_teams)
    bracket.organize()

    # Run simulations ---------------------------------------------
    championship_counts: Counter = Counter()
    round_advance: dict[str, Counter] = defaultdict(Counter)
    upset_counter: Counter = Counter()

    click.echo("Simulating ", nl=False)
    progress_step = max(simulations // 20, 1)

    for i in range(simulations):
        if i % progress_step == 0:
            click.echo(".", nl=False)

        result = _simulate_tournament(model, bracket)
        championship_counts[result["champion"]] += 1

        for round_name, winners in result["round_winners"].items():
            for w in winners:
                round_advance[round_name][w] += 1

        if show_upsets:
            for upset in result["upsets"]:
                desc = (
                    f"({upset['winner_seed']}) {upset['winner']} over "
                    f"({upset['loser_seed']}) {upset['loser']} "
                    f"[{upset['round']}]"
                )
                upset_counter[desc] += 1

    click.echo(" done!\n")

    # Print results -----------------------------------------------
    _print_championship_odds(championship_counts, simulations)
    _print_most_likely_bracket(round_advance, simulations)

    if show_upsets:
        _print_upsets(upset_counter, simulations)

    # Save results ------------------------------------------------
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "simulations": simulations,
            "model": model.name,
            "championship_odds": {
                name: count / simulations
                for name, count in championship_counts.most_common()
            },
            "round_advancement": {
                round_name: {
                    name: count / simulations
                    for name, count in counter.most_common()
                }
                for round_name, counter in round_advance.items()
            },
        }
        if show_upsets:
            output_data["top_upsets"] = [
                {"description": desc, "frequency": count / simulations}
                for desc, count in upset_counter.most_common(30)
            ]

        with open(out, "w") as f:
            json.dump(output_data, f, indent=2)
        click.secho(f"\nResults saved to {out}", fg="green")

    click.echo()
    click.secho("Simulation complete.", fg="green", bold=True)


if __name__ == "__main__":
    simulate_bracket()
