"""Bracket simulation engine for March Madness.

Provides deterministic and Monte Carlo simulation of an entire tournament
bracket, from the First Four through the Championship game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from config import RANDOM_SEED, ROUNDS
from data.schema import TeamData, TournamentBracket

if TYPE_CHECKING:
    from prediction.matchup import MatchupPredictor


# Mapping from round index (0-based after First Four) to display name.
_ROUND_SEQUENCE = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "Championship",
]

_REGIONS = ["East", "West", "South", "Midwest"]

# Final Four pairings (by convention): East vs West, South vs Midwest
_SEMIFINAL_PAIRINGS = [("East", "West"), ("South", "Midwest")]


class BracketSimulator:
    """Simulate the NCAA tournament bracket using a :class:`MatchupPredictor`."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _simulate_round(
        matchups: list[tuple[TeamData, TeamData]],
        predictor: "MatchupPredictor",
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> list[TeamData]:
        """Simulate a round and return the list of winners.

        Parameters
        ----------
        matchups : list of (team_a, team_b) tuples
        predictor : MatchupPredictor
        deterministic : bool
            If True the higher-probability team always wins.
            If False, the winner is sampled proportionally to win probability.
        rng : numpy Generator, optional
            Used only when ``deterministic`` is False.
        """
        winners: list[TeamData] = []
        for team_a, team_b in matchups:
            result = predictor.predict(team_a, team_b)
            prob_a = result["win_probability_a"]

            if deterministic:
                winners.append(team_a if prob_a >= 0.5 else team_b)
            else:
                if rng is None:
                    rng = np.random.default_rng(RANDOM_SEED)
                winners.append(
                    team_a if rng.random() < prob_a else team_b
                )
        return winners

    @staticmethod
    def _pair_winners(winners: list[TeamData]) -> list[tuple[TeamData, TeamData]]:
        """Pair adjacent winners for the next round."""
        return [
            (winners[i], winners[i + 1]) for i in range(0, len(winners), 2)
        ]

    def _simulate_region(
        self,
        bracket: TournamentBracket,
        region: str,
        predictor: "MatchupPredictor",
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[TeamData, dict]:
        """Simulate all intra-region rounds and return the region champion.

        Returns
        -------
        champion : TeamData
        round_results : dict  mapping round name -> list of result dicts
        """
        round_results: dict[str, list] = {}

        # Round of 64 matchups
        matchups = bracket.get_matchups_round1(region)
        if not matchups:
            raise ValueError(f"No teams found in region '{region}'.")

        for rnd_idx, round_name in enumerate(_ROUND_SEQUENCE[:4]):
            # Round of 64 -> Round of 32 -> Sweet 16 -> Elite 8
            winners = self._simulate_round(
                matchups, predictor, deterministic=deterministic, rng=rng
            )
            # Store results for this round
            rnd_records = []
            for (team_a, team_b), winner in zip(matchups, winners):
                pred = predictor.predict(team_a, team_b)
                pred["winner"] = winner.name
                rnd_records.append(pred)
            round_results[round_name] = rnd_records

            if len(winners) == 1:
                break
            matchups = self._pair_winners(winners)

        champion = winners[0]
        return champion, round_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simulate_bracket(
        self,
        bracket: TournamentBracket,
        predictor: "MatchupPredictor",
    ) -> dict:
        """Simulate the entire tournament deterministically.

        Always picks the team with the higher predicted win probability.

        Returns
        -------
        dict
            ``regions``  : per-region round results
            ``final_four`` : Final Four results
            ``championship`` : Championship result
            ``champion`` : name of the predicted champion
        """
        bracket.organize()

        all_results: dict = {"regions": {}, "final_four": [], "championship": None, "champion": None}
        region_champions: dict[str, TeamData] = {}

        for region in _REGIONS:
            champion, rnd_results = self._simulate_region(
                bracket, region, predictor, deterministic=True
            )
            region_champions[region] = champion
            all_results["regions"][region] = rnd_results

        # --- Final Four ---
        ff_winners: list[TeamData] = []
        for region_a, region_b in _SEMIFINAL_PAIRINGS:
            team_a = region_champions[region_a]
            team_b = region_champions[region_b]
            result = predictor.predict(team_a, team_b)
            winner = team_a if result["win_probability_a"] >= 0.5 else team_b
            result["winner"] = winner.name
            result["round"] = "Final Four"
            all_results["final_four"].append(result)
            ff_winners.append(winner)

        # --- Championship ---
        team_a, team_b = ff_winners
        result = predictor.predict(team_a, team_b)
        winner = team_a if result["win_probability_a"] >= 0.5 else team_b
        result["winner"] = winner.name
        result["round"] = "Championship"
        all_results["championship"] = result
        all_results["champion"] = winner.name

        return all_results

    def monte_carlo_simulate(
        self,
        bracket: TournamentBracket,
        predictor: "MatchupPredictor",
        n_simulations: int = 10_000,
    ) -> dict:
        """Run *n_simulations* stochastic bracket simulations.

        In each simulation, game outcomes are sampled randomly with
        probability equal to the predicted win probability.  Uses numpy
        for vectorized random draws for speed.

        Returns
        -------
        dict
            ``team_probabilities`` : dict mapping team name -> dict with
                ``championship_prob``, ``final_four_prob``,
                ``sweet_16_prob``, ``elite_8_prob``
            ``most_common_champion`` : str
            ``n_simulations`` : int
        """
        bracket.organize()

        # Collect every team name for counters
        all_teams: set[str] = set()
        for teams in bracket.regions.values():
            for t in teams:
                all_teams.add(t.name)

        counts: dict[str, dict[str, int]] = {
            name: {
                "championship": 0,
                "final_four": 0,
                "elite_8": 0,
                "sweet_16": 0,
            }
            for name in all_teams
        }

        # Pre-generate all random numbers we will need.
        # Upper bound: 4 regions * 4 rounds * 8 games + 2 FF + 1 champ = 131
        # but actual count is 63 games per simulation.
        rng = np.random.default_rng(RANDOM_SEED)
        # We draw per-simulation so the rng stays deterministic.

        for _ in range(n_simulations):
            region_champions: dict[str, TeamData] = {}

            for region in _REGIONS:
                matchups = bracket.get_matchups_round1(region)
                # Round of 64
                winners = self._simulate_round(
                    matchups, predictor, deterministic=False, rng=rng
                )
                # Round of 32
                matchups = self._pair_winners(winners)
                winners = self._simulate_round(
                    matchups, predictor, deterministic=False, rng=rng
                )
                for w in winners:
                    counts[w.name]["sweet_16"] += 1

                # Sweet 16
                matchups = self._pair_winners(winners)
                winners = self._simulate_round(
                    matchups, predictor, deterministic=False, rng=rng
                )
                for w in winners:
                    counts[w.name]["elite_8"] += 1

                # Elite 8
                matchups = self._pair_winners(winners)
                winners = self._simulate_round(
                    matchups, predictor, deterministic=False, rng=rng
                )
                region_champions[region] = winners[0]
                counts[winners[0].name]["final_four"] += 1

            # Final Four
            ff_winners: list[TeamData] = []
            for region_a, region_b in _SEMIFINAL_PAIRINGS:
                team_a = region_champions[region_a]
                team_b = region_champions[region_b]
                result = predictor.predict(team_a, team_b)
                prob_a = result["win_probability_a"]
                winner = team_a if rng.random() < prob_a else team_b
                ff_winners.append(winner)

            # Championship
            team_a, team_b = ff_winners
            result = predictor.predict(team_a, team_b)
            prob_a = result["win_probability_a"]
            champion = team_a if rng.random() < prob_a else team_b
            counts[champion.name]["championship"] += 1

        # Convert counts to probabilities
        team_probabilities: dict[str, dict[str, float]] = {}
        for name, cnts in counts.items():
            team_probabilities[name] = {
                "championship_prob": round(cnts["championship"] / n_simulations, 4),
                "final_four_prob": round(cnts["final_four"] / n_simulations, 4),
                "elite_8_prob": round(cnts["elite_8"] / n_simulations, 4),
                "sweet_16_prob": round(cnts["sweet_16"] / n_simulations, 4),
            }

        # Most common champion
        most_common_champion = max(
            counts, key=lambda name: counts[name]["championship"]
        )

        return {
            "team_probabilities": team_probabilities,
            "most_common_champion": most_common_champion,
            "n_simulations": n_simulations,
        }

    def get_most_likely_bracket(
        self,
        bracket: TournamentBracket,
        predictor: "MatchupPredictor",
    ) -> dict:
        """Return the deterministic bracket that always picks the higher
        probability team.  Alias for :meth:`simulate_bracket`."""
        return self.simulate_bracket(bracket, predictor)

    def get_upset_specials(
        self,
        bracket: TournamentBracket,
        predictor: "MatchupPredictor",
        threshold: float = 0.35,
    ) -> list[dict]:
        """Find first-round matchups where the lower seed has a realistic
        chance of pulling an upset.

        An *upset special* is defined as a game where the lower seed
        (higher seed number) has a win probability >= ``threshold``.

        Returns
        -------
        list of dict
            Each dict contains the matchup prediction plus an
            ``upset_probability`` key.
        """
        bracket.organize()
        upsets: list[dict] = []

        for region in _REGIONS:
            matchups = bracket.get_matchups_round1(region)
            for team_a, team_b in matchups:
                result = predictor.predict(team_a, team_b)

                # Identify underdog: the team with the higher seed number
                if team_a.seed > team_b.seed:
                    underdog_prob = result["win_probability_a"]
                    underdog = team_a
                    favorite = team_b
                elif team_b.seed > team_a.seed:
                    underdog_prob = result["win_probability_b"]
                    underdog = team_b
                    favorite = team_a
                else:
                    # Same seed (e.g., First Four): skip
                    continue

                if underdog_prob >= threshold:
                    upsets.append(
                        {
                            "region": region,
                            "favorite": favorite.name,
                            "favorite_seed": favorite.seed,
                            "underdog": underdog.name,
                            "underdog_seed": underdog.seed,
                            "upset_probability": round(underdog_prob, 4),
                            "predicted_margin": result["predicted_margin"],
                            "confidence_level": result["confidence_level"],
                            "key_factors": result["key_factors"],
                        }
                    )

        # Sort by upset probability descending (most likely upsets first)
        upsets.sort(key=lambda u: u["upset_probability"], reverse=True)
        return upsets
