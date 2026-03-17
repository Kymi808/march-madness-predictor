"""High-level tournament orchestrator for March Madness prediction.

Ties together model loading/training, bracket construction, simulation,
and result export into a single convenient interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from config import MODEL_DIR, OUTPUT_DIR, ROUNDS
from data.schema import TeamData, TournamentBracket
from prediction.matchup import MatchupPredictor
from prediction.bracket import BracketSimulator

if TYPE_CHECKING:
    from models.base_model import BaseMarchMadnessModel


class TournamentRunner:
    """End-to-end tournament simulation runner.

    Usage
    -----
    >>> runner = TournamentRunner()
    >>> results = runner.run(teams)
    >>> runner.print_bracket(results)
    >>> runner.export_results(results, "output.json")
    """

    def __init__(self) -> None:
        self.simulator = BracketSimulator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(model_path: str) -> "BaseMarchMadnessModel":
        """Load a persisted model from disk."""
        from models.base_model import BaseMarchMadnessModel

        return BaseMarchMadnessModel.load(model_path)

    @staticmethod
    def _build_bracket(teams: list[TeamData]) -> TournamentBracket:
        """Construct a :class:`TournamentBracket` from a list of teams."""
        bracket = TournamentBracket(teams=list(teams))
        bracket.organize()
        return bracket

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        teams: list[TeamData],
        model_path: str | None = None,
        model: "BaseMarchMadnessModel | None" = None,
        pipeline=None,
        n_simulations: int = 10_000,
    ) -> dict:
        """Run a full tournament prediction.

        Parameters
        ----------
        teams : list[TeamData]
            The tournament field (typically 64 or 68 teams).
        model_path : str, optional
            Path to a saved model file.  Loaded via
            ``BaseMarchMadnessModel.load``.
        model : BaseMarchMadnessModel, optional
            An already-instantiated (and trained) model.  Takes precedence
            over *model_path* when both are supplied.
        pipeline : FeatureEngineeringPipeline, optional
            Feature engineering pipeline passed through to the predictor.
        n_simulations : int
            Number of Monte Carlo simulations (default 10 000).

        Returns
        -------
        dict
            ``deterministic`` : result of :meth:`BracketSimulator.simulate_bracket`
            ``monte_carlo``   : result of :meth:`BracketSimulator.monte_carlo_simulate`
            ``upset_specials`` : result of :meth:`BracketSimulator.get_upset_specials`
        """
        # Resolve the model
        if model is None:
            if model_path is None:
                raise ValueError(
                    "Either 'model' or 'model_path' must be provided."
                )
            model = self._load_model(model_path)

        predictor = MatchupPredictor(model=model, pipeline=pipeline)
        bracket = self._build_bracket(teams)

        deterministic = self.simulator.simulate_bracket(bracket, predictor)
        monte_carlo = self.simulator.monte_carlo_simulate(
            bracket, predictor, n_simulations=n_simulations
        )
        upset_specials = self.simulator.get_upset_specials(bracket, predictor)

        return {
            "deterministic": deterministic,
            "monte_carlo": monte_carlo,
            "upset_specials": upset_specials,
        }

    # ------------------------------------------------------------------
    # Bracket display
    # ------------------------------------------------------------------
    @staticmethod
    def print_bracket(results: dict) -> None:
        """Pretty-print the deterministic bracket to stdout."""
        det = results.get("deterministic", results)

        print("=" * 72)
        print("  MARCH MADNESS BRACKET PREDICTION")
        print("=" * 72)

        # Per-region results
        regions = det.get("regions", {})
        for region_name, rounds in regions.items():
            print(f"\n{'─' * 36}")
            print(f"  {region_name.upper()} REGION")
            print(f"{'─' * 36}")
            for round_name, games in rounds.items():
                print(f"\n  {round_name}:")
                for game in games:
                    winner = game.get("winner", "?")
                    prob_a = game.get("win_probability_a", 0)
                    prob_b = game.get("win_probability_b", 0)
                    margin = game.get("predicted_margin", 0)
                    team_a = game.get("team_a", "?")
                    team_b = game.get("team_b", "?")
                    marker_a = " *" if winner == team_a else ""
                    marker_b = " *" if winner == team_b else ""
                    print(
                        f"    {team_a}{marker_a} ({prob_a:.1%}) vs "
                        f"{team_b}{marker_b} ({prob_b:.1%})  "
                        f"[margin: {abs(margin):.1f}]"
                    )

        # Final Four
        print(f"\n{'=' * 36}")
        print("  FINAL FOUR")
        print(f"{'=' * 36}")
        for game in det.get("final_four", []):
            winner = game.get("winner", "?")
            team_a = game.get("team_a", "?")
            team_b = game.get("team_b", "?")
            marker_a = " *" if winner == team_a else ""
            marker_b = " *" if winner == team_b else ""
            print(
                f"  {team_a}{marker_a} vs {team_b}{marker_b}  "
                f"-> Winner: {winner}"
            )

        # Championship
        champ = det.get("championship", {})
        if champ:
            print(f"\n{'=' * 36}")
            print("  CHAMPIONSHIP")
            print(f"{'=' * 36}")
            team_a = champ.get("team_a", "?")
            team_b = champ.get("team_b", "?")
            winner = champ.get("winner", "?")
            print(f"  {team_a} vs {team_b}  -> CHAMPION: {winner}")

        # Monte Carlo summary (if present)
        mc = results.get("monte_carlo")
        if mc:
            print(f"\n{'=' * 72}")
            print(
                f"  MONTE CARLO SIMULATION  "
                f"({mc['n_simulations']:,} simulations)"
            )
            print(f"{'=' * 72}")
            print(f"  Most likely champion: {mc['most_common_champion']}")

            # Top 10 championship contenders
            probs = mc["team_probabilities"]
            ranked = sorted(
                probs.items(),
                key=lambda kv: kv[1]["championship_prob"],
                reverse=True,
            )
            print("\n  Top 10 Championship Contenders:")
            for i, (name, p) in enumerate(ranked[:10], 1):
                print(
                    f"    {i:>2}. {name:<25s}  "
                    f"Champ: {p['championship_prob']:>6.1%}  "
                    f"FF: {p['final_four_prob']:>6.1%}  "
                    f"E8: {p['elite_8_prob']:>6.1%}  "
                    f"S16: {p['sweet_16_prob']:>6.1%}"
                )

        # Upset specials (if present)
        upsets = results.get("upset_specials")
        if upsets:
            print(f"\n{'=' * 72}")
            print("  UPSET SPECIALS")
            print(f"{'=' * 72}")
            for u in upsets:
                print(
                    f"  [{u['region']}] #{u['underdog_seed']} {u['underdog']} "
                    f"over #{u['favorite_seed']} {u['favorite']}  "
                    f"(upset prob: {u['upset_probability']:.1%})"
                )

        print()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    @staticmethod
    def export_results(results: dict, path: str) -> None:
        """Serialize tournament results to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path.  Parent directories are created
            automatically if they do not exist.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # numpy types are not JSON-serializable, so use a custom encoder.
        class _Encoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np

                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, cls=_Encoder)

        print(f"Results exported to {out_path}")

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    def compare_models(
        self,
        teams: list[TeamData],
        models: list["BaseMarchMadnessModel"],
        pipeline=None,
        n_simulations: int = 10_000,
    ) -> dict:
        """Run the tournament with each model and compare predictions.

        Parameters
        ----------
        teams : list[TeamData]
        models : list[BaseMarchMadnessModel]
            Trained models to compare.
        pipeline : FeatureEngineeringPipeline, optional
        n_simulations : int

        Returns
        -------
        dict
            Mapping ``model_name`` -> full results dict (same structure
            returned by :meth:`run`).  Also includes a ``summary`` key
            with a side-by-side comparison table.
        """
        comparison: dict = {}
        summaries: list[dict] = []

        for mdl in models:
            result = self.run(
                teams,
                model=mdl,
                pipeline=pipeline,
                n_simulations=n_simulations,
            )
            comparison[mdl.name] = result

            # Build a quick summary row
            mc = result["monte_carlo"]
            det = result["deterministic"]
            summaries.append(
                {
                    "model": mdl.name,
                    "deterministic_champion": det["champion"],
                    "mc_most_likely_champion": mc["most_common_champion"],
                    "mc_champion_prob": mc["team_probabilities"]
                    .get(mc["most_common_champion"], {})
                    .get("championship_prob", 0),
                    "num_upsets_flagged": len(result["upset_specials"]),
                }
            )

        comparison["summary"] = summaries

        # Print comparison table
        print(f"\n{'=' * 72}")
        print("  MODEL COMPARISON")
        print(f"{'=' * 72}")
        header = (
            f"  {'Model':<25s} {'Det. Champion':<22s} "
            f"{'MC Champion':<22s} {'MC Prob':>8s} {'Upsets':>6s}"
        )
        print(header)
        print(f"  {'─' * 67}")
        for s in summaries:
            print(
                f"  {s['model']:<25s} {s['deterministic_champion']:<22s} "
                f"{s['mc_most_likely_champion']:<22s} "
                f"{s['mc_champion_prob']:>7.1%} "
                f"{s['num_upsets_flagged']:>6d}"
            )
        print()

        return comparison
