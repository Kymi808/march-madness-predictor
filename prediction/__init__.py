"""Prediction module for March Madness tournament simulation.

Exports the three main classes:

- :class:`MatchupPredictor` -- predict individual head-to-head matchups
- :class:`BracketSimulator` -- simulate a full tournament bracket
- :class:`TournamentRunner` -- high-level orchestrator
"""

from prediction.matchup import MatchupPredictor
from prediction.bracket import BracketSimulator
from prediction.tournament import TournamentRunner

__all__ = [
    "MatchupPredictor",
    "BracketSimulator",
    "TournamentRunner",
]
