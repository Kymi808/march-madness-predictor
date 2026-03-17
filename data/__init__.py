"""Data models and team datasets for March Madness prediction."""

from data.schema import TeamData, MatchupData, TournamentBracket
from data.teams_2026 import get_all_teams, get_team_by_name, get_team_by_seed

__all__ = [
    "TeamData",
    "MatchupData",
    "TournamentBracket",
    "get_all_teams",
    "get_team_by_name",
    "get_team_by_seed",
]
