"""Data schemas for team features, matchups, and tournament structure."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class TeamData:
    """Comprehensive data model for a March Madness team.

    Captures all features needed for prediction: core stats, advanced metrics,
    rankings, sentiment, betting lines, historical tournament performance,
    player metrics, and coaching data.
    """

    # --- Identity ---
    name: str = ""
    seed: int = 0
    region: str = ""  # East, West, South, Midwest
    conference: str = ""
    auto_bid: bool = False  # True if conference tournament champion

    # --- Record ---
    wins: int = 0
    losses: int = 0
    conference_wins: int = 0
    conference_losses: int = 0
    road_wins: int = 0
    road_losses: int = 0
    neutral_wins: int = 0
    neutral_losses: int = 0
    quad1_wins: int = 0
    quad1_losses: int = 0
    quad2_wins: int = 0
    quad2_losses: int = 0
    quad3_wins: int = 0
    quad3_losses: int = 0
    quad4_wins: int = 0
    quad4_losses: int = 0
    last_10_wins: int = 0
    last_10_losses: int = 0

    # --- Offensive Stats (per game) ---
    points_per_game: float = 0.0
    field_goal_pct: float = 0.0
    three_point_pct: float = 0.0
    free_throw_pct: float = 0.0
    offensive_rebounds_pg: float = 0.0
    assists_pg: float = 0.0
    turnovers_pg: float = 0.0
    three_pointers_made_pg: float = 0.0

    # --- Defensive Stats (per game) ---
    points_allowed_pg: float = 0.0
    opponent_fg_pct: float = 0.0
    opponent_three_pct: float = 0.0
    blocks_pg: float = 0.0
    steals_pg: float = 0.0
    defensive_rebounds_pg: float = 0.0

    # --- Advanced Metrics (KenPom-style) ---
    adj_offensive_efficiency: float = 0.0  # Points per 100 possessions (adjusted)
    adj_defensive_efficiency: float = 0.0  # Points allowed per 100 possessions (adjusted)
    adj_tempo: float = 0.0  # Possessions per 40 minutes (adjusted)
    adj_net_efficiency: float = 0.0  # AdjOE - AdjDE
    effective_fg_pct: float = 0.0  # (FG + 0.5 * 3FG) / FGA
    turnover_pct: float = 0.0  # Turnovers per 100 possessions
    offensive_rebound_pct: float = 0.0  # % of available offensive rebounds grabbed
    free_throw_rate: float = 0.0  # FTA / FGA
    opponent_effective_fg_pct: float = 0.0
    opponent_turnover_pct: float = 0.0
    opponent_offensive_rebound_pct: float = 0.0
    opponent_free_throw_rate: float = 0.0

    # --- Rankings ---
    kenpom_rank: int = 0
    net_rank: int = 0
    ap_rank: int = 0  # 0 if unranked
    coaches_rank: int = 0  # 0 if unranked
    bpi_rank: int = 0
    sagarin_rank: int = 0
    kpi_rank: int = 0
    sor_rank: int = 0  # Strength of Record

    # --- Strength of Schedule ---
    sos_rank: int = 0
    sos_rating: float = 0.0
    non_conference_sos: float = 0.0
    conference_strength: float = 0.0  # Overall conference RPI/NET

    # --- Momentum / Form ---
    win_streak: int = 0  # Current win streak (negative for losing streak)
    last_5_margin: float = 0.0  # Average margin in last 5 games
    last_10_margin: float = 0.0  # Average margin in last 10 games
    conference_tournament_result: str = ""  # "champion", "runner-up", "semifinal", etc.
    days_since_last_game: int = 0
    games_played_last_30: int = 0

    # --- Historical Tournament Performance ---
    tournament_appearances_last_5yr: int = 0
    sweet_16_appearances_last_5yr: int = 0
    final_four_appearances_last_5yr: int = 0
    championship_appearances_last_5yr: int = 0
    championships_last_10yr: int = 0
    avg_seed_last_5yr: float = 0.0  # 0 if no appearances
    historical_upset_rate: float = 0.0  # Rate of winning as underdog in tournament

    # --- Player Metrics ---
    returning_minutes_pct: float = 0.0  # % of minutes from last season returning
    top_scorer_ppg: float = 0.0
    top_scorer_experience: int = 0  # Years in college
    roster_experience: float = 0.0  # Average years of college experience
    bench_minutes_pct: float = 0.0  # % of minutes from bench
    injury_impact: float = 0.0  # 0-1, 0 = no injuries, 1 = devastating
    nba_prospect_count: int = 0  # Number of projected NBA draft picks
    transfer_portal_additions: int = 0

    # --- Coaching ---
    coach_tournament_games: int = 0
    coach_tournament_wins: int = 0
    coach_final_fours: int = 0
    coach_championships: int = 0
    coach_years_at_school: int = 0
    coach_career_win_pct: float = 0.0

    # --- Sentiment Features ---
    public_pick_pct: float = 0.0  # % of public brackets picking this team (round-dependent)
    expert_pick_pct: float = 0.0  # % of expert brackets
    twitter_sentiment: float = 0.0  # -1 to 1, aggregate Twitter/X sentiment
    reddit_sentiment: float = 0.0  # -1 to 1, aggregate Reddit sentiment
    news_sentiment: float = 0.0  # -1 to 1, aggregate news article sentiment
    social_media_volume: float = 0.0  # Normalized volume of mentions
    preseason_ranking: int = 0  # AP preseason rank (0 if unranked)
    ranking_trajectory: float = 0.0  # Positive = improving, negative = declining

    # --- Betting / Market ---
    championship_odds: float = 0.0  # Decimal odds to win championship
    implied_probability: float = 0.0  # Implied probability from odds
    spread_consistency: float = 0.0  # How often team covers the spread
    over_under_avg: float = 0.0  # Average total in games
    ats_record_wins: int = 0  # Against the spread wins
    ats_record_losses: int = 0

    # --- Clutch / Situational ---
    close_game_wins: int = 0  # Games decided by 5 or fewer points
    close_game_losses: int = 0
    overtime_record_wins: int = 0
    overtime_record_losses: int = 0
    scoring_margin: float = 0.0  # Average scoring margin
    first_half_margin: float = 0.0
    second_half_margin: float = 0.0

    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def conference_win_pct(self) -> float:
        total = self.conference_wins + self.conference_losses
        return self.conference_wins / total if total > 0 else 0.0

    def coach_tournament_win_pct(self) -> float:
        total = self.coach_tournament_games
        return self.coach_tournament_wins / total if total > 0 else 0.0

    def ats_pct(self) -> float:
        total = self.ats_record_wins + self.ats_record_losses
        return self.ats_record_wins / total if total > 0 else 0.0

    def close_game_pct(self) -> float:
        total = self.close_game_wins + self.close_game_losses
        return self.close_game_wins / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_feature_vector(self) -> list[float]:
        """Convert to numeric feature vector for ML models."""
        d = self.to_dict()
        features = []
        skip_fields = {"name", "region", "conference", "conference_tournament_result", "auto_bid"}
        for key, val in d.items():
            if key in skip_fields:
                continue
            if isinstance(val, bool):
                features.append(float(val))
            elif isinstance(val, (int, float)):
                features.append(float(val))
        return features

    @staticmethod
    def feature_names() -> list[str]:
        """Return names of numeric features in the same order as to_feature_vector."""
        sample = TeamData()
        d = sample.to_dict()
        names = []
        skip_fields = {"name", "region", "conference", "conference_tournament_result", "auto_bid"}
        for key, val in d.items():
            if key in skip_fields:
                continue
            if isinstance(val, (bool, int, float)):
                names.append(key)
        return names


@dataclass
class MatchupData:
    """Data for a head-to-head matchup prediction."""

    team_a: TeamData = field(default_factory=TeamData)
    team_b: TeamData = field(default_factory=TeamData)
    round_name: str = ""
    neutral_site: bool = True

    # Derived matchup features
    seed_diff: float = 0.0
    efficiency_diff: float = 0.0
    tempo_diff: float = 0.0
    experience_diff: float = 0.0
    coach_experience_diff: float = 0.0

    def compute_diffs(self):
        """Compute differential features between two teams."""
        self.seed_diff = self.team_a.seed - self.team_b.seed
        self.efficiency_diff = self.team_a.adj_net_efficiency - self.team_b.adj_net_efficiency
        self.tempo_diff = self.team_a.adj_tempo - self.team_b.adj_tempo
        self.experience_diff = self.team_a.roster_experience - self.team_b.roster_experience
        self.coach_experience_diff = (
            self.team_a.coach_tournament_games - self.team_b.coach_tournament_games
        )

    def to_feature_vector(self) -> list[float]:
        """Create feature vector from the difference of team features."""
        a_features = self.team_a.to_feature_vector()
        b_features = self.team_b.to_feature_vector()
        diff_features = [a - b for a, b in zip(a_features, b_features)]
        combined = a_features + b_features + diff_features
        combined.append(float(self.neutral_site))
        return combined


@dataclass
class TournamentBracket:
    """Represents the full 68-team tournament bracket."""

    teams: list[TeamData] = field(default_factory=list)
    regions: dict = field(default_factory=lambda: {
        "East": [], "West": [], "South": [], "Midwest": []
    })
    first_four: list[tuple] = field(default_factory=list)

    def organize(self):
        """Sort teams into regions and identify First Four matchups."""
        for team in self.teams:
            if team.region in self.regions:
                self.regions[team.region].append(team)
        for region in self.regions.values():
            region.sort(key=lambda t: t.seed)

    def get_region(self, region_name: str) -> list[TeamData]:
        return self.regions.get(region_name, [])

    def get_matchups_round1(self, region_name: str) -> list[tuple]:
        """Standard 1v16, 2v15, ... 8v9 matchups for a region."""
        teams = self.get_region(region_name)
        seed_map = {t.seed: t for t in teams}
        matchups = []
        for high, low in [(1, 16), (2, 15), (3, 14), (4, 13),
                          (5, 12), (6, 11), (7, 10), (8, 9)]:
            if high in seed_map and low in seed_map:
                matchups.append((seed_map[high], seed_map[low]))
        return matchups

    def to_json(self) -> str:
        data = {
            "regions": {
                region: [t.to_dict() for t in teams]
                for region, teams in self.regions.items()
            }
        }
        return json.dumps(data, indent=2)
