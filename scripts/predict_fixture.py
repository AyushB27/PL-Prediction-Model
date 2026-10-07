"""Interactive CLI tool to predict any Premier League match using the best-performing models."""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "final_model_data.csv"

OUTCOMES = ["H", "D", "A"]
TEAM_ALIASES = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton",
    "burnley": "Burnley",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "leicester": "Leicester",
    "liverpool": "Liverpool",
    "man city": "Man City",
    "manchester city": "Man City",
    "man united": "Man United",
    "manchester united": "Man United",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "nottingham forest": "Nott'm Forest",
    "nott'm forest": "Nott'm Forest",
    "forest": "Nott'm Forest",
    "southampton": "Southampton",
    "sunderland": "Sunderland",
    "tottenham": "Tottenham",
    "spurs": "Tottenham",
    "west ham": "West Ham",
    "wolves": "Wolves",
    "wolverhampton": "Wolves",
}

# Best performing feature tier: Tier 2 (+ Elo Quality)
FEATURES_TIER2 = [
    "Home_AvgGoalsScored_Last5", "Home_AvgGoalsConceded_Last5",
    "Home_AvgSoT_Last5", "Home_AvgSoTConceded_Last5", "Home_AvgCorners_Last5",
    "Away_AvgGoalsScored_Last5", "Away_AvgGoalsConceded_Last5",
    "Away_AvgSoT_Last5", "Away_AvgSoTConceded_Last5", "Away_AvgCorners_Last5",
    "Home_Elo_PreMatch", "Away_Elo_PreMatch", "Elo_Difference",
    "Home_Elo_Momentum_Last5", "Away_Elo_Momentum_Last5",
]


def resolve_team(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]
    for key, canonical in TEAM_ALIASES.items():
        if cleaned in key or key in cleaned:
            return canonical
    raise ValueError(f"Team '{name}' not recognized. Available teams include:\n" +
                     ", ".join(sorted(set(TEAM_ALIASES.values()))))


def get_latest_team_stats(matches: pd.DataFrame, team: str) -> Dict[str, float]:
    """Retrieve the most recent pre-match form and Elo stats for a team."""
    team_matches = matches.loc[(matches.HomeTeam == team) | (matches.AwayTeam == team)].copy()
    if team_matches.empty:
        raise ValueError(f"No match history found for {team}.")
    last_row = team_matches.sort_values("Date").iloc[-1]
    is_home = (last_row.HomeTeam == team)
    prefix = "Home_" if is_home else "Away_"

    return {
        "AvgGoalsScored_Last5": last_row[f"{prefix}AvgGoalsScored_Last5"],
        "AvgGoalsConceded_Last5": last_row[f"{prefix}AvgGoalsConceded_Last5"],
        "AvgSoT_Last5": last_row[f"{prefix}AvgSoT_Last5"],
        "AvgSoTConceded_Last5": last_row[f"{prefix}AvgSoTConceded_Last5"],
        "AvgCorners_Last5": last_row[f"{prefix}AvgCorners_Last5"],
        "Elo": last_row[f"{prefix}Elo_PreMatch"],
        "Elo_Momentum": last_row[f"{prefix}Elo_Momentum_Last5"],
    }


def dixon_coles_matrix(lambda_h: float, lambda_a: float, rho: float = -0.10, max_goals: int = 6) -> np.ndarray:
    goals = np.arange(max_goals)
    matrix = np.outer(poisson.pmf(goals, lambda_h), poisson.pmf(goals, lambda_a))
    matrix[0, 0] *= max(0.0, 1.0 - (lambda_h * lambda_a * rho))
    matrix[0, 1] *= max(0.0, 1.0 + (lambda_h * rho))
    matrix[1, 0] *= max(0.0, 1.0 + (lambda_a * rho))
    matrix[1, 1] *= max(0.0, 1.0 - rho)
    matrix /= matrix.sum()
    return matrix


def predict_match(home_team: str, away_team: str) -> None:
    matches = pd.read_csv(DATA_FILE)
    home_canonical = resolve_team(home_team)
    away_canonical = resolve_team(away_team)

    if home_canonical == away_canonical:
        print("Error: Home and Away teams must be different.")
        return

    home_stats = get_latest_team_stats(matches, home_canonical)
    away_stats = get_latest_team_stats(matches, away_canonical)

    # Train Poisson GLMs on full historical dataset using Tier 2 features
    train_x = sm.add_constant(matches[FEATURES_TIER2], has_constant="add")
    home_glm = sm.GLM(matches.FTHG, train_x, family=sm.families.Poisson()).fit(disp=False)
    away_glm = sm.GLM(matches.FTAG, train_x, family=sm.families.Poisson()).fit(disp=False)

    fixture_features = pd.DataFrame([{
        "const": 1.0,
        "Home_AvgGoalsScored_Last5": home_stats["AvgGoalsScored_Last5"],
        "Home_AvgGoalsConceded_Last5": home_stats["AvgGoalsConceded_Last5"],
        "Home_AvgSoT_Last5": home_stats["AvgSoT_Last5"],
        "Home_AvgSoTConceded_Last5": home_stats["AvgSoTConceded_Last5"],
        "Home_AvgCorners_Last5": home_stats["AvgCorners_Last5"],
        "Away_AvgGoalsScored_Last5": away_stats["AvgGoalsScored_Last5"],
        "Away_AvgGoalsConceded_Last5": away_stats["AvgGoalsConceded_Last5"],
        "Away_AvgSoT_Last5": away_stats["AvgSoT_Last5"],
        "Away_AvgSoTConceded_Last5": away_stats["AvgSoTConceded_Last5"],
        "Away_AvgCorners_Last5": away_stats["AvgCorners_Last5"],
        "Home_Elo_PreMatch": home_stats["Elo"],
        "Away_Elo_PreMatch": away_stats["Elo"],
        "Elo_Difference": home_stats["Elo"] - away_stats["Elo"],
        "Home_Elo_Momentum_Last5": home_stats["Elo_Momentum"],
        "Away_Elo_Momentum_Last5": away_stats["Elo_Momentum"],
    }])

    lambda_h = float(home_glm.predict(fixture_features)[0])
    lambda_a = float(away_glm.predict(fixture_features)[0])

    score_matrix = dixon_coles_matrix(lambda_h, lambda_a, rho=-0.10)
    p_home = float(np.tril(score_matrix, k=-1).sum())
    p_draw = float(np.trace(score_matrix))
    p_away = float(np.triu(score_matrix, k=1).sum())

    # Find top 3 most probable scorelines
    scorelines = []
    for h in range(score_matrix.shape[0]):
        for a in range(score_matrix.shape[1]):
            scorelines.append(((h, a), score_matrix[h, a]))
    scorelines.sort(key=lambda item: item[1], reverse=True)

    print("\n" + "=" * 65)
    print(f"PREMIER LEAGUE FIXTURE FORECAST: {home_canonical} vs {away_canonical}")
    print("=" * 65)
    print(f"Pre-Match Elo Rating:      {home_canonical}: {home_stats['Elo']:.0f} | {away_canonical}: {away_stats['Elo']:.0f}")
    print(f"Recent Goals/Match (L5):   {home_canonical}: {home_stats['AvgGoalsScored_Last5']:.2f} | {away_canonical}: {away_stats['AvgGoalsScored_Last5']:.2f}")
    print(f"Expected Goals (xG Rate):  {home_canonical}: {lambda_h:.2f} | {away_canonical}: {lambda_a:.2f}")
    print("-" * 65)
    print(f"WIN PROBABILITIES (Dixon-Coles Model):")
    print(f"  • {home_canonical} Win: {p_home * 100:.1f}%")
    print(f"  • Draw:                 {p_draw * 100:.1f}%")
    print(f"  • {away_canonical} Win: {p_away * 100:.1f}%")
    print("-" * 65)
    print("MOST LIKELY SCORELINES:")
    for (h, a), prob in scorelines[:3]:
        print(f"  {home_canonical} {h} - {a} {away_canonical}  ({prob * 100:.1f}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        home = sys.argv[1]
        away = sys.argv[2]
    else:
        home = input("Enter Home Team: ")
        away = input("Enter Away Team: ")
    predict_match(home, away)
