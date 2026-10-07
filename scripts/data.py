"""Build leakage-safe, pre-match features for the Premier League dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = PROCESSED_DATA_DIR / "merged_sorted_premier_league.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "final_model_data.csv"

DEFAULT_FORM = {
    "GoalsScored": 1.35,
    "GoalsConceded": 1.35,
    "SoT": 4.5,
    "SoTConceded": 4.5,
    "Corners": 5.0,
    "XG": 1.35,
    "XGConceded": 1.35,
    "OpponentAdjustedXG": 1.35,
}


def team_pre_match_features(matches: pd.DataFrame, team: str, has_xg: bool) -> pd.DataFrame:
    """Return rolling form for each fixture, calculated strictly from prior matches."""
    team_matches = matches.loc[(matches.HomeTeam == team) | (matches.AwayTeam == team)].copy()
    team_matches = team_matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    is_home = team_matches.HomeTeam.eq(team)

    values = {
        "GoalsScored": np.where(is_home, team_matches.FTHG, team_matches.FTAG),
        "GoalsConceded": np.where(is_home, team_matches.FTAG, team_matches.FTHG),
        "SoT": np.where(is_home, team_matches.HST, team_matches.AST),
        "SoTConceded": np.where(is_home, team_matches.AST, team_matches.HST),
        "Corners": np.where(is_home, team_matches.HC, team_matches.AC),
    }

    if has_xg:
        values.update({
            "XG": np.where(is_home, team_matches.HXG, team_matches.AXG),
            "XGConceded": np.where(is_home, team_matches.AXG, team_matches.HXG),
            "OpponentAdjustedXG": np.where(
                is_home,
                team_matches.HXG * team_matches.Away_Elo_PreMatch / 1500,
                team_matches.AXG * team_matches.Home_Elo_PreMatch / 1500,
            ),
        })

    for name, value in values.items():
        # .shift(1) prevents lookahead bias: the match being predicted cannot inform its own form
        team_matches[f"Avg{name}_Last5"] = (
            pd.Series(value, index=team_matches.index)
            .rolling(window=5, min_periods=1)
            .mean()
            .shift(1)
            .fillna(DEFAULT_FORM[name])
        )

    if has_xg:
        for name in ("XG", "XGConceded"):
            team_matches[f"Venue_Avg{name}_Last5"] = (
                pd.Series(values[name], index=team_matches.index)
                .groupby(is_home)
                .transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
                .fillna(DEFAULT_FORM[name])
            )
        team_matches["EWM_XG_Last10"] = (
            pd.Series(values["XG"], index=team_matches.index)
            .ewm(span=10, adjust=False)
            .mean()
            .shift(1)
            .fillna(DEFAULT_FORM["XG"])
        )

    team_matches["DrawRate_Last5"] = (
        team_matches.FTR.eq("D").rolling(5, min_periods=1).mean().shift(1).fillna(0.25)
    )
    team_matches["RestDays"] = team_matches.Date.diff().dt.days.fillna(7).clip(lower=0)
    team_matches["Matches_Last14Days"] = [
        sum((team_matches.Date.iloc[:i] >= date - pd.Timedelta(days=14)))
        for i, date in enumerate(team_matches.Date)
    ]
    team_matches["Team"] = team
    return team_matches


def add_dynamic_elo(matches: pd.DataFrame, base_k: float = 20, base_rating: float = 1500) -> pd.DataFrame:
    """Add chronological pre-match Elo ratings carrying across seasons with goal-difference margins."""
    teams = pd.unique(pd.concat([matches.HomeTeam, matches.AwayTeam]))
    ratings = dict.fromkeys(teams, base_rating)
    home_pre, away_pre, home_momentum, away_momentum = [], [], [], []
    changes = {team: [] for team in teams}

    for row in matches.itertuples(index=False):
        home_rating, away_rating = ratings[row.HomeTeam], ratings[row.AwayTeam]
        home_pre.append(home_rating)
        away_pre.append(away_rating)
        home_momentum.append(np.mean(changes[row.HomeTeam][-5:]) if changes[row.HomeTeam] else 0.0)
        away_momentum.append(np.mean(changes[row.AwayTeam][-5:]) if changes[row.AwayTeam] else 0.0)

        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[row.FTR]
        goal_diff = abs(row.FTHG - row.FTAG)
        multiplier = 1.0 if goal_diff <= 1 else 1.5 if goal_diff == 2 else (11 + goal_diff) / 8
        adjustment = base_k * multiplier * (actual_home - expected_home)

        ratings[row.HomeTeam] = home_rating + adjustment
        ratings[row.AwayTeam] = away_rating - adjustment
        changes[row.HomeTeam].append(adjustment)
        changes[row.AwayTeam].append(-adjustment)

    matches["Home_Elo_PreMatch"], matches["Away_Elo_PreMatch"] = home_pre, away_pre
    matches["Home_Elo_Momentum_Last5"], matches["Away_Elo_Momentum_Last5"] = home_momentum, away_momentum
    matches["Elo_Difference"] = matches.Home_Elo_PreMatch - matches.Away_Elo_PreMatch
    return matches


def main() -> None:
    matches = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HST", "AST", "HC", "AC"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if matches.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate match keys found; cannot safely merge features.")

    matches = matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    matches = add_dynamic_elo(matches)
    has_xg = {"HXG", "AXG"}.issubset(matches.columns)

    teams = pd.unique(pd.concat([matches.HomeTeam, matches.AwayTeam]))
    feature_rows = pd.concat([team_pre_match_features(matches, team, has_xg) for team in teams])
    feature_columns = [
        c for c in feature_rows.columns
        if c.startswith(("Avg", "Venue_", "EWM_", "DrawRate", "RestDays", "Matches_"))
    ]

    team_features = feature_rows[["Date", "Team", *feature_columns]]
    home_features = team_features.rename(columns={"Team": "HomeTeam", **{c: f"Home_{c}" for c in feature_columns}})
    away_features = team_features.rename(columns={"Team": "AwayTeam", **{c: f"Away_{c}" for c in feature_columns}})

    matches["Date"] = matches.Date.dt.strftime("%Y-%m-%d")
    home_features["Date"] = pd.to_datetime(home_features.Date).dt.strftime("%Y-%m-%d")
    away_features["Date"] = pd.to_datetime(away_features.Date).dt.strftime("%Y-%m-%d")

    matches = matches.merge(home_features, on=["Date", "HomeTeam"], how="left", validate="one_to_one")
    matches = matches.merge(away_features, on=["Date", "AwayTeam"], how="left", validate="one_to_one")

    if matches.filter(regex=r"^(Home|Away)_Avg").isna().any().any():
        raise ValueError("Feature merge left missing pre-match values.")

    matches.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(matches)} matches with pre-match features to {OUTPUT_FILE.name}.")


if __name__ == "__main__":
    main()
