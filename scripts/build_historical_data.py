"""Standardize and validate EPL fixtures before joining historical Understat xG."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
XG_FILE = PROCESSED_DATA_DIR / "understat_xg_2014_2025.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "merged_sorted_premier_league.csv"
STANDARD_COLUMNS = {
    "MatchDate": "Date", "FullTimeHomeGoals": "FTHG", "FullTimeAwayGoals": "FTAG", "FullTimeResult": "FTR",
    "HalfTimeHomeGoals": "HTHG", "HalfTimeAwayGoals": "HTAG", "HalfTimeResult": "HTR",
    "HomeShots": "HS", "AwayShots": "AS", "HomeShotsOnTarget": "HST", "AwayShotsOnTarget": "AST",
    "HomeCorners": "HC", "AwayCorners": "AC", "HomeFouls": "HF", "AwayFouls": "AF",
    "HomeYellowCards": "HY", "AwayYellowCards": "AY", "HomeRedCards": "HR", "AwayRedCards": "AR",
}
FIXTURE_TO_CANONICAL = {"QPR": "Queens Park Rangers", "West Brom": "West Bromwich Albion"}


def archive_fixtures() -> pd.DataFrame:
    archive = pd.read_csv(RAW_DATA_DIR / "epl_final.csv")
    archive = archive[(archive.Season >= "2014/15") & (archive.Season <= "2023/24")].rename(columns=STANDARD_COLUMNS)
    archive[["HomeTeam", "AwayTeam"]] = archive[["HomeTeam", "AwayTeam"]].replace(FIXTURE_TO_CANONICAL)
    archive["Season"] = archive.Season.map(lambda value: f"{value[:4]}-{int(value[-2:]) + 2000}")
    return archive


def recent_fixtures(filename: str, season: str) -> pd.DataFrame:
    fixtures = pd.read_csv(RAW_DATA_DIR / filename)
    fixtures["Season"] = season
    return fixtures


def main() -> None:
    fixtures = pd.concat([archive_fixtures(), recent_fixtures("2425.csv", "2024-2025"), recent_fixtures("2526.csv", "2025-2026")], ignore_index=True)
    fixtures["Date"] = pd.to_datetime(fixtures.Date).dt.strftime("%Y-%m-%d")
    fixtures = fixtures.sort_values(["Date", "HomeTeam", "AwayTeam"])
    if fixtures.groupby("Season").size().ne(380).any() or fixtures.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Fixture data must have exactly 380 unique matches per season.")
    xg = pd.read_csv(XG_FILE).rename(columns={"Date": "XG_Date"})
    if xg.duplicated(["Season", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("xG data has duplicate season/team match keys.")
    merged = fixtures.merge(xg, on=["HomeTeam", "AwayTeam", "Season"], how="left", validate="one_to_one")
    missing = merged.HXG.isna() | merged.AXG.isna()
    if missing.any():
        sample = merged.loc[missing, ["Season", "Date", "HomeTeam", "AwayTeam"]].head().to_dict("records")
        raise ValueError(f"Missing xG for {missing.sum()} fixtures; sample: {sample}")
    date_differences = merged.Date.ne(merged.XG_Date).sum()
    if date_differences:
        print(f"Validated {date_differences} fixtures with differing archive/Understat dates using unique season/team keys.")
    merged = merged.drop(columns="XG_Date")
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(merged)} standardized xG-complete fixtures to {OUTPUT_FILE.name}.")


if __name__ == "__main__":
    main()
