"""Compute and compare Premier League standings (2025-2026) based on prediction models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "outputs" / "model_predictions.csv"
OUTPUT_TABLE_FILE = PROJECT_ROOT / "data" / "outputs" / "league_table_2025_26.csv"


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found at {PREDICTIONS_FILE}. "
            "Please run 'python scripts/run_experiments.py' first."
        )
    return pd.read_csv(PREDICTIONS_FILE)


def compute_actual_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the true, ground-truth league table for 2025-2026."""
    home = df[["HomeTeam", "FTHG", "FTAG", "FTR"]].rename(
        columns={"HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA"}
    )
    home["W"] = (home["FTR"] == "H").astype(int)
    home["D"] = (home["FTR"] == "D").astype(int)
    home["L"] = (home["FTR"] == "A").astype(int)
    home["Pts"] = home["W"] * 3 + home["D"]

    away = df[["AwayTeam", "FTAG", "FTHG", "FTR"]].rename(
        columns={"AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA"}
    )
    away["W"] = (away["FTR"] == "A").astype(int)
    away["D"] = (away["FTR"] == "D").astype(int)
    away["L"] = (away["FTR"] == "H").astype(int)
    away["Pts"] = away["W"] * 3 + away["D"]

    table = pd.concat([home, away]).groupby("Team", as_index=False).sum()
    table["P"] = table["W"] + table["D"] + table["L"]
    table["GD"] = table["GF"] - table["GA"]
    table = table.sort_values(
        by=["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    table.index += 1
    table.index.name = "Pos"
    return table[["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]


def compute_expected_points_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Probabilistic Expected Points (xPts) table.
    
    xPoints:
      Home xPts = 3 * P(Home) + 1 * P(Draw)
      Away xPts = 3 * P(Away) + 1 * P(Draw)
    """
    home = df.assign(
        Team=df.HomeTeam,
        xPts=3 * df.P_Home + df.P_Draw,
        xW=df.P_Home,
        xD=df.P_Draw,
        xL=df.P_Away,
        xGF=df.FTHG,
        xGA=df.FTAG,
    )[["Team", "xW", "xD", "xL", "xGF", "xGA", "xPts"]]

    away = df.assign(
        Team=df.AwayTeam,
        xPts=3 * df.P_Away + df.P_Draw,
        xW=df.P_Away,
        xD=df.P_Draw,
        xL=df.P_Home,
        xGF=df.FTAG,
        xGA=df.FTHG,
    )[["Team", "xW", "xD", "xL", "xGF", "xGA", "xPts"]]

    table = pd.concat([home, away]).groupby("Team", as_index=False).sum()
    table["P"] = 38
    table["xGD"] = table["xGF"] - table["xGA"]
    table = table.sort_values(
        by=["xPts", "xGD", "xGF", "Team"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    table.index += 1
    table.index.name = "Pos"
    return table[["Team", "P", "xW", "xD", "xL", "xGF", "xGA", "xGD", "xPts"]]


def compute_comparison_table(actual: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Generate a side-by-side comparison table highlighting over/underperformance."""
    act = actual.reset_index().rename(columns={"Pos": "Act_Pos", "Pts": "Act_Pts"})
    exp = expected.reset_index().rename(columns={"Pos": "Exp_Pos", "xPts": "Exp_Pts"})

    merged = pd.merge(act, exp[["Team", "Exp_Pos", "Exp_Pts"]], on="Team")
    merged["Pos_Diff"] = merged["Exp_Pos"] - merged["Act_Pos"]
    merged["Pts_Diff"] = merged["Act_Pts"] - merged["Exp_Pts"]

    merged = merged.sort_values("Act_Pos").reset_index(drop=True)
    merged.index += 1
    merged.index.name = "Pos"
    return merged[["Team", "Act_Pos", "Exp_Pos", "Pos_Diff", "Act_Pts", "Exp_Pts", "Pts_Diff"]]


def format_table_output(comp: pd.DataFrame, expected: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("PREMIER LEAGUE 2025-2026: EXPECTED POINTS (xPts) STANDINGS")
    print("=" * 90)

    exp_display = expected.copy()
    exp_display["xW"] = exp_display["xW"].map(lambda x: f"{x:.1f}")
    exp_display["xD"] = exp_display["xD"].map(lambda x: f"{x:.1f}")
    exp_display["xL"] = exp_display["xL"].map(lambda x: f"{x:.1f}")
    exp_display["xPts"] = exp_display["xPts"].map(lambda x: f"{x:.1f}")

    print(exp_display.to_string())

    print("\n" + "=" * 90)
    print("STANDINGS COMPARISON: ACTUAL vs. MODEL EXPECTATION (xPts)")
    print("=" * 90)

    comp_display = comp.copy()
    comp_display["Exp_Pts"] = comp_display["Exp_Pts"].map(lambda x: f"{x:.1f}")
    comp_display["Pts_Diff"] = comp_display["Pts_Diff"].map(lambda x: f"{x:+.1f}")
    comp_display["Pos_Diff"] = comp_display["Pos_Diff"].map(
        lambda x: f"+{x}" if x > 0 else (str(x) if x < 0 else "=")
    )

    print(comp_display.to_string())
    print("=" * 90)

    # Key Insights
    top_over = comp.sort_values("Pts_Diff", ascending=False).iloc[0]
    top_under = comp.sort_values("Pts_Diff", ascending=True).iloc[0]

    print("\nMODEL INSIGHTS:")
    print(f"  • Champion:            {actual_team(comp, 1)} (Predicted 1st: {expected.iloc[0]['Team']})")
    print(f"  • UCL Qualifiers:      {', '.join(comp.iloc[:4]['Team'].tolist())}")
    print(f"  • Relegated (Bottom 3):{', '.join(comp.iloc[-3:]['Team'].tolist())}")
    print(f"  • Top Overperformer:   {top_over['Team']} ({top_over['Pts_Diff']:+.1f} pts above expected)")
    print(f"  • Top Underperformer:  {top_under['Team']} ({top_under['Pts_Diff']:+.1f} pts below expected)\n")


def actual_team(comp: pd.DataFrame, pos: int) -> str:
    match = comp[comp["Act_Pos"] == pos]
    return match["Team"].values[0] if not match.empty else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Premier League 2025-26 Table from predictions.")
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_TABLE_FILE),
        help="Path to save the generated table CSV"
    )
    args = parser.parse_args()

    preds = load_predictions()
    actual_table = compute_actual_table(preds)
    expected_table = compute_expected_points_table(preds)
    comparison_table = compute_comparison_table(actual_table, expected_table)

    format_table_output(comparison_table, expected_table)

    # Save to CSV
    comparison_table.to_csv(args.output)
    print(f"Saved complete standings comparison to {Path(args.output).name}.\n")


if __name__ == "__main__":
    main()
