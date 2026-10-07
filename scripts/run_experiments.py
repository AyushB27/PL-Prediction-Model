"""Run progressive feature ablation and model benchmark across Premier League seasons."""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "final_model_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
EXPERIMENT_RESULTS_FILE = OUTPUT_DIR / "ablation_experiment_results.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "model_predictions.csv"

OUTCOMES = np.array(["H", "D", "A"])  # standard order: Home, Draw, Away
HOLDOUT_SEASON = "2025-2026"
VALIDATION_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]

# Feature Tiers
TIER_1_FORM = [
    "Home_AvgGoalsScored_Last5", "Home_AvgGoalsConceded_Last5",
    "Home_AvgSoT_Last5", "Home_AvgSoTConceded_Last5", "Home_AvgCorners_Last5",
    "Away_AvgGoalsScored_Last5", "Away_AvgGoalsConceded_Last5",
    "Away_AvgSoT_Last5", "Away_AvgSoTConceded_Last5", "Away_AvgCorners_Last5",
]

TIER_2_ELO = TIER_1_FORM + [
    "Home_Elo_PreMatch", "Away_Elo_PreMatch", "Elo_Difference",
    "Home_Elo_Momentum_Last5", "Away_Elo_Momentum_Last5",
]

TIER_3_XG = TIER_2_ELO + [
    "Home_AvgXG_Last5", "Home_AvgXGConceded_Last5",
    "Away_AvgXG_Last5", "Away_AvgXGConceded_Last5",
    "Home_Venue_AvgXG_Last5", "Home_Venue_AvgXGConceded_Last5",
    "Away_Venue_AvgXG_Last5", "Away_Venue_AvgXGConceded_Last5",
    "Home_EWM_XG_Last10", "Away_EWM_XG_Last10",
    "Home_AvgOpponentAdjustedXG_Last5", "Away_AvgOpponentAdjustedXG_Last5",
]

TIER_4_SCHEDULING = TIER_3_XG + [
    "Home_RestDays", "Away_RestDays",
    "Home_Matches_Last14Days", "Away_Matches_Last14Days",
    "Home_DrawRate_Last5", "Away_DrawRate_Last5",
]

FEATURE_TIERS = {
    "Tier 1: Basic Form": TIER_1_FORM,
    "Tier 2: + Elo Quality": TIER_2_ELO,
    "Tier 3: + Expected Goals": TIER_3_XG,
    "Tier 4: + Scheduling Context": TIER_4_SCHEDULING,
}


# --- Probability Metrics ---
def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared difference between outcome indicators and probability vectors."""
    one_hot = (y_true[:, None] == OUTCOMES).astype(float)
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute standard football evaluation metrics."""
    # Ensure probabilities are clipped for numerical stability in log loss
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    y_prob_clipped /= y_prob_clipped.sum(axis=1, keepdims=True)

    # Scikit-learn log_loss expects probabilities ordered by sorted class labels ('A', 'D', 'H')
    labels_sorted = ["A", "D", "H"]
    prob_sorted = y_prob_clipped[:, [2, 1, 0]]
    ll = log_loss(y_true, prob_sorted, labels=labels_sorted)
    brier = multiclass_brier_score(y_true, y_prob_clipped)

    preds = OUTCOMES[y_prob_clipped.argmax(axis=1)]
    acc = accuracy_score(y_true, preds)

    # Draw recall: proportion of actual draws correctly identified
    draw_mask = (y_true == "D")
    draw_recall = float((preds[draw_mask] == "D").mean()) if draw_mask.any() else 0.0

    return {
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc * 100, 2),
        "draw_recall": round(draw_recall * 100, 2),
    }


# --- Dixon-Coles Low-Score Correction ---
def dixon_coles_tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - (lambda_h * lambda_a * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 0:
        return 1.0 + (lambda_a * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def poisson_probabilities(lambda_h: np.ndarray, lambda_a: np.ndarray, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    """Compute match outcome probabilities [P(H), P(D), P(A)] from expected goals."""
    goals = np.arange(max_goals)
    probs = []
    for lh, la in zip(lambda_h, lambda_a):
        p_home_goals = poisson.pmf(goals, lh)
        p_away_goals = poisson.pmf(goals, la)
        matrix = np.outer(p_home_goals, p_away_goals)

        if rho != 0.0:
            for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
                matrix[x, y] *= max(0.0, dixon_coles_tau(x, y, lh, la, rho))

        matrix /= matrix.sum()
        p_home = np.tril(matrix, k=-1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, k=1).sum()
        probs.append((p_home, p_draw, p_away))
    return np.asarray(probs)


# --- Model Training Wrappers ---
def fit_predict_poisson(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], with_dc: bool = False
) -> np.ndarray:
    train_x = sm.add_constant(train_df[features], has_constant="add")
    test_x = sm.add_constant(test_df[features], has_constant="add")

    model_home = sm.GLM(train_df.FTHG, train_x, family=sm.families.Poisson()).fit(disp=False)
    model_away = sm.GLM(train_df.FTAG, test_x if False else train_x, family=sm.families.Poisson()).fit(disp=False)

    lambda_h = model_home.predict(test_x).to_numpy()
    lambda_a = model_away.predict(test_x).to_numpy()

    rho = -0.10 if with_dc else 0.0
    return poisson_probabilities(lambda_h, lambda_a, rho=rho)


def fit_predict_logistic(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    model.fit(train_df[features], train_df.FTR)
    raw_probs = model.predict_proba(test_df[features])
    classes = list(model.classes_)
    return raw_probs[:, [classes.index(c) for c in OUTCOMES]]


def fit_predict_rf(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(train_df[features], train_df.FTR)
    raw_probs = model.predict_proba(test_df[features])
    classes = list(model.classes_)
    return raw_probs[:, [classes.index(c) for c in OUTCOMES]]


def fit_predict_hgb(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    weights = compute_sample_weight("balanced", train_df.FTR)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=42)
    model.fit(train_df[features], train_df.FTR, sample_weight=weights)
    raw_probs = model.predict_proba(test_df[features])
    classes = list(model.classes_)
    return raw_probs[:, [classes.index(c) for c in OUTCOMES]]


MODEL_REGISTRY: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, List[str]], np.ndarray]] = {
    "Poisson GLM": lambda tr, te, feat: fit_predict_poisson(tr, te, feat, with_dc=False),
    "Dixon-Coles": lambda tr, te, feat: fit_predict_poisson(tr, te, feat, with_dc=True),
    "Logistic Regression": fit_predict_logistic,
    "Random Forest": fit_predict_rf,
    "HistGradientBoosting": fit_predict_hgb,
}


def build_league_table(predictions: pd.DataFrame) -> pd.DataFrame:
    home = predictions.assign(
        Team=predictions.HomeTeam,
        Points=np.select([predictions.Predicted_FTR.eq("H"), predictions.Predicted_FTR.eq("D")], [3, 1], default=0),
        GF=predictions.FTHG, GA=predictions.FTAG,
    )[["Team", "Points", "GF", "GA"]]
    away = predictions.assign(
        Team=predictions.AwayTeam,
        Points=np.select([predictions.Predicted_FTR.eq("A"), predictions.Predicted_FTR.eq("D")], [3, 1], default=0),
        GF=predictions.FTAG, GA=predictions.FTHG,
    )[["Team", "Points", "GF", "GA"]]
    table = pd.concat([home, away]).groupby("Team", as_index=False).sum()
    table["GD"] = table.GF - table.GA
    table = table.sort_values(["Points", "GD", "GF", "Team"], ascending=[False, False, False, True]).reset_index(drop=True)
    table.index += 1
    return table


def main() -> None:
    print("=" * 80)
    print("PREMIER LEAGUE OUTCOME PREDICTION: PROGRESSIVE FEATURE ABLATION STUDY")
    print("=" * 80)

    matches = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(matches)} matches across {matches.Season.nunique()} seasons (2014-2026).")
    print(f"Validation Folds: {VALIDATION_SEASONS} | Final Holdout: {HOLDOUT_SEASON}\n")

    train_all = matches[matches.Season.lt(HOLDOUT_SEASON)].copy()
    holdout = matches[matches.Season.eq(HOLDOUT_SEASON)].copy()

    # Historical Empirical Baseline (predicting historical class proportions)
    baseline_probs = train_all.FTR.value_counts(normalize=True)[OUTCOMES].to_numpy()
    baseline_holdout_prob = np.tile(baseline_probs, (len(holdout), 1))
    baseline_metrics = calculate_metrics(holdout.FTR.to_numpy(), baseline_holdout_prob)

    results = []

    # Run Walk-Forward Validation & Holdout for all (Tier, Model) combinations
    for tier_name, features in FEATURE_TIERS.items():
        print(f"Evaluating {tier_name} ({len(features)} features)...")
        for model_name, model_fn in MODEL_REGISTRY.items():
            # 1. Walk-Forward Cross Validation
            oof_probs = []
            oof_actual = []
            for val_season in VALIDATION_SEASONS:
                fold_train = matches[matches.Season.lt(val_season)]
                fold_val = matches[matches.Season.eq(val_season)]
                probs = model_fn(fold_train, fold_val, features)
                oof_probs.append(probs)
                oof_actual.append(fold_val.FTR.to_numpy())

            oof_probs_arr = np.vstack(oof_probs)
            oof_actual_arr = np.concatenate(oof_actual)
            val_metrics = calculate_metrics(oof_actual_arr, oof_probs_arr)

            # 2. Final Holdout Season (2025-2026)
            holdout_probs = model_fn(train_all, holdout, features)
            hold_metrics = calculate_metrics(holdout.FTR.to_numpy(), holdout_probs)

            results.append({
                "Model": model_name,
                "Feature Tier": tier_name,
                "Num Features": len(features),
                "CV Log Loss": val_metrics["log_loss"],
                "CV Brier Score": val_metrics["brier_score"],
                "CV Accuracy (%)": val_metrics["accuracy"],
                "CV Draw Recall (%)": val_metrics["draw_recall"],
                "Holdout Log Loss": hold_metrics["log_loss"],
                "Holdout Brier": hold_metrics["brier_score"],
                "Holdout Accuracy (%)": hold_metrics["accuracy"],
                "Holdout Draw Recall (%)": hold_metrics["draw_recall"],
            })

    # Add Baseline
    results.append({
        "Model": "Empirical Baseline",
        "Feature Tier": "Tier 0 (Historical Prior)",
        "Num Features": 0,
        "CV Log Loss": np.nan,
        "CV Brier Score": np.nan,
        "CV Accuracy (%)": np.nan,
        "CV Draw Recall (%)": np.nan,
        "Holdout Log Loss": baseline_metrics["log_loss"],
        "Holdout Brier": baseline_metrics["brier_score"],
        "Holdout Accuracy (%)": baseline_metrics["accuracy"],
        "Holdout Draw Recall (%)": baseline_metrics["draw_recall"],
    })

    results_df = pd.DataFrame(results).sort_values("Holdout Log Loss", ascending=True).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(EXPERIMENT_RESULTS_FILE, index=False)

    print("\n" + "=" * 105)
    print("EXPERIMENTAL RESULTS SUMMARY (Sorted by Holdout Log Loss):")
    print("=" * 105)
    cols_display = [
        "Model", "Feature Tier", "Holdout Log Loss", "Holdout Brier",
        "Holdout Accuracy (%)", "Holdout Draw Recall (%)", "CV Log Loss"
    ]
    print(results_df[cols_display].to_string(index=False))
    print("=" * 105)
    print(f"Results written to: {EXPERIMENT_RESULTS_FILE.name}")

    # Best model prediction generation for Holdout
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_tier_name = best_row["Feature Tier"]
    best_features = FEATURE_TIERS[best_tier_name]
    best_fn = MODEL_REGISTRY[best_model_name]

    final_probs = best_fn(train_all, holdout, best_features)
    holdout_out = holdout.copy()
    holdout_out[["P_Home", "P_Draw", "P_Away"]] = final_probs
    holdout_out["Predicted_FTR"] = OUTCOMES[final_probs.argmax(axis=1)]
    holdout_out["Correct"] = holdout_out.Predicted_FTR == holdout_out.FTR

    holdout_out.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nBest Overall Model: {best_model_name} with {best_tier_name}")
    print(f"Saved holdout match predictions to {PREDICTIONS_FILE.name}.")

    print("\nBacktest Implied League Table (2025-26):")
    print(build_league_table(holdout_out).head(10).to_string())


if __name__ == "__main__":
    main()
