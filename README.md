# Premier League Match Predictions & Feature Ablation Study

A rigorous statistical and machine learning pipeline for predicting English Premier League (EPL) match outcomes (**Home Win / Draw / Away Win**) and goal distributions across 12 seasons (**2014–2015 to 2025–2026**, 4,560 matches).

This repository investigates two central research questions:
1. **Statistical vs. Machine Learning Models**: *Do domain-specific football models (Poisson GLM, Dixon–Coles) produce better-calibrated probabilities than general machine learning classifiers (Logistic Regression, Random Forest, HistGradientBoosting)?*
2. **Progressive Feature Ablation**: *How much information is actually needed to predict Premier League outcomes, and does adding scheduling context and expected goals (xG) improve probabilistic calibration or introduce noise?*

---

## 📊 Core Research Findings

* **Statistical models outperform general ML**: **Dixon–Coles** (Holdout Log Loss **1.0319**) and **Poisson GLM** (**1.0347**) consistently beat **Logistic Regression** (1.0479), **Random Forest** (1.0529), and **HistGradientBoosting** (1.1142) on probability quality.
* **Elo ratings are the single highest-leverage feature**: Introducing continuous pre-match Elo ratings (Tier 2) produces the largest reduction in log loss across every single model family (e.g. Dixon-Coles drops from 1.0486 to 1.0319).
* **Expected Goals (xG) improves validation quality**: Adding underlying shot quality (Tier 3) delivers the best Out-of-Fold Cross-Validation Log Loss (0.9593).
* **Scheduling and fatigue context add noise**: Adding rest days and 14-day fixture congestion (Tier 4) slightly increases holdout log loss, demonstrating that over-indexing on short-term scheduling introduces variance.

---

## 🔬 Experimental Architecture

### 1. Progressive Feature Tiers
All rolling metrics strictly enforce zero lookahead leakage by using chronological `.shift(1)` — the fixture being predicted never informs its own pre-match form.

```
Tier 1: Basic Form (10 features)
  ├── Rolling 5-match goals scored & conceded
  └── Rolling 5-match shots on target & corners
        │
        ▼ (+ Team Strength)
Tier 2: Elo Quality (15 features)
  ├── Pre-match dynamic Elo ratings (continuous across seasons)
  ├── Elo rating difference
  └── Elo momentum (rate of rating change over last 5 fixtures)
        │
        ▼ (+ Underlying Quality)
Tier 3: Expected Goals (27 features)
  ├── Rolling 5-match xG and xG conceded
  ├── Venue-specific home/away rolling xG
  ├── 10-match exponentially weighted moving average (EMA) xG
  └── Opponent-adjusted xG based on opposing Elo
        │
        ▼ (+ Fatigue & Context)
Tier 4: Scheduling Context (33 features)
  ├── Rest days between matches
  ├── Fixture congestion (matches played in the last 14 days)
  └── 5-match rolling draw rate
```

### 2. Evaluated Models
- **Poisson GLM**: Fits independent log-linear generalized linear models for home and away scoring rates ($\lambda_H, \lambda_A$), evaluating bivariate Poisson scoreline matrices up to 10 goals.
- **Dixon–Coles**: Applies the bivariate Dixon–Coles correction factor ($\rho = -0.10$) to account for real-world interdependence in low-scoring match states ($0\text{--}0, 1\text{--}0, 0\text{--}1, 1\text{--}1$).
- **Logistic Regression**: Scaled multinomial logistic baseline with $L_2$ regularization.
- **Random Forest**: 300-tree bagged ensemble with `min_samples_leaf=8` and balanced class weights.
- **HistGradientBoosting**: Histogram-binned gradient boosted decision trees capturing non-linear threshold effects.
- **Empirical Baseline**: Historical unconditioned class frequency prior ($~42.6\%$ Home, $27.4\%$ Draw, $30.0\%$ Away).

### 3. Chronological Walk-Forward Validation
- **Training Set**: 2014–15 up to season $T-1$.
- **Validation Folds**: Walk-forward cross-validation across `2021–2022`, `2022–2023`, `2023–2024`, and `2024–2025`.
- **Holdout Test**: Full 380 matches of season `2025–2026`.

---

## 🏆 Full Ablation Results Matrix

Evaluated on the 2025–2026 Holdout Season (sorted by Holdout Log Loss):

| Model | Feature Tier | Features | Holdout Log Loss | Holdout Brier | Holdout Accuracy | CV Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dixon-Coles** | **Tier 2: + Elo Quality** | 15 | **1.0319** | **0.6203** | **48.16%** | 0.9641 |
| **Dixon-Coles** | Tier 3: + Expected Goals | 27 | 1.0335 | 0.6214 | 47.11% | **0.9596** |
| **Poisson GLM** | Tier 2: + Elo Quality | 15 | 1.0347 | 0.6222 | 48.16% | 0.9634 |
| **Poisson GLM** | Tier 3: + Expected Goals | 27 | 1.0366 | 0.6234 | 47.11% | **0.9593** |
| **Dixon-Coles** | Tier 4: + Scheduling Context | 33 | 1.0376 | 0.6228 | 46.84% | 0.9619 |
| **Poisson GLM** | Tier 4: + Scheduling Context | 33 | 1.0407 | 0.6248 | 46.84% | 0.9616 |
| **Logistic Regression** | Tier 2: + Elo Quality | 15 | 1.0479 | 0.6297 | 45.53% | 0.9900 |
| **Dixon-Coles** | Tier 1: Basic Form | 10 | 1.0486 | 0.6319 | 47.63% | 0.9985 |
| **Logistic Regression** | Tier 4: + Scheduling Context | 33 | 1.0493 | 0.6315 | 44.74% | 0.9902 |
| **Poisson GLM** | Tier 1: Basic Form | 10 | 1.0505 | 0.6331 | 47.63% | 0.9965 |
| **Random Forest** | Tier 2: + Elo Quality | 15 | 1.0529 | 0.6352 | 46.58% | 0.9882 |
| **Logistic Regression** | Tier 3: + Expected Goals | 27 | 1.0534 | 0.6334 | 44.47% | 0.9864 |
| **Random Forest** | Tier 4: + Scheduling Context | 33 | 1.0575 | 0.6373 | 44.47% | 0.9795 |
| **Random Forest** | Tier 3: + Expected Goals | 27 | 1.0583 | 0.6379 | 46.58% | 0.9789 |
| **Logistic Regression** | Tier 1: Basic Form | 10 | 1.0684 | 0.6452 | 41.05% | 1.0313 |
| **Random Forest** | Tier 1: Basic Form | 10 | 1.0782 | 0.6526 | 43.95% | 1.0314 |
| *Empirical Baseline* | *Tier 0 (Historical Prior)* | 0 | 1.0835 | 0.6557 | 42.63% | — |
| **HistGradientBoosting** | Tier 4: + Scheduling Context | 33 | 1.1142 | 0.6699 | 42.63% | 1.0424 |
| **HistGradientBoosting** | Tier 2: + Elo Quality | 15 | 1.1151 | 0.6700 | 44.47% | 1.0514 |
| **HistGradientBoosting** | Tier 1: Basic Form | 10 | 1.1157 | 0.6752 | 39.21% | 1.0942 |
| **HistGradientBoosting** | Tier 3: + Expected Goals | 27 | 1.1236 | 0.6769 | 42.89% | 1.0433 |

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/AyushB27/PL-Prediction-Model.git
cd PL-Prediction-Model
pip install -r requirements.txt
```

### 2. Feature Generation (Optional)
Generate all leakage-free pre-match features:
```bash
python scripts/data.py
```

### 3. Run the Ablation Benchmark
Execute the full 4-tier × 5-model evaluation suite across walk-forward folds and holdout:
```bash
python scripts/run_experiments.py
```
This generates:
- `data/outputs/ablation_experiment_results.csv` (complete metric table)
- `data/outputs/model_predictions.csv` (fixture-level holdout predictions from the winning model)

### 4. Interactive Match Forecaster
Predict any fixture with the interactive CLI tool:
```bash
python scripts/predict_fixture.py "Arsenal" "Chelsea"
```

Output:
```text
=================================================================
PREMIER LEAGUE FIXTURE FORECAST: Arsenal vs Chelsea
=================================================================
Pre-Match Elo Rating:      Arsenal: 1823 | Chelsea: 1614
Recent Goals/Match (L5):   Arsenal: 1.40 | Chelsea: 0.80
Expected Goals (xG Rate):  Arsenal: 2.05 | Chelsea: 0.76
-----------------------------------------------------------------
WIN PROBABILITIES (Dixon-Coles Model):
  • Arsenal Win: 65.9%
  • Draw:        22.0%
  • Chelsea Win: 12.1%
-----------------------------------------------------------------
MOST LIKELY SCORELINES:
  Arsenal 2 - 0 Chelsea  (12.9%)
  Arsenal 1 - 0 Chelsea  (11.6%)
  Arsenal 1 - 1 Chelsea  (10.5%)
=================================================================
```

---

## 📁 Repository Structure

```
PL-Prediction-Model/
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                                 # Historical source match archives
│   ├── processed/
│   │   ├── merged_sorted_premier_league.csv # 4,560 canonical fixtures (2014-2026)
│   │   ├── understat_xg_2014_2025.csv       # Match-level Understat xG metrics
│   │   └── final_model_data.csv             # Leakage-safe, shifted feature dataset
│   └── outputs/
│       ├── ablation_experiment_results.csv  # Full 4-Tier x 5-Model metric benchmark
│       └── model_predictions.csv            # 2025-2026 holdout predictions
└── scripts/
    ├── build_historical_data.py             # Harmonizes raw archives with Understat xG
    ├── data.py                              # Computes rolling form, Elo, and fatigue
    ├── run_experiments.py                   # Main experiment & ablation pipeline
    └── predict_fixture.py                   # Interactive fixture prediction CLI
```
