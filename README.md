# WC26 Dixon-Coles Engine

A regularized Dixon-Coles match-outcome model and Monte Carlo tournament simulator, built end-to-end for the FIFA World Cup 2026 — from raw historical data to a validated, competition-schema submission for all 104 matches.

`Python` `NumPy` `SciPy` `Pandas` `Statistical Modeling` `Maximum Likelihood Estimation` `Monte Carlo Simulation`

---

## Why this project

Predicting a single football match is a straightforward statistics problem; predicting an entire 104-match tournament is a complex systems problem. The model output must propagate correctly through a stateful simulation, remain numerically stable across thousands of Monte Carlo runs, and pass a strict competition schema involving complex bracket routing.

This repository contains the complete, end-to-end pipeline: data acquisition, feature engineering, statistical modeling, rigorous backtesting, tournament simulation, and automated submission auditing.

## Engineering highlights

- **Statistical modeling from first principles** — implemented the Dixon-Coles bivariate Poisson model (Dixon & Coles, 1997) directly from the paper, including the low-score dependence correction.
- **Custom maximum likelihood estimation** — parameters (attack, defense, home advantage) are fit via `scipy.optimize` with a hand-derived negative log-likelihood, identifiability constraints, and fallback optimizers (L-BFGS-B → SLSQP).
- **Context-aware weighting** — historical results are weighted by recency (730-day half-life) and competitive tier (World Cup > continental championship > qualifier > friendly).
- **Regularization & numerical stability** — L2 penalties on attack/defense parameters and log-space clipping prevent overflow and overfitting on sparse data.
- **Rigorous backtesting** — fitted parameters are validated out-of-sample against World Cup 2022 results using Brier score, log-loss, and Ranked Probability Score (RPS), not just raw accuracy.
- **Monte Carlo tournament simulation** — a 10,000-iteration simulator resolves the entire bracket, producing consensus semifinalists and full champion probability distributions rather than a fragile single-point prediction.

## Pipeline architecture

```
┌─────────────────┐       ┌──────────────────┐     ┌────────────────────────┐
│  Kaggle: intl.  │─────▶│  Feature build   │────▶│  Dixon-Coles fit       │
│  results 1872-  │      │  (Elo ratings,   │      │  (time decay, tier     │
│  present        │      │  goal averages)  │      │  weights, L2 reg)      │
└─────────────────┘      └──────────────────┘      └──────────┬─────────────┘
                                                              │
                          ┌───────────────────────────────────┴───────────────────────┐
                          ▼                                                           ▼
                ┌───────────────────┐                                     ┌──────────────────────┐
                │  Backtest vs.     │                                     │  10,000-run Monte    │
                │  World Cup 2022   │                                     │  Carlo tournament    │
                │  (Brier, log-loss,│                                     │  simulation          │
                │  RPS)             │                                     │  (group → knockout)  │
                └───────────────────┘                                     └──────────┬───────────┘
                                                                                     │
                                                                                     ▼
                                                                        ┌──────────────────────┐
                                                                        │  Submission compile  │
                                                                        │  + schema/sanity     │
                                                                        │  audit               │
                                                                        └──────────────────────┘
```

## The model

### Dixon-Coles bivariate Poisson

Home and away goals are modeled as Poisson-distributed, with means derived from team-specific attack and defense strengths:

```
lambda (home goals) = exp(attack_home + defense_away + home_advantage)
mu     (away goals) = exp(attack_away + defense_home)
```

The plain bivariate Poisson assumes home and away goals are independent, underestimating the frequency of low-scoring draws. Dixon-Coles corrects this with a dependence adjustment `tau(x, y, rho)` applied to 0-0, 1-0, 0-1, and 1-1 scorelines:

| Scoreline | Adjustment |
|---|---|
| 0-0 | `1 - lambda · mu · rho` |
| 1-0 | `1 + mu · rho` |
| 0-1 | `1 + lambda · rho` |
| 1-1 | `1 - rho` |
| other | `1.0` |

### Fitting methodology

Parameters are estimated by maximizing the weighted log-likelihood across historical matches, subject to:

- **Time decay** — `weight = exp(-ln(2) · days_ago / 730)`
- **Tournament-tier weighting** — World Cup (1.0), continental (0.8), qualifiers (0.5), friendlies (0.1)
- **Identifiability constraint** — `sum(attack) = 0` in log space, resolved via L-BFGS-B

Two implementations are provided:

| File | Purpose |
|---|---|
| `src/models/dixon_coles.py` | Clean, reusable `DixonColesMatchPredictor` class (`fit`, `predict_score_probs`), covered by unit tests. The reference implementation. |
| `src/models/fit_dixon_coles_regularized.py` | Production fitting script with the full weighting scheme, regularization, and diagnostics used to generate final parameters. |

## Repository structure

```
.
├── src/
│   ├── data_loader.py                      # Path-safe CSV loader
│   ├── data/
│   │   └── make_dataset.py                 # Kaggle API ingestion
│   ├── features/
│   │   └── build_features.py               # Elo ratings + goals-for/against
│   ├── models/
│   │   ├── dixon_coles.py                  # Core model class (unit-tested)
│   │   ├── fit_dixon_coles_regularized.py  # Production fitting script
│   │   ├── evaluate_backtest.py            # Out-of-sample validation vs. 2022
│   │   ├── predict_group_stage.py          # [DEV] Lightweight Elo-heuristic predictions
│   │   ├── predict_knockouts.py            # [DEV] Lightweight bracket heuristic predictions
│   │   ├── run_ev_simulator.py             # [PROD] Canonical 10k Monte Carlo simulator
│   │   └── compile_submission.py           # Merges outputs into final submission
│   └── scripts/
│       └── audit_submission.py             # Schema/business-rule validation
├── data/
│   ├── group_fixtures.csv                  # Matches 1-72 schedule
│   └── knockout_slots.csv                  # Matches 73-104 bracket slot definitions
├── tests/
│   ├── test_dixon_coles.py                 # Model correctness (tau adjustment, etc.)
│   └── test_data_loader.py                 # I/O contract and error handling
├── notebook.ipynb                          # End-to-end exploratory workflow
├── requirements.txt
└── submissions/
    └── final_ev_submission.csv             # Final canonical submission artifact
```

## Getting started

```bash
git clone https://github.com/MercuryConnor/wc26-dixon-coles-engine.git
cd wc26-dixon-coles-engine
pip install -r requirements.txt
```

A configured Kaggle API token (`~/.kaggle/kaggle.json`) is required for data acquisition.

### Running the full pipeline

Run each step from the project root so package-relative imports resolve correctly:

```bash
# 1. Pull historical match results & build features
python -m src.data.make_dataset
python -m src.features.build_features

# 2. Fit the regularized Dixon-Coles model & backtest
python -m src.models.fit_dixon_coles_regularized
python -m src.models.evaluate_backtest

# 3. Generate the canonical submission via Monte Carlo simulation
python -m src.models.run_ev_simulator

# 4. Audit the final submission against the competition schema
python -m src.scripts.audit_submission
```

## Engineering decisions & known constraints

- **Canonical prediction path** — two prediction paths exist. `predict_group_stage.py` uses a lightweight Elo-differential heuristic for fast iteration/debugging. `run_ev_simulator.py` uses the fully fitted Dixon-Coles parameters and is the canonical path used to generate the final submission.
- **Knockout variance trade-off** — knockout-stage expected goals in the simulator are rounded deterministically per run rather than sampled, trading some high-scoring knockout variance for strict topological bracket stability across 10,000 iterations.
- **Name normalization** — team-name matching between historical datasets and the 2026 fixture list relies on exact string matching. Unmatched teams receive a baseline Elo of 1500 (flagged for future normalization work).

## References

Dixon, M.J. and Coles, S.G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market." *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.
