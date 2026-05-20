# Empirical Asset Pricing: ML-Driven Dollar-Neutral Equity Strategy

> A machine learning approach to long-short equity portfolio construction, with rigorous out-of-sample validation, statistical significance testing, and transaction cost analysis.

**Authors:** Thanushree Keshava Murthy · Agniva Mondal (equal contribution)
**Course:** Purdue University, Krannert School of Management

---

## Headline Results

| Metric | LightGBM Strategy | ElasticNet (linear baseline) |
|---|---|---|
| Annualized Return | **12.28%** | -0.27% |
| Annualized Volatility | 15.47% | 8.72% |
| **Sharpe Ratio** | **0.82** | 0.01 |
| Sortino Ratio | 2.16 | — |
| Maximum Drawdown | -18.73% | -20.30% |
| Annual Turnover | 11.1× | 11.1× |
| **Monte Carlo p-value** | **0.032** | — |

**Key takeaway:** The tree-based model (LightGBM) extracted real predictive signal from firm characteristics, while the linear baseline (ElasticNet) did not. The strategy's Sharpe ratio passes a 1,000-iteration permutation test at the 5% significance level.

**95% bootstrap confidence interval on Sharpe:** [0.46, 1.66]
**Test period:** 84 months out-of-sample

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Why This Matters](#why-this-matters)
3. [Methodology](#methodology)
4. [Data](#data)
5. [Pipeline](#pipeline)
6. [Detailed Results](#detailed-results)
7. [Robustness Tests](#robustness-tests)
8. [Limitations](#limitations)
9. [How to Run](#how-to-run)
10. [Repository Structure](#repository-structure)

---

## What This Project Does

This project builds and tests a **dollar-neutral long-short equity strategy** that uses machine learning to predict next-month stock returns from firm-level characteristics (size, value, momentum, profitability, investment, and many others — 174 features in total).

A dollar-neutral strategy buys a basket of stocks the model predicts will outperform and sells short an equal-dollar basket it predicts will underperform. If the predictions have any real edge, the long minus short return should be positive on average, with low correlation to the broader market.

The project follows the methodology of Gu, Kelly, and Xiu (2020) *"Empirical Asset Pricing via Machine Learning"* — a foundational paper in the modern quantitative finance literature — and adds explicit statistical significance testing, regime-stability analysis, and transaction cost sensitivity.

---

## Why This Matters

Most undergraduate and master's-level finance projects calculate a Sharpe ratio on a backtest and declare victory. This project goes further:

- **Out-of-sample validation:** No peeking at future data. The model trains on rolling 10-year windows, tunes on 5-year validation, and tests on the next 1-year window.
- **Statistical significance:** A Sharpe of 0.8 means nothing if a random strategy gets the same number 50% of the time. We run 1,000 Monte Carlo permutations to test whether the result is statistically distinguishable from luck.
- **Bootstrap confidence intervals:** Single point estimates are misleading. We compute block-bootstrap 95% CIs on Sharpe to show the realistic range.
- **Regime stability:** Strategies that work in one market environment often fail in another. We split the test period and re-evaluate.
- **Transaction cost realism:** A paper Sharpe of 1.0 can collapse to 0.2 once you pay 20 basis points per trade. We sweep this explicitly.

---

## Methodology

### Models

Two models are trained and compared:

1. **ElasticNet (linear baseline)** — L1 + L2 regularized linear regression, the simplest credible model for high-dimensional firm characteristics.
2. **LightGBM (tree ensemble)** — gradient-boosted decision trees, well-suited to capturing nonlinear interactions across characteristics.

The comparison is intentional: a tree model that beats a linear baseline is evidence that nonlinearities matter. A tree model that does not beat the baseline is evidence that the signal (if any) is linear.

### Rolling Backtest Windows

```
| ----------- TRAIN (10 yr) ----------- | -- VAL (5 yr) -- | -- TEST (1 yr) -- |
        |--- next window: TRAIN ---|------ VAL ------|--- TEST ---|
```

Windows roll forward by 1 test year at a time. This ensures:
- No future information leaks into training
- Hyperparameter tuning uses validation data only
- Each year of test data is evaluated by a fresh model

### Portfolio Construction

Each month:
1. Predict next-month excess return for every stock
2. Rank stocks by predicted return
3. **Long** the top 10% (391 stocks per month on average)
4. **Short** the bottom 10% (equal dollar amount)
5. Apply 36-month rolling volatility targeting (10% annualized vol target, 3× leverage cap)
6. Hold for one month, then rebalance

### Cross-Sectional Preprocessing

Within each calendar month (cross-sectional), features are:
1. Winsorized at the 1st and 99th percentiles (to limit outlier influence)
2. Z-score standardized (mean 0, std 1)
3. Median-imputed for missing values

This is done per-month to avoid look-ahead bias — only contemporaneous information is used to standardize each month's data.

---

## Data

The dataset was provided via a private course Kaggle competition by the instructor.

**Schema:** CRSP-style monthly equity panel
- `permno` — stable stock identifier
- `date` — month-end date
- `ret_excess` — monthly excess return over the risk-free rate
- 174 firm characteristics (pre-normalized to [-1, 1])

**Dimensions:** 642,848 stock-month observations spanning 265 months (~22 years).

**Note on time period:** The data covers a historical sample (approximately 1974–1996 based on date stamps in outputs). Results would need to be re-validated on modern data before any practical use. The methodology is the contribution; the specific numbers are sample-dependent.

**Data is not redistributed in this repository** since it was provided under course terms. To run the notebook, you would need either the original course dataset or a comparable panel (e.g., Open Source Asset Pricing by Chen & Zimmermann, or the JKP Global Factor Data).

---

## Pipeline

```
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  1. Load + Clean   │   │  2. Cross-Section  │   │  3. Rolling Split  │
│                    │ → │     Preprocess     │ → │   (10/5/1 yr)      │
│  Parquet → Panel   │   │  Winsorize+ZScore  │   │  No leakage        │
└────────────────────┘   └────────────────────┘   └────────────────────┘
                                                            ↓
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ 6. Statistical     │   │  5. Portfolio      │   │  4. Train Models   │
│    Validation      │ ← │   Construction     │ ← │  ElasticNet + LGBM │
│  MC + Bootstrap    │   │  Long-Short 10%    │   │  Hyperparam Tune   │
└────────────────────┘   └────────────────────┘   └────────────────────┘
```

---

## Detailed Results

### Model Comparison

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Turnover |
|---|---|---|---|---|---|---|
| **LightGBM (raw)** | **12.28%** | 15.47% | **0.82** | 2.16 | -18.73% | 11.1× |
| LightGBM (vol-targeted) | 9.70% | 14.83% | 0.69 | 1.64 | -18.73% | 11.1× |
| ElasticNet | -0.27% | 8.72% | 0.01 | — | -20.30% | 11.1× |

**Interpretation:** The linear model fails to extract signal. The tree model produces an economically meaningful return-to-risk profile. The vol-targeted version (the more practical implementation) lowers absolute returns but improves risk-adjusted comparability.

### Monte Carlo Permutation Test

A 1,000-iteration permutation test shuffles the time ordering of predictions while preserving the cross-sectional structure. If the strategy's Sharpe is real signal rather than luck, the actual Sharpe should be in the right tail of the null distribution.

| Metric | Value |
|---|---|
| Actual Sharpe (vol-targeted) | 0.6936 |
| Empirical p-value | **0.032** |
| P(null Sharpe > 0) | 0.523 |
| P(null Sharpe > 1) | 0.005 |

The strategy passes the test at the 5% level: only 3.2% of random permutations produced a Sharpe as high or higher.

### Alpha Regression

Regressing strategy returns on the market factor (CAPM alpha):

| Metric | Value |
|---|---|
| Monthly Alpha | 0.86% |
| Alpha t-statistic | 1.83 |
| Sharpe 95% CI | [0.46, 1.66] |

Alpha is marginally significant (t ≈ 1.83, just below the conventional 2.0 threshold), and the Sharpe confidence interval excludes zero but is wide — reflecting the realistic uncertainty around a single-sample backtest.

---

## Robustness Tests

### Transaction Cost Sensitivity

| Round-trip cost | Net Sharpe |
|---|---|
| 0 bps | 0.69 |
| 5 bps | 0.66 |
| 10 bps | 0.62 |
| 20 bps | 0.54 |

The strategy remains positive at realistic institutional cost levels (5–10 bps). At 20 bps (closer to retail) it weakens substantially but is still positive.

### Drawdown Analysis

| Model | Max DD | Longest DD (months) | Worst 1-month | Worst 3-month |
|---|---|---|---|---|
| LightGBM | -18.7% | 25 | -7.1% | -16.0% |
| ElasticNet | -20.3% | 82 | -13.7% | -17.3% |

LightGBM recovers from drawdowns faster (25 months vs 82) and has milder tail months.

---

## Limitations

This section is included because mature analysts disclose what they don't know.

1. **Sample period is historical.** The data covers approximately 1974–1996. Markets have changed: HFT, ETF dominance, factor crowding, and regime shifts after 2008 are not represented. Results may not generalize to recent data.

2. **No post-2000 regime test.** The course dataset did not include modern data, so the regime analysis cannot validate stability across the 2008 GFC, COVID, or the 2022 rate cycle.

3. **High turnover (11×/year).** This is acceptable at institutional cost levels but would degrade significantly at retail trading costs.

4. **Single asset class.** Only US equities. Strategy may not generalize to other markets or asset classes.

5. **Survivorship bias not explicitly addressed.** Standard CRSP data has known biases; this analysis inherits them.

6. **No live execution simulation.** Backtest assumes mid-quote fills with no slippage or market impact. Real implementation would face execution friction.

7. **One realization of luck.** Even a well-tested strategy can over-fit a specific historical sample. Out-of-sample testing on modern data is the next step.

---

## How to Run

### Prerequisites

- Python 3.9+
- An equity panel dataset matching the schema described in [Data](#data)
- (Recommended) Google Colab Pro or a machine with ≥16GB RAM

### Setup

```bash
git clone https://github.com/thanushreekmurthy1999-hub/empirical-asset-pricing-ml.git
cd empirical-asset-pricing-ml
pip install -r requirements.txt
```

### Running the Notebook

1. Open `Machine_Learning_Project.ipynb` in Jupyter or Colab
2. Update the `PARQUET_DIR` path to point to your data
3. Run all cells (full pipeline takes ~30–60 minutes depending on hardware)

### Notes

- The notebook caches intermediate results to avoid re-running expensive steps
- Monte Carlo permutation test runs 1,000 iterations by default; reduce if needed via `CONFIG["monte_carlo_sims"]`
- Plot output is controlled by `CONFIG["make_plots"]`

---

## Repository Structure

```
empirical-asset-pricing-ml/
├── Machine_Learning_Project.ipynb   # Full pipeline (load → preprocess → train → backtest → test)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Excludes data, figures, cache files
└── figures/                          # Generated charts (not committed)
```

---

## Methodology References

- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies.
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap.* Journal of the American Statistical Association.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

---

## Acknowledgments

This project was developed as part of the MS Business Analytics & Information Management program at Purdue University, under the guidance of [course instructor name optional]. Co-authored with [Agniva Mondal](https://github.com/agniva1534), equal contribution.

---

*Built as a course project in Empirical Asset Pricing. Not investment advice. Past performance does not guarantee future results.*
