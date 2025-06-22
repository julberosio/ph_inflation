# Replication Package for: Forecasting Philippine Inflation with a Bayesian Markov-Switching Dynamic Factor Model

## June 2025

This replication package accompanies Osio, Julber (2025). "Forecasting Philippine Inflation with a Bayesian Markov-Switching Dynamic Factor Model." Term paper, Barcelona School of Economics.

## Abstract in English

I estimate a Bayesian two-stage model to forecast Philippine inflation over the second quarter of 2025. A dynamic factor model extracts two latent macroeconomic drivers from a high-dimensional panel of monthly and quarterly variables, including a novel typhoon exposure index constructed from spatial wind field data. Inflation is modelled as a regime-switching linear function of these factors, with coefficients and variances that evolve according to a first-order Markov process. To improve short-term forecast accuracy, the regime model is estimated only on a post-break subsample, with the break identified endogenously based on smoothed inflation trends. Forecasts for April–June 2025 remain below 1.5\%, closely tracking realised values where available. The model is implemented via Gibbs sampling and validated through rolling cross-validation, achieving a mean absolute percentage error (MAPE) of 4.4\% for the forecast period.

JEL Classification: C11, C32, E31
Keywords: Bayesian factor model, inflation forecasting, regime switching

---

## 📁 Project Structure

```
project_root/
├── code/                      # All Python scripts for estimation, forecasting, and diagnostics
│   ├── main.py                # Orchestrates the full forecasting pipeline
│   ├── data.py                # Loads and merges monthly + quarterly datasets
│   ├── utils.py               # Transformations, plotting, forward-backward algorithm
│   ├── select_factors.py      # Bai and Ng (2002) information criteria
│   ├── factors.py             # Bayesian dynamic factor model (latent factors)
│   ├── ms_regression.py       # Markov-switching inflation regression
│   ├── forecast.py            # Forecast generation and simulation
│   └── diagnostics.py         # Accuracy metrics and cross-validation logic
├── data_prep/                 # Raw and preprocessed data files (CSV, R interpolations, etc.)
├── output/                    # All generated figures and tables (PDF format)
└── requirements.txt           # Python dependencies for reproducibility
```

---

## ▶️ How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the pipeline**

   Simply run:

   ```bash
   python main.py
   ```

   This performs the entire process:
   - Loads and transforms the dataset
   - Estimates the dynamic factor model
   - Trains the Markov-switching regression on post-break inflation data
   - Forecasts inflation over 2025Q2
   - Evaluates forecast performance
   - Saves all plots and tables to

---

## 📦 Requirements

See `requirements.txt`. The code runs on Python 3.8+ and uses:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- No proprietary packages

---

## 🧠 Model Overview

- **Stage 1: Dynamic Factor Model**
  - Bayesian estimation of latent factors summarising macroeconomic indicators
  - Factor model includes AR(1) dynamics
  - Implemented in `factors.py`

- **Stage 2: Regime-Switching Inflation Model**
  - Inflation depends on the latent factor via regime-specific linear equations
  - Regimes switch via a Markov process
  - Trained only on post-break period (identified endogenously)
  - Implemented in `ms_regression.py`

---

## 📈 Output

Running `main.py` produces:

| Output File                     | Description                                |
|--------------------------------|--------------------------------------------|
| `latent_factors.pdf`           | Estimated latent factor over time          |
| `inflation_histogram.pdf`      | Histogram and KDE of inflation             |
| `regime_probs_with_shade.pdf`  | Smoothed regime probabilities              |
| `trend_break_detection.pdf`    | Inflation, trend, slope and trend break    |
| `forecast_plot.pdf`            | Inflation: historical, forecast, realised  |
| `cv_errors_over_time.pdf`      | Rolling RMSE and MAE across windows        |
| `cv_error_histograms.pdf`      | Histograms of RMSE and MAE                 |
| `cv_results.csv`               | Complete results of cross validation       |


All outputs are stored in the `output/` directory.

---

## 📝 Data Sources

Gahtan, J. et al. (2024). *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01*. NOAA National Centers for Environmental Information.
Knapp, Kenneth R et al. (2010). “The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data”. In: *Bulletin of the American Meteorological Society* 91.3, pp. 363–376.

---

## 🛠 Reproducibility

- All steps are automated in `main.py`
- Data cleaning and interpolation (e.g. Denton method for unemployment) are documented in `data_prep/`
- Modular codebase allows for easy adaptation to other countries or variables

---

Author: **Julber Osio**  
June 2025
