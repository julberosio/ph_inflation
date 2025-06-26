import numpy as np
import pandas as pd

# Modular codes
from data import load_and_merge
from utils import (
    series_transformations,
    plot_latent_factors,
    plot_regime_probs,
    plot_forecasts,
    debug_forecast_inputs,
    plot_inflation_histogram,
    plot_trend_and_slope
)
from select_factors import bai_ng_criteria
from factors import BayesianFactorModel
from ms_regression import fit_markov_switching_regression
from forecast import forecast_inflation, simulate_forecast_distribution
import diagnostics

# Random seed for reproducibility
np.random.seed(10)

# 1. Load and preprocess data
df = load_and_merge('monthly_vars.csv', 'quarterly_vars.csv')
df = df[df.index >= '2002-01-31']
print(df.columns) # Print columns to interpret factor loadings

# 2. Visual check of inflation distribution
plot_inflation_histogram(df)

# 3. Transform series
df_t = series_transformations(df)

# 4. Find optimal number of latent factors (Bai and Ng)
X = df_t.drop(columns=['inflation'])
optimal_factors = bai_ng_criteria(X, max_k=3)
print(f"Optimal factors: {optimal_factors}")
k_factors = optimal_factors['ICp1']

# 5. Estimate latent factors
bfm = BayesianFactorModel(X, n_factors=k_factors, ar_order=1)
bfm.run_gibbs(n_iter=200)
factor_df = bfm.get_latent_factors()
phi = bfm.get_transition_matrix()
factor_var = bfm.factor_var
print(bfm.lam) # Print factor loadings

# 6. Fit regime-switching regression model
post = fit_markov_switching_regression(factor_df, df['inflation'], k_regimes=2)
plot_trend_and_slope(df['inflation'], post['cutoff_index'][0])

# 7. Align regime probabilities over the full sample for plotting
full_probs = pd.DataFrame(
    np.nan,
    index=factor_df.index,
    columns=[f"Regime {i}" for i in range(post['s_prob_matrix'].shape[1])]
)
full_probs.loc[post['cutoff_index']] = post['s_prob_matrix']

# 8. Forecast inflation from cutoff date onwards
cutoff = pd.Timestamp("2025-03-31")
forecasts = forecast_inflation(
    factor_df, phi, factor_var, post,
    cutoff=cutoff,
    horizon=3)

# 9. Compare to realised inflation if available
realized = df['inflation'].reindex(forecasts.index)
comparison = forecasts.copy()
comparison['realized_inflation'] = realized
print("\nForecast vs Realised:")
print(comparison)

# 10. Debug internals for cutoff
debug_forecast_inputs(factor_df, phi, post, cutoff)

# 11. Save plots
plot_latent_factors(factor_df)
plot_regime_probs(
    full_probs.values,
    full_probs.index,
    cutoff_date=post['cutoff_index'][0])
plot_forecasts(
    forecasts,
    df['inflation'],
    start_date=pd.Timestamp("2002-01-01"),
    cutoff_date=post['cutoff_index'][0])

# 12. Diagnostics and cross validation
diagnostics.forecast_accuracy_metrics(forecasts, realized)
diagnostics.regime_persistence(post)

cv_results = diagnostics.cross_validation(df)
diagnostics.plot_cv_errors(cv_results)
diagnostics.plot_cv_error_histograms(cv_results)
