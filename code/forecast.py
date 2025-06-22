import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def forecast_inflation(
    factor_df: pd.DataFrame,
    phi: np.ndarray,
    factor_var: np.ndarray,
    post: dict,
    cutoff: pd.Timestamp,
    horizon: int = 3,
    stoch: bool = False
) -> pd.DataFrame:
    """
    Generate multi-step inflation forecasts using stochastic AR dynamics of latent factors
    and regime-switching regression posteriors.

    Args:
        factor_df (pd.DataFrame): Latent factors time series indexed by date (T x K).
        phi (np.ndarray): Transition matrix for latent factors (K x K).
        factor_var (np.ndarray): Innovation variances for each latent factor (K,).
        post (dict): Posterior estimates from MS regression including:
            - 'alpha': regime intercepts (M,)
            - 'beta': regime-specific slope vectors (M x K)
            - 'sigma2': regime-specific variances (M,)
            - 'P': regime transition matrix (M x M)
        cutoff (pd.Timestamp): Last date used for model training.
        horizon (int): Forecast horizon in months.
        stoch (bool): Adds a shock to the f matrix for stochastic forecast if True.

    Returns:
        pd.DataFrame: Forecasted inflation indexed by forecast dates.

    Procedure:
        1. Extract last estimated latent factor vector f_T at cutoff.
        2. Iterate h = 1 to horizon:
            a. Project latent factor f_{T+h} = phi @ f_{T+h-1} + ε_h, where ε_h ~ N(0, factor_var)
            b. Compute regime-specific forecasts: μ_j = α_j + β_j @ f_{T+h}
            c. Compute regime weights from σ²_j-adjusted squared deviation
            d. Aggregate forecast as weighted average across regimes.
    """

    alpha = post['alpha']
    beta = post['beta']
    sigma2 = post['sigma2']          

    f = factor_df.loc[:cutoff].iloc[-1].values
    f = np.atleast_1d(f).flatten()

    # Ensure phi is square (K x K)
    if phi.shape[1] == 1:
        phi = np.diagflat(phi)

    forecasts = []
    idx = pd.period_range(cutoff, periods=horizon+1, freq='M')[1:].to_timestamp('M')

    for _ in range(horizon):
        if stoch:
            shock = np.random.normal(loc=0, scale=np.sqrt(factor_var), size=f.shape)
        else:
            shock = 0
        f = phi @ f + stoch

        mu = alpha + beta @ f
        mu_bar = mu.mean()
        weights = np.exp(-0.5 * ((mu - mu_bar) ** 2) / sigma2)
        weights /= weights.sum()

        forecast = np.dot(weights, mu)
        forecasts.append(forecast)

    return pd.DataFrame({'inflation_forecast': forecasts}, index=idx)


def simulate_forecast_distribution(
    factor_df, phi, factor_var, post,
    cutoff, horizon=6, n_sim=500
):
    """
    Simulate multiple inflation forecasts with stochastic factor innovations and plot a fan chart.

    Args:
        factor_df (pd.DataFrame): Latent factors.
        phi (np.ndarray): Transition matrix.
        factor_var (np.ndarray): Innovation variances.
        post (dict): Posterior from MS regression.
        cutoff (pd.Timestamp): Start date of forecast.
        horizon (int): Forecast length.
        n_sim (int): Number of Monte Carlo simulations.
    """

    all_paths = []
    original_state = np.random.get_state()  # Save RNG state

    for i in range(n_sim):
        np.random.seed(i)  # Different seed per simulation
        forecast = forecast_inflation(
            factor_df, phi, factor_var, post,
            cutoff=cutoff, horizon=horizon, stoch=True
        )
        all_paths.append(forecast)

    np.random.set_state(original_state)  # Restore RNG state

    forecasts = np.stack(all_paths)  # Shape: (n_sim, horizon)
    dates = pd.period_range(cutoff, periods=horizon+1, freq='M')[1:].to_timestamp('M')

    # Percentiles for fan (flatten to 1D for matplotlib)
    percentiles = np.percentile(forecasts, [10, 25, 50, 75, 90], axis=0)
    p10, p25, p50, p75, p90 = [p.flatten() for p in percentiles]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.fill_between(dates, p10, p90, color='orange', alpha=0.2, label='10–90%')
    plt.fill_between(dates, p25, p75, color='orange', alpha=0.4, label='25–75%')
    plt.plot(dates, p50, color='black', linewidth=1.5, label='Median Forecast')

    plt.title("Inflation Forecast Fan Chart")
    plt.ylabel("Inflation (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fan_plot.pdf")
    plt.close()
    print("Saved fan_plot.pdf")
