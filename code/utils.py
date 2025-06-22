import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def series_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to time series data.

    Rationale:
    - Economic aggregates or indices (forex rate, M3 money supply, remittances, loans, PPI, GDP, imports):
      use log diff scaled by 100 to approx. pct growth rate, which are stationary, interpretable as returns or growth
    - Policy rates (rrp) and unemployment rate (unemp):
      use simple differences to capture absolute change, which better reflect dynamics
    - Standardise all predictors (mean 0, std 1) excluding target inflation for comparability and to stabilise estimation

    Args:
        df (pd.DataFrame): Input data with raw variables

    Returns:
        pd.DataFrame: Transformed data with stationary predictors
    """
    df_t = df.copy()

    # Log differences for percentage growth rates:
    for col in ['forex', 'm3', 'remittances', 'loans', 'ppi', 'gdp', 'imports']:
        if col in df_t.columns:
            df_t[col] = 100 * np.log(df_t[col]).diff()

    # Simple first differences
    for col in ['rrp', 'unemp']:
        if col in df_t.columns:
            df_t[col] = df_t[col].diff()

    # Standardise predictors (mean 0, std 1)
    preds = df_t.columns.difference(['inflation'])
    df_t[preds] = (df_t[preds] - df_t[preds].mean()) / df_t[preds].std()

    return df_t


def forward_backward(log_lik: np.ndarray, P: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Forward-Backward algorithm for smoothing hidden Markov models (HMMs)

    This algorithm computes the posterior distribution over the hidden states (regimes) given the observed data likelihoods and the state transition probabilities, then samples a sequence of hidden states according to this posterior.

    Args:
    - log_lik (numpy.ndarray): Matrix of log-likelihoods (T time periods x M regimes).
      Each element: log_lik[t, j] = log P(observation at time t | state j).

    - P (numpy.ndarray): State transition probability matrix (M x M).
      Each element: P[i, j] = P(state j at time t | state i at time t-1).
      Rows sum to 1.

    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - s (numpy.ndarray): Array of sampled hidden states (regimes),
      where each s[t] is an integer in [0, M-1] representing the state at time t (T x 1)

    Procedure:

    1. Convert P to log scale to avoid numerical underflow: log_P = log(P + small_eps).

    2. Forward pass: compute alpha (log_alpha),
       where log_alpha[t, j] = log P(observations up to t, state j at time t).

       Initialisation (t=0):
         log_alpha[0] = log_lik[0]  (initial likelihood for each state at first time)

       Recursion (t=1 to T-1):
         For each state j,
         log_alpha[t, j] = log_lik[t, j] + logsumexp over i of [log_alpha[t-1, i] + log_P[i, j]]

       Explanation:
         - For each state j at time t, sum over all possible previous states i
           the joint log probability of observing data up to t-1 and transitioning from i to j, then add likelihood of data at time t for state j.

    3. Backward pass: compute beta (log_beta),
       where log_beta[t, i] = log P(observations after t | state i at time t).

       Initialisation (t=T-1):
         log_beta[T-1] = 0 (log(1)), no future observations after last time

       Recursion (t=T-2 down to 0):
         For each state i,
         log_beta[t, i] = logsumexp over j of [log_P[i, j] + log_lik[t+1, j] + log_beta[t+1, j]]

       Explanation:
         - For each state i at time t, sum over all next states j
           the joint log probability of transitioning from i to j, observing data at t+1, and the future data likelihoods after t+1.

    4. Combine forward and backward messages to compute smoothing probabilities:

       For each time t and state j,
       log_gamma[t, j] = log_alpha[t, j] + log_beta[t, j]

       Normalise by subtracting max log_gamma over states to improve numerical stability, then exponentiate and normalise to get gamma[t, j] = P(state j at t | all observations).

    5. Sample the hidden state sequence s of length T by sampling each s[t]
       independently from the categorical distribution with probabilities gamma[t].

    Note: Sampling states independently given gamma is a simplification often used for approximate inference.
    """

    if seed is not None:
        np.random.seed(seed)

    T, M = log_lik.shape # Log-likelihood matrix: T time, M regimes
    log_alpha = np.zeros((T, M)) # Forward probabilities in log
    log_beta = np.zeros((T, M)) # Backward probabilities in log
    log_P = np.log(P + 1e-12) # To prevent log(0)

    # Forward recursion initialisation at t=0
    log_alpha[0] = log_lik[0]

    # Forward recursion at t=1...T-1
    for t in range(1, T):
        for j in range(M):
            # For each state j at time t,
            # sum over previous states i the probability of arriving in j
            # from i, adding previous log_alpha and transition log-probabilities
            log_alpha[t, j] = log_lik[t, j] + np.logaddexp.reduce(log_alpha[t-1] + log_P[:, j])

    # Backward recursion initialisation at t=T-1
    log_beta[-1] = 0

    # Backward recursion at time t=T-2...0
    for t in reversed(range(T - 1)):
        for i in range(M):
            # For each state i at time t,
            # sum over next states j the product of transition prob,
            # likelihood of data at t+1, and future data likelihood
            log_beta[t, i] = np.logaddexp.reduce(log_P[i, :] + log_lik[t+1, :] + log_beta[t+1, :])

    # Combine forward and backward to get smoothing log probs
    log_gamma = log_alpha + log_beta
    # Subtract max log prob per time for numerical stability
    log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    # Normalise prob to sum to 1 per t
    gamma /= gamma.sum(axis=1, keepdims=True)

    # Sample states according to smoothing probabilities
    s = np.array([np.random.choice(M, p=gamma[t]) for t in range(T)])
    return s


def debug_forecast_inputs(factor_df, phi, post, cutoff):
    """
    Print detailed components used in one-step ahead inflation forecast at a given cutoff date.

    This function helps break down how the forecast is constructed:
    - Uses last latent factor vector
    - Applies AR transition to get f_{t+1}
    - Computes regime-specific inflation expectations
    - Weights them using posterior regime probabilities

    Args:
        factor_df (pd.DataFrame): Latent factors over time (T x K), indexed by date.
        phi (np.ndarray): Transition matrix for AR(K) latent factors (K x K).
        post (dict): Posterior estimates including:
            - 'alpha': regime intercepts (M,)
            - 'beta': regime slopes (M x K)
            - 'P': transition matrix (M x M)
            - 's_prob_matrix': smoothed regime probabilities (T x M)
            - 'cutoff_index': index of training sample (length T)
        cutoff (pd.Timestamp): Date at which to print 1-step forecast decomposition.

    Procedure:
        1. Find the local index in the regime matrix corresponding to cutoff.
        2. Extract regime probabilities at that date.
        3. Extract last latent factor vector \( f_t \) from factor_df.
        4. Compute \( f_{t+1} = \Phi f_t \).
        5. Compute regime-specific inflation means: \( \mu_j = \alpha_j + \beta_j^\top f_{t+1} \).
        6. Weight the means by posterior probabilities to get final forecast.
    """

    try:
        local_idx = post['cutoff_index'].get_loc(cutoff)
        p = post['s_prob_matrix'][local_idx]  # Regime weights
    except (IndexError, KeyError):
        print("Warning: cutoff index out of range for regime probabilities; skipping debug.")
        return

    alpha = post['alpha']
    beta = post['beta']
    f_t = factor_df.loc[:cutoff].iloc[-1].values
    f_next = phi @ f_t

    # Regime-specific means
    regime_means = alpha + beta @ f_next
    weighted_forecast = np.dot(p, regime_means)

    # Print results
    print("\nForecast Components")
    print(f"Last latent factor f_t: {np.round(f_t, 4)}")
    print(f"Forecasted f_t+1: {np.round(f_next, 4)}")
    print(f"\nRegime weights: {np.round(p, 3)}")
    print(f"Regime-specific means: {np.round(regime_means, 4)}")
    print("Regime intercepts (alpha):", post['alpha'])
    print("Regime slopes (beta):", post['beta'])
    print("Regime variances (sigma²):", post['sigma2'])
    print(f"\nWeighted forecast (1-step ahead): {weighted_forecast:.4f}")



def plot_latent_factors(factor_df: pd.DataFrame):
    """
    Plot latent factor time series.

    Args:
        factor_df (pd.DataFrame): DataFrame with latent factor columns indexed by time

    Saves:
        latent_factors.pdf
    """
    plt.figure(figsize=(10, 4))
    for col in factor_df.columns:
        plt.plot(factor_df.index.to_numpy(), factor_df[col].to_numpy(), label=col)
    plt.title("Latent Factors")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("latent_factors.pdf")
    plt.close()
    print("Saved latent_factors.pdf")


def plot_regime_probs(s_probs: np.ndarray, index: pd.Index, cutoff_date: pd.Timestamp = None):
    """
    Plot smoothed regime probabilities over time with optional shading for pre-training period.
    Args:
        s_probs (np.ndarray): T x M regime probability matrix
        index (pd.Index): Time index corresponding to s_probs rows
        cutoff_date (pd.Timestamp, optional): Date before which regime probs not estimated
    Saves:
        regime_probs_with_shade.pdf
    """
    min_len = min(len(index), len(s_probs))
    plt.figure(figsize=(12, 4))

    if cutoff_date is not None:
        # Shade area before cutoff to show pre-training period
        plt.axvspan(index.min(), cutoff_date, color='gray', alpha=0.3, label='Pre-training period')
        plt.axvline(cutoff_date, color='red', linestyle='--', label='MS training start')

    # Plot each regime prob
    for j in range(s_probs.shape[1]):
        plt.plot(index.to_numpy()[:min_len], s_probs[:min_len, j], label=f"Regime {j}")

    plt.title("Smoothed Regime Probabilities")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regime_probs_with_shade.pdf")
    plt.close()
    print("Saved regime_probs_with_shade.pdf")


def plot_forecasts(forecast_df: pd.DataFrame, inflation_series: pd.Series, start_date: pd.Timestamp = pd.Timestamp("2024-01-01"), cutoff_date: pd.Timestamp = None):
    """
    Plot historical inflation, forecasts, and realised inflation from a specified start date.
    Args:
        forecast_df (pd.DataFrame): Forecasted inflation indexed by date
        inflation_series (pd.Series): Actual inflation series
        start_date (pd.Timestamp): Date to start plot
        cutoff_date (pd.Timestamp, optional): Date before which regime probs not estimated
    Saves:
        forecast_plot.pdf
    """
    plt.figure(figsize=(10, 4))

    # Filter inflation from start_date
    inflation_zoom = inflation_series[inflation_series.index >= start_date]

    if cutoff_date is not None:
        plt.axvline(cutoff_date, color='red', linestyle='--', label='MS training start')

    # Plot historical inflation from start_date onwards
    plt.plot(inflation_zoom.index.to_numpy(), inflation_zoom.to_numpy(), label="Historical Inflation", linewidth=1.5)

    # Plot forecasted inflation
    plt.plot(forecast_df.index.to_numpy(), forecast_df.iloc[:, 0].to_numpy(), linestyle='-', color='orange', label="Forecasted Inflation")

    # Plot realised inflation for forecast months
    realised = inflation_series.reindex(forecast_df.index)
    plt.plot(realised.index.to_numpy(), realised.to_numpy(), linestyle='-', color='green', label="Realised Inflation (Forecast Months)")

    plt.title(f"Inflation: Historical, Forecast, and Realised (From {start_date.strftime('%Y-%m-%d')})")
    plt.xlabel("Date")
    plt.ylabel("Inflation (%)")
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_plot.pdf")
    plt.close()
    print("Saved forecast_plot.pdf")


def plot_inflation_histogram(df: pd.DataFrame):
    """
    Plot histogram and KDE of inflation series.
    Args:
        df (pd.DataFrame): DataFrame with 'inflation' column
    Saves:
        inflation_histogram.pdf
    """
    if 'inflation' not in df.columns:
        print("Inflation column not found for histogram plot.")
        return

    plt.figure(figsize=(6, 4))
    df['inflation'].dropna().plot(kind='hist', bins=30, density=True, alpha=0.7, color='skyblue')
    df['inflation'].dropna().plot(kind='kde', color='darkblue')
    plt.title("Histogram and KDE of Inflation")
    plt.xlabel("Inflation (%)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("inflation_histogram.pdf")
    plt.close()
    print("Saved inflation_histogram.pdf")


def plot_trend_and_slope(inflation: pd.Series, cutoff: pd.Timestamp):
    """
    Plot the smoothed inflation trend and its rolling slope to visualise the detected trend break.

    Args:
        inflation (pd.Series): Monthly inflation series (raw).
        cutoff (pd.Timestamp): Detected trend break date.
    """
    import matplotlib.pyplot as plt

    trend = inflation.rolling(6, min_periods=1).mean()
    slope = trend.diff().rolling(3, min_periods=1).mean()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top panel: raw vs smoothed inflation
    axs[0].plot(inflation.index, inflation, color='lightgrey', label='Inflation')
    axs[0].plot(trend.index, trend, color='blue', label='6-mo Moving Avg')
    axs[0].axvline(cutoff, color='red', linestyle='--', label='Trend Break')
    axs[0].legend()
    axs[0].set_title('Inflation and Smoothed Trend')

    # Bottom panel: slope of the smoothed trend
    axs[1].plot(slope.index, slope, color='purple', label='Slope of Trend')
    axs[1].axhline(-0.2, color='black', linestyle='--', label='Slope Threshold')
    axs[1].axvline(cutoff, color='red', linestyle='--')
    axs[1].legend()
    axs[1].set_title('Slope of 6-mo Trend (3-mo MA)')
    axs[1].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig("trend_break_detection.pdf")
    plt.close()
    print("Saved trend_break_detection.pdf")
