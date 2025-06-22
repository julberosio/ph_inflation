import numpy as np
import pandas as pd
from numpy.linalg import inv
from utils import forward_backward
import matplotlib.pyplot as plt


def gibbs_ms_regression(
    factor: np.ndarray,
    inflation: np.ndarray,
    n_regimes: int = 2,
    n_iter: int = 1200,
    burn: int = 200
) -> dict:
    """
    Gibbs sampler for Bayesian Markov-Switching regression with multivariate latent factors.

    Args:
        factor (np.ndarray): Latent factor matrix of shape (T time periods x K factors).
        inflation (np.ndarray): Inflation time series of length T.
        n_regimes (int): Number of Markov regimes (M).
        n_iter (int): Total number of Gibbs iterations.
        burn (int): Number of initial iterations to discard as burn-in.

    Returns:
        dict: Posterior summaries including:
            - 'alpha': regime intercepts (M,)
            - 'beta': regime-specific slopes (M x K)
            - 'sigma2': regime-specific error variances (M,)
            - 'P': regime transition matrix (M x M)
            - 's_prob_matrix': smoothed regime probabilities (T x M)
            - 'training_index': time index of training sample

    Procedure:
        1. Initialise parameters:
           - Intercepts (alpha), slopes (beta), variances (sigma2), regime states (s), and transition matrix (P).

        2. For each iteration:
            a. Compute log-likelihood of inflation under each regime using current parameters.
            b. Update transition matrix P using factor-dependent weights based on deviation from factor mean.
            c. Sample regime sequence s using the forward–backward algorithm.
            d. For each regime j:
                i. Extract inflation and factor values for observations in regime j.
                ii. Apply time-weighted Bayesian regression with Gaussian priors to sample (alpha_j, beta_j).
                iii. Update variance sigma2_j using inverse-gamma posterior from residuals.
            e. Update P by counting transitions and drawing from a Dirichlet posterior.

        3. After burn-in, store parameter draws and compute posterior means.
        4. Estimate smoothed regime probabilities as the average frequency of regime states across iterations.
    """

    T, K = factor.shape
    alpha = np.full(n_regimes, 2.0) # Initialise intercepts
    beta = np.full((n_regimes, K), 0.5) # Initialise slopes (one per factor)
    sigma2 = np.ones(n_regimes) # Regime-specific error variances
    P = np.full((n_regimes, n_regimes), 1.0 / n_regimes)  # Initial transition matrix
    s = np.random.randint(n_regimes, size=T) # Random initial state sequence

    # Store posterior draws
    draws = {"alpha": [], "beta": [], "sigma2": [], "P": [], "s": []}

    for it in range(n_iter):
        # Step 1: Compute log-likelihood matrix (T x M)
        ll = np.zeros((T, n_regimes))
        for j in range(n_regimes):
            mu_j = alpha[j] + factor @ beta[j]  # Predicted mean under regime j
            ll[:, j] = -0.5 * (np.log(2 * np.pi * sigma2[j]) +
                              ((inflation - mu_j) ** 2) / sigma2[j])

        # Step 2: Factor-weighted transition probability matrix
        f_dev = np.abs(factor - factor.mean(axis=0)) / (factor.std(axis=0) + 1e-5)
        f_weight = np.clip(1.0 - f_dev.mean(axis=1), 0.1, 0.9)  # T weights ∈ [0.1, 0.9]

        # Construct time-averaged matrix P
        p_base = np.full((n_regimes, n_regimes), 1.0 / n_regimes)
        for i in range(n_regimes):
            for j in range(n_regimes):
                p_base[i, j] = f_weight.mean() if i != j else 1.0 - f_weight.mean()
        P = np.array([np.random.dirichlet(p_base[i] + 1e-2) for i in range(n_regimes)])

        # Step 3: Sample regime states using forward-backward algorithm
        s = forward_backward(ll, P)

        # Step 4: Regime-wise regression updates
        for j in range(n_regimes):
            idx = (s == j)
            if idx.sum() < K + 2:
                continue  # Skip regime if not enough data

            yj = inflation[idx]
            Fj = factor[idx]
            Xj = np.column_stack([np.ones_like(yj), Fj])  # Add intercept column

            # Prior setup
            prior_mean = np.concatenate([[2.0], np.full(K, 0.5)])
            prior_var = np.full(K + 1, 0.25)
            V0 = np.diag(1 / prior_var)
            M0 = prior_mean

            # Weighted least squares (higher weight for recent obs)
            weights = np.linspace(0.5, 1.5, num=len(yj))
            W = np.diag(weights)

            # Posterior moments
            Vn = inv(V0 + Xj.T @ W @ Xj / sigma2[j])
            Mn = Vn @ (V0 @ M0 + Xj.T @ W @ yj / sigma2[j])

            # Sample from posterior
            ab = np.random.multivariate_normal(Mn, Vn)
            alpha[j], beta[j] = ab[0], ab[1:]

            # Update residual variance
            resid = weights * (yj - Xj @ ab)
            shape = 2.0 + len(resid) / 2
            scale = 1.0 + 0.5 * (resid @ resid)
            sigma2[j] = max(1e-2, scale / np.random.chisquare(2 * shape))

        # Step 5: Update transition matrix using state counts
        counts = np.zeros((n_regimes, n_regimes))
        for t in range(1, T):
            counts[s[t - 1], s[t]] += 1
        for i in range(n_regimes):
            P[i] = np.random.dirichlet(counts[i] + 1)

        # Store draws
        if it >= burn:
            draws["alpha"].append(alpha.copy())
            draws["beta"].append(beta.copy())
            draws["sigma2"].append(sigma2.copy())
            draws["P"].append(P.copy())
            draws["s"].append(s.copy())

    # Posterior means (except regime paths)
    post = {key: np.mean(val, axis=0) for key, val in draws.items() if key != "s"}
    s_arr = np.stack(draws["s"])
    post["s_prob_matrix"] = np.vstack([
        (s_arr == j).mean(axis=0) for j in range(n_regimes)
    ]).T
    post["training_index"] = np.arange(T)

    return post


def plot_trend_break(inflation_series: pd.Series, trend_break_date):
    """
    Plot inflation series with vertical line at detected trend break date.

    Args:
        inflation_series (pd.Series): Inflation time series indexed by date.
        trend_break_date (pd.Timestamp): Date of trend break to highlight.

    Saves:
        inflation_trend_break.pdf
    """

    plt.figure(figsize=(8, 3))
    plt.plot(inflation_series.index.to_numpy(), inflation_series.to_numpy(), label="Inflation", linewidth=1.5)
    if trend_break_date:
        plt.axvline(trend_break_date, color='red', linestyle='--', label='Trend Break')
    plt.title("Inflation Trend Break Used for Regime Model")
    plt.xlabel("Date")
    plt.ylabel("Inflation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("inflation_trend_break.pdf")
    plt.close()
    print("Saved inflation_trend_break.pdf")


def fit_markov_switching_regression(
    factor_df: pd.DataFrame,
    inflation: pd.Series,
    k_regimes: int = 2
) -> dict:
    """
    Detect structural break in inflation and estimate regime-switching model.

    Args:
        factor_df (pd.DataFrame): Latent factors (T x K).
        inflation (pd.Series): Inflation series (T,).
        k_regimes (int): Number of regimes.

    Returns:
        dict: Posterior estimates and smoothed regime probabilities.
    """

    # Detect trend break
    inf_smoothed = inflation.rolling(window=6, min_periods=1).mean()
    slope = inf_smoothed.diff().rolling(window=3).mean()
    trend_break = slope[slope < -0.19].last_valid_index()
    # print(f"Detected trend break: {trend_break}")

    # Restrict to post-break data
    if trend_break is not None:
        inflation = inflation[inflation.index >= trend_break]
        factor_df = factor_df.loc[inflation.index]

    # Drop missing
    df = pd.concat([inflation, factor_df], axis=1).dropna()
    f = df[factor_df.columns].values
    y = df[inflation.name].values

    # Run MS regression
    post = gibbs_ms_regression(f, y, n_regimes=k_regimes)
    post["cutoff_index"] = df.index
    return post
