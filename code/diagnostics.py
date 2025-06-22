import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

from factors import BayesianFactorModel
from ms_regression import fit_markov_switching_regression
from forecast import forecast_inflation


def forecast_accuracy_metrics(forecasts: pd.DataFrame, realized: pd.Series):
    """
    Compute forecast accuracy metrics comparing predicted and realised inflation.

    Args:
        forecasts (pd.DataFrame): Forecasted inflation indexed by date.
        realized (pd.Series): Actual inflation series indexed by date.

    Returns:
        dict: Dictionary containing RMSE, MAE, and MAPE of forecasts.
    """

    # Align and drop NaNs in either series
    common_idx = forecasts.index.intersection(realized.index)
    y_true = realized.loc[common_idx]
    y_pred = forecasts.loc[common_idx, forecasts.columns[0]]

    # Drop any NaNs in either series
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].values
    y_pred = y_pred[valid].values

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"Forecast Accuracy Metrics:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.2f}%")
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def regime_persistence(post: dict):
    """
    Print estimated regime transition matrix and regime persistence probabilities.

    Args:
        post (dict): Posterior dictionary from MS regression with key 'P' for transition matrix.
    """

    P = post['P']
    print("Estimated Regime Transition Matrix (Posterior Mean):")
    print(P)

    persistence = np.diag(P)
    for i, p in enumerate(persistence):
        print(f"Regime {i} persistence (prob stay same): {p:.3f}")


def regime_transition_counts(post: dict):
    """
    Compute and print counts of transitions between regimes over sampled state sequences.

    Args:
        post (dict): Posterior dictionary expected to have 's_arr' key (n_draws x T time periods).
    """

    s_arr = np.array(post.get('s_arr'))
    if s_arr is None:
        print("No regime state draws stored for transition counts.")
        return

    counts = np.zeros((post['P'].shape[0], post['P'].shape[0]))
    for s in s_arr:
        for t in range(1, len(s)):
            counts[s[t-1], s[t]] += 1

    print("Regime transition counts (sum over all Gibbs samples):")
    print(counts)


def cross_validation(
    df: pd.DataFrame,
    window_size: int = 90,
    forecast_horizon: int = 3,
    n_factors: int = 1,
    ar_order: int = 1,
    k_regimes: int = 2,
    gibbs_iters: int = 200,
    progress_every: int = 10,
    save_path: str = "cv_results.csv"
):
    """
    Perform rolling-window cross-validation of the full pipeline for forecasting inflation.

    Args:
        df (pd.DataFrame): Full dataset with 'inflation' and predictors.
        window_size (int): Size of training window in months.
        forecast_horizon (int): Forecast horizon in months.
        n_factors (int): Number of latent factors to extract.
        ar_order (int): AR order for latent factor dynamics.
        k_regimes (int): Number of Markov regimes.
        gibbs_iters (int): Gibbs sampler iterations.
        progress_every (int): Print progress every N folds.
        save_path (str): Where to save the CV results CSV.

    Returns:
        pd.DataFrame: Cross-validation results per fold.
    """

    print("Running cross validation...")
    results = []
    n_obs = len(df)
    total_folds = n_obs - window_size - forecast_horizon + 1

    for start_idx in range(total_folds):
        train = df.iloc[start_idx:start_idx + window_size]
        test = df.iloc[start_idx + window_size:start_idx + window_size + forecast_horizon]

        if test['inflation'].isna().all():
            print(f"Skipping fold starting at {train.index[0]}: all test inflation missing.")
            continue

        df_t = train.copy()
        predictors = df_t.columns.difference(['inflation'])
        df_t[predictors] = (df_t[predictors] - df_t[predictors].mean()) / df_t[predictors].std()
        df_t = df_t.dropna()

        if df_t.empty or len(df_t) < 10:
            print(f"Skipping fold starting at {train.index[0]}: insufficient data after standardisation.")
            continue

        X_train = df_t.drop(columns=['inflation'])

        bfm = BayesianFactorModel(X_train, n_factors=n_factors, ar_order=ar_order)
        bfm.run_gibbs(n_iter=gibbs_iters)
        factor_df = bfm.get_latent_factors()
        phi = bfm.get_transition_matrix()
        factor_var = bfm.factor_var  

        # Fit MS regression with full factor set
        post = fit_markov_switching_regression(factor_df, df_t['inflation'], k_regimes=k_regimes)
        cutoff = factor_df.index[-1]

        # Forecast using all latent factors
        forecasts = forecast_inflation(factor_df, phi, factor_var, post, cutoff=cutoff, horizon=forecast_horizon)

        y_true = test['inflation'].reindex(forecasts.index)
        valid_idx = y_true.dropna().index.intersection(forecasts.dropna().index)
        y_true = y_true.loc[valid_idx]
        y_pred = forecasts.loc[valid_idx, 'inflation_forecast']

        if len(y_true) == 0:
            print(f"Skipping fold starting at {train.index[0]}: no valid forecast/true overlap.")
            continue

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            'train_start': train.index[0],
            'train_end': train.index[-1],
            'test_start': test.index[0],
            'test_end': test.index[-1],
            'rmse': rmse,
            'mae': mae
        })

        if start_idx % progress_every == 0:
            print(f"Processed fold {start_idx + 1}/{total_folds}...")

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print(f"Saved cross-validation results to {save_path}")

    print("\nCross-Validation Summary Statistics:")
    for metric in ['rmse', 'mae']:
        print(f"{metric.upper()}: mean={results_df[metric].mean():.4f}, std={results_df[metric].std():.4f}, min={results_df[metric].min():.4f}, max={results_df[metric].max():.4f}")

    return results_df


def plot_cv_errors(cv_results: pd.DataFrame):
    """
    Plot RMSE and MAE over time from cross-validation results.

    Args:
        cv_results (pd.DataFrame): DataFrame containing CV errors and dates.

    Saves:
        cv_errors_over_time.pdf
    """

    plt.figure(figsize=(12, 5))

    # Plot RMSE and MAE over test start dates
    plt.plot(cv_results['test_start'], cv_results['rmse'], label='RMSE', marker='o')
    plt.plot(cv_results['test_start'], cv_results['mae'], label='MAE', marker='x')

    plt.title('Cross-Validation Errors Over Time')
    plt.xlabel('Test Period Start Date')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cv_errors_over_time.pdf')
    plt.close()
    print("Saved cv_errors_over_time.pdf")


def plot_cv_error_histograms(cv_results: pd.DataFrame):
    """
    Plot histograms of RMSE and MAE from cross-validation results.

    Args:
        cv_results (pd.DataFrame): DataFrame containing CV error metrics.

    Saves:
        cv_error_histograms.pdf
    """

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(cv_results['rmse'], bins=20, alpha=0.7, color='blue')
    plt.title('Histogram of RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(cv_results['mae'], bins=20, alpha=0.7, color='green')
    plt.title('Histogram of MAE')
    plt.xlabel('MAE')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('cv_error_histograms.pdf')
    plt.close()
    print("Saved cv_error_histograms.pdf")
