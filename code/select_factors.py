import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def bai_ng_criteria(X, max_k=8, col_thresh=0.1):
    """
    Compute Bai and Ng (2002) information criteria for choosing number of factors.

    Parameters:
    - X: numpy array or DataFrame of shape (T, N) with mean-zero data
    - max_k: maximum number of factors to test
    - col_thresh: maximum allowed share of missing values per column (default: 10%)

    Returns:
    - A dictionary with optimal number of factors under each IC
    """

    # Convert to DataFrame if necessary
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    # Step 1: Drop columns with too many missing values
    valid_cols = X_df.columns[X_df.isna().mean() < col_thresh]
    X_df = X_df[valid_cols]

    if X_df.shape[1] == 0:
        raise ValueError("No variables passed column missing-value filter.")

    # Step 2: Drop remaining rows with any missing values
    X_df = X_df.dropna()

    if X_df.shape[0] == 0:
        raise ValueError("No complete rows after dropping NaNs. Cannot compute PCA.")

    # PCA
    T, N = X_df.shape
    pca = PCA()
    pca.fit(X_df)

    ICp1, ICp2, ICp3 = [], [], []

    for k in range(1, min(max_k, N) + 1):
        X_hat = pca.transform(X_df)[:, :k] @ pca.components_[:k, :]
        residual = X_df.values - X_hat
        sigma2 = np.mean(residual ** 2)

        penalty1 = k * (N + T) / (N * T) * np.log(N * T / (N + T))
        penalty2 = k * np.log(min(N, T)) / min(N, T)
        penalty3 = k * np.log(min(N, T)) / (N * T)

        ICp1.append(np.log(sigma2) + penalty1)
        ICp2.append(np.log(sigma2) + penalty2)
        ICp3.append(np.log(sigma2) + penalty3)

    return {
        "ICp1": np.argmin(ICp1) + 1,
        "ICp2": np.argmin(ICp2) + 1,
        "ICp3": np.argmin(ICp3) + 1,
        # "n_obs": T,
        # "n_vars": N
    }
