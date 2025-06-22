import pandas as pd
import numpy as np
from numpy.linalg import inv, eig

class BayesianFactorModel:
    """
    Bayesian Factor Model for multivariate time series.

    Estimates latent factors explaining covariance among observed variables,
    modelling factors as AR processes with order `ar_order`.

    Attributes:
        data (pd.DataFrame): Observed data matrix (T time periods x N variables).
        n_factors (int): Number of latent factors to estimate.
        ar_order (int): Autoregressive order for latent factors.
        obs (np.ndarray): Observed data values as numpy array.
        T (int): Number of time points.
        N (int): Number of observed variables.
        factors (np.ndarray): Latent factors matrix (T x K factors).
        lam (np.ndarray): Factor loadings matrix (N x K).
        phi (np.ndarray): AR coefficients matrix (K x ar_order).
        factor_var (np.ndarray): Variances of latent factor innovations (K x 1).
    """
    def __init__(self, data: pd.DataFrame, n_factors: int = 1, ar_order: int = 1):
        # Store input data and parameters
        self.data = data
        self.n_factors = n_factors
        self.ar_order = ar_order

        # Convert data to numpy array
        self.obs = data.values
        self.T, self.N = self.obs.shape

        # Initialise latent factors randomly (standard normal) (T x K)
        self.factors = np.random.randn(self.T, self.n_factors)

        # Initialise factor loadings randomly (N x K)
        self.lam = np.random.randn(self.N, self.n_factors)

        # Initialise AR coefficients with zeros (K x ar_order)
        self.phi = np.zeros((self.n_factors, self.ar_order))

         # Initialise latent factor innovation variances to 1 (K x 1)
        self.factor_var = np.ones(self.n_factors)

    def sample_lambda(self):
        """
        Sample factor loadings conditional on current latent factors and observed data
        """
        for i in range(self.N): # For each variable
            y = self.obs[:, i] # Observed values of variable i over time
            valid = ~np.isnan(y) # Mask for non-missing observations
            X = self.factors[valid, :] # Latent factors corresponding to observed data
            Y = y[valid]

            if len(Y) < 3:
                continue # Skip if insufficient data for reliable estimation

            # Posterior covariance matrix V = inverse of (X'X + identity matrix)
            V = inv(X.T @ X + np.eye(self.n_factors))

            # Posterior mean M = V * X'Y
            M = V @ X.T @ Y

            # Sample from multivariate normal with mean M and covariance V
            self.lam[i, :] = M + np.random.multivariate_normal(np.zeros(self.n_factors), V)

    def sample_phi(self):
        """
        Sample AR coefficients for each latent factor with stationarity constraint
        """
        for k in range(self.n_factors): # For each factor
            Y = self.factors[self.ar_order:, k] # Dependent variable: factor values from lag ar_order onwards

            # Design matrix X with lagged factor values for AR(p) regression
            X = np.column_stack([
                self.factors[self.ar_order - i - 1:-i - 1, k]
                for i in range(self.ar_order)
            ])

            # Posterior covariance and mean of AR coefficients
            V = inv(X.T @ X + np.eye(self.ar_order))
            M = V @ X.T @ Y

            accepted = False
            # Sample AR coefficients ensuring stationarity via companion matrix eigenvalues
            while not accepted:
                phi_sample = M + np.random.multivariate_normal(np.zeros(self.ar_order), V)
                comp_matrix = np.vstack([phi_sample, np.eye(self.ar_order - 1, self.ar_order)])
                if max(abs(eig(comp_matrix)[0])) < 1:
                    self.phi[k, :] = phi_sample
                    accepted = True

    def sample_factors(self):
        """
        Sample latent factors conditional on AR coefficients and variances.
        """
        for t in range(self.ar_order, self.T): # From time ar_order to T
            for k in range(self.n_factors):
                # Compute mean using AR coefficients and lagged factors (reversed order)
                mean = np.dot(self.phi[k, :], self.factors[t - self.ar_order:t, k][::-1])

                # Sample factor value from normal distribution with computed mean and variance
                self.factors[t, k] = np.random.normal(loc=mean, scale=np.sqrt(self.factor_var[k]))

    def run_gibbs(self, n_iter=100):
        """
        Gibbs sampler: iterate sampling lambda, phi, and factors.
        """
        for _ in range(n_iter):
            self.sample_lambda()
            self.sample_phi()
            self.sample_factors()

    def get_latent_factors(self):
        """
        Return latent factors as DataFrame with original index and factor labels.
        """
        return pd.DataFrame(self.factors, index=self.data.index,
                            columns=[f'factor_{i+1}' for i in range(self.n_factors)])

    def get_transition_matrix(self):
        """
        Return copy of AR coefficient matrix phi.
        """
        if self.ar_order == 1:
            return np.diagflat(self.phi)
        return self.phi.copy()
