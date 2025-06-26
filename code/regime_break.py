import pandas as pd
import ruptures as rpt


def detect_recent_break(
    inflation: pd.Series,
    window_smooth: int = 6,
    window_slope: int = 3,
    penalty: int = 10,
    recent_cutoff: str = "2021-01-01") -> pd.Timestamp:
    """
    Detect recent structural break in smoothed inflation trend slope
    using ruptures (Bai-Perron-like method) with recency filter.

    Args:
        inflation (pd.Series): Monthly inflation series indexed by date.
        window_smooth (int): Window for smoothing inflation trend.
        window_slope (int): Window for computing trend slope.
        penalty (int): Penalty parameter for ruptures.
        recent_cutoff (str): ISO date string for recency filter.

    Returns:
        pd.Timestamp or None: Detected trend break date (if any)
    """
    # Step 1: Smooth inflation and compute slope
    inf_smoothed = inflation.rolling(window=window_smooth, min_periods=1).mean()
    slope = inf_smoothed.diff().rolling(window=window_slope).mean().dropna()

    # Step 2: Apply ruptures to slope series
    signal = inflation.dropna().values.reshape(-1, 1)
    model = rpt.Pelt(model="l2").fit(signal)
    breaks = model.predict(pen=penalty)

    # Step 3: Filter breaks by recency
    break_dates = slope.index[breaks[:-1]]  # exclude end split
    recent_cutoff = pd.Timestamp(recent_cutoff)
    recent_breaks = [d for d in break_dates if d >= recent_cutoff]

    print("All break dates:", break_dates)

    return recent_breaks[-2] if recent_breaks else None
