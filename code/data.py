import pandas as pd

def load_and_merge(monthly_path: str, quarterly_path: str) -> pd.DataFrame:
    """
    Load and merge monthly and quarterly datasets:
      - Reads two CSV files: monthly data and quarterly data
      - Parse ‘month_id’ as datetime index
      - Merge monthly and quarterly datasets on month-end dates
      - Preserve NaNs for Kalman handling of unbalanced data

    Args:
        monthly_path (str): path to monthly CSV (inflation, forex, ppi, typhoon, m3, loans, remittances, rrp, unemp)
        quarterly_path (str): path to quarterly CSV (gdp, imports)

    Returns:
        DataFrame with merged monthly and quarterly data, indexed at month-end frequency
    """
    # Read monthly data
    monthly = pd.read_csv(
        monthly_path, # File path
        parse_dates=['month_id'], # Convert to datetime object
        index_col='month_id', # Set as index
        thousands=',' # For parsing numbers with commas
    )
    monthly.index.name = 'Date'

    # Read quarterly data
    quarterly = pd.read_csv(
        quarterly_path,
        parse_dates=['month_id'],
        index_col='month_id',
        thousands=','
    )
    quarterly.index.name = 'Date' # Rename index

    # Merge: monthly as base, join quarterly, then align to month-end
    df = monthly.join(quarterly, how='outer').sort_index()

    # Convert index from PeriodIndex (monthly) to timestamp at month-end
    df.index = df.index.to_period('M').to_timestamp('M')

    return df
