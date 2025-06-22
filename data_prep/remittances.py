import pandas as pd

# Load file, no headers
file_path = "ofwp.xls"
df_raw = pd.read_excel(file_path, header=None)

# Slice data starting from row 6 where actual data begins (0-based index)
df = df_raw.iloc[5:, [0, 6]].copy()
df.columns = ['raw_date', 'remittances']

# Drop rows without dates
df = df[df['raw_date'].notna()]

# Try to parse month_id safely
df['month_id'] = pd.to_datetime(df['raw_date'], errors='coerce').dt.to_period('M')
df = df[df['month_id'].notna()]  # keep only rows with valid dates

# Force numeric conversion on remittance values
df['remittances'] = pd.to_numeric(df['remittances'], errors='coerce')

# Build full range
start = df['month_id'].min()
end = df['month_id'].max()
full_range = pd.period_range(start=start, end=end, freq='M')

# Reindex and keep NaNs where data is missing
df = df.set_index('month_id').reindex(full_range).reset_index().rename(columns={'index': 'month_id'})

# Save to CSV
df.to_csv("remittances.csv", index=False)
print("Saved to: remittances.csv")
