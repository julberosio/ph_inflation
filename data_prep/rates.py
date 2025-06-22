import pandas as pd

# Load file and correct sheet
file_path = "sdir.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="MONTHLY", header=None)

# Slice rows starting from the data block (around row 9)
df = df_raw.iloc[8:].copy()

# Extract year and month (year appears only once per year, needs forward fill)
df['raw_date'] = df.iloc[:, 1]
df['year'] = df['raw_date'].where(df['raw_date'].astype(str).str.match(r'^\d{4}$'))
df['year'] = df['year'].ffill()
df['month'] = df['raw_date'].where(df['raw_date'].astype(str).str.match(r'^[A-Za-z]{3}$'))

# Drop rows without valid month names
df = df[df['month'].notna()]

# Combine year and month to build full date
df['date_str'] = df['month'].astype(str) + ' ' + df['year'].astype(str)
df['month_id'] = pd.to_datetime(df['date_str'], errors='coerce').dt.to_period('M')

# Extract the relevant columns (rename clearly)
df_rates = df[['month_id', 24, 44, 68]].copy()
df_rates.columns = ['month_id', 'sda', 'rrp', 'ibcl']

# Clean up weird symbols and force numeric
for col in ['sda', 'rrp', 'ibcl']:
    df_rates[col] = pd.to_numeric(df_rates[col], errors='coerce')

# Reindex to full range and keep NaNs where data is missing
start = df_rates['month_id'].min()
end = df_rates['month_id'].max()
full_range = pd.period_range(start=start, end=end, freq='M')
df_rates = df_rates.set_index('month_id').reindex(full_range).reset_index().rename(columns={'index': 'month_id'})

df_rates = df_rates.ffill()

# Save
df_rates.to_csv("rates.csv", index=False)
print("Saved to: rates.csv")
