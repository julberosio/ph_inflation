import pandas as pd

# Load BSP Excel file
file_path = "pesodollar.xlsx"
df = pd.read_excel(file_path, sheet_name='monthly', skiprows=5)
df = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3']] # Select relevant columns
df.columns = ['year', 'month', 'usd_php'] # Rename
df = df.dropna(subset=['month', 'usd_php']) # Drop rows with missing month or exchange rate
df['year'] = df['year'].ffill() # Forward-fill year values

# Combine year and month into month_id
df['month_id'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-' + df['month'], format='%Y-%B', errors='coerce').dt.to_period('M')
df = df.dropna(subset=['month_id']) # Drop rows where conversion failed
df_fx = df[['month_id', 'usd_php']]

# Reindex
full_range = pd.period_range('1958-01', '2025-06', freq='M')
df_fx = df_fx.set_index('month_id').reindex(full_range).reset_index().rename(columns={'index': 'month_id'})

# Save to CSV
df_fx.to_csv("forex.csv", index=False)
print("Saved to: forex.csv")
