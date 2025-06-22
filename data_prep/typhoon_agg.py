import pandas as pd

# Load data
df = pd.read_csv("par_typhoons.csv")
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['month_id'] = df['ISO_TIME'].dt.to_period('M').astype(str)

# Drop rows without wind speed
df = df.dropna(subset=['TOK_WIND', 'ISO_TIME'])

# Get max wind per storm per month
df_storm_month = df.groupby(['SID', 'month_id'], as_index=False)['TOK_WIND'].max()

# Apply the exposure formula
# Threshold minimum is 33 knots, max is 152.3
df_storm_month['exposure'] = df_storm_month['TOK_WIND'].apply(
    lambda w: ((w - 33) / (152.3 - 33)) ** 2 if w >= 33 else 0
)

# Sum exposures per month
df_monthly = df_storm_month.groupby('month_id', as_index=False)['exposure'].sum()
df_monthly.rename(columns={'exposure': 'exposure_index'}, inplace=True)

# Fill in missing months
month_range = pd.date_range(start="1958-01", end="2025-06", freq='MS')
df_all_months = pd.DataFrame({'month_id': month_range.to_period('M').astype(str)})

df_final = df_all_months.merge(df_monthly, on='month_id', how='left')
df_final['exposure_index'] = df_final['exposure_index'].fillna(0)

# Save to CSV
df_final.to_csv("ph_typhoon_exposure.csv", index=False)
print("Saved to: ph_typhoon_exposure.csv")
