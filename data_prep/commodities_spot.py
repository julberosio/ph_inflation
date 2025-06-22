import pandas as pd

# Load monthly data (World Bank Pink Sheet)
url_monthly = "https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx"
wb_mon = pd.read_excel(url_monthly, sheet_name="Monthly Prices", skiprows=4)

# Select columns (rice, wheat, crude oil)
wb_mon = wb_mon[['Unnamed: 0', 'Rice, Thai 5% ', 'Wheat, US HRW', 'Crude oil, Brent']]
wb_mon.columns = ['Date', 'rice', 'wheat', 'brent_crude']
wb_mon['month_id'] = pd.to_datetime(wb_mon['Date'], format='%YM%m').dt.to_period('M')
wb_mon = wb_mon[['month_id', 'rice', 'wheat', 'brent_crude']]

# Reindex to entire time period and leave missing values as NaN
full_range = pd.period_range('1958-01', '2025-06', freq='M')
df = wb_mon.set_index('month_id').reindex(full_range).reset_index().rename(columns={'index': 'month_id'})

# Save as CSV
df.to_csv("commodities_spot.csv", index=False)
print("Saved to: commodities_spot.csv")
