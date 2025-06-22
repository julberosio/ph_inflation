import yfinance as yf
import pandas as pd

# Tickers
tickers = {
    'ZR=F': 'rice',
    'ZW=F': 'wheat',
    'BZ=F': 'brent_crude'
}

# Time period
start_date = '1958-01-01'
end_date = '2025-06-30'

data = pd.DataFrame()

for symbol, name in tickers.items():
    print(f"Fetching {name}...")
    df = yf.download(symbol, start=start_date, end=end_date, interval='1mo', auto_adjust=True)
    df = df[['Close']].rename(columns={'Close': name})
    data = pd.concat([data, df], axis=1)

# Format
data.index = data.index.to_period('M')
data = data.loc[~data.index.duplicated()]
data = data.reset_index().rename(columns={'index': 'month_id'})

# Save to CSV
data.to_csv("commodities.csv", index=False)

print("Saved to: commodities.csv")
