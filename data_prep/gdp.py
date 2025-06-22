import pandas as pd

# Load PSA Excel file
file_path = "psa_gdp.xlsx"
df_raw = pd.read_excel(file_path, sheet_name=0, header=None)

# Extract and forward-fill sparse year headers
years = df_raw.iloc[35, 1:].ffill()
quarters = df_raw.iloc[36, 1:]

# Generate full quarter label
quarter_labels = [
    f"{int(y)}Q{int(str(q).strip()[-1])}"
    for y, q in zip(years, quarters)
    if pd.notna(y) and pd.notna(q)
]
n_quarters = len(quarter_labels)

# Extract GDP rows
df_data = df_raw.iloc[38:46, 1:n_quarters + 1].copy()
df_data.insert(0, "variable", df_raw.iloc[38:46, 0].values)

# Assign full column names
df_data.columns = ["variable"] + quarter_labels

# Reshape to long format
df_long = df_data.melt(id_vars="variable", var_name="quarter", value_name="value")

# Convert quarter to end-of-quarter month
df_long["month_id"] = pd.PeriodIndex(df_long["quarter"], freq="Q").asfreq("M", how="end")

# Pivot to wide format
df_final = df_long.pivot_table(index="month_id", columns="variable", values="value", aggfunc="first").reset_index()

# Rename final columns
df_final = df_final.rename(columns={
    "01. Household final consumption expenditure": "gdp_household",
    "02. Government final consumption expenditure": "gdp_gov",
    "03. Gross capital formation": "gdp_capital",
    "04. Exports of goods and services": "gdp_exports",
    "05. Less : Imports of goods and services": "gdp_imports",
    "06. Statistical discrepancy": "gdp_se",
    "Gross Domestic Product": "gdp_total"
})

# Save to CSV
df_final.to_csv("gdp.csv", index=False)
print("Saved to: gdp.csv")
